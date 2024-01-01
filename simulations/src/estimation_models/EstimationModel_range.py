# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

# Tudatpy imports
import tudatpy
from tudatpy import util
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import estimation
from tudatpy.kernel.numerical_simulation import propagation_setup, environment_setup, estimation_setup
from tudatpy.kernel.astro import time_conversion, element_conversion, frame_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Own
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dynamic_models import validation_LUMIO
from dynamic_models.low_fidelity.integration_settings import *
from dynamic_models.high_fidelity.point_mass import *
from dynamic_models.high_fidelity.point_mass_srp import *
from dynamic_models.high_fidelity.spherical_harmonics import *
from dynamic_models.high_fidelity.spherical_harmonics_srp import *

class EstimationModel:

    def __init__(self, dynamic_model):

        # Loading dynamic model
        self.dynamic_model = dynamic_model

        # Defining basis for observations
        self.bias_range = 10.0
        self.bias_doppler = 0.001
        self.noise_range = 2.98
        self.noise_doppler = 0.00097
        self.observation_step_size_range = 600
        self.observation_step_size_doppler = 600
        self.observation_times_range = np.arange(self.dynamic_model.simulation_start_epoch+500, self.dynamic_model.simulation_end_epoch-500, self.observation_step_size_range)
        # self.observation_times_doppler = np.arange(self.dynamic_model.simulation_start_epoch+500, self.dynamic_model.simulation_end_epoch-500, self.observation_step_size_doppler)


    def update_environment_settings(self):

        self.dynamic_model.set_environment_settings()

        # Create default body settings
        self.dynamic_model.body_settings = environment_setup.get_default_body_settings(
                                                self.dynamic_model.bodies_to_create,
                                                self.dynamic_model.global_frame_origin,
                                                self.dynamic_model.global_frame_orientation)

        self.dynamic_model.body_settings.add_empty_settings(self.dynamic_model.name_ELO)
        self.dynamic_model.body_settings.add_empty_settings(self.dynamic_model.name_LPO)

        # Create spacecraft bodies
        for index, body in enumerate(self.dynamic_model.bodies_to_propagate):
            self.dynamic_model.bodies.create_empty_body(body)
            self.dynamic_model.bodies.get_body(body).mass = self.dynamic_model.bodies_mass[index]
            self.dynamic_model.body_settings.get(body).ephemeris_settings = environment_setup.ephemeris.tabulated(
                validation_LUMIO.get_reference_state_history(self.dynamic_model.simulation_start_epoch_MJD,
                                                             self.dynamic_model.propagation_time,
                                                             satellite=body,
                                                             get_dict=True,
                                                             get_full_history=True),
                self.dynamic_model.global_frame_origin,
                self.dynamic_model.global_frame_orientation)

        # Update environment with reference states as ephermeris settings
        self.dynamic_model.bodies = environment_setup.create_system_of_bodies(self.dynamic_model.body_settings)




    def set_observation_settings(self):

        self.update_environment_settings()

        # Define the uplink link ends for one-way observable
        link_ends = {observation.transmitter: observation.body_origin_link_end_id(self.dynamic_model.name_ELO),
                     observation.receiver: observation.body_origin_link_end_id(self.dynamic_model.name_LPO)}
        self.link_definition = observation.LinkDefinition(link_ends)

        # Define settings for light-time calculations
        light_time_correction_settings = observation.first_order_relativistic_light_time_correction(self.dynamic_model.bodies_to_create)

        # Define settings for range and doppler bias
        range_bias_settings = observation.absolute_bias([self.bias_range])
        doppler_bias_settings = observation.absolute_bias([self.bias_doppler])

        # Create observation settings for each link/observable
        self.observation_settings_list = [observation.one_way_range(self.link_definition,
                                                                    light_time_correction_settings = [light_time_correction_settings],
                                                                    bias_settings = range_bias_settings)]

        # Define observation simulation times for each link
        self.observation_simulation_settings = [observation.tabulated_simulation_settings(observation.one_way_range_type,
                                                                                          self.link_definition,
                                                                                          self.observation_times_range)]

        # Add noise levels to observations
        observation.add_gaussian_noise_to_observable(
            self.observation_simulation_settings,
            self.noise_range,
            observation.one_way_range_type)

        # Create viability settings
        viability_setting_list = [observation.body_occultation_viability([self.dynamic_model.name_ELO, self.dynamic_model.name_LPO], self.dynamic_model.name_secondary)]

        observation.add_viability_check_to_all(
            self.observation_simulation_settings,
            viability_setting_list)


    def set_parameters_to_estimate(self):

        self.set_observation_settings()
        self.dynamic_model.set_propagator_settings()

        # Setup parameters settings to propagate the state transition matrix
        self.parameter_settings = estimation_setup.parameter.initial_states(self.dynamic_model.propagator_settings, self.dynamic_model.bodies)

        # Add estimated parameters to the sensitivity matrix that will be propagated
        # self.parameter_settings.append(estimation_setup.parameter.gravitational_parameter(self.dynamic_model.name_primary))
        # self.parameter_settings.append(estimation_setup.parameter.gravitational_parameter(self.dynamic_model.name_secondary))
        # self.parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(self.link_definition, observation.one_way_range_type))
        # self.parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(self.link_definition, observation.one_way_instantaneous_doppler_type))

        # Depending on the dynamic model
        # self.parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient(self.dynamic_model.name_ELO))
        # self.parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient(self.dynamic_model.name_LPO))

        # Create the parameters that will be estimated
        self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.dynamic_model.bodies)
        self.original_parameter_vector = self.parameters_to_estimate.parameter_vector


    def set_simulated_observations(self):

        self.set_parameters_to_estimate()
        self.dynamic_model.set_propagator_settings()

        # Create the estimator
        self.estimator = numerical_simulation.Estimator(
            self.dynamic_model.bodies,
            self.parameters_to_estimate,
            self.observation_settings_list,
            self.dynamic_model.propagator_settings)

        # Simulate required observations
        self.simulated_observations = estimation.simulate_observations(
            self.observation_simulation_settings,
            self.estimator.observation_simulators,
            self.dynamic_model.bodies)

        self.single_observation_set_lists = [self.simulated_observations.get_single_link_and_type_observations(observation.one_way_range_type, self.link_definition)]


    def get_estimation_results(self, apriori_covariance=np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2, maximum_iterations=2):

        self.set_simulated_observations()

        # Create input object for the estimation
        convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=maximum_iterations)
        estimation_input = estimation.EstimationInput(observations_and_times=self.simulated_observations,
                                                      convergence_checker=convergence_checker,
                                                      inverse_apriori_covariance=np.linalg.inv(apriori_covariance))

        # Set methodological options
        estimation_input.define_estimation_settings(save_state_history_per_iteration=False)

        # Define weighting of the observations in the inversion
        weights_per_observable = {estimation_setup.observation.one_way_range_type: self.noise_range**-2}
        estimation_input.set_constant_weight_per_observable(weights_per_observable)

        # Run the estimation
        with util.redirect_std(redirect_out=True):
            estimation_output = self.estimator.perform_estimation(estimation_input)
        # initial_states_updated = self.parameters_to_estimate.parameter_vector

        # Propagate formal errors and covariance over the course of estimation window
        output_times = np.arange(self.dynamic_model.simulation_start_epoch, self.dynamic_model.simulation_end_epoch, 60)

        propagated_formal_errors = estimation.propagate_formal_errors_split_output(
            initial_covariance=apriori_covariance,
            state_transition_interface=self.estimator.state_transition_interface,
            output_times=output_times)

        propagated_covariance = estimation.propagate_covariance_split_output(
            initial_covariance=apriori_covariance,
            state_transition_interface=self.estimator.state_transition_interface,
            output_times=output_times)

        len_range = len(self.observation_times_range)
        len_doppler = len(self.observation_times_doppler)
        len_parameters = len(self.original_parameter_vector)

        information_matrix_history = dict()
        covariance_history = dict()

        weighted_design_matrix_history_range = np.stack([estimation_output.weighted_design_matrix[:len_range, :]], axis=1)
        covariance = np.empty((len_range, len_parameters, len_parameters))
        information_matrix = covariance.copy()
        for index, weighted_design_matrix in enumerate(weighted_design_matrix_history_range):
            information_matrix[index] = np.dot(weighted_design_matrix.T, weighted_design_matrix) + np.linalg.inv(apriori_covariance)
            covariance[index] = np.linalg.inv(information_matrix[index])
        covariance_history[observation.one_way_range_type] = dict(zip(self.observation_times_range, covariance))
        information_matrix_history[observation.one_way_range_type] = dict(zip(self.observation_times_range, information_matrix))

        self.observations = dict()
        for single_observation_set_list in self.single_observation_set_lists:
            for single_observation_set in single_observation_set_list:
                self.observations[single_observation_set.observable_type] = dict(zip(single_observation_set.observation_times, single_observation_set.concatenated_observations))

        # ax = plt.figure()
        # plt.plot(self.observations[observation.one_way_instantaneous_doppler_type].keys(), self.observations[observation.one_way_instantaneous_doppler_type].values(), color="red", marker="o")
        # plt.show()

        return estimation_output, \
               propagated_formal_errors, \
               propagated_covariance, \
               covariance_history, \
               information_matrix_history, \
               self.observations


    def get_propagated_orbit(self):

        self.set_simulated_observations()

        return self.estimator.variational_solver.dynamics_simulator, self.estimator.variational_solver


model = high_fidelity_point_mass_srp_01.HighFidelityDynamicModel(60390, 28)
# model = LowFidelityDynamicModel.LowFidelityDynamicModel(60390, 14)
estimation_model = EstimationModel(model)

estimation_output = estimation_model.get_estimation_results()[0]

parameter_history = estimation_output.parameter_history
residual_history = estimation_output.residual_history
covariance = estimation_output.covariance
formal_errors = estimation_output.formal_errors
print(parameter_history)
print(residual_history, np.shape(residual_history))
print(covariance)
print(formal_errors)

propagated_formal_errors = estimation_model.get_estimation_results()[1]
observations = estimation_model.get_estimation_results()[-1]
observations_range = observations[observation.one_way_range_type]

# print(propagated_formal_errors[0], np.shape(propagated_formal_errors[0]))
# print(propagated_formal_errors[1], np.shape(propagated_formal_errors[1]))

ax = plt.figure(figsize=(6.5,6))
plt.plot(propagated_formal_errors[0], np.vstack(propagated_formal_errors[1])[:,:6])
plt.plot(observations_range.keys(), observations_range.values())
# plt.plot(observations_doppler.keys(), observations_doppler.values())
plt.show()