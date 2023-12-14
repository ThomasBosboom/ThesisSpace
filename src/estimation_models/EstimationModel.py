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
from dynamic_models.low_fidelity import low_fidelity
from dynamic_models.high_fidelity.point_mass import *
from dynamic_models.high_fidelity.point_mass_srp import *
from dynamic_models.high_fidelity.spherical_harmonics import *
from dynamic_models.high_fidelity.spherical_harmonics_srp import *

class EstimationModel:

    def __init__(self, parent_instance):

        self.parent_instance = parent_instance


    def set_observation_settings(self):

        # Define the uplink link ends for one-way observable
        link_ends = {observation.transmitter: observation.body_origin_link_end_id(self.parent_instance.name_ELO),
                     observation.receiver: observation.body_origin_link_end_id(self.parent_instance.name_LPO)
        }
        self.link_definition = observation.LinkDefinition(link_ends)

        # Define settings for light-time calculations
        light_time_correction_settings = observation.first_order_relativistic_light_time_correction(self.parent_instance.bodies_to_create)

        # Define settings for range and doppler bias
        self.bias_level_range = 1.0E1
        range_bias_settings = observation.absolute_bias([self.bias_level_range])

        self.bias_level_doppler = 1.0E-3
        doppler_bias_settings = observation.absolute_bias([self.bias_level_doppler])

        # Create observation settings for each link/observable
        self.observation_settings_list = [observation.one_way_range(self.link_definition,
                                                                    light_time_correction_settings = [light_time_correction_settings],
                                                                    bias_settings = range_bias_settings),
                                          observation.one_way_doppler_instantaneous(self.link_definition,
                                                                    light_time_correction_settings = [light_time_correction_settings],
                                                                    bias_settings = doppler_bias_settings)
        ]

        # Define observation simulation times for each link
        self.observation_times_range = np.linspace(self.parent_instance.simulation_start_epoch+1000, self.parent_instance.simulation_end_epoch-1000, 1000)
        self.observation_times_doppler = np.linspace(self.parent_instance.simulation_start_epoch+1000, self.parent_instance.simulation_end_epoch-1000, 2000)

        # print(self.parent_instance.simulation_start_epoch, self.parent_instance.simulation_end_epoch)

        self.observation_simulation_settings = [observation.tabulated_simulation_settings(observation.one_way_range_type,
                                                                                          self.link_definition,
                                                                                          self.observation_times_range),
                                                observation.tabulated_simulation_settings(observation.one_way_instantaneous_doppler_type,
                                                                                          self.link_definition,
                                                                                          self.observation_times_doppler)
        ]

        # Add noise levels to observations
        self.noise_level_range = 2.98 # 102.44
        observation.add_gaussian_noise_to_observable(
            self.observation_simulation_settings,
            self.noise_level_range,
            observation.one_way_range_type
        )

        self.noise_level_doppler = 0.00097
        observation.add_gaussian_noise_to_observable(
            self.observation_simulation_settings,
            self.noise_level_doppler,
            observation.one_way_instantaneous_doppler_type
        )


    def set_viability_settings(self):

        self.set_observation_settings()

        # Create viability settings
        viability_setting_list = [observation.body_occultation_viability([self.parent_instance.name_ELO, self.parent_instance.name_LPO], self.parent_instance.name_secondary)]

        observation.add_viability_check_to_all(
            self.observation_simulation_settings,
            viability_setting_list
        )


    def set_simulated_observations(self):

        self.set_viability_settings()
        self.parent_instance.set_propagator_settings()

        # Setup parameters settings to propagate the state transition matrix
        self.parameter_settings = estimation_setup.parameter.initial_states(self.parent_instance.propagator_settings, self.parent_instance.bodies)

        # Add estimated parameters to the sensitivity matrix that will be propagated
        # self.parameter_settings.append(estimation_setup.parameter.gravitational_parameter(self.parent_instance.name_primary))
        # self.parameter_settings.append(estimation_setup.parameter.gravitational_parameter(self.parent_instance.name_secondary))
        # self.parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(self.link_definition, observation.one_way_range_type))
        # self.parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(self.link_definition, observation.one_way_instantaneous_doppler_type))

        # Depending on the dynamic model
        # self.parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient(self.parent_instance.name_ELO))
        # self.parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient(self.parent_instance.name_LPO))

        # Create the parameters that will be estimated
        self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.parent_instance.bodies)
        self.original_parameter_vector = self.parameters_to_estimate.parameter_vector


    # Create the estimator
        self.estimator = numerical_simulation.Estimator(
            self.parent_instance.bodies,
            self.parameters_to_estimate,
            self.observation_settings_list,
            self.parent_instance.propagator_settings)

        # Simulate required observations
        self.simulated_observations = estimation.simulate_observations(
            self.observation_simulation_settings,
            self.estimator.observation_simulators,
            self.parent_instance.bodies)

        self.single_observation_set_lists = [self.simulated_observations.get_single_link_and_type_observations(observation.one_way_range_type, self.link_definition),
                                             self.simulated_observations.get_single_link_and_type_observations(observation.one_way_instantaneous_doppler_type, self.link_definition),
                                            ]

        self.observations = dict()
        for single_observation_set_list in self.single_observation_set_lists:
            for single_observation_set in single_observation_set_list:
                self.observations[single_observation_set.observable_type] = dict(zip(single_observation_set.observation_times, single_observation_set.concatenated_observations))

        # ax = plt.figure()
        # plt.plot(self.observations[observation.one_way_instantaneous_doppler_type].keys(), self.observations[observation.one_way_instantaneous_doppler_type].values(), color="red", marker="o")
        # plt.show()

        return self.observations


    def get_estimation_results(self):

        self.set_simulated_observations()

        print('Running propagation...')
        with util.redirect_std():
            estimator = numerical_simulation.Estimator(self.parent_instance.bodies,
                                                       self.parameters_to_estimate,
                                                       self.observation_settings_list,
                                                       self.parent_instance.propagator_settings)

        # Create input object for the estimation
        apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
        convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=2)
        estimation_input = estimation.EstimationInput(observations_and_times=self.simulated_observations,
                                                      convergence_checker=convergence_checker,
                                                      inverse_apriori_covariance=np.linalg.inv(apriori_covariance)
                                                      )
        # Set methodological options
        estimation_input.define_estimation_settings(save_state_history_per_iteration=False)

        # Define weighting of the observations in the inversion
        weights_per_observable = {estimation_setup.observation.one_way_range_type: self.noise_level_range**-2,
                                  estimation_setup.observation.one_way_instantaneous_doppler_type: self.noise_level_doppler**-2}
        estimation_input.set_constant_weight_per_observable(weights_per_observable)

        # Run the estimation
        with util.redirect_std(redirect_out=False):
            estimation_output = estimator.perform_estimation(estimation_input)
        initial_states_updated = self.parameters_to_estimate.parameter_vector

        # Propagate formal errors and covariance over the course of estimation window
        output_times = np.linspace(self.parent_instance.simulation_start_epoch, self.parent_instance.simulation_end_epoch, 1000)

        propagated_formal_errors = estimation.propagate_formal_errors_split_output(
            initial_covariance=apriori_covariance,
            state_transition_interface=estimator.state_transition_interface,
            output_times=output_times)

        propagated_covariance = estimation.propagate_covariance_split_output(
            initial_covariance=apriori_covariance,
            state_transition_interface=estimator.state_transition_interface,
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

        weighted_design_matrix_history_doppler = np.stack([estimation_output.weighted_design_matrix[len_range:, :]], axis=1)
        covariance = np.empty((len_doppler, len_parameters, len_parameters))
        information_matrix = covariance.copy()
        for index, weighted_design_matrix in enumerate(weighted_design_matrix_history_doppler):
            information_matrix[index] = np.dot(weighted_design_matrix.T, weighted_design_matrix) + np.linalg.inv(apriori_covariance)
            covariance[index] = np.linalg.inv(information_matrix[index])
        covariance_history[observation.one_way_instantaneous_doppler_type] = dict(zip(self.observation_times_doppler, covariance))
        information_matrix_history[observation.one_way_instantaneous_doppler_type] = dict(zip(self.observation_times_doppler, information_matrix))

        return initial_states_updated, \
               estimation_output.correlations, estimation_output.covariance, \
               estimation_output.inverse_covariance, \
               propagated_formal_errors, propagated_covariance, \
               covariance_history, information_matrix_history



    def get_propagated_orbit_from_estimator(self):

        self.set_simulated_observations()

        # # Extract the simulation results
        # self.epochs                          = np.vstack(list(self.estimator.variational_solver.state_history.keys()))
        # self.state_history                   = np.vstack(list(self.estimator.variational_solver.state_history.values()))
        # self.dependent_variables_history     = np.vstack(list(self.estimator.variational_solver.dynamics_simulator.dependent_variable_history.values()))
        # self.state_transition_matrix_history = np.vstack(list(self.estimator.variational_solver.state_transition_matrix_history.values())).reshape((np.shape(self.state_history)[0], np.shape(self.state_history)[1], np.shape(self.state_history)[1]))


        # print(np.shape(self.epochs), np.shape(self.dependent_variables_history))
        return self.estimator.variational_solver, self.estimator.variational_solver.dynamics_simulator


# model = high_fidelity_point_mass_srp_01.HighFidelityDynamicModel(60390, 28)
# # model = LowFidelityDynamicModel.LowFidelityDynamicModel(60390, 14)
# estimation_model = EstimationModel(model)

# covariance_dict = estimation_model.get_estimation_results()

# print(covariance_dict)

# covariance_history = np.stack(covariance_dict.values())

# print(covariance_history)
# # estimation_result = estimation_model.get_estimation_results()

# # information_matrix_history  = estimation_result[-1]

# ax = plt.figure(figsize=(6.5,6))
# plt.plot(covariance_history)
# plt.plot()
# plt.show()