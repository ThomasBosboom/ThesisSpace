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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# Own
from dynamic_models import validation_LUMIO
from dynamic_models.full_fidelity import *
from dynamic_models.low_fidelity.three_body_problem import *
from dynamic_models.high_fidelity.point_mass import *
from dynamic_models.high_fidelity.point_mass_srp import *
from dynamic_models.high_fidelity.spherical_harmonics import *
from dynamic_models.high_fidelity.spherical_harmonics_srp import *



class EstimationModel:

    def __init__(self, dynamic_model, observation_model):

        # Loading dynamic model
        self.dynamic_model = dynamic_model
        self.truth_model = observation_model

        # Defining basis for observations
        self.bias_range = 10.0
        self.bias_doppler = 0.001
        self.noise_range = 2.98
        self.noise_doppler = 0.00097
        self.observation_step_size_range = 1000
        self.observation_step_size_doppler = 1000
        self.observation_times_range = np.arange(self.dynamic_model.simulation_start_epoch+500, self.dynamic_model.simulation_end_epoch-500, self.observation_step_size_range)
        self.observation_times_doppler = np.arange(self.dynamic_model.simulation_start_epoch+500, self.dynamic_model.simulation_end_epoch-500, self.observation_step_size_doppler)


    def set_observation_model_settings(self):

        self.dynamic_model.set_environment_settings()

        # Define the uplink link ends for one-way observable
        link_ends = {observation.receiver: observation.body_origin_link_end_id(self.dynamic_model.name_ELO),
                     observation.transmitter: observation.body_origin_link_end_id(self.dynamic_model.name_LPO)}
        self.link_definition = observation.LinkDefinition(link_ends)

        self.link_definition = {
            "LPF": self.link_definition,
            "LUMIO": self.link_definition,
        }

        # Define settings for light-time calculations
        light_time_correction_settings = observation.first_order_relativistic_light_time_correction(self.dynamic_model.bodies_to_create)

        # Define settings for range and doppler bias
        range_bias_settings = observation.absolute_bias([self.bias_range])
        doppler_bias_settings = observation.absolute_bias([self.bias_doppler])

        # Create observation settings for each link/observable
        self.observation_settings_list = list()
        for body in self.link_definition.keys():
            self.observation_settings_list.extend([observation.one_way_range(self.link_definition[body],
                                                                        light_time_correction_settings = [light_time_correction_settings],
                                                                        bias_settings = range_bias_settings),
                                                   observation.one_way_doppler_instantaneous(self.link_definition[body],
                                                                        light_time_correction_settings = [light_time_correction_settings],
                                                                        bias_settings = doppler_bias_settings)])


    def set_observation_simulation_settings(self):

        self.set_observation_model_settings()

        # Define observation simulation times for each link
        self.observation_simulation_settings = list()
        for body in self.link_definition.keys():
            self.observation_simulation_settings.extend([observation.tabulated_simulation_settings(observation.one_way_range_type,
                                                                                                self.link_definition[body],
                                                                                                self.observation_times_range,
                                                                                                reference_link_end_type = observation.transmitter),
                                                         observation.tabulated_simulation_settings(observation.one_way_instantaneous_doppler_type,
                                                                                                self.link_definition[body],
                                                                                                self.observation_times_doppler,
                                                                                                reference_link_end_type = observation.transmitter)])

        # Add noise levels to observations
        observation.add_gaussian_noise_to_observable(
            self.observation_simulation_settings,
            self.noise_range,
            observation.one_way_range_type)

        observation.add_gaussian_noise_to_observable(
            self.observation_simulation_settings,
            self.noise_doppler,
            observation.one_way_instantaneous_doppler_type)

        # Create viability settings
        viability_setting_list = [observation.body_occultation_viability([self.dynamic_model.name_ELO, self.dynamic_model.name_LPO], self.dynamic_model.name_secondary)]

        observation.add_viability_check_to_all(
            self.observation_simulation_settings,
            viability_setting_list)


    def observation_simulators(self):

        self.set_observation_simulation_settings()
        self.truth_model.set_propagator_settings()

        # Create or update the ephemeris of all propagated bodies (here: LPF and LUMIO) to match the propagated results
        self.truth_model.propagator_settings.processing_settings.set_integrated_result = True

        # Run propagation
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(self.truth_model.bodies, self.truth_model.propagator_settings)
        state_history_simulated_observations = dynamics_simulator.state_history

        # Create observation simulators
        self.observation_simulators = estimation_setup.create_observation_simulators(
            self.observation_settings_list, self.truth_model.bodies)


    def set_parameters_to_estimate(self):

        self.observation_simulators()
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
        self.truth_parameters = self.parameters_to_estimate.parameter_vector


    def set_simulated_observations(self):

        self.set_parameters_to_estimate()

        # Get LPF and LUMIO simulated observations as ObservationCollection
        self.simulated_observations = estimation.simulate_observations(
            self.observation_simulation_settings,
            self.observation_simulators,
            self.truth_model.bodies)

        # Get sorted observation sets
        self.sorted_observation_sets = self.simulated_observations.sorted_observation_sets
        # ax = plt.figure()
        # for observable_type, observation_sets in self.sorted_observation_sets.items():
        #     for observation_set in observation_sets.values():
        #         for single_observation_set in observation_set:
                    # print(observable_type, observation_set, single_observation_set)
                    # plt.plot(single_observation_set.observation_times, single_observation_set.concatenated_observations)
        # plt.show()


    def get_estimation_results(self, apriori_covariance=np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2, maximum_iterations=2):

        self.set_simulated_observations()

        # Create the estimator
        self.estimator = numerical_simulation.Estimator(
            self.dynamic_model.bodies,
            self.parameters_to_estimate,
            self.observation_settings_list,
            self.dynamic_model.propagator_settings)

        # Create input object for the estimation
        convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=maximum_iterations)
        estimation_input = estimation.EstimationInput(observations_and_times=self.simulated_observations,
                                                      convergence_checker=convergence_checker,
                                                      inverse_apriori_covariance=np.linalg.inv(apriori_covariance))

        # Set methodological options
        estimation_input.define_estimation_settings(save_state_history_per_iteration=False)

        # Define weighting of the observations in the inversion
        weights_per_observable = {estimation_setup.observation.one_way_range_type: self.noise_range**-2,
                                  estimation_setup.observation.one_way_instantaneous_doppler_type: self.noise_doppler**-2}
        estimation_input.set_constant_weight_per_observable(weights_per_observable)

        # Run the estimation
        with util.redirect_std(redirect_out=True):
            estimation_output = self.estimator.perform_estimation(estimation_input)
        updated_parameters = self.parameters_to_estimate.parameter_vector
        estimation_error = self.truth_parameters - updated_parameters

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


        # Generate information and covariance histories based on all the combinations of observables and link definitions
        total_information_dict = self.sorted_observation_sets.copy()
        total_covariance_dict = self.sorted_observation_sets.copy()
        len_obs_list = []
        for i, (observable_type, observation_sets) in enumerate(self.sorted_observation_sets.items()):
            for j, observation_set in enumerate(observation_sets.values()):
                for k, single_observation_set in enumerate(observation_set):

                    epochs = single_observation_set.observation_times
                    len_obs_list.append(len(epochs))

                    weighted_design_matrix_history = np.stack([estimation_output.weighted_design_matrix[sum(len_obs_list[:-1]):sum(len_obs_list), :]], axis=1)

                    information_dict = dict()
                    total_information = 0
                    for index, weighted_design_matrix in enumerate(weighted_design_matrix_history):
                        epoch = epochs[index]
                        current_information = total_information + np.dot(weighted_design_matrix.T, weighted_design_matrix)
                        information_dict[epoch] = current_information
                        measured_information = current_information

                    covariance_dict = dict()
                    for key in information_dict:
                        information_dict[key] = information_dict[key] + np.linalg.inv(apriori_covariance)
                        covariance_dict[key] = np.linalg.inv(information_dict[key])

                    total_covariance_dict[observable_type][0][k] = covariance_dict
                    total_information_dict[observable_type][0][k] = information_dict

        # fig = plt.figure()
        # for i, (observable_type, information_sets) in enumerate(total_information_dict.items()):
        #     for k in range(1,2,1):
        #         information_dict = total_covariance_dict[observable_type][0][k]

        #         epochs = np.array(list(information_dict.keys()))
        #         information_matrix_history = np.array(list(information_dict.values()))
        #         observability_lpf = np.max(np.linalg.eigvals(information_matrix_history[:,0:3,0:3]), axis=1, keepdims=True)
        #         observability_lumio = np.max(np.linalg.eigvals(information_matrix_history[:,6:9,6:9]), axis=1, keepdims=True)
        #         # observability_lpf = np.max(np.linalg.eigvals(information_matrix_history[:,3:6,3:6]), axis=1, keepdims=True)
        #         # observability_lumio = np.max(np.linalg.eigvals(information_matrix_history[:,9:12,9:12]), axis=1, keepdims=True)

        #         plt.plot(epochs, observability_lpf, label="LPF")
        #         plt.plot(epochs, observability_lumio, label="LUMIO")
        #         plt.yscale("log")
        #         plt.grid()

        # plt.legend()
        # plt.show()

        return estimation_output, \
               propagated_formal_errors, propagated_covariance, \
               total_covariance_dict, total_information_dict, \
               self.sorted_observation_sets


    def get_propagated_orbit(self):

        self.set_simulated_observations()

        return self.estimator.variational_solver.dynamics_simulator, self.estimator.variational_solver



custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
                                1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])

# dynamic_model = low_fidelity.LowFidelityDynamicModel(60390, 14, custom_initial_state=custom_initial_state)
# truth_model = low_fidelity.LowFidelityDynamicModel(60390, 14, custom_initial_state=custom_initial_state)
dynamic_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(60390, 14)
truth_model = high_fidelity_point_mass_srp_01.HighFidelityDynamicModel(60390, 14)
estimation_model = EstimationModel(dynamic_model, truth_model)

estimation_output = estimation_model.get_estimation_results()[0]

parameter_history = estimation_output.parameter_history
residual_history = estimation_output.residual_history
covariance = estimation_output.covariance
formal_errors = estimation_output.formal_errors
print(estimation_output.weighted_design_matrix, np.shape(estimation_output.weighted_design_matrix))
