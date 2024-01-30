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
from dynamic_models import validation
from dynamic_models.full_fidelity import *
from dynamic_models.low_fidelity.three_body_problem import *
from dynamic_models.high_fidelity.point_mass import *
from dynamic_models.high_fidelity.point_mass_srp import *
from dynamic_models.high_fidelity.spherical_harmonics import *
from dynamic_models.high_fidelity.spherical_harmonics_srp import *



class EstimationModel:

    def __init__(self, dynamic_model, truth_model, apriori_covariance=None):

        # Loading dynamic model
        self.dynamic_model = dynamic_model
        self.truth_model = truth_model

        # Loading apriori covariance
        self.apriori_covariance = apriori_covariance

        # Defining basis for observations
        self.bias_range = 10.0
        self.bias_doppler = 0.001
        self.noise_range = 2.98
        self.noise_doppler = 0.00097
        self.observation_step_size_range = 1000
        self.observation_step_size_doppler = 1000
        self.retransmission_delay = 6
        self.integration_time = 0.5
        self.time_drift_bias = 6.9e-8

        # Creating observation time vector
        self.observation_times_range = np.arange(self.dynamic_model.simulation_start_epoch, self.dynamic_model.simulation_end_epoch, self.observation_step_size_range)
        self.observation_times_doppler = np.arange(self.dynamic_model.simulation_start_epoch+30, self.dynamic_model.simulation_end_epoch-30, self.observation_step_size_doppler)


    def set_observation_model_settings(self):

        self.dynamic_model.set_environment_settings()

        # Define the uplink link ends for one-way observable
        link_ends = {observation.transmitter: observation.body_origin_link_end_id(self.dynamic_model.name_LPO),
                     observation.retransmitter: observation.body_origin_link_end_id(self.dynamic_model.name_ELO),
                     observation.receiver: observation.body_origin_link_end_id(self.dynamic_model.name_LPO)}
        self.link_definition = observation.LinkDefinition(link_ends)

        self.link_definition = {"two_way_system": self.link_definition}

        # Define settings for light-time calculations
        light_time_correction_settings = list()
        correcting_bodies = list(set(self.truth_model.bodies_to_create).intersection(self.dynamic_model.bodies_to_create))
        for correcting_body in correcting_bodies:
            light_time_correction_settings.append(observation.first_order_relativistic_light_time_correction(correcting_bodies))

        # Define settings for range and doppler bias
        range_bias_settings = observation.combined_bias([observation.absolute_bias([self.bias_range]),
                                                         observation.time_drift_bias(bias_value = np.array([self.time_drift_bias]),
                                                                                     time_link_end = observation.transmitter,
                                                                                     ref_epoch = self.dynamic_model.simulation_start_epoch)])
        doppler_bias_settings = observation.absolute_bias([self.bias_doppler])

        # Create observation settings for each link/observable
        self.observation_settings_list = list()
        for link in self.link_definition.keys():
            self.observation_settings_list.extend([observation.two_way_range(self.link_definition[link],
                                                                        light_time_correction_settings = light_time_correction_settings,
                                                                        bias_settings = range_bias_settings),
                                                   observation.two_way_doppler_averaged(self.link_definition[link],
                                                                        light_time_correction_settings = light_time_correction_settings,
                                                                        bias_settings = doppler_bias_settings)])


    def set_observation_simulation_settings(self):

        self.set_observation_model_settings()

        # Define observation simulation times for each link
        self.observation_simulation_settings = list()
        for link in self.link_definition.keys():
            self.observation_simulation_settings.extend([observation.tabulated_simulation_settings(observation.n_way_range_type,
                                                                                                self.link_definition[link],
                                                                                                self.observation_times_range,
                                                                                                reference_link_end_type = observation.transmitter),
                                                         observation.tabulated_simulation_settings(observation.n_way_averaged_doppler_type,
                                                                                                self.link_definition[link],
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

        # Provide ancillary settings for n-way observables
        observation.two_way_range_ancilliary_settings(retransmission_delay = self.retransmission_delay)
        observation.n_way_doppler_ancilliary_settings(integration_time = self.integration_time,
                                                      link_end_delays = [self.retransmission_delay])

        # Create viability settings
        viability_setting_list = [observation.body_occultation_viability([self.dynamic_model.name_ELO, self.dynamic_model.name_LPO], self.dynamic_model.name_secondary)]

        observation.add_viability_check_to_all(
            self.observation_simulation_settings,
            viability_setting_list)


    def set_observation_simulators(self):

        self.set_observation_simulation_settings()
        self.truth_model.set_propagator_settings()

        # Create or update the ephemeris of all propagated bodies (here: LPF and LUMIO) to match the propagated results
        self.truth_model.propagator_settings.processing_settings.set_integrated_result = True

        # Run propagation
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(self.truth_model.bodies, self.truth_model.propagator_settings)
        # self.state_history_simulated_observations = dynamics_simulator.state_history

        # dynamics_simulator = numerical_simulation.create_dynamics_simulator(self.truth_model.bodies, self.truth_model.propagator_settings)
        # self.state_history_dynamic = dynamics_simulator.state_history

        # state_history_observations = np.stack(list(dynamics_simulator.state_history.values()))


        # Create observation simulators
        self.observation_simulators = estimation_setup.create_observation_simulators(
            self.observation_settings_list, self.truth_model.bodies)


    def set_simulated_observations(self):

        self.set_observation_simulators()

        # Get LPF and LUMIO simulated observations as ObservationCollection
        self.simulated_observations = estimation.simulate_observations(
            self.observation_simulation_settings,
            self.observation_simulators,
            self.truth_model.bodies)

        # Get sorted observation sets
        self.sorted_observation_sets = self.simulated_observations.sorted_observation_sets


    def set_parameters_to_estimate(self, estimated_initial_state=None):

        self.set_simulated_observations()
        self.dynamic_model.set_propagator_settings(estimated_initial_state=estimated_initial_state)

        # print(self.dynamic_model.propagator_settings)
        # dynamics_simulator = numerical_simulation.create_dynamics_simulator(self.dynamic_model.bodies, self.dynamic_model.propagator_settings)
        # state_history_dynamic_model = np.stack(list(dynamics_simulator.state_history.values()))
        # epochs = np.stack(list(dynamics_simulator.state_history.keys()))
        # print("initial_state before update 2", epochs[0], state_history_dynamic_model[0,:])
        # plt.plot(epochs, state_history_dynamic_model[:,6:9])

        # Setup parameters settings to propagate the state transition matrix
        self.parameter_settings = estimation_setup.parameter.initial_states(self.dynamic_model.propagator_settings, self.dynamic_model.bodies)

        # Add estimated parameters to the sensitivity matrix that will be propagated
        # self.parameter_settings.append(estimation_setup.parameter.gravitational_parameter(self.dynamic_model.name_primary))
        # self.parameter_settings.append(estimation_setup.parameter.gravitational_parameter(self.dynamic_model.name_secondary))
        # self.parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(self.link_definition, observation.n_way_range_type))
        # self.parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(self.link_definition, observation.n_way_averaged_doppler_type))

        # Depending on the dynamic model
        # self.parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient(self.dynamic_model.name_ELO))
        # self.parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient(self.dynamic_model.name_LPO))

        # Create the parameters that will be estimated
        self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.dynamic_model.bodies)
        # self.truth_parameters = self.parameters_to_estimate.parameter_vector



    def set_estimator_settings(self, estimated_initial_state=None, maximum_iterations=3):

        self.set_parameters_to_estimate(estimated_initial_state=estimated_initial_state)

        # Create the estimator
        self.estimator = numerical_simulation.Estimator(
            self.dynamic_model.bodies,
            self.parameters_to_estimate,
            self.observation_settings_list,
            self.dynamic_model.propagator_settings)

        # Create input object for the estimation
        convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=maximum_iterations)
        if self.apriori_covariance is None:
            self.estimation_input = estimation.EstimationInput(observations_and_times=self.simulated_observations,
                                                               convergence_checker=convergence_checker)
        else:
            self.estimation_input = estimation.EstimationInput(observations_and_times=self.simulated_observations,
                                                               convergence_checker=convergence_checker,
                                                               inverse_apriori_covariance=np.linalg.inv(self.apriori_covariance))

        # Set methodological options
        self.estimation_input.define_estimation_settings(reintegrate_variational_equations=False,
                                                         save_state_history_per_iteration=True)

        # Define weighting of the observations in the inversion
        weights_per_observable = {estimation_setup.observation.n_way_range_type: self.noise_range**-2,
                                  estimation_setup.observation.n_way_averaged_doppler_type: self.noise_doppler**-2}
        self.estimation_input.set_constant_weight_per_observable(weights_per_observable)


    def get_estimation_results(self):

        self.set_estimator_settings()

        # Run the estimation
        estimation_output = self.estimator.perform_estimation(self.estimation_input)

        # Generate information and covariance histories based on all the combinations of observables and link definitions
        total_information_dict = dict()
        total_covariance_dict = dict()
        len_obs_list = []
        for i, (observable_type, observation_sets) in enumerate(self.sorted_observation_sets.items()):
            total_information_dict[observable_type] = dict()
            total_covariance_dict[observable_type] = dict()
            for j, observation_set in enumerate(observation_sets.values()):
                total_information_dict[observable_type][j] = list()
                total_covariance_dict[observable_type][j] = list()
                for k, single_observation_set in enumerate(observation_set):

                    epochs = single_observation_set.observation_times
                    len_obs_list.append(len(epochs))

                    weighted_design_matrix_history = np.stack([estimation_output.weighted_design_matrix[sum(len_obs_list[:-1]):sum(len_obs_list), :]], axis=1)

                    information_dict = dict()
                    total_information = 0
                    for index, weighted_design_matrix in enumerate(weighted_design_matrix_history):
                        epoch = epochs[index]
                        current_information = np.dot(weighted_design_matrix.T, weighted_design_matrix)
                        information_dict[epoch] = current_information
                        total_information = current_information

                    covariance_dict = dict()
                    for key in information_dict:
                        if self.apriori_covariance is not None:
                            information_dict[key] = information_dict[key] + np.linalg.inv(self.apriori_covariance)
                        covariance_dict[key] = np.linalg.inv(information_dict[key])

                    total_information_dict[observable_type][j].append(information_dict)
                    total_covariance_dict[observable_type][j].append(covariance_dict)


        # Update the estimator with the latest parameter values to generate dynamics simulators with latest dynamics for next batch


        fig = plt.figure()

        import Interpolator
        epochs, state_history_dynamic_model_initial, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.001).get_propagation_results(self.dynamic_model,estimated_initial_state=estimation_output.parameter_history[:,0])
        plt.plot(epochs, state_history_dynamic_model_initial[:,:3], color="red")
        epochs, state_history_dynamic_model_final, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.001).get_propagation_results(self.dynamic_model,estimated_initial_state=estimation_output.parameter_history[:,-1])
        plt.plot(epochs, state_history_dynamic_model_final[:,:3], color="red", ls="--")
        epochs, state_history_observations, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.001).get_propagation_results(self.truth_model,estimated_initial_state=None)
        plt.plot(epochs, state_history_observations[:,:3], color="blue")

        # plt.show()

        fig = plt.figure()
        plt.plot(epochs, np.linalg.norm(state_history_dynamic_model_initial[:,:3]-state_history_dynamic_model_final[:,:3], axis=1))
        plt.plot(epochs, np.linalg.norm(state_history_observations[:,:3]-state_history_dynamic_model_initial[:,:3], axis=1))
        plt.plot(epochs, np.linalg.norm(state_history_observations[:,:3]-state_history_dynamic_model_final[:,:3], axis=1))
        plt.yscale("log")

        fig1_3d = plt.figure()
        ax = fig1_3d.add_subplot(111, projection='3d')
        plt.plot(state_history_observations[:,0], state_history_observations[:,1], state_history_observations[:,2], label="LPF truth", color="red")
        plt.plot(state_history_observations[:,6], state_history_observations[:,7], state_history_observations[:,8], label="LUMIO truth", color="blue")
        plt.plot(state_history_dynamic_model_final[:,0], state_history_dynamic_model_final[:,1], state_history_dynamic_model_final[:,2], label="LPF final", color="red")
        plt.plot(state_history_dynamic_model_final[:,6], state_history_dynamic_model_final[:,7], state_history_dynamic_model_final[:,8], label="LUMIO final", color="blue")
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.legend(loc="upper right")
        plt.show()

        return estimation_output, \
               total_covariance_dict, total_information_dict, \
               self.sorted_observation_sets, \
            #    self.estimator.variational_solver.dynamics_simulator, self.estimator.variational_solver


    def get_propagation_simulator(self, estimated_initial_state=None):

        self.set_simulated_observations()

        return self.estimator.variational_solver.dynamics_simulator, self.estimator.variational_solver



# custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
#                                 1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])
# dynamic_model = low_fidelity.LowFidelityDynamicModel(60390, 14, custom_initial_state=custom_initial_state)
# truth_model = low_fidelity.LowFidelityDynamicModel(60390, 14, custom_initial_state=custom_initial_state)
dynamic_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(60390, 14)
truth_model = high_fidelity_point_mass_05.HighFidelityDynamicModel(60390, 14)

apriori_covariance=np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
estimation_model = EstimationModel(dynamic_model, truth_model, apriori_covariance=apriori_covariance)

estimation_output = estimation_model.get_estimation_results()[0]

parameter_history = estimation_output.parameter_history
residual_history = estimation_output.residual_history
covariance = estimation_output.covariance
formal_errors = estimation_output.formal_errors

fig = plt.figure()
simulation_results_per_iteration = estimation_output.simulation_results_per_iteration
base_state_history = simulation_results_per_iteration[0].dynamics_results.state_history
base_epochs = np.stack(list(base_state_history.keys()))
base_state_history = np.stack(list(base_state_history.values()))
for simulation_result_per_iteration in simulation_results_per_iteration:
    state_history = simulation_result_per_iteration.dynamics_results.state_history
    epochs = np.stack(list(state_history.keys()))
    state_history = np.stack(list(state_history.values()))

    # print(state_history, np.shape(state_history))
    plt.plot(epochs, np.linalg.norm(state_history[:,6:9]-base_state_history[:,6:9], axis=1))

    print(state_history[0,:]-base_state_history[0,:])

# plt.yscale("log")
plt.show()



# fig = plt.figure()
# dynamics_simulator = estimation_model.get_estimation_results()[-2]
# print(dynamics_simulator)

# state_history = dynamics_simulator.state_history
# print(state_history, np.shape(state_history))

# epochs = np.stack(list(state_history.keys()))
# state_history = np.stack(list(state_history.values()))

# print(state_history, np.shape(state_history))

# # plt.plot(state_history[:,6], state_history[:,7])
# plt.plot(epochs, np.linalg.norm(state_history[:,0:3], axis=1))

# # plt.yscale("log")
# plt.show()