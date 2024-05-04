# General imports
import numpy as np
import os
import sys

# Tudatpy imports
from tudatpy import util
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

class EstimationModel:

    def __init__(self, dynamic_model, truth_model, apriori_covariance=None, initial_estimation_error=None, **kwargs):

        # Loading dynamic model
        self.dynamic_model = dynamic_model
        self.truth_model = truth_model

        # Setting up initial state error properties
        self.apriori_covariance = apriori_covariance
        self.initial_estimation_error = initial_estimation_error

        # Defining basis for observations
        self.bias_range = 0
        self.noise_range = 2.98
        self.observation_step_size_range = 600
        self.retransmission_delay = 0.5e-10
        self.integration_time = 0.5e-10
        self.time_drift_bias = 6.9e-20
        self.maximum_iterations = 4
        self.margin = 120

        # Flexible initialization using optional parameters and default values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


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

        # Create observation settings for each link/observable
        self.observation_settings_list = list()
        for link in self.link_definition.keys():
            self.observation_settings_list.extend([observation.two_way_range(self.link_definition[link],
                                                                        light_time_correction_settings = light_time_correction_settings,
                                                                        bias_settings = range_bias_settings)])


    def set_observation_simulation_settings(self):

        self.set_observation_model_settings()

        self.observation_times_range = np.arange(self.dynamic_model.simulation_start_epoch+self.margin, self.dynamic_model.simulation_end_epoch-self.margin, self.observation_step_size_range)

        # Define observation simulation times for each link
        self.observation_simulation_settings = list()
        for link in self.link_definition.keys():
            self.observation_simulation_settings.extend([observation.tabulated_simulation_settings(observation.n_way_range_type,
                                                                                                self.link_definition[link],
                                                                                                self.observation_times_range,
                                                                                                reference_link_end_type = observation.transmitter)])

        # Add noise levels to observations
        observation.add_gaussian_noise_to_observable(
            self.observation_simulation_settings,
            self.noise_range,
            observation.n_way_range_type)

        # Provide ancillary settings for n-way observables
        observation.two_way_range_ancilliary_settings(retransmission_delay = self.retransmission_delay)

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
        # state_history_simulated_observations = self.dynamics_simulator.state_history

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


    def set_parameters_to_estimate(self):

        self.set_simulated_observations()
        self.dynamic_model.set_propagator_settings()

        # Setup parameters settings to propagate the state transition matrix
        self.parameter_settings = estimation_setup.parameter.initial_states(self.dynamic_model.propagator_settings, self.dynamic_model.bodies)

        # Create the parameters that will be estimated
        self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.dynamic_model.bodies)

        # self.set_simulated_observations()
        # self.truth_model.set_propagator_settings()
        # self.dynamic_model.set_propagator_settings()

        # # Setup parameters settings to propagate the state transition matrix
        # self.parameter_settings = estimation_setup.parameter.initial_states(self.truth_model.propagator_settings, self.truth_model.bodies)

        # # Create the parameters that will be estimated
        # self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.truth_model.bodies)


    def set_estimator_settings(self):

        self.set_parameters_to_estimate()

        # Create the estimator
        self.estimator = numerical_simulation.Estimator(
            self.dynamic_model.bodies,
            self.parameters_to_estimate,
            self.observation_settings_list,
            self.dynamic_model.propagator_settings)

        # # Save the true parameters to later analyse the error
        # self.truth_parameters = self.parameters_to_estimate.parameter_vector

        # # Perturb the initial state estimate from the truth
        # self.perturbed_parameters = self.truth_parameters.copy()
        # if self.initial_estimation_error is not None:
        #     self.perturbed_parameters += self.initial_estimation_error
        # self.parameters_to_estimate.parameter_vector[:12] = self.perturbed_parameters

        # print("Truth at start of arc: ", self.truth_parameters)
        # print("Estimate at start of arc: ", self.parameters_to_estimate.parameter_vector)


        # print("Dynamic model initial state: \n", self.dynamic_model.initial_state)
        # print("Truth model initial state: \n", self.truth_model.initial_state)
        # print("DIFFERENCE: \n", self.dynamic_model.initial_state-self.truth_model.initial_state)

        # # # Save the true parameters to later analyse the error
        # self.truth_parameters = self.parameters_to_estimate.parameter_vector

        # # Perturb the initial state estimate from the truth
        # self.perturbed_parameters = self.truth_parameters.copy()
        # if self.initial_estimation_error is not None:
        #     self.perturbed_parameters += self.initial_estimation_error
        # self.parameters_to_estimate.parameter_vector = self.perturbed_parameters

        # print("Truth at start of arc: ", self.truth_parameters)
        # print("Estimate at start of arc: ", self.parameters_to_estimate.parameter_vector)


        # Create input object for the estimation
        convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=self.maximum_iterations,
                                                                        minimum_residual_change = 1.5*self.noise_range)
        if self.apriori_covariance is None:
            self.estimation_input = estimation.EstimationInput(observations_and_times=self.simulated_observations,
                                                               convergence_checker=convergence_checker)
        else:
            self.estimation_input = estimation.EstimationInput(observations_and_times=self.simulated_observations,
                                                               convergence_checker=convergence_checker,
                                                               inverse_apriori_covariance=np.linalg.inv(self.apriori_covariance))

        # Set methodological options
        self.estimation_input.define_estimation_settings(reintegrate_variational_equations=False,
                                                         save_state_history_per_iteration=False)

        # Define weighting of the observations in the inversion
        weights_per_observable = {estimation_setup.observation.n_way_range_type: self.noise_range**-2}
        self.estimation_input.set_constant_weight_per_observable(weights_per_observable)


    def get_estimation_results(self, redirect_out=False):

        self.set_estimator_settings()

        # od_error = self.parameters_to_estimate.parameter_vector-self.truth_parameters
        # print("Estimation error before estimation: \n", od_error)
        # Run the estimation
        with util.redirect_std(redirect_out=redirect_out):
            self.estimation_output = self.estimator.perform_estimation(self.estimation_input)

        # od_error = self.estimation_output.parameter_history[:, self.estimation_output.best_iteration]-self.truth_model.initial_state
        # print("Estimation error after estimation: \n", od_error)
        # print("LUMIO pos 3D OD error: \n", np.linalg.norm(od_error[6:9]))

        return self



# from src.dynamic_models.HF.PMSRP import *
# import Interpolator
# from matplotlib import pyplot as plt

# initial_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
# apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2

# for propagation_time in np.arange(0.1, 0.5, 0.1):

#     print("NEXT LOOP ==============")
#     print(propagation_time)

#     for dynamic_model in [PMSRP01.HighFidelityDynamicModel(60390, propagation_time)]:

#         estimation_model = EstimationModel(dynamic_model,
#                                            dynamic_model,
#                                            apriori_covariance=apriori_covariance,
#                                            initial_estimation_error=initial_estimation_error,
#                                            maximum_iterations=10,
#                                            noise_range=1,
#                                            observation_step_size_range=600)

#         estimation_model = estimation_model.get_estimation_results()

#         epochs_truth, state_history_truth, dependent_variables_history_truth = \
#             Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.01).get_propagation_results(dynamic_model,
#                                                                                         custom_initial_state=estimation_model.truth_parameters,
#                                                                                         custom_propagation_time=dynamic_model.propagation_time,
#                                                                                         solve_variational_equations=False)

#         epochs_estimated, state_history_estimated, dependent_variables_history_estimated = \
#             Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.01).get_propagation_results(dynamic_model,
#                                                                                         # custom_initial_state=estimation_model.estimation_output.parameter_history[:, estimation_model.estimation_output.best_iteration],
#                                                                                         custom_initial_state=estimation_model.parameters_to_estimate.parameter_vector,
#                                                                                         custom_propagation_time=dynamic_model.propagation_time,
#                                                                                         solve_variational_equations=False)

#         est_error_0 = state_history_estimated[0,:]-state_history_truth[0,:]
#         est_error = state_history_estimated[-1,:]-state_history_truth[-1,:]
#         print("Estimation error x0: \n", est_error_0)
#         print("Estimation error xf: \n", est_error)

#         # print(np.linalg.norm(est_error_0[0:3]), np.linalg.norm(est_error_0[3:6]), np.linalg.norm(est_error_0[6:9]), np.linalg.norm(est_error_0[9:12]))
#         # print(np.linalg.norm(est_error[0:3]), np.linalg.norm(est_error[3:6]), np.linalg.norm(est_error[6:9]), np.linalg.norm(est_error[9:12]))
#         # print("Formal errors: \n", 3*np.sqrt(np.diagonal(estimation_model.estimation_output.covariance)))

#         plt.plot(epochs_truth-dynamic_model.simulation_start_epoch_MJD, state_history_estimated-state_history_truth)

#         final_residuals = estimation_model.estimation_output.final_residuals
#         observation_times = np.array(estimation_model.simulated_observations.concatenated_times)

#         fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))

#         ax1.scatter((observation_times - observation_times[0]) / (3600*24),
#                     final_residuals)

#         ax1.set_title("Observations as a function of time")
#         ax1.set_xlabel(r'Time [days]')
#         ax1.set_ylabel(r'Final Residuals [m]')
#         plt.tight_layout()
#         # plt.show()

#         fig, ax = plt.subplots(1, 1, figsize=(9, 6))
#         for i, (observable_type, information_sets) in enumerate(estimation_model.sorted_observation_sets.items()):
#             for j, observation_set in enumerate(information_sets.values()):
#                 for k, single_observation_set in enumerate(observation_set):
#                     observation_times = np.array(single_observation_set.observation_times)
#                     observation_times = observation_times - observation_times[0]
#                     ax.scatter(observation_times, single_observation_set.concatenated_observations)

# plt.show()

