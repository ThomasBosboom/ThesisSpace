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
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

class EstimationModel:

    def __init__(self, dynamic_model, truth_model, **kwargs):

        # Loading dynamic model
        self.dynamic_model = dynamic_model
        self.truth_model = truth_model

        # Setting up initial state error properties
        self.apriori_covariance = None
        # self.initial_estimation_error = None

        # Defining basis for observations
        self.bias_range = 0
        self.noise_range = 1
        self.observation_step_size_range = 300
        self.total_observation_count = None
        self.retransmission_delay = 0.5e-10
        self.integration_time = 0.5e-10
        self.time_drift_bias = 6.9e-20
        self.maximum_iterations = 10
        self.margin = 120
        self.redirect_out = True
        self.seed = 0

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
                                                                        # bias_settings = range_bias_settings
                                                                        )])


    def set_observation_simulation_settings(self):

        self.set_observation_model_settings()

        self.observation_times_range = np.arange(self.dynamic_model.simulation_start_epoch+self.margin, self.dynamic_model.simulation_end_epoch-self.margin, self.observation_step_size_range)
        if self.total_observation_count is not None:
            self.observation_times_range = np.linspace(self.dynamic_model.simulation_start_epoch+self.margin, self.dynamic_model.simulation_end_epoch-self.margin, self.total_observation_count)
            # print((self.observation_times_range[-1]-self.observation_times_range[0])/len(self.observation_times_range))
        print(len(self.observation_times_range), self.observation_times_range[0], self.observation_times_range[-1])

        # Define observation simulation times for each link
        self.observation_simulation_settings = list()
        for link in self.link_definition.keys():
            self.observation_simulation_settings.extend([observation.tabulated_simulation_settings(observation.n_way_range_type,
                                                                                                self.link_definition[link],
                                                                                                self.observation_times_range,
                                                                                                reference_link_end_type = observation.transmitter)])

        # # Add noise levels to observations
        # observation.add_gaussian_noise_to_observable(
        #     self.observation_simulation_settings,
        #     self.noise_range,
        #     observation.n_way_range_type)

        # Add noise levels to observations
        # seed = int(self.dynamic_model.simulation_start_epoch)
        rng = np.random.default_rng(seed=self.seed)
        save_noise = []
        def range_noise_function(time):
            noise = rng.normal(loc=0.0, scale=self.noise_range, size=1)
            save_noise.append(noise)
            # print(noise[-1], seed)
            return noise

        observation.add_noise_function_to_observable(
            self.observation_simulation_settings,
            range_noise_function,
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


    def set_estimator_settings(self):

        self.set_parameters_to_estimate()

        # Create the estimator
        self.estimator = numerical_simulation.Estimator(
            self.dynamic_model.bodies,
            self.parameters_to_estimate,
            self.observation_settings_list,
            self.dynamic_model.propagator_settings)

        # Save the true parameters to later analyse the error
        self.truth_parameters = self.parameters_to_estimate.parameter_vector

        # Create input object for the estimation
        convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=self.maximum_iterations)
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


    def get_estimation_results(self):

        self.set_estimator_settings()

        # Run the estimation
        with util.redirect_std(redirect_out=self.redirect_out):
            self.estimation_output = self.estimator.perform_estimation(self.estimation_input)

        return self



from src.dynamic_models.HF.PMSRP import *
import Interpolator
from matplotlib import pyplot as plt
import copy
from tests import utils
from src import StationKeeping

orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0
# initial_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5, 5, 5, 1e-5, 1e-5, 1e-5])*1
# apriori_covariance = np.diag([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3,  5, 5, 5, 1e-5, 1e-5, 1e-5])*1**2
lpf_estimation_error = np.array([5e1, 5e1, 5e1, 1e-4, 1e-4, 1e-4])*1
# lpf_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
lumio_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
initial_estimation_error = np.concatenate((lpf_estimation_error, lumio_estimation_error))
apriori_covariance = np.diag(initial_estimation_error**2)
mission_start_epoch = 60390
# redirect_out = False
num_runs = 2

print(f"Runs: {num_runs}, start epoch: {mission_start_epoch}")
print("========================================================")

for noise_range in [1]:
    for propagation_time in [0.1, 0.2, 0.5, 1.0]:

        est_error_0_list = []
        delta_v_list = []
        for run in range(num_runs):

            truth_model = PMSRP01.HighFidelityDynamicModel(mission_start_epoch, propagation_time)
            truth_model.initial_state += orbit_insertion_error
            dynamic_model = PMSRP01.HighFidelityDynamicModel(mission_start_epoch, propagation_time, custom_initial_state=truth_model.initial_state+initial_estimation_error)

            # dynamic_model.initial_state = truth_model.initial_state + initial_estimation_error
            # od_error = dynamic_model.initial_state-truth_model.initial_state
            # print(od_error)
            # print(dynamic_model.initial_state)

            # dynamic_model_objects = utils.get_dynamic_model_objects(mission_start_epoch, propagation_time,
            #                                                         custom_initial_state=dynamic_model.initial_state,
            #                                                         custom_model_dict={"HF": ['PMSRP']},
            #                                                         custom_model_list=[0])

            # dynamic_model = dynamic_model_objects["HF"]["PMSRP"][0]

            # print(f"============ Run {run}, Noise {noise_range}, propagation time {propagation_time}")

            od_error = dynamic_model.initial_state-truth_model.initial_state
            # print(od_error)
            # print(dynamic_model.initial_state)


            estimation_model = EstimationModel(dynamic_model, truth_model,
                                                apriori_covariance=apriori_covariance,
                                                initial_estimation_error=initial_estimation_error,
                                                redirect_out=False,
                                                noise_range = noise_range,
                                                total_observation_count=150,
                                                # observation_step_size_range = 300,
                                                margin=120,
                                                orbit_insertion_error=orbit_insertion_error,
                                                seed=0
                                                )

            # print("Details: ", vars(estimation_model))
            # print(estimation_model.observation_step_size_range)
            estimation_model = estimation_model.get_estimation_results()
            estimation_output = estimation_model.estimation_output
            best_iteration = estimation_model.estimation_output.best_iteration

            epochs_truth, state_history_truth, dependent_variables_history_truth, state_transition_matrix_history_truth = \
            Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.01).get_propagation_results(truth_model,
                                                                                        custom_initial_state=truth_model.initial_state,
                                                                                        custom_propagation_time=truth_model.propagation_time,
                                                                                        solve_variational_equations=True)

            epochs_estimated, state_history_estimated, dependent_variables_history_estimated = \
            Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.01).get_propagation_results(dynamic_model,
                                                                                        custom_initial_state=estimation_model.estimation_output.parameter_history[:, best_iteration],
                                                                                        # custom_initial_state=estimation_model.parameters_to_estimate.parameter_vector,
                                                                                        custom_propagation_time=dynamic_model.propagation_time,
                                                                                        solve_variational_equations=False)

            est_error_0 = state_history_estimated[0,:]-state_history_truth[0,:]
            est_error = state_history_estimated[-1,:]-state_history_truth[-1,:]
            formal_errors = estimation_output.formal_errors


            if run != 0:
                est_error_0_list.append(np.linalg.norm(est_error_0[6:9]))

            residual_history = estimation_output.residual_history
            final_residuals = estimation_output.final_residuals
            parameter_history = estimation_output.parameter_history

            # print(np.linalg.norm(est_error_0[6:9]))
            # print("initial residuals: ", np.sqrt(np.mean(residual_history[:, 0]**2)), "initial OD error: ", np.linalg.norm(est_error_0[6:9]))
            # print("final residuals: ", np.sqrt(np.mean(residual_history[:, best_iteration]**2)), "initial OD error: ", np.linalg.norm(est_error[6:9]))
            # print("initial OD error Rlpf, Vlpf, Rlumio, Vlumio: \n", np.linalg.norm(od_error[0:3]), np.linalg.norm(od_error[3:6]), np.linalg.norm(od_error[6:9]), np.linalg.norm(od_error[9:12]))
            # print("estimated OD error Rlpf, Vlpf, Rlumio, Vlumio: \n", np.linalg.norm(est_error_0[0:3]), np.linalg.norm(est_error_0[3:6]), np.linalg.norm(est_error_0[6:9]), np.linalg.norm(est_error_0[9:12]))
            # print("propagated estimated OD error Rlpf, Vlpf, Rlumio, Vlumio: \n", np.linalg.norm(est_error[0:3]), np.linalg.norm(est_error[3:6]), np.linalg.norm(est_error[6:9]), np.linalg.norm(est_error[9:12]))
            # print("formal errors OD error Rlpf, Vlpf, Rlumio, Vlumio: \n", np.linalg.norm(formal_errors[0:3]), np.linalg.norm(formal_errors[3:6]), np.linalg.norm(formal_errors[6:9]), np.linalg.norm(formal_errors[9:12]))


            # print(np.linalg.norm(parameter_history[:, estimation_model.estimation_output.best_iteration][6:9]-parameter_history[:, 0][6:9]))
            # print("propagated estimated hsitory: ", state_history_estimated[-1,:])
            dynamic_model_objects = utils.get_dynamic_model_objects(mission_start_epoch+propagation_time, 3,
                                                                    custom_initial_state=state_history_estimated[-1,:],
                                                                    custom_model_dict={"HF": ['PMSRP']},
                                                                    custom_model_list=[0])

            dynamic_model_station_keeping = dynamic_model_objects["HF"]["PMSRP"][0]
            # dynamic_model = PMSRP01.HighFidelityDynamicModel(mission_start_epoch, 3, custom_initial_state=estimation_model.estimation_output.parameter_history[:, best_iteration])
            # print("custom_initial_state: ", dynamic_model_station_keeping.custom_initial_state)
            station_keeping = StationKeeping.StationKeeping(dynamic_model_station_keeping, step_size=0.01)
            delta_v, dispersion = station_keeping.get_corrected_state_vector(cut_off_epoch=0,
                                                                            correction_epoch=0,
                                                                            target_point_epochs=[3])

            delta_v_list.append(np.linalg.norm(delta_v))
            est_error_0_list.append(np.linalg.norm(est_error_0[6:9]))

            formal_errors = estimation_output.formal_errors

            propagated_covariance_final = dict()
            propagated_formal_errors_final = dict()
            for i in range(len(epochs_truth)):
                propagated_covariance = state_transition_matrix_history_truth[i] @ estimation_output.covariance @ state_transition_matrix_history_truth[i].T
                propagated_covariance_final.update({epochs_truth[i]: propagated_covariance})
                propagated_formal_errors_final.update({epochs_truth[i]: np.sqrt(np.diagonal(propagated_covariance))})

            epochs = np.stack(list(propagated_formal_errors_final.keys()))
            formal_errors = np.stack(list(propagated_formal_errors_final.values()))

            fig, ax = plt.subplots(2, 1, figsize=(9, 8))
            ax[0].plot(epochs, formal_errors[:, 0:3])
            ax[1].plot(epochs, formal_errors[:, 3:6])


            # observation_times = np.array(estimation_model.simulated_observations.concatenated_times)

            # fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))

            # ax1.scatter((observation_times - observation_times[0]) / (3600*24),
            #         final_residuals)

            # ax1.set_title("Observations as a function of time")
            # ax1.set_xlabel(r'Time [days]')
            # ax1.set_ylabel(r'Final Residuals [m]')
            # plt.tight_layout()
            # # plt.show()

            # fig, ax = plt.subplots(1, 1, figsize=(9, 6))
            # for i, (observable_type, information_sets) in enumerate(estimation_model.sorted_observation_sets.items()):
            #     for j, observation_set in enumerate(information_sets.values()):
            #         for k, single_observation_set in enumerate(observation_set):
            #             observation_times = np.array(single_observation_set.observation_times)
            #             observation_times = observation_times - observation_times[0]
            #             ax.scatter(observation_times, single_observation_set.concatenated_observations)

            print("Delta v: ", np.linalg.norm(delta_v), "\n", \
                  "Residuals: ", np.sqrt(np.mean(residual_history[:, best_iteration]**2)), \
                  "Estimation error: ", np.linalg.norm(est_error_0[0:3]), np.linalg.norm(est_error_0[3:6]), np.linalg.norm(est_error_0[6:9]), np.linalg.norm(est_error_0[9:12]),
            # print("estimated OD error Rlpf, Vlpf, Rlumio, Vlumio: \n", np.linalg.norm(est_error_0[0:3]), np.linalg.norm(est_error_0[3:6]), np.linalg.norm(est_error_0[6:9]), np.linalg.norm(est_error_0[9:12]), \
                        "\n", np.linalg.norm(dispersion[0:3]))
        # print(f"Noise {noise_range}, propagation time {propagation_time}: | OD error: Mean: ", np.mean(est_error_0_list), "Std: ", np.std(est_error_0_list))
        print(f"Noise {noise_range}, propagation time {propagation_time}: | OD error: Mean: ", np.mean(est_error_0_list), "Std: ", np.std(est_error_0_list), " | Delta V: Mean: ", np.mean(delta_v_list), "Std: ", np.std(delta_v_list))

plt.show()