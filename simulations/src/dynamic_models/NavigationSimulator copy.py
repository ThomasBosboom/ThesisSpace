# Standard
import os
import sys
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d

# Tudatpy
from tudatpy.kernel.numerical_simulation import estimation
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from tests import utils
from src.dynamic_models import validation
from src.dynamic_models import Interpolator, StationKeeping
from src.dynamic_models.full_fidelity.full_fidelity import *
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model


class NavigationSimulator():

    def __init__(self, observation_windows, dynamic_model_list, truth_model_list, sigma_number=3, step_size=0.01, station_keeping_epoch=4, target_point_epochs=[3]):

        self.observation_windows = observation_windows
        self.dynamic_model_list = dynamic_model_list
        self.truth_model_list = truth_model_list

        self.batch_start_times = np.array([t[0] for t in self.observation_windows])
        self.batch_end_times = np.array([t[1] for t in self.observation_windows])

        self.mission_start_epoch = 60390

        self.initial_state_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
        self.apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2

        self.sigma_number = sigma_number
        self.step_size = step_size
        self.station_keeping_epoch = station_keeping_epoch
        self.target_point_epochs = target_point_epochs


    def perform_navigation(self, include_station_keeping=True):

        mission_epoch = self.mission_start_epoch

        # Initial batch timing settings
        batch_times = np.round(self.batch_end_times - self.batch_start_times, 2)
        propagation_times = np.round(np.concatenate((np.diff(self.batch_start_times), [batch_times[-1]])), 2)

        # Settings of the dynamic model and truth model
        model_type = self.dynamic_model_list[0]
        model_name = self.dynamic_model_list[1]
        model_number = self.dynamic_model_list[2]
        model_type_truth = self.truth_model_list[0]
        model_name_truth = self.truth_model_list[1]
        model_number_truth = self.truth_model_list[2]

        # Initialize the navigation loop
        custom_initial_state = None
        custom_initial_state_truth = None
        batch_count = 0
        full_estimation_error_dict = dict()
        full_reference_state_deviation_dict = dict()
        full_propagated_covariance_dict = dict()
        full_propagated_formal_errors_dict = dict()
        full_state_history_reference_dict = dict()
        full_state_history_truth_dict = dict()
        full_state_history_initial_dict = dict()
        full_state_history_final_dict = dict()
        while mission_epoch < self.batch_end_times[-1]:

            print(f"Estimation of arc {batch_count}, duration {batch_times[batch_count]} days: {self.batch_start_times[batch_count]} until {self.batch_end_times[batch_count]}")

            # Define dynamic models and select one to test the estimation on
            dynamic_model_objects = utils.get_dynamic_model_objects(self.batch_start_times[batch_count],
                                                                    batch_times[batch_count],
                                                                    package_dict=None,
                                                                    get_only_first=False,
                                                                    custom_initial_state=custom_initial_state)

            dynamic_model = dynamic_model_objects[model_type][model_name][model_number]

            # Obtain the initial state of the whole simulation once
            if batch_count == 0:
                epochs, state_history_initialize, dependent_variables_history_initialize = \
                    Interpolator.Interpolator(epoch_in_MJD=False, step_size=self.step_size).get_propagation_results(dynamic_model,
                                                                                                            custom_initial_state=custom_initial_state,
                                                                                                            custom_propagation_time=batch_times[batch_count],
                                                                                                            solve_variational_equations=False)

                custom_initial_state = state_history_initialize[0,:] + self.initial_state_error



            truth_model = copy.copy(dynamic_model_objects[model_type_truth][model_name_truth][model_number_truth])
            truth_model.custom_initial_state = custom_initial_state_truth

            dynamic_model.custom_propagation_time = batch_times[batch_count]
            dynamic_model_objects = {model_type: {model_name: [dynamic_model]}}
            # print(dynamic_model_objects)

            # Read out attributes of the estimation model object if necessary
            # estimation_model_objects = utils.get_estimation_model_objects(dynamic_model_objects,
            #                                                                 custom_truth_model=self.truth_model,
            #                                                                 apriori_covariance=self.apriori_covariance,
            #                                                                 initial_state_error=self.initial_state_error)

            # estimation_model_object = estimation_model_objects[model_type][model_name][model_number]

            # Obtain estimation results for given batch
            estimation_model_objects_results = utils.get_estimation_model_results(dynamic_model_objects,
                                                                                get_only_first=False,
                                                                                custom_truth_model=truth_model,
                                                                                apriori_covariance=self.apriori_covariance,
                                                                                initial_state_error=self.initial_state_error)


            # Extract all the results
            estimation_model_objects_result = estimation_model_objects_results[model_type][model_name][0]
            estimation_output = estimation_model_objects_result[0]
            parameter_history = estimation_output.parameter_history
            final_covariance = estimation_output.covariance
            formal_errors = estimation_output.formal_errors

            # Define the times
            end_of_batch = utils.convert_MJD_to_epoch(self.batch_end_times[batch_count], full_array=False)

            # Update the reference orbit that the estimated orbit should follow
            state_history_reference = list()
            for body in dynamic_model.bodies_to_propagate:
                state_history_reference.append(validation.get_reference_state_history(self.batch_start_times[batch_count],
                                                                                    propagation_times[batch_count],
                                                                                        satellite=body,
                                                                                        step_size=self.step_size,
                                                                                        get_full_history=True))

            state_history_reference = np.concatenate(state_history_reference, axis=1)

            # Obtain the state histories of the truth, current and the newly estimated trajectories
            epochs, state_history_truth, dependent_variables_history_truth = \
                Interpolator.Interpolator(epoch_in_MJD=False, step_size=self.step_size).get_propagation_results(truth_model,
                                                                                                        custom_initial_state=custom_initial_state_truth,
                                                                                                        custom_propagation_time=propagation_times[batch_count],
                                                                                                        solve_variational_equations=False)

            epochs, state_history_initial, dependent_variables_history_initial, state_transition_history_initial = \
                Interpolator.Interpolator(epoch_in_MJD=False, step_size=self.step_size).get_propagation_results(dynamic_model,
                                                                                                    custom_initial_state=custom_initial_state,
                                                                                                    custom_propagation_time=propagation_times[batch_count],
                                                                                                    solve_variational_equations=True)

            epochs, state_history_final, dependent_variables_history_final, state_transition_history_final = \
                Interpolator.Interpolator(epoch_in_MJD=False, step_size=self.step_size).get_propagation_results(dynamic_model,
                                                                                                    custom_initial_state=parameter_history[:,-1],
                                                                                                    custom_propagation_time=propagation_times[batch_count],
                                                                                                    solve_variational_equations=True)

            # Save the propagated histories of the uncertainties
            propagated_covariance_initial = dict()
            propagated_formal_errors_initial = dict()
            for i in range(len(epochs)):
                propagated_covariance = state_transition_history_initial[i] @ self.apriori_covariance @ state_transition_history_initial[i].T
                propagated_covariance_initial.update({epochs[i]: propagated_covariance})
                propagated_formal_errors_initial.update({epochs[i]: np.sqrt(np.diagonal(propagated_covariance))})

            propagated_covariance_final= dict()
            propagated_formal_errors_final = dict()
            for i in range(len(epochs)):
                propagated_covariance = state_transition_history_final[i] @ estimation_output.covariance @ state_transition_history_final[i].T
                propagated_covariance_final.update({epochs[i]: propagated_covariance})
                propagated_formal_errors_final.update({epochs[i]: np.sqrt(np.diagonal(propagated_covariance))})

            for key, value in propagated_covariance_final.items():
                if key > end_of_batch:
                    propagated_covariance_initial[key] = value

            for key, value in propagated_formal_errors_final.items():
                if key > end_of_batch:
                    propagated_formal_errors_initial[key] = value

            for i, epoch in enumerate(epochs):
                if epoch > end_of_batch:
                    state_history_initial[i] = state_history_final[i]

            full_estimation_error_dict.update(dict(zip(epochs, state_history_initial-state_history_truth)))
            full_reference_state_deviation_dict.update(dict(zip(epochs, state_history_initial-state_history_reference)))
            full_propagated_covariance_dict.update(propagated_covariance_initial)
            full_propagated_formal_errors_dict.update(propagated_formal_errors_initial)
            full_state_history_reference_dict.update(dict(zip(epochs, state_history_reference)))
            full_state_history_truth_dict.update(dict(zip(epochs, state_history_truth)))
            full_state_history_initial_dict.update(dict(zip(epochs, state_history_initial)))
            full_state_history_final_dict.update(dict(zip(epochs, state_history_final)))

            self.initial_state_error = state_history_final[-1,:]-state_history_truth[-1,:]
            custom_initial_state = state_history_final[-1,:]
            custom_initial_state_truth = state_history_truth[-1,:]

            # Add process noise element to the apriori covariance for the next estimation arc
            process_noise = np.random.normal(loc=0, scale=0*np.abs(self.initial_state_error), size=self.initial_state_error.shape)
            process_noise_covariance = np.outer(process_noise, process_noise)
            # process_noise_covariance = np.diag([10e0, 10e0, 10e0, 1e-2, 1e-2, 1e-2, 10e0, 10e0, 10e0, 1e-2, 1e-2, 1e-2])**2
            # print(process_noise_covariance)


            # print(process_noise_covariance)
            self.apriori_covariance = np.stack(list(propagated_covariance_final.values()))[-1] + process_noise_covariance

            mission_epoch += propagation_times[batch_count]
            batch_count += 1

            # Set condition on the maximum amount of error allowed before the station-keeping should take place
            if include_station_keeping:
                if (mission_epoch-self.mission_start_epoch)%self.station_keeping_epoch == 0 and mission_epoch != self.mission_start_epoch:

                    lists = [[0, self.target_point_epochs]]
                    for i, list1 in enumerate(lists):
                        dynamic_model.simulation_start_epoch_MJD = self.batch_start_times[batch_count]
                        start_time = time.time()
                        station_keeping = StationKeeping.StationKeeping(dynamic_model, custom_initial_state=custom_initial_state, custom_propagation_time=max(list1[1]), step_size=self.step_size)
                        delta_v = station_keeping.get_corrected_state_vector(cut_off_epoch=list1[0], correction_epoch=list1[0], target_point_epochs=list1[1])
                        lists[i].append(time.time()-start_time)

                    # Generate random noise to simulate station-keeping errors
                    delta_v_noise = np.random.normal(loc=0, scale=0.0000001*np.abs(delta_v), size=delta_v.shape)
                    custom_initial_state[9:12] += delta_v + delta_v_noise
                    custom_initial_state_truth[9:12] += delta_v

                    delta_v_uncertainty = np.zeros((12,))
                    delta_v_uncertainty[9:12] = delta_v_noise
                    # print(np.outer(delta_v_uncertainty, delta_v_uncertainty))
                    self.apriori_covariance += np.outer(delta_v_uncertainty, delta_v_uncertainty)


        return full_estimation_error_dict, full_reference_state_deviation_dict, full_propagated_covariance_dict, full_propagated_formal_errors_dict,\
               full_state_history_reference_dict, full_state_history_reference_dict, full_state_history_initial_dict, full_state_history_final_dict


    def plot_navigation_results(self, full_estimation_error_dict, full_reference_state_deviation_dict, full_propagated_covariance_dict, full_propagated_formal_errors_dict):

        propagated_covariance_epochs = np.stack(list(full_propagated_covariance_dict.keys()))
        propagated_covariance_history = np.stack(list(full_propagated_covariance_dict.values()))
        full_estimation_error_epochs = np.stack(list(full_estimation_error_dict.keys()))
        full_estimation_error_history = np.stack(list(full_estimation_error_dict.values()))
        full_reference_state_deviation_epochs = np.stack(list(full_reference_state_deviation_dict.keys()))
        full_reference_state_deviation_history = np.stack(list(full_reference_state_deviation_dict.values()))
        full_propagated_formal_errors_epochs = np.stack(list(full_propagated_formal_errors_dict.keys()))
        full_propagated_formal_errors_history = np.stack(list(full_propagated_formal_errors_dict.values()))

        full_estimation_error_history = np.array([interp1d(full_estimation_error_epochs, state, kind='linear', fill_value='extrapolate')(propagated_covariance_epochs) for state in full_estimation_error_history.T]).T


        # show areas where there are no observations:
        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        reference_epoch_array = mission_start_epoch*np.ones(np.shape(full_propagated_formal_errors_epochs))
        ax[0].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, 3*full_propagated_formal_errors_history[:,0:3], label=[r"$3\sigma_{x}$", r"$3\sigma_{y}$", r"$3\sigma_{z}$"])
        ax[1].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, 3*full_propagated_formal_errors_history[:,6:9], label=[r"$3\sigma_{x}$", r"$3\sigma_{y}$", r"$3\sigma_{z}$"])
        for j in range(2):
            for i, gap in enumerate(observation_windows):
                ax[j].axvspan(
                    xmin=gap[0]-mission_start_epoch,
                    xmax=gap[1]-mission_start_epoch,
                    color="gray",
                    alpha=0.1,
                    label="Observation window" if i == 0 else None)
                ax[j].set_ylabel(r"$\sigma$ [m]")
                ax[j].grid(alpha=0.5, linestyle='--')
        ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
        ax[0].set_title("LPF")
        ax[1].set_title("LUMIO")
        fig.suptitle("Propagated formal errors")
        plt.legend()
        # plt.show()

        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        reference_epoch_array = mission_start_epoch*np.ones(np.shape(full_propagated_formal_errors_epochs))
        ax[0].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, sigma_number*np.linalg.norm(full_propagated_formal_errors_history[:, 0:3], axis=1))
        ax[1].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, sigma_number*np.linalg.norm(full_propagated_formal_errors_history[:, 6:9], axis=1))
        for j in range(2):
            for i, gap in enumerate(observation_windows):
                ax[j].axvspan(
                    xmin=gap[0]-mission_start_epoch,
                    xmax=gap[1]-mission_start_epoch,
                    color="gray",
                    alpha=0.1,
                    label="Observation window" if i == 0 else None)
                ax[j].set_ylabel("3D RSS OD \n position uncertainty [m]")
                ax[j].grid(alpha=0.5, linestyle='--')
                ax[j].set_yscale("log")
        ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
        ax[0].set_title("LPF")
        ax[1].set_title("LUMIO")
        fig.suptitle("Total position uncertainty")
        plt.legend()
        # plt.show()


        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        reference_epoch_array = mission_start_epoch*np.ones(np.shape(full_reference_state_deviation_epochs))
        for j in range(2):
            labels = ["x", "y", "z"]
            for i in range(3):
                ax[j].plot(utils.convert_epochs_to_MJD(full_reference_state_deviation_epochs)-reference_epoch_array, full_reference_state_deviation_history[:,6*j+i], label=labels[i])
            for i, gap in enumerate(observation_windows):
                ax[j].axvspan(
                    xmin=gap[0]-mission_start_epoch,
                    xmax=gap[1]-mission_start_epoch,
                    color="gray",
                    alpha=0.1,
                    label="Observation window" if i == 0 else None)
            ax[j].set_ylabel(r"$\mathbf{r}-\hat{\mathbf{r}}_{ref}$ [m]")
            ax[j].grid(alpha=0.5, linestyle='--')
        ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
        ax[0].set_title("LPF")
        ax[1].set_title("LUMIO")
        fig.suptitle("Deviation from reference orbit")
        plt.legend()
        # plt.show()

        fig, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
        reference_epoch_array = mission_start_epoch*np.ones(np.shape(propagated_covariance_epochs))
        for k in range(2):
            for j in range(2):
                colors = ["red", "green", "blue"]
                symbols = [[r"x", r"y", r"z"], [r"v_{x}", r"v_{y}", r"v_{z}"]]
                ylabels = [r"$\mathbf{r}-\hat{\mathbf{r}}$ [m]", r"$\mathbf{v}-\hat{\mathbf{v}}$ [m]"]
                for i in range(3):
                    sigma = sigma_number*full_propagated_formal_errors_history[:, 3*k+6*j+i]

                    ax[k][j].plot(utils.convert_epochs_to_MJD(propagated_covariance_epochs)-reference_epoch_array, sigma, color=colors[i], ls="--", label=f"$3\sigma_{{{symbols[k][i]}}}$", alpha=0.3)
                    ax[k][j].plot(utils.convert_epochs_to_MJD(propagated_covariance_epochs)-reference_epoch_array, -sigma, color=colors[i], ls="-.", alpha=0.3)
                    ax[k][j].plot(utils.convert_epochs_to_MJD(propagated_covariance_epochs)-reference_epoch_array, full_estimation_error_history[:,3*k+6*j+i], color=colors[i], label=f"${symbols[k][i]}-\hat{{{symbols[k][i]}}}$")
                ax[k][0].set_ylabel(ylabels[k])
                ax[k][j].grid(alpha=0.5, linestyle='--')
                for i, gap in enumerate(observation_windows):
                    ax[k][j].axvspan(
                        xmin=gap[0]-mission_start_epoch,
                        xmax=gap[1]-mission_start_epoch,
                        color="gray",
                        alpha=0.1,
                        label="Observation window" if i == 0 else None)

                # ax[0][0].set_ylim(-1000, 1000)
                # ax[1][0].set_ylim(-0.3, 0.3)

                ax[-1][j].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")

                # Set y-axis tick label format to scientific notation with one decimal place
                ax[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

            ax[k][0].set_title("LPF")
            ax[k][1].set_title("LUMIO")
            ax[k][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left')

        fig.suptitle("Estimation errors")
        plt.tight_layout()
        plt.show()

# batch_start_times = np.array([60390, 60394.7, 60401.5, 60406.5])
# batch_end_times = np.array([60392.5, 60397.2, 60404, 60409])

# observation_windows = list(zip(batch_start_times, batch_end_times))
# sigma_number = 3

# mission_time = 14
# mission_start_epoch = 60390
# mission_end_epoch = mission_start_epoch + mission_time
# mission_epoch = mission_start_epoch

# # Initial batch timing settings
# propagation_time = 1
# batch_start_times = np.arange(mission_start_epoch, mission_end_epoch, propagation_time)
# batch_end_times = np.arange(propagation_time+mission_start_epoch, propagation_time+mission_end_epoch, propagation_time)
# observation_windows = list(zip(batch_start_times, batch_end_times))

# dynamic_model_list = ["low_fidelity", "three_body_problem", 0]
# truth_model_list = ["low_fidelity", "three_body_problem", 0]
# dynamic_model_list = ["high_fidelity", "point_mass", 0]
# truth_model_list = ["high_fidelity", "point_mass", 7]


# navigation_simulator = NavigationSimulator(observation_windows, dynamic_model_list, truth_model_list)
# full_estimation_error_dict, full_reference_state_deviation_dict, full_propagated_covariance_dict, full_propagated_formal_errors_dict = navigation_simulator.perform_navigation()
# navigation_simulator.plot_navigation_results(full_estimation_error_dict, full_reference_state_deviation_dict, full_propagated_covariance_dict, full_propagated_formal_errors_dict)

# plt.show()