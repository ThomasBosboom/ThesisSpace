# Standard
import os
import sys
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d


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

    def __init__(self, observation_windows, dynamic_model_list, truth_model_list, step_size=0.01, station_keeping_epoch=4, target_point_epochs=[3], include_station_keeping=True, exclude_first_manouvre=False):

        # Miscellaneous
        self.step_size = step_size

        # Managing the dynamic model specifications
        self.model_type, self.model_name, self.model_number = dynamic_model_list[0], dynamic_model_list[1], dynamic_model_list[2]
        self.model_type_truth, self.model_name_truth, self.model_number_truth = truth_model_list[0], truth_model_list[1], truth_model_list[2]

        # Station keeping parameters
        self.include_station_keeping = include_station_keeping
        self.station_keeping_epoch = station_keeping_epoch
        self.target_point_epochs = target_point_epochs
        self.exclude_first_manouvre = exclude_first_manouvre

        self.custom_initial_state = None
        self.custom_initial_state_truth = None
        self.initial_state_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])/10
        self.apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2

        # Managing the timing aspect of the navigation arcs
        self.observation_windows = observation_windows
        self.mission_start_time = 60390
        self.mission_time = self.observation_windows[-1][-1]-self.mission_start_time
        self.mission_end_time = self.mission_start_time + self.mission_time

        self.batch_start_times = np.array([t[0] for t in self.observation_windows])
        self.batch_end_times = np.array([t[1] for t in self.observation_windows])

        self.times = list(set([self.mission_start_time] + [item for sublist in self.observation_windows for item in sublist] + [self.mission_end_time]))

        self.station_keeping_epochs = []
        if self.include_station_keeping:
            for station_keeping_epoch in range(self.mission_start_time, self.mission_end_time, 4):
                if station_keeping_epoch != self.mission_start_time:
                    self.station_keeping_epochs.append(station_keeping_epoch)
            if self.exclude_first_manouvre:
                self.station_keeping_epochs.remove(60394)

        print(self.station_keeping_epochs)
        self.times.extend(self.station_keeping_epochs)
        self.times = np.round(sorted(list(set(self.times))), 2)
        print(self.times)


    def perform_navigation(self):

        estimation_arc = 0
        navigation_arc = 0
        full_estimation_error_dict = dict()
        full_reference_state_deviation_dict = dict()
        full_propagated_covariance_dict = dict()
        full_propagated_formal_errors_dict = dict()
        full_state_history_reference_dict = dict()
        full_state_history_truth_dict = dict()
        full_state_history_initial_dict = dict()
        full_state_history_final_dict = dict()
        delta_v_dict = dict()
        print("Starting navigation simulation. \n ======================")
        for t, time in enumerate(self.times):

            navigation_arc_duration = np.round(np.diff(self.times)[navigation_arc], 2)

            print(f"Start of navigation arc {navigation_arc} at {time} for {navigation_arc_duration} days")

            # print(f"EPOCH AND VALUE OF FIRST CUSTOM STATE OF ARC {navigation_arc}: ", self.custom_initial_state)

            if navigation_arc_duration == 0 or time == self.times[-1]:
                continue

            # Define dynamic and truth models to calculate the relevant histories
            dynamic_model_objects = utils.get_dynamic_model_objects(time, navigation_arc_duration, custom_initial_state=self.custom_initial_state)
            dynamic_model = dynamic_model_objects[self.model_type][self.model_name][self.model_number]

            truth_model_objects = utils.get_dynamic_model_objects(time, navigation_arc_duration, custom_initial_state=self.custom_initial_state_truth)
            truth_model = truth_model_objects[self.model_type_truth][self.model_name_truth][self.model_number_truth]


            ##############################################################
            #### PROPAGATIONS OF STATE HISTORIES #########################
            ##############################################################

            # Obtain the initial state of the whole simulation once
            if navigation_arc == 0:
                epochs, state_history_initialize, dependent_variables_history_initialize = \
                    Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model, solve_variational_equations=False)

                self.custom_initial_state_truth = state_history_initialize[0,:]
                self.custom_initial_state = self.custom_initial_state_truth + self.initial_state_error

            dynamic_model.custom_initial_state = self.custom_initial_state
            truth_model.custom_initial_state = self.custom_initial_state_truth

            # Update the reference orbit that the estimated orbit should follow
            state_history_reference = list()
            for body in dynamic_model.bodies_to_propagate:
                state_history_reference.append(validation.get_reference_state_history(time,
                                                                                    navigation_arc_duration,
                                                                                        satellite=body,
                                                                                        step_size=self.step_size,
                                                                                        get_full_history=True))
            state_history_reference = np.concatenate(state_history_reference, axis=1)

            epochs, state_history_initial, dependent_variables_history_initial, state_transition_history_initial = \
                Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model,
                                                                                                    custom_initial_state=self.custom_initial_state,
                                                                                                    custom_propagation_time=navigation_arc_duration,
                                                                                                    solve_variational_equations=True)

            epochs, state_history_truth, dependent_variables_history_truth = \
                Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(truth_model,
                                                                                                        custom_initial_state=self.custom_initial_state_truth,
                                                                                                        custom_propagation_time=navigation_arc_duration,
                                                                                                        solve_variational_equations=False)

            # Save the propagated histories of the uncertainties
            propagated_covariance_initial = dict()
            propagated_formal_errors_initial = dict()
            for i in range(len(epochs)):
                propagated_covariance = state_transition_history_initial[i] @ self.apriori_covariance @ state_transition_history_initial[i].T
                propagated_covariance_initial.update({epochs[i]: propagated_covariance})
                propagated_formal_errors_initial.update({epochs[i]: np.sqrt(np.diagonal(propagated_covariance))})


            ##############################################################
            #### LOOP FOR ESTIMATION ARCS ################################
            ##############################################################
            estimation_arc_activated = False
            if time in self.batch_start_times:

                estimation_arc_activated = True

                observation_window = self.observation_windows[estimation_arc]
                estimation_arc_duration = np.around(observation_window[1]-observation_window[0], 2)

                print(f"Start of estimation arc {estimation_arc} at navigation arc {navigation_arc},  at {time} for {estimation_arc_duration} days")

                # Define dynamic models and select one to test the estimation on
                dynamic_model_objects = utils.get_dynamic_model_objects(time,
                                                                        estimation_arc_duration,
                                                                        custom_model_dict=None,
                                                                        get_only_first=False,
                                                                        custom_initial_state=self.custom_initial_state)
                dynamic_model = dynamic_model_objects[self.model_type][self.model_name][self.model_number]

                # print(f"USE INITIAL STATE IN ESTIMATION ARC {estimation_arc}: \n", self.custom_initial_state)
                # print(f"CHECK WITH VALUE IN DYNAMIC MODEL {dynamic_model}: \n", self.custom_initial_state==dynamic_model.custom_initial_state)
                # print(f"CHECK WITH VALUE IN TRUTH MODEL {truth_model}: \n", self.custom_initial_state_truth==truth_model.custom_initial_state)
                # print("START THE FORMER ESTIMATOR")

                # Obtain estimation results for given batch and extract results of the estimation arc
                estimation_model_objects_results = utils.get_estimation_model_results({self.model_type: {self.model_name: [dynamic_model]}},
                                                                                    get_only_first=False,
                                                                                    custom_truth_model=truth_model,
                                                                                    apriori_covariance=self.apriori_covariance,
                                                                                    initial_state_error=self.initial_state_error)
                # print("START THE LATTER ESTIMATOR")
                # test = estimation_model.EstimationArc(dynamic_model, truth_model, apriori_covariance=self.apriori_covariance, initial_state_error=self.initial_state_error).get_estimation_results()
                # print(test)
                # plt.show()

                estimation_model_objects_result = estimation_model_objects_results[self.model_type][self.model_name][0]
                estimation_output = estimation_model_objects_result[0]
                parameter_history = estimation_output.parameter_history
                final_covariance = estimation_output.covariance
                formal_errors = estimation_output.formal_errors
                best_iteration = estimation_output.best_iteration

                # print(f"Diff initial/final parameter arc {estimation_arc}: \n", parameter_history[:,best_iteration]-parameter_history[:,0])

                epochs, state_history_final, dependent_variables_history_final, state_transition_history_final = \
                    Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model,
                                                                                                    custom_initial_state=parameter_history[:,best_iteration],
                                                                                                    custom_propagation_time=estimation_arc_duration,
                                                                                                    solve_variational_equations=True)

                propagated_covariance_final = dict()
                propagated_formal_errors_final = dict()
                for i in range(len(epochs)):
                    propagated_covariance = state_transition_history_final[i] @ estimation_output.covariance @ state_transition_history_final[i].T
                    propagated_covariance_final.update({epochs[i]: propagated_covariance})
                    propagated_formal_errors_final.update({epochs[i]: np.sqrt(np.diagonal(propagated_covariance))})

                # Save the history
                full_state_history_final_dict.update(dict(zip(epochs, state_history_final)))

                estimation_arc += 1

            # Update the values for the next navigation arc
            self.custom_initial_state_truth = state_history_truth[-1,:]

            if estimation_arc_activated:
                self.custom_initial_state = state_history_final[-1,:]
                self.initial_state_error = state_history_final[-1,:]-state_history_truth[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_final.values()))[-1]
            else:
                # print("state history inital final entry: ", state_history_initial[-1,:])
                self.custom_initial_state = state_history_initial[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_initial.values()))[-1]

            process_noise_sigmas = np.array([1e2, 1e2, 1e2, 1e-3, 1e-3, 1e-3, 1e2, 1e2, 1e2, 1e-3, 1e-3, 1e-3])/10
            process_noise = np.random.normal(scale=process_noise_sigmas, size=len(process_noise_sigmas))
            # process_noise = process_noise_sigmas
            # print(process_noise)

            if estimation_arc_activated:
                # self.custom_initial_state += process_noise
                self.apriori_covariance += np.outer(process_noise, process_noise)

            # Save histories for reading out later
            full_estimation_error_dict.update(dict(zip(epochs, state_history_initial-state_history_truth)))
            full_reference_state_deviation_dict.update(dict(zip(epochs, state_history_initial-state_history_reference)))
            full_propagated_covariance_dict.update(propagated_covariance_initial)
            full_propagated_formal_errors_dict.update(propagated_formal_errors_initial)
            full_state_history_reference_dict.update(dict(zip(epochs, state_history_reference)))
            full_state_history_truth_dict.update(dict(zip(epochs, state_history_truth)))
            full_state_history_initial_dict.update(dict(zip(epochs, state_history_initial)))

            # print("Dict of estimation error: ", full_estimation_error_dict)

            print(f"end of navigation arc {navigation_arc}")

            ##############################################################
            #### STATION KEEPING #########################################
            ##############################################################

            if self.include_station_keeping:
                if self.times[navigation_arc+1] in self.station_keeping_epochs:

                    lists = [[0, self.target_point_epochs]]
                    for i, list1 in enumerate(lists):
                        dynamic_model.simulation_start_epoch_MJD = self.times[navigation_arc+1]
                        station_keeping = StationKeeping.StationKeeping(dynamic_model, custom_initial_state=self.custom_initial_state, custom_propagation_time=max(list1[1]), step_size=self.step_size)
                        delta_v = station_keeping.get_corrected_state_vector(cut_off_epoch=list1[0], correction_epoch=list1[0], target_point_epochs=list1[1])

                    # Generate random noise to simulate station-keeping errors
                    delta_v_noise = np.random.normal(loc=0, scale=0.00000000002*np.abs(delta_v), size=delta_v.shape)
                    self.custom_initial_state[9:12] += delta_v
                    self.custom_initial_state_truth[9:12] += delta_v + delta_v_noise

                    print(f"Correction at {self.times[navigation_arc+1]}: ", delta_v)

                    delta_v_dict.update({self.times[navigation_arc+1]: delta_v})

                    # delta_v_uncertainty = np.zeros((12,))
                    # delta_v_uncertainty[9:12] = delta_v_noise
                    # self.apriori_covariance += np.outer(delta_v_uncertainty, delta_v_uncertainty)

            # print("CURRENT LENGTH OF DICT: ", len(full_state_history_initial_dict.keys()))
            if navigation_arc<len(self.times)-2:
                navigation_arc += 1
            else:
                print("Navigation simulation has ended.")
                break

        return full_estimation_error_dict, full_reference_state_deviation_dict, full_propagated_covariance_dict, full_propagated_formal_errors_dict,\
               full_state_history_reference_dict, full_state_history_truth_dict, full_state_history_initial_dict, full_state_history_final_dict, delta_v_dict



# List of tuples representing regions
# observation_windows = [(60390, 60391), (60391, 60392), (60392, 60393), (60393, 60394), (60394, 60395), (60395, 60396), (60396, 60397), (60397, 60398), (60398, 60399)]
# observation_windows = [(60392, 60394), (60396, 60398), (60400, 60402), (60404, 60406), (60408, 60410), (60412, 60414)]
# observation_windows = [(60393, 60394), (60397, 60398), (60401, 60402), (60405, 60406), (60409, 60410), (60413, 60414)]
# observation_windows = [(60390, 60391), (60392, 60393), (60394, 60395), (60396, 60397), (60398, 60399), (60400, 60401)]
# observation_windows = [(60391, 60392), (60393, 60394), (60395, 60396), (60397, 60398), (60399, 60400), (60401, 60402),
#                         (60401, 60402), (60403, 60404), (60405, 60406), (60407, 60408),(60409, 60410), (60411, 60412)
#                       ]
# observation_windows = [(60392, 60394), (60396, 60398), (60400, 60402), (60404, 60406), (60408, 60410), (60412, 60414)]
# observation_windows = [(60391, 60393), (60395, 60397), (60399, 60400), (60403, 60405), (60407, 60409), (60411, 60413)]
# observation_windows = [(60391, 60394), (60395, 60398), (60399, 60402), (60403, 60406), (60407, 60410), (60411, 60414)]
# observation_windows = [(60391, 60392), (60393, 60394), (60395, 60396), (60397, 60398)]


# dynamic_model_list = ["low_fidelity", "three_body_problem", 0]
# truth_model_list = ["low_fidelity", "three_body_problem", 0]
# # dynamic_model_list = ["high_fidelity", "point_mass", 0]
# # truth_model_list = ["high_fidelity", "point_mass", 0]
# # dynamic_model_list = ["high_fidelity", "spherical_harmonics_srp", 3]
# # truth_model_list = ["high_fidelity", "spherical_harmonics_srp", 3]
# mission_start_time = 60390


# navigation_simulator = NavigationSimulator(observation_windows, dynamic_model_list, truth_model_list, include_station_keeping=True, exclude_first_manouvre=True)
# full_estimation_error_dict, full_reference_state_deviation_dict, full_propagated_covariance_dict, full_propagated_formal_errors_dict,\
#                full_state_history_reference_dict, full_state_history_truth_dict, full_state_history_initial_dict, full_state_history_final_dict, delta_v_dict = navigation_simulator.perform_navigation()

# propagated_covariance_epochs = np.stack(list(full_propagated_covariance_dict.keys()))
# propagated_covariance_history = np.stack(list(full_propagated_covariance_dict.values()))
# full_estimation_error_epochs = np.stack(list(full_estimation_error_dict.keys()))
# full_estimation_error_history = np.stack(list(full_estimation_error_dict.values()))
# full_reference_state_deviation_epochs = np.stack(list(full_reference_state_deviation_dict.keys()))
# full_reference_state_deviation_history = np.stack(list(full_reference_state_deviation_dict.values()))
# full_propagated_formal_errors_epochs = np.stack(list(full_propagated_formal_errors_dict.keys()))
# full_propagated_formal_errors_history = np.stack(list(full_propagated_formal_errors_dict.values()))

# full_state_history_reference_history = np.stack(list(full_state_history_reference_dict.values()))
# full_state_history_truth_history = np.stack(list(full_state_history_truth_dict.values()))
# full_state_history_initial_history = np.stack(list(full_state_history_initial_dict.values()))
# full_state_history_initial_epochs = np.stack(list(full_state_history_initial_dict.keys()))
# full_state_history_final_history = np.stack(list(full_state_history_final_dict.values()))
# full_estimation_error_history = np.array([interp1d(full_estimation_error_epochs, state, kind='linear', fill_value='extrapolate')(propagated_covariance_epochs) for state in full_estimation_error_history.T]).T


# print("================")

# print("total delta_v: ", delta_v_dict)


# # print("keys estimation error: ", [key for key in full_estimation_error_dict.keys()])
# # print("keys truth history error: ", [key for key in full_state_history_truth_dict.keys()])
# # print("keys truth initial error: ", [key for key in full_state_history_initial_dict.keys()])

# # print("diff estimation and truth: ", np.array([key for key in full_estimation_error_dict.keys()])-np.array([key for key in full_state_history_truth_dict.keys()]))
# # print("diff estimation and initial: ", np.array([key for key in full_estimation_error_dict.keys()])-np.array([key for key in full_state_history_initial_dict.keys()]))
# # print("diff truth and initial: ", np.array([key for key in full_state_history_truth_dict.keys()])-np.array([key for key in full_state_history_initial_dict.keys()]))
# # print("length of the estimation dict: ", np.shape(np.array([key for key in full_estimation_error_dict.keys()])))
# # print("length of the initial dict: ", np.shape(np.array([key for key in full_state_history_initial_dict.keys()])))
# # print("length of the truth dict: ", np.shape(np.array([key for key in full_state_history_truth_dict.keys()])))

# fig, ax = plt.subplots(3, 1, figsize=(9, 5), sharex=True)
# plt.title("position")
# ax[0].plot(full_state_history_initial_epochs, full_state_history_initial_history[:,8]-full_state_history_truth_history[:,8], label="error", color="blue", marker="+")
# ax[1].plot(full_state_history_initial_epochs, full_state_history_initial_history[:,8], label="estimated", color="red", marker="+")
# ax[2].plot(full_state_history_initial_epochs, full_state_history_truth_history[:,8], label="truth", color="green", marker="+")
# # ax[0].set_xlim(198, 202)
# # plt.show()

# fig, ax = plt.subplots(3, 1, figsize=(9, 5), sharex=True)
# plt.title("velocity")
# ax[0].plot(full_state_history_initial_epochs, full_state_history_initial_history[:,11]-full_state_history_truth_history[:,11], label="error", color="blue", marker="+")
# ax[1].plot(full_state_history_initial_epochs, full_state_history_initial_history[:,11], label="estimated", color="red", marker="+")
# ax[2].plot(full_state_history_initial_epochs, full_state_history_truth_history[:,11], label="truth", color="green", marker="+")
# # ax[0].set_xlim(198, 202)
# # plt.show()



# fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
# # reference_epoch_array = mission_start_time*np.ones(np.shape(full_reference_state_deviation_epochs))
# for j in range(2):
#     labels = ["x", "y", "z"]
#     start_epoch = full_reference_state_deviation_epochs[0]
#     relative_epoch = full_reference_state_deviation_epochs-full_reference_state_deviation_epochs[0]
#     for i in range(3):
#         ax[j].plot(relative_epoch, full_reference_state_deviation_history[:,6*j+i], label=labels[i])
#     for i, gap in enumerate(observation_windows):
#         ax[j].axvspan(
#             xmin=gap[0]-start_epoch,
#             xmax=gap[1]-start_epoch,
#             color="gray",
#             alpha=0.1,
#             label="Observation window" if i == 0 else None)
#     ax[j].set_ylabel(r"$\mathbf{r}-\hat{\mathbf{r}}_{ref}$ [m]")
#     ax[j].grid(alpha=0.5, linestyle='--')
# ax[-1].set_xlabel(f"Time since MJD {mission_start_time} [days]")
# ax[0].set_title("LPF")
# ax[1].set_title("LUMIO")
# fig.suptitle("Deviation from reference orbit")
# plt.legend()
# # plt.show()

# fig1_3d = plt.figure()
# ax = fig1_3d.add_subplot(111, projection='3d')
# ax.plot(full_state_history_reference_history[:,0], full_state_history_reference_history[:,1], full_state_history_reference_history[:,2], label="LPF ref", color="green")
# ax.plot(full_state_history_reference_history[:,6], full_state_history_reference_history[:,7], full_state_history_reference_history[:,8], label="LUMIO ref", color="green")
# ax.plot(full_state_history_initial_history[:,0], full_state_history_initial_history[:,1], full_state_history_initial_history[:,2], label="LPF estimated")
# ax.plot(full_state_history_initial_history[:,6], full_state_history_initial_history[:,7], full_state_history_initial_history[:,8], label="LUMIO estimated")
# # ax.plot(full_state_history_final_history[:,0], full_state_history_final_history[:,1], full_state_history_final_history[:,2], label="LPF estimated")
# # ax.plot(full_state_history_final_history[:,6], full_state_history_final_history[:,7], full_state_history_final_history[:,8], label="LUMIO estimated")
# ax.plot(full_state_history_truth_history[:,0], full_state_history_truth_history[:,1], full_state_history_truth_history[:,2], label="LPF truth", color="black", ls="--")
# ax.plot(full_state_history_truth_history[:,6], full_state_history_truth_history[:,7], full_state_history_truth_history[:,8], label="LUMIO truth", color="black", ls="--")
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')
# plt.legend()




# # show areas where there are no observations:
# fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
# start_epoch = full_propagated_formal_errors_epochs[0]
# relative_epochs = full_propagated_formal_errors_epochs-start_epoch
# ax[0].plot(relative_epochs, 3*full_propagated_formal_errors_history[:,0:3], label=[r"$3\sigma_{x}$", r"$3\sigma_{y}$", r"$3\sigma_{z}$"])
# ax[1].plot(relative_epochs, 3*full_propagated_formal_errors_history[:,6:9], label=[r"$3\sigma_{x}$", r"$3\sigma_{y}$", r"$3\sigma_{z}$"])
# for j in range(2):
#     for i, gap in enumerate(observation_windows):
#         ax[j].axvspan(
#             xmin=gap[0]-start_epoch,
#             xmax=gap[1]-start_epoch,
#             color="gray",
#             alpha=0.1,
#             label="Observation window" if i == 0 else None)
#         ax[j].set_ylabel(r"$\sigma$ [m]")
#         ax[j].grid(alpha=0.5, linestyle='--')
#         ax[j].set_yscale("log")
# ax[-1].set_xlabel(f"Time since MJD {mission_start_time} [days]")
# ax[0].set_title("LPF")
# ax[1].set_title("LUMIO")
# fig.suptitle("Propagated formal errors")
# plt.legend()

# fig, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
# for k in range(2):
#     for j in range(2):
#         colors = ["red", "green", "blue"]
#         symbols = [[r"x", r"y", r"z"], [r"v_{x}", r"v_{y}", r"v_{z}"]]
#         ylabels = [r"$\mathbf{r}-\hat{\mathbf{r}}$ [m]", r"$\mathbf{v}-\hat{\mathbf{v}}$ [m]"]
#         for i in range(3):
#             sigma = 3*full_propagated_formal_errors_history[:, 3*k+6*j+i]
#             start_epoch = propagated_covariance_epochs[0]
#             relative_epochs = propagated_covariance_epochs-start_epoch
#             ax[k][j].plot(relative_epochs, sigma, color=colors[i], ls="--", label=f"$3\sigma_{{{symbols[k][i]}}}$", alpha=0.3)
#             ax[k][j].plot(relative_epochs, -sigma, color=colors[i], ls="-.", alpha=0.3)
#             ax[k][j].plot(relative_epochs, full_estimation_error_history[:,3*k+6*j+i], color=colors[i], label=f"${symbols[k][i]}-\hat{{{symbols[k][i]}}}$")
#         ax[k][0].set_ylabel(ylabels[k])
#         ax[k][j].grid(alpha=0.5, linestyle='--')
#         for i, gap in enumerate(observation_windows):
#             ax[k][j].axvspan(
#                 xmin=gap[0]-start_epoch,
#                 xmax=gap[1]-start_epoch,
#                 color="gray",
#                 alpha=0.1,
#                 label="Observation window" if i == 0 else None)

#         # ax[0][0].set_ylim(-1000, 1000)
#         # ax[1][0].set_ylim(-0.3, 0.3)

#         ax[-1][j].set_xlabel(f"Time since MJD {mission_start_time} [days]")

#         # Set y-axis tick label format to scientific notation with one decimal place
#         ax[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         ax[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

#     ax[k][0].set_title("LPF")
#     ax[k][1].set_title("LUMIO")
#     ax[k][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left')

# fig.suptitle("Estimation errors")
# plt.tight_layout()

# plt.show()
