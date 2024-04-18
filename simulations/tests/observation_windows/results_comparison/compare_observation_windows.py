# Standard
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils, helper_functions
from src import NavigationSimulator, PlotNavigationResults


#################################################################
###### Compare results observation windows ######################
#################################################################

### Compare difference timing cases
# dynamic_model_list = ["HF", "PMSRP", 0]
# truth_model_list = ["HF", "PMSRP", 0]
num_runs = 1
# custom_station_keeping_error =  2e-2
# custom_range_noise = 2.98
# custom_target_point_epoch = 3
# custom_initial_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
# custom_apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
# custom_orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])
simulation_start_epoch = 60390


# Collect a series of observation window sets to compare
observation_windows_list = []
# observation_windows_list.append(helper_functions.get_custom_observation_windows(3, 3, 3, 0.1))
# observation_windows_list.append([(60390.38, 60390.51), (60390.83, 60390.96), (60391.28, 60391.41), (60391.73, 60391.86), (60392.18, 60392.31), (60392.63, 60392.76), (60393.08, 60393.21), (60393.53, 60393.66), (60393.98, 60394.11), (60394.43, 60394.56), (60394.88, 60395.01), (60395.33, 60395.46), (60395.78, 60395.91), (60396.23, 60396.36), (60396.68, 60396.81), (60397.13, 60397.26), (60397.58, 60397.71), (60398.03, 60398.16), (60398.48, 60398.61), (60398.93, 60399.06), (60399.38, 60399.51), (60399.83, 60399.96), (60400.28, 60400.41), (60400.73, 60400.86), (60401.18, 60401.31), (60401.63, 60401.76), (60402.08, 60402.21), (60402.53, 60402.66), (60402.98, 60403.11), (60403.43, 60403.56), (60403.88, 60404.01), (60404.33, 60404.46), (60404.78, 60404.91), (60405.23, 60405.36), (60405.68, 60405.81), (60406.13, 60406.26), (60406.58, 60406.71), (60407.03, 60407.16), (60407.48, 60407.61), (60407.93, 60408.06), (60408.38, 60408.51),
# (60408.83, 60408.96), (60409.28, 60409.41), (60409.73, 60409.86), (60410.18, 60410.31), (60410.63, 60410.76), (60411.08, 60411.21), (60411.53, 60411.66), (60411.98, 60412.11), (60412.43, 60412.56), (60412.88, 60413.01), (60413.33, 60413.46), (60413.78, 60413.91), (60414.23, 60414.36), (60414.68, 60414.81), (60415.13, 60415.26), (60415.58, 60415.71), (60416.03, 60416.16), (60416.48, 60416.61), (60416.93, 60417.06), (60417.38, 60417.51), (60417.83, 60417.96)])
observation_windows_list.append(helper_functions.get_custom_observation_windows(28, 3, 1, 1, simulation_start_epoch=simulation_start_epoch))

# Run the navigation routine using given settings
navigation_results_list = []
navigation_simulator_list = []
navigation_output_list = []
for run in range(num_runs):
    for i, observation_windows in enumerate(observation_windows_list):

        print("Running with observation windows: \n", observation_windows)

        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows)

        navigation_output = navigation_simulator.perform_navigation()
        navigation_results = navigation_output.navigation_results
        navigation_simulator = navigation_output.navigation_simulator

        delta_v = navigation_output.navigation_results[8][1]
        delta_v_per_skm = np.linalg.norm(delta_v, axis=1)
        objective_value = np.sum(delta_v_per_skm)
        print(f"Objective: \n", delta_v_per_skm, objective_value)
        print("End of objective calculation ===============")

        navigation_results_list.append(navigation_output.navigation_results)
        navigation_simulator_list.append(navigation_simulator)
        navigation_output_list.append(navigation_output)


fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
for i, navigation_results in enumerate(navigation_results_list):

    print(f"Delta V for case {i}: \n: ", np.linalg.norm(navigation_results_list[i][8][1], axis=1), np.sum(np.linalg.norm(navigation_results_list[i][8][1], axis=1)))

    # ax[i].plot(navigation_results[3][0]-navigation_results[3][0][0], np.linalg.norm(navigation_results[3][1][:, :3], axis=1))

# plt.show()


for navigation_output in navigation_output_list:
    plot_navigation_results = PlotNavigationResults.PlotNavigationResults(navigation_output)
    plot_navigation_results.plot_estimation_error_history()
    plot_navigation_results.plot_uncertainty_history()
    plot_navigation_results.plot_reference_deviation_history()
    plot_navigation_results.plot_full_state_history()
    plot_navigation_results.plot_formal_error_history()
    plot_navigation_results.plot_observations()
    plot_navigation_results.plot_observability()
    plot_navigation_results.plot_od_error_delta_v_relation()
    plot_navigation_results.plot_correlation_history()

# plt.show()


fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
# fig1, ax1 = plt.subplots(1, 1, figsize=(12, 5), sharex=True)
for navigation_output in navigation_output_list:

    navigation_simulator = navigation_output.navigation_simulator
    navigation_results = navigation_output.navigation_results

    delta_v_history = navigation_simulator.delta_v_dict
    od_error_history = navigation_simulator.full_estimation_error_dict
    estimation_arc_results_dict = navigation_simulator.estimation_arc_results_dict

    full_reference_state_deviation_dict = navigation_simulator.full_reference_state_deviation_dict

    delta_v = []
    od_error_history_at_delta_v = []
    reference_deviation_at_delta_v = []
    for key, value in delta_v_history.items():
        index = min(od_error_history.keys(), key=lambda x: abs(x - key))
        od_error_history_at_delta_v.append(od_error_history[index])
        reference_deviation_at_delta_v.append(full_reference_state_deviation_dict[index])
        delta_v.append(value)

    delta_v = np.array(delta_v)
    od_error_history_at_delta_v = np.array(od_error_history_at_delta_v)
    reference_deviation_at_delta_v = np.array(reference_deviation_at_delta_v)

    abs_delta_v_history = np.linalg.norm(delta_v[:, :3], axis=1)
    abs_pos_od_error_history = np.linalg.norm(od_error_history_at_delta_v[:, 6:9], axis=1)
    abs_pos_deviation_history = np.linalg.norm(reference_deviation_at_delta_v[:, 6:9], axis=1)

    axs[0].scatter(abs_delta_v_history, abs_pos_od_error_history, label=str(navigation_simulator.estimation_arc_durations[-1]))
    axs[1].scatter(abs_delta_v_history, abs_pos_deviation_history, label=str(navigation_simulator.estimation_arc_durations[-1]))


axs[1].set_xlabel(r"||$\Delta V$|| [m/s]")
axs[0].set_ylabel(r"||$\hat{\mathbf{r}}-\mathbf{r}$|| [m]")
axs[1].set_ylabel(r"||$\mathbf{r}-\mathbf{r}_{ref}$|| [m]")
axs[0].set_title("Maneuver cost versus OD error")
axs[1].set_title("Maneuver cost versus reference orbit deviation")
axs[0].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

fig.suptitle("Relations between SKM cost for run of 28 days")
plt.legend()
plt.show()


utils.save_figure_to_folder(figs=[fig], labels=["CostVersusError"], custom_sub_folder_name=file_name)







# mission_start_epoch = 60390
# fig, ax = plt.subplots(len(observation_windows_list), 1, figsize=(12, 5), sharex=True)
# color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# for j in range(len(ax)):
#     for i, gap in enumerate(observation_windows_list[j]):
#         ax[j].axvspan(
#             xmin=gap[0]-mission_start_epoch,
#             xmax=gap[1]-mission_start_epoch,
#             color=color_cycle[i],
#             alpha=0.1,
#             label=f"Arc {i}" if j == 0 else None)
#     for i, epoch in enumerate([windows[1] for windows in observation_windows_list[j]]):
#         station_keeping_epoch = epoch - 60390
#         ax[j].axvline(x=station_keeping_epoch, color='black', linestyle='--', label="SKM" if i==0 else None)

#     ax[j].grid(alpha=0.5, linestyle='--')

#     # Set y-axis tick label format to scientific notation with one decimal place
#     # ax[j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#     # ax[j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# ax[0].set_ylabel("Range [m]")
# ax[1].set_ylabel("Observation Residual [m]")
# ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
# ax[0].legend(bbox_to_anchor=(1, 1.04), loc='upper left')

# # fig.suptitle(f"Intersatellite range observations \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
# plt.tight_layout()
# plt.show()











