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
from src.optimization_models import OptimizationModel
from src import NavigationSimulator, PlotNavigationResults


#################################################################
###### Compare results observation windows ######################
#################################################################


### Extract information from specific folder and run
# folder_name = "PM01_PM01_thr3_skm3_dur28_od1_num1_max20_test"
# batch_name = "09040214"
# folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "variable_arc\data", folder_name)

# concatenated_json = helper_functions.concatenate_json_files(folder_path, batch=batch_name)
# monte_carlo_stats_dict = helper_functions.get_monte_carlo_stats_dict(concatenated_json)

# Extract information directly
# file_name = f"{batch_name}_stats_{folder_name}.json"
# file_path = os.path.join(folder_path, file_name)
# monte_carlo_stats_dict = helper_functions.load_json_file(file_path)

# file_name = f"{batch_name}_run_0_{folder_name}.json"
# file_path = os.path.join(folder_path, file_name)
# run_data = helper_functions.load_json_file(file_path)

# observation_windows_optim = run_data["final_result"]["observation_windows"]
# station_keeping_epochs_optim = run_data["final_result"]["skm_epochs"]
observation_windows_optim = [(60390, 60397), (60400.0, 60401.22475752616), (60404.22475752616, 60405.066958155025), (60408.066958155025, 60408.566958155025), (60411.566958155025, 60412.98370281383), (60415.98370281383, 60416.79029821602)]

### Compare difference timing cases
dynamic_model_list = ["HF", "PMSRP", 0]
truth_model_list = ["HF", "PMSRP", 0]
threshold = 1
skm_to_od_duration = 3
duration = 28
od_duration = 1
custom_station_keeping_error = 1e-10
custom_target_point_epoch = 3
custom_initial_estimation_error = np.array([5e0, 5e0, 5e0, 1e-5, 1e-5, 1e-5, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])*0
custom_apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
custom_orbit_insertion_error = np.array([1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e1, 1e1, 1e1, 1e-4, 1e-4, 1e-4])*1e2
# orbit_insertion_error = np.array([1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e1, 1e1, 1e1, 1e-4, 1e-4, 1e-4])*1e0


# Collect a series of observation window sets to compare
observation_windows_list = []
observation_windows_list.append(helper_functions.get_custom_observation_windows(28, 3, 1, 0.1))
# observation_windows_list.append(helper_functions.get_custom_observation_windows(28, 3, 1, 0.2))
# observation_windows_list.append(helper_functions.get_custom_observation_windows(28, 3, 1, 0.5))
observation_windows_list.append(helper_functions.get_custom_observation_windows(28, 3, 1, 1))
observation_windows_list.append(helper_functions.get_custom_observation_windows(28, 3, 1, 1.5))


navigation_results_list = []
navigation_simulator_list = []
navigation_output_list = []
for i, observation_windows in enumerate(observation_windows_list):

    print("Running with observation windows: \n", observation_windows)

    station_keeping_epochs = [windows[1] for windows in observation_windows]

    # custom_orbit_insertion_error = np.random.normal(loc=0, scale=orbit_insertion_error)
    # print(custom_orbit_insertion_error)

    navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                    dynamic_model_list,
                                                                    truth_model_list,
                                                                    step_size=1e-2,
                                                                    station_keeping_epochs=station_keeping_epochs,
                                                                    target_point_epochs=[custom_target_point_epoch],
                                                                    custom_station_keeping_error=custom_station_keeping_error,
                                                                    custom_initial_estimation_error=custom_initial_estimation_error,
                                                                    custom_apriori_covariance=custom_apriori_covariance,
                                                                    custom_orbit_insertion_error=custom_orbit_insertion_error)

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
    # formal_errror_history = navigation_results[3][0]
    # ax[i].plot(navigation_results[3][0]-navigation_results[3][0][0], np.linalg.norm(navigation_results[3][1][:, :3], axis=1))

# plt.show()


for navigation_output in navigation_output_list:
    plot_navigation_results = PlotNavigationResults.PlotNavigationResults(navigation_output)
    plot_navigation_results.plot_estimation_error_history()
    plot_navigation_results.plot_uncertainty_history()
    # plot_navigation_results.plot_reference_deviation_history()
    # # plot_navigation_results.plot_full_state_history()
    # plot_navigation_results.plot_formal_error_history()
    # # plot_navigation_results.plot_correlation_history()
    # plot_navigation_results.plot_observations()
    # # plot_navigation_results.plot_observability()
    # plot_navigation_results.plot_od_error_delta_v_relation()

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











