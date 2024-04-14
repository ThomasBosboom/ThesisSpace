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
folder_name = "PM01_PM01_thr3_skm3_dur28_od1_num1_max20_test"
batch_name = "09040214"
folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "variable_arc\data", folder_name)

# concatenated_json = helper_functions.concatenate_json_files(folder_path, batch=batch_name)
# monte_carlo_stats_dict = helper_functions.get_monte_carlo_stats_dict(concatenated_json)

# Extract information directly
# file_name = f"{batch_name}_stats_{folder_name}.json"
# file_path = os.path.join(folder_path, file_name)
# monte_carlo_stats_dict = helper_functions.load_json_file(file_path)

file_name = f"{batch_name}_run_0_{folder_name}.json"
file_path = os.path.join(folder_path, file_name)
run_data = helper_functions.load_json_file(file_path)

observation_windows_optim = run_data["final_result"]["observation_windows"]
# station_keeping_epochs_optim = run_data["final_result"]["skm_epochs"]
observation_windows_optim = [(60390, 60397), (60400.0, 60401.22475752616), (60404.22475752616, 60405.066958155025), (60408.066958155025, 60408.566958155025), (60411.566958155025, 60412.98370281383), (60415.98370281383, 60416.79029821602)]

### Compare difference timing cases
dynamic_model_list = ["HF", "PMSRP", 0]
truth_model_list = ["HF", "PMSRP", 0]
threshold = 7
skm_to_od_duration = 3
duration = 28
od_duration = 1
custom_station_keeping_error = 1e-2
custom_target_point_epoch = 3
custom_initial_estimation_error = np.array([5e0, 5e0, 5e0, 1e-5, 1e-5, 1e-5, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])*1e0
custom_apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
custom_orbit_insertion_error = np.array([1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e1, 1e1, 1e1, 1e-4, 1e-4, 1e-4])*1e2


# Collect a series of observation window sets to compare
observation_windows_list = []
# observation_windows_continuous = helper_functions.get_custom_observation_windows(duration, 0, threshold, od_duration)
observation_windows_constant = helper_functions.get_custom_observation_windows(28, 1, 7, 0.25)
# observation_windows_list.append(observation_windows_continuous)
# observation_windows_list.append(observation_windows_constant)
# observation_windows_list.append(observation_windows_optim)

observation_windows_1 = helper_functions.get_custom_observation_windows(28, 1, 7, 0.25)
observation_windows_2 = helper_functions.get_custom_observation_windows(28, 2, 7, 0.25)
observation_windows_3 = helper_functions.get_custom_observation_windows(28, 3, 7, 0.25)
observation_windows_4 = helper_functions.get_custom_observation_windows(28, 4, 7, 0.25)
# observation_windows_list.append(observation_windows_1)
observation_windows_list.append(observation_windows_2)
observation_windows_list.append(observation_windows_3)
observation_windows_list.append(observation_windows_4)
observation_windows_1 = helper_functions.get_custom_observation_windows(28, 1, 7, 0.5)
observation_windows_2 = helper_functions.get_custom_observation_windows(28, 2, 7, 0.5)
observation_windows_3 = helper_functions.get_custom_observation_windows(28, 3, 7, 0.5)
observation_windows_4 = helper_functions.get_custom_observation_windows(28, 4, 7, 0.5)
# observation_windows_list.append(observation_windows_1)
observation_windows_list.append(observation_windows_2)
observation_windows_list.append(observation_windows_3)
observation_windows_list.append(observation_windows_4)
# observation_windows_1 = helper_functions.get_custom_observation_windows(28, 1, 7, 1)
observation_windows_2 = helper_functions.get_custom_observation_windows(28, 2, 7, 1)
observation_windows_3 = helper_functions.get_custom_observation_windows(28, 3, 7, 1)
observation_windows_4 = helper_functions.get_custom_observation_windows(28, 4, 7, 1)
# observation_windows_list.append(observation_windows_1)
observation_windows_list.append(observation_windows_2)
observation_windows_list.append(observation_windows_3)
observation_windows_list.append(observation_windows_4)


navigation_results_list = []
navigation_simulator_list = []
navigation_output_list = []
for i, observation_windows in enumerate(observation_windows_list):

    print("Running with observation windows: \n", observation_windows)

    station_keeping_epochs = [windows[1] for windows in observation_windows]

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



for i, navigation_results in enumerate(navigation_results_list):

    print(f"Delta V for case {i}: \n: ", np.linalg.norm(navigation_results_list[i][8][1], axis=1), np.sum(np.linalg.norm(navigation_results_list[i][8][1], axis=1)))


for navigation_output in navigation_output_list:
    plot_navigation_results = PlotNavigationResults.PlotNavigationResults(navigation_output)
    plot_navigation_results.plot_estimation_error_history()
    # plot_navigation_results.plot_uncertainty_history()
    plot_navigation_results.plot_reference_deviation_history()
    # plot_navigation_results.plot_full_state_history()
    plot_navigation_results.plot_formal_error_history()
    # plot_navigation_results.plot_correlation_history()
    plot_navigation_results.plot_observations()
    # plot_navigation_results.plot_observability()

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











