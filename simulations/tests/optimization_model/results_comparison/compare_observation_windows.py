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

from tests import utils
import helper_functions
from src.optimization_models import OptimizationModel
from src import NavigationSimulator



#################################################################
###### Compare results observation windows ######################
#################################################################


### Extract information from specific folder and run
folder_name = "PM01_PM01_thr3_skm3_dur28_od1_num1_max20_test"
batch_name = "09040214"
folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "optimization_routines\data", folder_name)

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

### Compare difference timing cases
dynamic_model_list = ["HF", "PM", 0]
truth_model_list = ["HF", "PM", 0]
threshold = 3
skm_to_od_duration = 3
duration = 28
od_duration = 1
observation_windows_list = []
observation_windows_continuous = helper_functions.get_custom_observation_windows(duration, 0, threshold, od_duration)
observation_windows_constant = helper_functions.get_custom_observation_windows(duration, 3, threshold, od_duration)
# observation_windows_list.append(observation_windows_continuous)
observation_windows_list.append(observation_windows_constant)
observation_windows_list.append(observation_windows_optim)

navigation_results_list = []
for i, observation_windows in enumerate(observation_windows_list):

    print("Running with observation windows: \n", observation_windows)
    station_keeping_epochs = [windows[1] for windows in observation_windows]
    target_point_epochs = [observation_windows[1][0]-observation_windows[0][1]]
    print(target_point_epochs)

    navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                    dynamic_model_list,
                                                                    truth_model_list,
                                                                    station_keeping_epochs=station_keeping_epochs,
                                                                    target_point_epochs=target_point_epochs,
                                                                    step_size=1e-3)

    navigation_results = navigation_simulator.get_navigation_results()

    delta_v = navigation_results[8][1]
    objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
    print(f"Objective: \n", delta_v, objective_value)
    print("End of objective calculation ===============")

    navigation_results_list.append(navigation_results)

print("Delta V for continuous case: \n: ", navigation_results_list[0][8], np.sum(np.linalg.norm(navigation_results_list[0][8][1], axis=1)))
print("Delta V for constant OD case: \n: ", navigation_results_list[1][8], np.sum(np.linalg.norm(navigation_results_list[1][8][1], axis=1)))
print("Delta V for variable OD case: \n: ", navigation_results_list[2][8], np.sum(np.linalg.norm(navigation_results_list[2][8][1], axis=1)))

# from src import PlotNavigationResults

for navigation_results in navigation_results_list:
    results_dict = {"HF": {"PM": [navigation_results]}}
    PlotNavigationResults.PlotNavigationResults(results_dict).plot_estimation_error_history()
    # PlotNavigationResults.PlotNavigationResults(results_dict).plot_uncertainty_history()
    PlotNavigationResults.PlotNavigationResults(results_dict).plot_reference_deviation_history()
    PlotNavigationResults.PlotNavigationResults(results_dict).plot_full_state_history()
    PlotNavigationResults.PlotNavigationResults(results_dict).plot_formal_error_history()
    # PlotNavigationResults.PlotNavigationResults(results_dict).plot_correlation_history()
    # PlotNavigationResults.PlotNavigationResults(results_dict).plot_observations()
    # PlotNavigationResults.PlotNavigationResults(results_dict).plot_observability()

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











