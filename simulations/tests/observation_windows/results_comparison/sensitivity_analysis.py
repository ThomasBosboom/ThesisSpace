# Standard
import os
import sys
import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Define current time
current_time = datetime.now().strftime("%d%m%H%M")

# from tests import utils, helper_functions
import comparison_helper_functions
from src import NavigationSimulator, NavigationSimulatorBase, PlotNavigationResults
from tests import utils



#################################################################
###### Define the observation windows ###########################
#################################################################

duration = 28
numruns = 2

# Collect a series of observation window sets to compare
observation_windows_settings = {
    "Constant": [
        (comparison_helper_functions.get_constant_arc_observation_windows(duration, skm_to_arc_duration=3.5, threshold=0.5, arc_duration=0.5), numruns),
    ]
}

print(observation_windows_settings)


# #################################################################
# ###### Window-based sensitivity analysis ########################
# #################################################################

# sensitivity_settings = {
#     # "threshold": [0.1, 0.2, 0.3],
#     "skm_to_arc_duration": [2, 3],
#     # "arc_duration": [0.1, 0.2]
# }

# navigation_outputs_observation_window_sensitivity = comparison_helper_functions.generate_navigation_outputs_observation_window_sensitivity(duration, numruns, observation_windows_settings, sensitivity_settings)
# print("navigation_outputs: ", navigation_outputs_observation_window_sensitivity)


# #################################################################
# ###### Auxiliary settings-based sensitivity analysis ############
# #################################################################

# sensitivity_settings = {
#     "noise_range": [1, 50],
#     # "total_observation_count": [5, 50, 100],
#     # "target_point_epochs": [[3], [2, 3], [1, 2, 3], [4]],
#     # "station_keeping_error": [0.00, 0.01, 0.02, 0.05],
#     # "delta_v_min": [0.00, 0.001, 0.01, 0.02, 0.03],
#     # "observation_step_size_range": [60, 120, 300, 600],
#     # "mission_start_epoch": [60390, 60395, 60400]
# }

# auxiliary_settings = {
#     "noise_range": 10,
#     "redirect_out": False,
#     "total_observation_count": 50,
#     "propagate_dynamics_linearly": False
# }

# navigation_outputs_parameter_sensitivity = comparison_helper_functions.generate_navigation_outputs_parameter_sensitivity(observation_windows_settings,
#                                                                                                                sensitivity_settings,
#                                                                                                                 # **auxiliary_settings
#                                                                                                                 )

# print(navigation_outputs_observation_window_sensitivity)


# navigation_outputs_sensitivity = comparison_helper_functions.generate_navigation_outputs_sensitivity([navigation_outputs_observation_window_sensitivity, navigation_outputs_parameter_sensitivity])



sensitivity_settings = {
    "skm_to_arc_duration": [2, 3],
    # "mission_start_epoch": [60400],
    # "arc_duration": [0.1, 0.2, 0.3, 0.4, 0.5, 1],
    # "noise_range": [1, 5, 10, 50]
}

navigation_outputs_sensitivity = comparison_helper_functions.generate_navigation_outputs_sensitivity(duration, numruns, observation_windows_settings, sensitivity_settings)



#################################################################
###### Plot results of sensitivity analysis #####################
#################################################################


ylabels = ["3D RSS OD \nposition uncertainty [m]", "3D RSS OD \nvelocity uncertainty [m/s]"]
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_style_cycle = ["solid", "dashed", "dashdot"]
for type_index, (window_type, navigation_outputs_sensitivity_types) in enumerate(navigation_outputs_sensitivity.items()):

    color = color_cycle[int(type_index%len(color_cycle))]
    for sensitivity_type_index, (sensitivity_type, navigation_outputs_sensitivity_cases) in enumerate(navigation_outputs_sensitivity_types.items()):

        fig, axs = plt.subplots(3, 1, figsize=(12, 7))
        axs_twin = axs[0].twinx()

        shades = color_cycle
        # shades = [mcolors.to_rgb(color)[:-1] + (l,) for l in np.linspace(0.1, 0.9, len(navigation_outputs_sensitivity_cases))]

        # shapes = np.linspace(1, 0.5, len(navigation_outputs_sensitivity_cases))
        delta_v_runs_dict_sensitivity_case = {}
        for index, navigation_outputs_sensitivity_case in enumerate(navigation_outputs_sensitivity_cases):

            # line_style = line_style_cycle[int(case_index%len(line_style_cycle))]
            full_propagated_formal_errors_histories = []
            full_reference_state_deviation_histories = []
            delta_v_runs_dict = {}
            for run_index, (run, navigation_output) in enumerate(navigation_outputs_sensitivity_case.items()):

                print(window_type, sensitivity_type, navigation_outputs_sensitivity_cases)

                # Extracting the relevant objects
                navigation_results = navigation_output.navigation_results
                navigation_simulator = navigation_output.navigation_simulator

                # Extracting the relevant results from objects
                for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):
                    if end_epoch in navigation_simulator.delta_v_dict.keys():

                        delta_v = np.linalg.norm(navigation_simulator.delta_v_dict[end_epoch])

                        if end_epoch in delta_v_runs_dict:
                            delta_v_runs_dict[end_epoch].append(delta_v)
                        else:
                            delta_v_runs_dict[end_epoch] = [delta_v]

                    if run_index==0:

                        axs[0].axvspan(
                            xmin=start_epoch-navigation_simulator.mission_start_epoch,
                            xmax=end_epoch-navigation_simulator.mission_start_epoch,
                            color=shades[index],
                            alpha=0.2,
                            # label=f"Observation window" if window_index==0 and case_index==0 else None
                            )

                        axs[1].axvspan(
                            xmin=start_epoch-navigation_simulator.mission_start_epoch,
                            xmax=end_epoch-navigation_simulator.mission_start_epoch,
                            color=shades[index],
                            alpha=0.2,
                            # label=f"Observation window" if window_index==0 and case_index==0 else None
                            )

                full_propagated_formal_errors_epochs = navigation_results[3][0]
                full_propagated_formal_errors_history = navigation_results[3][1]
                relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch
                full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

                full_estimation_error_epochs = navigation_results[0][0]
                full_estimation_error_history = navigation_results[0][1]

                full_reference_state_deviation_epochs = navigation_results[1][0]
                full_reference_state_deviation_history = navigation_results[1][1]
                full_reference_state_deviation_histories.append(full_reference_state_deviation_history)

                if run_index == 0:

                    for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):
                        station_keeping_epoch = epoch - navigation_simulator.mission_start_epoch

                        axs[0].axvline(x=station_keeping_epoch,
                                            color='black',
                                            linestyle='--',
                                            alpha=0.3,
                                            zorder=0,
                                            label="SKM" if i==0 and index==0 else None)

                        axs[1].axvline(x=station_keeping_epoch,
                                            color='black',
                                            linestyle='--',
                                            alpha=0.3,
                                            zorder=0,
                                            label="SKM" if i==0 and index==0 else None)

                axs_twin.plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 6:9], axis=1),
                                color=shades[index],
                                # ls=line_style,
                                alpha=0.2,
                                # label=f"{sensitivity_settings[sensitivity_type][index]}"
                                )

                axs_twin.plot(relative_epochs, np.linalg.norm(full_estimation_error_history[:, 6:9], axis=1),
                                color=shades[index],
                                # ls='--',
                                alpha=0.2)

                axs[1].plot(relative_epochs, np.linalg.norm(full_reference_state_deviation_history[:, 6:9], axis=1),
                                color=shades[index],
                                # ls='--',
                                alpha=0.2
                                )

            mean_full_propagated_formal_errors_histories = np.mean(np.array(full_propagated_formal_errors_histories), axis=0)
            axs_twin.plot(relative_epochs, 3*np.linalg.norm(mean_full_propagated_formal_errors_histories[:, 6:9], axis=1),
                            color=shades[index],
                            # ls=line_style,
                            alpha=0.8,
                            # label=f"{sensitivity_settings[sensitivity_type][index]}"
                            )

            mean_full_reference_state_deviation_histories = np.mean(np.array(full_reference_state_deviation_histories), axis=0)
            axs[1].plot(relative_epochs, np.linalg.norm(mean_full_reference_state_deviation_histories[:, 6:9], axis=1),
                            color=shades[index],
                            # ls=line_style,
                            alpha=0.8,
                            # label=f"{sensitivity_settings[sensitivity_type][index]}"
                            )

            # Plot the station keeping costs standard deviations
            for delta_v_runs_dict_index, (end_epoch, delta_v_runs) in enumerate(delta_v_runs_dict.items()):
                mean_delta_v = np.mean(delta_v_runs)
                std_delta_v = np.std(delta_v_runs)
                axs[0].bar(end_epoch-navigation_simulator.mission_start_epoch, mean_delta_v,
                        color=shades[index],
                        alpha=0.6,
                        width=0.2,
                        yerr=std_delta_v,
                        capsize=4,
                        label=f"{window_type}" if navigation_outputs_sensitivity_case==0 and delta_v_runs_dict_index==0 else None)

            delta_v_runs_dict_sensitivity_case[str(sensitivity_settings[sensitivity_type][index])] = delta_v_runs_dict

        # Plot the total delta v results per sensitivity case
        axs[2].grid(alpha=0.5, linestyle='--', zorder=0)
        axs[2].set_title("Parameter sensitivity")
        axs[2].set_ylabel("Parameter value")
        axs[2].set_xlabel(r"Total $||\Delta V||$ [m/s]")

        axs[1].grid(alpha=0.5, linestyle='--', zorder=0)
        axs[1].set_title("Dispersion from reference orbit")
        axs[1].set_ylabel(r"||$\hat{\mathbf{r}}-\mathbf{r}_{ref}$|| [m]")
        axs[0].set_xlabel(f"Time since MJD start epoch [days]", fontsize="small")
        axs[1].set_xlabel(f"Time since MJD start epoch [days]", fontsize="small")

        stats_types = {}
        for index, (type_key, type_value) in enumerate(delta_v_runs_dict_sensitivity_case.items()):

            runs = len(list(type_value.values())[0])
            sums = {}
            for run in range(runs):
                value_list = []
                for key, value in delta_v_runs_dict_sensitivity_case[type_key].items():
                    value_list.append(value[run])
                sums[run] = value_list

            for key, value in sums.items():
                sums[key] = np.sum(value)

            all_values = list(sums.values())
            stats_types[type_key] = [np.mean(all_values), np.std(all_values)]

            print(stats_types)

            axs[2].barh(type_key, np.mean(all_values),
                color=shades[index],
                # width=0.2,
                xerr=np.std(all_values),
                capsize=4,
                label=f"{sensitivity_settings[sensitivity_type][index]}"
                )

        # axs[0].set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]", fontsize="small")
        axs[0].set_ylabel(r"$||\Delta V||$ [m/s]")
        axs[0].grid(alpha=0.5, linestyle='--', zorder=0)
        axs[0].set_title("Station keeping costs")
        axs_twin.set_ylabel(ylabels[0])
        # axs[0].set_yscale("log")
        axs_twin.set_yscale("log")
        # axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(navigation_outputs_sensitivity_cases)+1, fontsize="small")
        # axs_twin.legend(loc='upper center', bbox_to_anchor=(0.5, -0.23), ncol=len(navigation_outputs_sensitivity_cases)+1, fontsize="small")
        axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.23), ncol=len(navigation_outputs_sensitivity_cases)+1, fontsize="small")
        plt.tight_layout()
        utils.save_figure_to_folder(figs=[fig], labels=[f"{current_time}_sensitivity_analysis_{sensitivity_type}"], custom_sub_folder_name=file_name)


    plt.show()
