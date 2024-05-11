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

# Collect a series of observation window sets to compare
observation_windows_settings = {
    "Perilune": [
        (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.1, threshold=0.5, pass_interval=4), 5),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(5, margin=0.1, threshold=0.2, pass_interval=1), 1),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(5, margin=0.1, threshold=0.2, pass_interval=2), 1),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(5, margin=0.1, threshold=0.2, pass_interval=4), 1),
    ],
    # "Apolune": [
    #     # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.1, threshold=0.5, pass_interval=7, apolune=True), 5),
    # ],
    # "Random": [
    #     # (comparison_helper_functions.get_random_arc_observation_windows(28, skm_to_od_duration_vars=[3.5, 0.1], threshold_vars=[0.5, 0.001], od_duration_vars=[0.5, 0.1], seed=0), 5),
    # ],
    # # "Continuous": [
    # #     # (comparison_helper_functions.get_constant_arc_observation_windows(28, skm_to_od_duration=0.1, threshold=0.1, od_duration=0.1), 1)
    # # ],
    # "Constant": [
    #     # (comparison_helper_functions.get_constant_arc_observation_windows(28, skm_to_od_duration=3.5, threshold=0.5, od_duration=0.5), 5),
    # ]
}

print(observation_windows_settings)


#################################################################
###### Post processing of sensitivity analysis ##################
#################################################################

sensitivity_dict = {
    "noise_range": [1, 5, 10, 50, 100],
    # "target_point_epochs": [[3]]
}

navigation_outputs_sensitivity = comparison_helper_functions.generate_navigation_outputs_sensitivity(observation_windows_settings, sensitivity_dict)
print(navigation_outputs_sensitivity)

ylabels = ["3D RSS OD \n position uncertainty [m]", "3D RSS OD \n velocity uncertainty [m/s]"]
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_style_cycle = ["solid", "dashed", "dashdot"]
for type_index, (window_type, navigation_outputs_sensitivity_types) in enumerate(navigation_outputs_sensitivity.items()):

    # fig, axs = plt.subplots(len(navigation_outputs_sensitivity_types.keys()), 4, figsize=(14, 7), sharex=True)
    # axs_twin = axs.twinx()

    color = color_cycle[int(type_index%len(color_cycle))]
    for sensitivity_type_index, (sensitivity_type, navigation_outputs_sensitivity_cases) in enumerate(navigation_outputs_sensitivity_types.items()):

        fig, axs = plt.subplots(figsize=(12, 4), sharex=True)
        axs_twin = axs.twinx()

        shades = [mcolors.to_rgb(color)[:-1] + (l,) for l in np.linspace(0.5, 0.9, len(navigation_outputs_sensitivity_cases))]
        # shapes = np.linspace(1, 0.5, len(navigation_outputs_sensitivity_cases))
        for index, navigation_outputs_sensitivity_case in enumerate(navigation_outputs_sensitivity_cases):

            # line_style = line_style_cycle[int(case_index%len(line_style_cycle))]
            full_propagated_formal_errors_histories = []
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

                        axs.axvspan(
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

                if run_index == 0:

                    for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):
                        station_keeping_epoch = epoch - navigation_simulator.mission_start_epoch

                        axs.axvline(x=station_keeping_epoch,
                                            color='black',
                                            linestyle='--',
                                            alpha=0.3,
                                            label="SKM" if i==0 and index==0 else None)

                axs_twin.plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 6:9], axis=1),
                                color=shades[index],
                                # ls=line_style,
                                alpha=0.2,
                                # label=f"{sensitivity_dict[sensitivity_type][index]}"
                                )

            mean_full_propagated_formal_errors_histories = np.mean(np.array(full_propagated_formal_errors_histories), axis=0)
            axs_twin.plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 6:9], axis=1),
                            color=shades[index],
                            # ls=line_style,
                            alpha=1,
                            label=f"{sensitivity_dict[sensitivity_type][index]}")


            # Plot the station keeping costs standard deviations
            for delta_v_runs_dict_index, (end_epoch, delta_v_runs) in enumerate(delta_v_runs_dict.items()):
                mean_delta_v = np.mean(delta_v_runs)
                std_delta_v = np.std(delta_v_runs)
                axs.bar(end_epoch-navigation_simulator.mission_start_epoch, mean_delta_v,
                        color=shades[index],
                        width=0.2,
                        yerr=std_delta_v,
                        capsize=4,
                        label=f"{window_type}" if navigation_outputs_sensitivity_case==0 and delta_v_runs_dict_index==0 else None)

        axs.set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")
        axs.set_ylabel(r"$||\Delta V||$ [m/s]")
        axs.grid(alpha=0.5, linestyle='--', zorder=0)
        axs.set_title("Station keeping costs")
        axs_twin.set_ylabel(ylabels[0])
        axs.set_yscale("log")
        axs_twin.set_yscale("log")
        # axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(navigation_outputs_sensitivity_cases)+1, fontsize="small")
        axs_twin.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(navigation_outputs_sensitivity_cases)+1, fontsize="small")
        plt.tight_layout()
        # utils.save_figure_to_folder(figs=[fig2], labels=[current_time+"_uncertainty_history"], custom_sub_folder_name=file_name)


    plt.show()



    # color = color_cycle[int(type_index%len(color_cycle))]

    # for case_index, window_case in enumerate(navigation_outputs_cases):

    #     line_style = line_style_cycle[int(case_index%len(line_style_cycle))]

    #     full_propagated_formal_errors_histories = []
    #     delta_v_runs_dict = {}
    #     for run_index, (run, navigation_output) in enumerate(window_case.items()):

    #         print(f"Results for {window_type} window_case {case_index} run {run}:")

    #         # Extracting the relevant objects
    #         navigation_results = navigation_output.navigation_results
    #         navigation_simulator = navigation_output.navigation_simulator










        ### Bar chart of the total station-keeping costs
        # fig, ax = plt.subplots(figsize=(10, 4))
        # objective_value_results = comparison_helper_functions.generate_objective_value_results(navigation_outputs)
        # comparison_helper_functions.bar_plot(ax, objective_value_results, bar_labeler=None)
        # utils.save_figure_to_folder(figs=[fig], labels=[current_time+"_objective_value_results"], custom_sub_folder_name=file_name)
        # # plt.show()

        # fig, axs = plt.subplots(figsize=(12, 4), sharex=True)
        # axs_twin = axs.twinx()
        # ylabels = ["3D RSS OD \n position uncertainty [m]", "3D RSS OD \n velocity uncertainty [m/s]"]
        # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # line_style_cycle = ["solid", "dashed", "dashdot"]
        # for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):

        #     color = color_cycle[int(type_index%len(color_cycle))]

        #     for case_index, window_case in enumerate(navigation_outputs_cases):

        #         line_style = line_style_cycle[int(case_index%len(line_style_cycle))]

        #         full_propagated_formal_errors_histories = []
        #         delta_v_runs_dict = {}
        #         for run_index, (run, navigation_output) in enumerate(window_case.items()):

        #             print(f"Results for {window_type} window_case {case_index} run {run}:")

        #             # Extracting the relevant objects
        #             navigation_results = navigation_output.navigation_results
        #             navigation_simulator = navigation_output.navigation_simulator

        #             # Extracting the relevant results from objects
        #             for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):
        #                 if end_epoch in navigation_simulator.delta_v_dict.keys():

        #                     delta_v = np.linalg.norm(navigation_simulator.delta_v_dict[end_epoch])

        #                     if end_epoch in delta_v_runs_dict:
        #                         delta_v_runs_dict[end_epoch].append(delta_v)
        #                     else:
        #                         delta_v_runs_dict[end_epoch] = [delta_v]

        #                 if run_index==0:

        #                     axs.axvspan(
        #                         xmin=start_epoch-navigation_simulator.mission_start_epoch,
        #                         xmax=end_epoch-navigation_simulator.mission_start_epoch,
        #                         color=color,
        #                         alpha=0.2,
        #                         # label=f"Observation window" if window_index==0 and case_index==0 else None
        #                         )

        #             full_propagated_formal_errors_epochs = navigation_results[3][0]
        #             full_propagated_formal_errors_history = navigation_results[3][1]
        #             relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch
        #             full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

        #             if run_index == 0:

        #                 for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):
        #                     station_keeping_epoch = epoch - navigation_simulator.mission_start_epoch

        #                     axs.axvline(x=station_keeping_epoch,
        #                                         color='black',
        #                                         linestyle='--',
        #                                         alpha=0.3,
        #                                         label="SKM" if k == 0 and j == 1 and i==0 else None)

        #                     axs_twin.plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 6:9], axis=1),
        #                                     color=color,
        #                                     ls=line_style,
        #                                     alpha=0.1)

        #         # Plot the station keeping costs standard deviations
        #         for delta_v_runs_dict_index, (end_epoch, delta_v_runs) in enumerate(delta_v_runs_dict.items()):
        #             mean_delta_v = np.mean(delta_v_runs)
        #             std_delta_v = np.std(delta_v_runs)
        #             axs.bar(end_epoch-navigation_simulator.mission_start_epoch, mean_delta_v,
        #                     color=color,
        #                     width=0.2,
        #                     yerr=std_delta_v,
        #                     capsize=4,
        #                     label=f"{window_type}" if case_index==0 and delta_v_runs_dict_index==0 else None)

        # axs.set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")
        # axs.set_ylabel(r"$||\Delta V||$ [m/s]")
        # axs.grid(alpha=0.5, linestyle='--', zorder=0)
        # axs.set_title("Station keeping costs")
        # axs_twin.set_ylabel(ylabels[0])
        # axs.set_yscale("log")
        # axs_twin.set_yscale("log")
        # axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(navigation_outputs.keys()), fontsize="small")
        # plt.tight_layout()
        # utils.save_figure_to_folder(figs=[fig2], labels=[current_time+"_uncertainty_history"], custom_sub_folder_name=file_name)



        # plt.show()









# ############################################################
# #### Compare uncertainties #################################
# ############################################################

# fig, axs = plt.subplots(2, 2, figsize=(11, 4), sharex=True)
# detailed_results = [["Perilune", "Apolune", "Random"], [0], [0]]
# color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# line_style_cycle = ["solid", "dashed", "dashdot"]
# ylabels = ["3D RSS OD \n position uncertainty [m]", "3D RSS OD \n velocity uncertainty [m/s]"]
# for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):

#     color = color_cycle[int(type_index%len(color_cycle))]
#     for case_index, window_case in enumerate(navigation_outputs_cases):

#         line_style = line_style_cycle[int(case_index%len(line_style_cycle))]
#         full_propagated_formal_errors_histories = []
#         for run_index, (run, navigation_output) in enumerate(window_case.items()):

#             # Plotting detailed results for the specified models
#             if window_type in detailed_results[0]:
#                 if case_index in detailed_results[1]:
#                     if run_index in detailed_results[2]:

#                         plot_navigation_results = PlotNavigationResults.PlotNavigationResults(navigation_output)
#                         plot_navigation_results.plot_estimation_error_history()
#                         plot_navigation_results.plot_uncertainty_history()
#                         plot_navigation_results.plot_reference_deviation_history()
#                         # plot_navigation_results.plot_full_state_history()
#                         # plot_navigation_results.plot_formal_error_history()
#                         # plot_navigation_results.plot_observations()
#                         # plot_navigation_results.plot_observability()
#                         # plot_navigation_results.plot_od_error_delta_v_relation()
#                         # plot_navigation_results.plot_correlation_history()

#             alpha = 0.3

#             # print(f"Results for {window_type} window_case {case_index} run {run}:")

#             # Extracting the relevant objects
#             navigation_results = navigation_output.navigation_results
#             navigation_simulator = navigation_output.navigation_simulator

#             # Extract the relevant information from the objects
#             full_propagated_formal_errors_epochs = navigation_results[3][0]
#             full_propagated_formal_errors_history = navigation_results[3][1]
#             propagated_covariance_epochs = navigation_results[2][0]
#             relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch

#             full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

#             # Plot observation windows
#             for k in range(2):
#                 for j in range(2):

#                     if run_index==0:

#                         for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):

#                             axs[k][j].axvspan(
#                                 xmin=start_epoch-navigation_simulator.mission_start_epoch,
#                                 xmax=end_epoch-navigation_simulator.mission_start_epoch,
#                                 color=color,
#                                 alpha=0.2,
#                                 label=f"Observation window" if k==0 and j==1 and window_index==0 and case_index==0 else None
#                                 )

#                         for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):

#                             axs[k][j].axvline(x=epoch - navigation_simulator.mission_start_epoch,
#                                                 color='black',
#                                                 linestyle='--',
#                                                 alpha=0.3,
#                                                 label="SKM" if k==0 and j==1 and i==0 and case_index==0 else None
#                                                 )

#                     axs[k][j].plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 3*k+6*j:3*k+6*j+3], axis=1),
#                                     # label=window_type if case_index==0 and run_index==0 else None,
#                                     color=color,
#                                     ls=line_style,
#                                     alpha=alpha)

#         mean_full_propagated_formal_errors_histories = np.mean(np.array(full_propagated_formal_errors_histories), axis=0)
#         for k in range(2):
#             for j in range(2):
#                 axs[k][j].plot(relative_epochs, 3*np.linalg.norm(mean_full_propagated_formal_errors_histories[:, 3*k+6*j:3*k+6*j+3], axis=1),
#                     label=f"{window_type}, case {case_index+1}",
#                     color=color,
#                     ls=line_style,
#                     alpha=1)

# for k in range(2):
#     for j in range(2):
#         axs[k][0].set_ylabel(ylabels[k])
#         axs[k][j].grid(alpha=0.5, linestyle='--', zorder=0)
#         axs[k][j].set_yscale("log")
#         axs[k][0].set_title("LPF")
#         axs[k][1].set_title("LUMIO")
#         axs[-1][j].set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")

# axs[0][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
# fig.suptitle(f"Total 3D RSS 3$\sigma$ uncertainty \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
# plt.tight_layout()


