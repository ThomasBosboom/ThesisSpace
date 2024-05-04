# Standard
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
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
        (comparison_helper_functions.get_orbit_based_arc_observation_windows(4, margin=0.05, threshold=0.1, pass_interval=0), 1),
        (comparison_helper_functions.get_orbit_based_arc_observation_windows(4, margin=0.05, threshold=0.5, pass_interval=0), 1),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(14, margin=0.1, threshold=1, pass_interval=4), 5),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(14, margin=0.1, threshold=1, pass_interval=8), 5),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.05, threshold=0.5, pass_interval=4), 5),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.08, threshold=0.5, pass_interval=4), 5),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.1, threshold=0.5, pass_interval=4), 5),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(14, margin=0.1, threshold=0.1, pass_interval=4), 5),
    ],
    "Apolune": [
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(4, margin=0.05, threshold=2, pass_interval=0, apolune=True), 1),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.05, threshold=0.5, pass_interval=4, apolune=True), 5),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.08, threshold=0.5, pass_interval=4, apolune=True), 5),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.1, threshold=0.5, pass_interval=4, apolune=True), 5),
    ],
    "Random": [
        # (comparison_helper_functions.get_random_arc_observation_windows(14, [2, 0.01], [2, 0.01], [2, 0.01], seed=0), 1),
        # (comparison_helper_functions.get_random_arc_observation_windows(8, [1, 0.5], [0.5, 0.01], [0.2, 0.01], seed=1), 1),
        # (comparison_helper_functions.get_random_arc_observation_windows(28, [2, 0.5], [0.5, 0.01], [0.1, 0.01], seed=1), 5),
        # (comparison_helper_functions.get_random_arc_observation_windows(28, [2, 0.5], [0.5, 0.01], [0.1, 0.01], seed=2), 5),
        # (comparison_helper_functions.get_random_arc_observation_windows(14, [2, 0.5], [1, 0.5], [0.1, 0.01], seed=3), 3),
    ],
    "Continuous": [
        # (comparison_helper_functions.get_constant_arc_observation_windows(10, skm_to_od_duration=0.1, threshold=0.1, od_duration=0.1), 1)
        # (comparison_helper_functions.get_constant_arc_observation_windows(14, 3, 0.5, 0.5), 1)
    ],
    "Constant": [
        # (comparison_helper_functions.get_constant_arc_observation_windows(20, skm_to_od_duration=1, threshold=2, od_duration=2), 1),
        # (comparison_helper_functions.get_constant_arc_observation_windows(28, 3, 1, 0.5), 10),
        # (comparison_helper_functions.get_constant_arc_observation_windows(28, 3, 1, 1), 10),
    ]
}

print(observation_windows_settings)


#################################################################
###### Post processing of the navigation results ################
#################################################################

# Run the navigation routine using given settings
navigation_outputs = comparison_helper_functions.generate_navigation_outputs(observation_windows_settings,
                                                                             mission_start_epoch=60390,
                                                                             noise_range=2.98,
                                                                             observation_step_size_range=600,
                                                                             maximum_iterations=5,
                                                                             station_keeping_error=0,
                                                                             target_point_epochs = [3],
                                                                             include_station_keeping=True,
                                                                             orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0,
                                                                             initial_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3]),
                                                                             apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2)

# Generate results
objective_value_results           = comparison_helper_functions.generate_objective_value_results(navigation_outputs)
orbit_determination_error_results = comparison_helper_functions.generate_orbit_determination_error_results(navigation_outputs)
reference_orbit_deviation_results = comparison_helper_functions.generate_reference_orbit_deviation_results(navigation_outputs)
global_uncertainty_results        = comparison_helper_functions.generate_global_uncertainty_results(navigation_outputs)

# Save results
utils.save_dicts_to_folder(dicts=[observation_windows_settings], labels=[current_time+"_observation_windows_settings"], custom_sub_folder_name=file_name)
utils.save_dicts_to_folder(dicts=[objective_value_results], labels=[current_time+"_objective_value_results"], custom_sub_folder_name=file_name)
utils.save_dicts_to_folder(dicts=[orbit_determination_error_results], labels=[current_time+"_orbit_determination_error_results"], custom_sub_folder_name=file_name)
utils.save_dicts_to_folder(dicts=[reference_orbit_deviation_results], labels=[current_time+"_reference_orbit_deviation_results"], custom_sub_folder_name=file_name)
utils.save_dicts_to_folder(dicts=[global_uncertainty_results], labels=[current_time+"_global_uncertainty_results"], custom_sub_folder_name=file_name)


fig, ax = plt.subplots(figsize=(10, 4))
comparison_helper_functions.bar_plot(ax, objective_value_results, bar_labeler=None)
utils.save_figure_to_folder(figs=[fig], labels=[current_time+"_objective_value_results"], custom_sub_folder_name=file_name)
# plt.show()


# Plot OD errors given the arcs
# fig, axs = plt.subplots(4, 1, figsize=(12, 5), sharex=True)
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
fig3, axs3 = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
fig4, axs4 = plt.subplots(figsize=(12, 3), sharex=True)
axs4_twin = axs4.twinx()
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_style_cycle = ["solid", "dashed", "dashdot"]
for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):

    color = color_cycle[int(type_index%len(color_cycle))]

    for case_index, window_case in enumerate(navigation_outputs_cases):

        line_style = line_style_cycle[int(case_index%len(line_style_cycle))]

        full_propagated_formal_errors_histories = []
        for run_index, (run, navigation_output) in enumerate(window_case.items()):

            alpha = 0.3

            print(f"Results for {window_type} window_case {case_index} run {run}:")

            # Extracting the relevant objects
            navigation_results = navigation_output.navigation_results
            navigation_simulator = navigation_output.navigation_simulator

            # Extracting the relevant results from objects
            delta_v_per_skm_list = np.linalg.norm(navigation_results[8][1], axis=1).tolist()
            print("Size list: ", delta_v_per_skm_list)

            orbit_determination_errors = {}
            for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):

                # Calculate the absolute difference between each value and the target
                epochs = list(navigation_simulator.full_estimation_error_dict.keys())
                differences = [abs(epoch - end_epoch) for epoch in epochs]
                closest_index = differences.index(min(differences))
                end_epoch = epochs[closest_index]

                full_estimation_error_dict = navigation_simulator.full_estimation_error_dict
                orbit_determination_error = full_estimation_error_dict[end_epoch]
                orbit_determination_errors.update({end_epoch: list(orbit_determination_error)})


                # axs[0].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[0:3]), color=color, alpha=alpha, label=f"{window_type}" if window_index==0 and run_index==0 else None)
                # axs[1].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[3:6]), color=color, alpha=alpha)
                # axs[2].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[6:9]), color=color, alpha=alpha)
                # axs[3].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[9:12]), color=color, alpha=alpha)
                # axs[3].set_xlabel(f"Epoch since {navigation_simulator.mission_start_epoch} [MJD]")

                axs2[0][0].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[0:3]), color=color, alpha=alpha)
                axs2[1][0].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[3:6]), color=color, alpha=alpha)
                axs2[0][1].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[6:9]), color=color, alpha=alpha)
                axs2[1][1].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[9:12]), color=color, alpha=alpha)

                for k in range(2):
                    for j in range(2):
                        for i in range(3):
                            axs3[k][j].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.abs(orbit_determination_error[3*k+6*j+i]), color=color, alpha=alpha)


                axs4_twin.bar(end_epoch-navigation_simulator.mission_start_epoch, delta_v_per_skm_list[window_index], color=color, width=0.2)

                for k in range(2):
                    for j in range(2):

                        if run_index==0:

                            axs2[k][j].axvspan(
                                xmin=start_epoch-navigation_simulator.mission_start_epoch,
                                xmax=end_epoch-navigation_simulator.mission_start_epoch,
                                color=color,
                                alpha=0.2,
                                label=f"Observation window" if k == 0 and j == 1 and window_index==0 and case_index==0 else None
                                )

                            axs3[k][j].axvspan(
                                xmin=start_epoch-navigation_simulator.mission_start_epoch,
                                xmax=end_epoch-navigation_simulator.mission_start_epoch,
                                color=color,
                                alpha=0.2,
                                label=f"Observation window" if k == 0 and j == 1 and window_index==0 and case_index==0 else None
                                )

                            if k == 0 and j == 0:

                                axs4.axvspan(
                                    xmin=start_epoch-navigation_simulator.mission_start_epoch,
                                    xmax=end_epoch-navigation_simulator.mission_start_epoch,
                                    color=color,
                                    alpha=0.2,
                                    label=f"Observation window" if k == 0 and j == 1 and window_index==0 and case_index==0 else None
                                )

            print(list(orbit_determination_errors.values()))
            # colors = ["red", "green", "blue"]
            # symbols = [[r"x", r"y", r"z"], [r"v_{x}", r"v_{y}", r"v_{z}"]]
            ylabels = ["3D RSS OD \n position uncertainty [m]", "3D RSS OD \n velocity uncertainty [m/s]"]

            full_propagated_formal_errors_epochs = navigation_results[3][0]
            full_propagated_formal_errors_history = navigation_results[3][1]
            propagated_covariance_epochs = navigation_results[2][0]
            relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch

            full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

            for k in range(2):
                for j in range(2):

                    if run_index == 0:

                        for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):
                            station_keeping_epoch = epoch - navigation_simulator.mission_start_epoch
                            axs2[k][j].axvline(x=station_keeping_epoch,
                                                color='black',
                                                linestyle='--',
                                                alpha=0.3,
                                                label="SKM" if k == 0 and j == 1 and i==0 else None
                                                )

                            axs3[k][j].axvline(x=station_keeping_epoch,
                                                color='black',
                                                linestyle='--',
                                                alpha=0.3,
                                                label="SKM" if k == 0 and j == 1 and i==0 else None
                                                )

                            axs4.axvline(x=station_keeping_epoch,
                                                color='black',
                                                linestyle='--',
                                                alpha=0.3,
                                                label="SKM" if k == 0 and j == 1 and i==0 else None
                                                )

                    if run_index == 0 and case_index == 0:

                        axs2[k][0].set_ylabel(ylabels[k])
                        axs2[k][j].grid(alpha=0.5, linestyle='--')
                        axs2[k][j].set_yscale("log")
                        axs2[k][0].set_title("LPF")
                        axs2[k][1].set_title("LUMIO")
                        axs2[-1][j].set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")

                        axs3[k][0].set_ylabel(ylabels[k])
                        axs3[k][j].grid(alpha=0.5, linestyle='--')
                        axs3[k][j].set_yscale("log")
                        axs3[k][0].set_title("LPF")
                        axs3[k][1].set_title("LUMIO")
                        axs3[-1][j].set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")

                        axs4.set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")
                        axs4_twin.set_ylabel(r"$||\Delta V||$ [m/s]")
                        axs4.grid(alpha=0.5, linestyle='--')
                        axs4.set_title("Station keeping costs")
                        axs4.set_ylabel(ylabels[0])
                        axs4.set_yscale("log")
                        axs4_twin.set_yscale("log")


                    axs2[k][j].plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 3*k+6*j:3*k+6*j+3], axis=1),
                                    # label=window_type if case_index==0 and run_index==0 else None,
                                    color=color,
                                    ls=line_style,
                                    alpha=alpha)

                    axs3[k][j].plot(relative_epochs, 3*np.abs(full_propagated_formal_errors_history[:, 3*k+6*j:3*k+6*j+3]),
                                    # label=window_type if case_index==0 and run_index==0 else None,
                                    color=color,
                                    ls=line_style,
                                    alpha=alpha)

                    axs4.plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 6:9], axis=1),
                                    # label=window_type if case_index==0 and run_index==0 else None,
                                    color=color,
                                    ls=line_style,
                                    alpha=0.1)

        for k in range(2):
            for j in range(2):

                mean_full_propagated_formal_errors_histories = np.mean(np.array(full_propagated_formal_errors_histories), axis=0)
                axs2[k][j].plot(relative_epochs, 3*np.linalg.norm(mean_full_propagated_formal_errors_histories[:, 3*k+6*j:3*k+6*j+3], axis=1),
                    label="Mean",
                    color=color,
                    ls=line_style,
                    alpha=1)


# axs2[0][1].legend(title="Details", bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

fig2.suptitle(f"Total 3D RSS 3$\sigma$ uncertainty \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
plt.tight_layout()


utils.save_figure_to_folder(figs=[fig2], labels=[current_time+"_uncertainty_history"], custom_sub_folder_name=file_name)


# axs[0].legend(title="Setup", bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
# plt.legend()
# plt.show()



# Plot specific runs
objective_value_results = {}
for window_type in navigation_outputs.keys():

    objective_value_results_per_window_case = []
    for window_case, navigation_output_list in enumerate(navigation_outputs[window_type]):

        objective_values = []
        for run, navigation_output in navigation_output_list.items():

            print(f"Results for {window_type} window_case {window_case} run {run}:")

            # Plotting results
            plot_navigation_results = PlotNavigationResults.PlotNavigationResults(navigation_output)
            plot_navigation_results.plot_estimation_error_history()
            # plot_navigation_results.plot_uncertainty_history()
            plot_navigation_results.plot_reference_deviation_history()
            # plot_navigation_results.plot_full_state_history()
            # plot_navigation_results.plot_formal_error_history()
            # plot_navigation_results.plot_observations()
            # plot_navigation_results.plot_observability()
            # plot_navigation_results.plot_od_error_delta_v_relation()
            # plot_navigation_results.plot_correlation_history()

            # plt.show()

plt.show()