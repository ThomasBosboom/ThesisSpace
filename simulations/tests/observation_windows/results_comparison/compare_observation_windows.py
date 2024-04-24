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
    "continuous_arc": [
        # (comparison_helper_functions.get_constant_arc_observation_windows(28, 0, 1, 1), 1)
    ],
    "constant_arc": [
        # (comparison_helper_functions.get_constant_arc_observation_windows(28, 3, 1, 0.1), 30),
        # (comparison_helper_functions.get_constant_arc_observation_windows(28, 3, 1, 0.5), 10),
        # (comparison_helper_functions.get_constant_arc_observation_windows(28, 3, 1, 1), 10),
    ],
    "perilune_arc": [
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(14, margin=0.1, step_size=0.01, threshold=0, pass_interval=0), 2),
        (comparison_helper_functions.get_orbit_based_arc_observation_windows(6, margin=0.1, step_size=0.01, threshold=0.5, pass_interval=6), 2),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.08, step_size=0.01, threshold=0.5, pass_interval=6), 10),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.05, step_size=0.01, threshold=0.5, pass_interval=6), 10),


    ],
    "apolune_arc": [
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.1, step_size=0.01, threshold=0.5, pass_interval=6, apolune=True), 10),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.08, step_size=0.01, threshold=0.5, pass_interval=6, apolune=True), 10),
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.05, step_size=0.01, threshold=0.5, pass_interval=6, apolune=True), 10),
    ]
}

print(observation_windows_settings)



#################################################################
###### Post processing of the navigation results ################
#################################################################

# Run the navigation routine using given settings
initial_estimation_error_sigmas = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])*10
orbit_insertion_error_sigmas = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])
navigation_outputs = comparison_helper_functions.generate_navigation_outputs(observation_windows_settings,
                                                                             initial_estimation_error_sigmas=initial_estimation_error_sigmas,
                                                                             orbit_insertion_error_sigmas=orbit_insertion_error_sigmas)

print(navigation_outputs)

utils.save_dicts_to_folder(dicts=[observation_windows_settings], labels=[current_time+"_observation_windows_settings"], custom_sub_folder_name=file_name)


### Extracting the relevant information for each NavigationOutput object


# Get objective value history
objective_value_results = {}
for window_type in navigation_outputs.keys():

    objective_value_results_per_window_case = []
    for window_case, navigation_output_list in enumerate(navigation_outputs[window_type]):

        objective_values = []
        for run, navigation_output in navigation_output_list.items():

            print(f"Results for {window_type} window_case {window_case} run {run}:")

            # Extracting the relevant objects
            navigation_results = navigation_output.navigation_results
            navigation_simulator = navigation_output.navigation_simulator

            # Extracting the relevant results from objects
            delta_v = navigation_results[8][1]
            delta_v_per_skm = np.linalg.norm(delta_v, axis=1)
            objective_value = np.sum(delta_v_per_skm)
            objective_values.append(objective_value)

            print("Objective: ", delta_v_per_skm, objective_value)

        objective_value_results_per_window_case.append((len(objective_values),
                                                    min(objective_values),
                                                    max(objective_values),
                                                    np.mean(objective_values),
                                                    np.std(objective_values),
                                                    objective_values))

    objective_value_results[window_type] = objective_value_results_per_window_case


print(objective_value_results)


utils.save_dicts_to_folder(dicts=[objective_value_results], labels=[current_time+"_objective_value_results"], custom_sub_folder_name=file_name)





# Plot OD errors given the arcs
fig, axs = plt.subplots(4, 1, figsize=(12, 5), sharex=True)
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
j = 0
for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):

    objective_value_results_per_window_case = []
    for case_index, window_case in enumerate(navigation_outputs_cases):

        color = color_cycle[j]
        j += 1

        objective_values = []
        for run_index, (run, navigation_output) in enumerate(window_case.items()):

            print(f"Results for {window_type} window_case {case_index} run {run}:")

            # Extracting the relevant objects
            navigation_results = navigation_output.navigation_results
            navigation_simulator = navigation_output.navigation_simulator

            alpha = np.random.randint(98,100)/100

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

                axs[0].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[0:3]), color=color, alpha=alpha, label=f"{window_type}" if window_index==0 and run_index==0 else None)
                axs[1].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[3:6]), color=color, alpha=alpha)
                axs[2].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[6:9]), color=color, alpha=alpha)
                axs[3].scatter(end_epoch-navigation_simulator.mission_start_epoch, np.linalg.norm(orbit_determination_error[9:12]), color=color, alpha=alpha)
                axs[3].set_xlabel(f"Epoch since {navigation_simulator.mission_start_epoch} [MJD]")

                labels = [r"$||\hat{\mathbf{r}}_{0, LPF}-\mathbf{r}_{LPF}||$",
                          r"$||\hat{\mathbf{v}}_{0, LPF}-\mathbf{v}_{LPF}||$",
                          r"$||\hat{\mathbf{r}}_{0, LUMIO}-\mathbf{r}_{LUMIO}||$",
                          r"$||\hat{\mathbf{v}}_{0, LUMIO}-\mathbf{v}_{LUMIO}||$"]

                for i in range(len(axs)):
                    axs[i].set_ylabel(labels[i])
                    axs[i].grid(alpha=0.3)
                    if run_index == 0:
                        axs[i].axvspan(
                            xmin=start_epoch-navigation_simulator.mission_start_epoch,
                            xmax=end_epoch-navigation_simulator.mission_start_epoch,
                            color=color,
                            alpha=0.1,
                            # label="Observation window" if type_index == 0 else None
                            )



            full_propagated_formal_errors_epochs = navigation_results[3][0]
            full_propagated_formal_errors_history = navigation_results[3][1]
            propagated_covariance_epochs = navigation_results[2][0]
            relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch

            # Plot the estimation error history
            for k in range(2):
                for j in range(2):
                    colors = ["red", "green", "blue"]
                    symbols = [[r"x", r"y", r"z"], [r"v_{x}", r"v_{y}", r"v_{z}"]]
                    ylabels = ["3D RSS OD \n position uncertainty [m]", "3D RSS OD \n velocity uncertainty [m/s]"]
                    axs2[k][j].plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 3*k+6*j:3*k+6*j+3], axis=1), label=navigation_simulator.model_name)

            for k in range(2):
                for j in range(2):
                    for i, gap in enumerate(navigation_simulator.observation_windows):
                        axs2[k][j].axvspan(
                            xmin=gap[0]-navigation_simulator.mission_start_epoch,
                            xmax=gap[1]-navigation_simulator.mission_start_epoch,
                            color="gray",
                            alpha=0.1,
                            label="Observation window" if i == 0 else None)
                    for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):
                        station_keeping_epoch = epoch - navigation_simulator.mission_start_epoch
                        axs2[k][j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.7, label="SKM" if i==0 else None)
                    axs2[k][0].set_ylabel(ylabels[k])
                    axs2[k][j].grid(alpha=0.5, linestyle='--')
                    axs2[k][j].set_yscale("log")
                    axs2[k][0].set_title("LPF")
                    axs2[k][1].set_title("LUMIO")

                    # Set y-axis tick label format to scientific notation with one decimal place
                    axs2[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    axs2[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                    axs2[-1][j].set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")

axs2[0][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left')

fig2.suptitle(f"Total 3D RSS 3$\sigma$ uncertainty \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
plt.tight_layout()







axs[0].legend(title="Setup", bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
# plt.legend()
plt.show()




# # Plot specific runs
# objective_value_results = {}
# for window_type in navigation_outputs.keys():

#     objective_value_results_per_window_case = []
#     for window_case, navigation_output_list in enumerate(navigation_outputs[window_type]):

#         objective_values = []
#         for run, navigation_output in navigation_output_list.items():

#             print(f"Results for {window_type} window_case {window_case} run {run}:")

#             # Plotting results
#             plot_navigation_results = PlotNavigationResults.PlotNavigationResults(navigation_output)
#             plot_navigation_results.plot_estimation_error_history()
#             plot_navigation_results.plot_uncertainty_history()
#             # plot_navigation_results.plot_reference_deviation_history()
#             plot_navigation_results.plot_full_state_history()
#             plot_navigation_results.plot_formal_error_history()
#             # plot_navigation_results.plot_observations()
#             # plot_navigation_results.plot_observability()
#             # plot_navigation_results.plot_od_error_delta_v_relation()
#             # plot_navigation_results.plot_correlation_history()

#             # plt.show()

# plt.show()





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











