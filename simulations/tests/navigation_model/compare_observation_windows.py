# Standard
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
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

numruns = 1
duration = 14

# Collect a series of observation window sets to compare
observation_windows_settings = {
    "Perilune": [
        (comparison_helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.1, threshold=0.2, pass_interval=8), numruns),
    ],
    # "Apolune": [
    #     (comparison_helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.1, threshold=0.2, pass_interval=8, apolune=True), numruns),
    # ],
    # "Random": [
    #     (comparison_helper_functions.get_random_arc_observation_windows(duration, arc_interval_vars=[3, 1], threshold_vars=[1, 0.1], arc_duration_vars=[1, 0.1], seed=0), numruns),
    # ],
    # "Constant": [
    #     (comparison_helper_functions.get_constant_arc_observation_windows(duration, arc_interval=3, threshold=1, arc_duration=1), numruns),
    # ]
}



observation_windows_settings = {
    # "Random1": [
    #     (comparison_helper_functions.get_random_arc_observation_windows(duration, arc_interval_vars=[3, 0.001], threshold_vars=[1, 0.3], arc_duration_vars=[1, 0.3], seed=0), numruns),
    # ],
    # "Random2": [
    #     (comparison_helper_functions.get_random_arc_observation_windows(duration, arc_interval_vars=[3, 0.001], threshold_vars=[1, 0.3], arc_duration_vars=[1, 0.3], seed=1), numruns),
    # ],
    # "Random3": [
    #     (comparison_helper_functions.get_random_arc_observation_windows(duration, arc_interval_vars=[3, 0.001], threshold_vars=[1, 0.3], arc_duration_vars=[1, 0.3], seed=2), numruns),
    # ],
    "Constant": [
        (comparison_helper_functions.get_constant_arc_observation_windows(duration, arc_interval=3, threshold=1, arc_duration=1), numruns),
    ]
}

# observation_windows_settings = {
#     # "Perilune": [
#     #     (comparison_helper_functions.get_orbit_based_arc_observation_windows(15, margin=0.05, threshold=0.1, pass_interval=7), 1),
#     # ],
#     # "Apolune": [
#     #     (comparison_helper_functions.get_orbit_based_arc_observation_windows(15, margin=0.05, threshold=0.1, pass_interval=7, apolune=True), 1),
#     # ],
#     "Constant": [
#         (comparison_helper_functions.get_constant_arc_observation_windows(28, arc_interval=3, threshold=1, arc_duration=0.1), 2),
#     ],
#     "Constant2": [
#         (comparison_helper_functions.get_constant_arc_observation_windows(28, arc_interval=3, threshold=1, arc_duration=0.5), 2),
#     ],
#     "Constant3": [
#         (comparison_helper_functions.get_constant_arc_observation_windows(28, arc_interval=3, threshold=1, arc_duration=1), 2),
#     ],
#     "Constant4": [
#         (comparison_helper_functions.get_constant_arc_observation_windows(28, arc_interval=3, threshold=1, arc_duration=1.5), 2),
#     ]

# }
# }

print(observation_windows_settings)



#################################################################
###### Post processing of the navigation results ################
#################################################################

auxilary_settings = {
    "orbit_insertion_error": np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0,
    "initial_estimation_error": np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3]),
    "apriori_covariance": np.diag([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])**2
}

# Run the navigation routine using given settings
navigation_outputs = comparison_helper_functions.generate_navigation_outputs(observation_windows_settings, **auxilary_settings)


############################################################
###### Total maneuvre cost #################################
############################################################

### Bar chart of the total station-keeping costs
fig, ax = plt.subplots(figsize=(10, 4))
objective_value_results = comparison_helper_functions.generate_objective_value_results(navigation_outputs)
comparison_helper_functions.bar_plot(ax, objective_value_results, bar_labeler=None)
utils.save_figure_to_folder(figs=[fig], labels=[current_time+"_objective_value_results"], custom_sub_folder_name=file_name)
# plt.show()


############################################################
###### Plotting detailed results ###########################
############################################################

print("Plotting results...")

detailed_results = [["Random3", "Constant"], [0], [0]]
for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):
    for case_index, window_case in enumerate(navigation_outputs_cases):
        for run_index, (run, navigation_output) in enumerate(window_case.items()):

            # Plotting detailed results for the specified models
            if window_type in detailed_results[0]:
                if case_index in detailed_results[1]:
                    if run_index in detailed_results[2]:

                        plot_navigation_results = PlotNavigationResults.PlotNavigationResults(navigation_output)
                        plot_navigation_results.plot_estimation_error_history()
                        # plot_navigation_results.plot_uncertainty_history()
                        plot_navigation_results.plot_dispersion_history()
                        # plot_navigation_results.plot_dispersion_to_estimation_error_history()
                        # plot_navigation_results.plot_full_state_history()
                        # plot_navigation_results.plot_formal_error_history()
                        plot_navigation_results.plot_observations()
                        # plot_navigation_results.plot_observability()
                        plot_navigation_results.plot_od_error_dispersion_relation()
                        plot_navigation_results.plot_correlation_history()


############################################################
###### Compare uncertainties ###############################
############################################################

fig, axs = plt.subplots(2, 2, figsize=(12.5, 5), sharex=True)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_style_cycle = ["solid", "dashed", "dashdot"]
ylabels = ["3D RSS OD position \nuncertainty [m]", "3D RSS OD velocity \nuncertainty [m/s]"]
for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):

    color = color_cycle[int(type_index%len(color_cycle))]
    for case_index, window_case in enumerate(navigation_outputs_cases):

        line_style = line_style_cycle[int(case_index%len(line_style_cycle))]
        full_propagated_formal_errors_histories = []
        for run_index, (run, navigation_output) in enumerate(window_case.items()):

            # print(f"Results for {window_type} window_case {case_index} run {run}:")

            # Extracting the relevant objects
            navigation_results = navigation_output.navigation_results
            navigation_simulator = navigation_output.navigation_simulator

            # Extract the relevant information from the objects
            full_propagated_formal_errors_epochs = navigation_results[3][0]
            full_propagated_formal_errors_history = navigation_results[3][1]
            relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch

            full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

            # Plot observation windows
            if run_index==0:

                for k in range(2):
                    for j in range(2):

                        for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):

                            axs[k][j].axvline(x=epoch - navigation_simulator.mission_start_epoch,
                                                color='black',
                                                linestyle='--',
                                                alpha=0.3,
                                                label="SKM" if k==0 and j==1 and i==0 and type_index==0 else None
                                                )

                        for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):

                            axs[k][j].axvspan(
                                xmin=start_epoch-navigation_simulator.mission_start_epoch,
                                xmax=end_epoch-navigation_simulator.mission_start_epoch,
                                color=color,
                                alpha=0.2,
                                label=f"Tracking arc" if k==0 and j==1 and window_index==0 and case_index==0 else None
                                )



                        # Plot the results of the first run
                        # axs[k][j].plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 3*k+6*j:3*k+6*j+3], axis=1),
                        #                 # label=window_type if case_index==0 and run_index==0 else None,
                        #                 color=color,
                        #                 ls=line_style,
                        #                 alpha=0.1
                        #                 )

        mean_full_propagated_formal_errors_histories = np.mean(np.array(full_propagated_formal_errors_histories), axis=0)
        for k in range(2):
            for j in range(2):
                axs[k][j].plot(relative_epochs, 3*np.linalg.norm(mean_full_propagated_formal_errors_histories[:, 3*k+6*j:3*k+6*j+3], axis=1),
                    # label=f"{window_type}, case {case_index+1}",
                    label=f"{window_type}",
                    color=color,
                    ls=line_style,
                    alpha=1)

for k in range(2):
    for j in range(2):
        axs[k][0].set_ylabel(ylabels[k])
        axs[k][j].grid(alpha=0.5, linestyle='--', zorder=0)
        axs[k][j].set_yscale("log")
        axs[k][0].set_title("LPF")
        axs[k][1].set_title("LUMIO")
        axs[-1][j].set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")

axs[0][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(navigation_outputs.keys()), fontsize='small')
fig.suptitle(f"Total 3D RSS 3$\sigma$ uncertainty \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
plt.tight_layout()

utils.save_figure_to_folder(figs=[fig], labels=[current_time+"_compare_uncertainties"], custom_sub_folder_name=file_name)



############################################################
###### Compare maneuvre costs windows ######################
############################################################

fig, axs = plt.subplots(figsize=(12, 4), sharex=True)
axs_twin = axs.twinx()
ylabels = ["3D RSS OD \n position uncertainty [m]", "3D RSS OD \n velocity uncertainty [m/s]"]
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_style_cycle = ["solid", "dashed", "dashdot"]
for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):

    color = color_cycle[int(type_index%len(color_cycle))]

    for case_index, window_case in enumerate(navigation_outputs_cases):

        line_style = line_style_cycle[int(case_index%len(line_style_cycle))]

        full_propagated_formal_errors_histories = []
        delta_v_runs_dict = {}
        for run_index, (run, navigation_output) in enumerate(window_case.items()):

            # print(f"Results for {window_type} window_case {case_index} run {run}:")

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
                        color=color,
                        alpha=0.2,
                        # label=f"Observation window" if window_index==0 and case_index==0 else None
                        )

            full_propagated_formal_errors_epochs = navigation_results[3][0]
            full_propagated_formal_errors_history = navigation_results[3][1]
            relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch
            full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

            full_estimation_error_epochs = navigation_results[0][0]
            full_estimation_error_history = navigation_results[0][1]

            if run_index == 0:

                for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):
                    station_keeping_epoch = epoch - navigation_simulator.mission_start_epoch

                    axs.axvline(x=station_keeping_epoch,
                                        color='black',
                                        linestyle='--',
                                        alpha=0.3,
                                        label="SKM" if k == 0 and j == 1 and i==0 else None)

                axs_twin.plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 6:9], axis=1),
                                color=color,
                                ls=line_style,
                                alpha=0.3)

            axs_twin.plot(relative_epochs, np.linalg.norm(full_estimation_error_history[:, 6:9], axis=1),
                            color=color,
                            ls='--',
                            alpha=0.2)

        # Plot the station keeping costs standard deviations
        for delta_v_runs_dict_index, (end_epoch, delta_v_runs) in enumerate(delta_v_runs_dict.items()):
            mean_delta_v = np.mean(delta_v_runs)
            std_delta_v = np.std(delta_v_runs)
            axs.bar(end_epoch-navigation_simulator.mission_start_epoch, mean_delta_v,
                    color=color,
                    width=0.2,
                    yerr=std_delta_v,
                    capsize=4,
                    label=f"{window_type}" if case_index==0 and delta_v_runs_dict_index==0 else None)

axs.set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")
axs.set_ylabel(r"$||\Delta V||$ [m/s]")
axs.grid(alpha=0.5, linestyle='--', zorder=0)
axs.set_title("Station keeping costs")
axs_twin.set_ylabel(ylabels[0])
axs.set_yscale("log")
axs_twin.set_yscale("log")
axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(navigation_outputs.keys()), fontsize="small")
plt.tight_layout()
utils.save_figure_to_folder(figs=[fig], labels=[current_time+"_compare_maneuvre_cost"], custom_sub_folder_name=file_name)



############################################################
###### Compare Monte Carlo estimation errors ###############
############################################################
rows = len(navigation_outputs.keys())
fig, axs = plt.subplots(rows, 4, figsize=(13, 3*rows), sharex=True)
if len(navigation_outputs.keys())==1:
    axs = np.array([axs])
label_index = 0
detailed_results = [["Perilune", "Apolune", "Random"], [0], [0]]
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_style_cycle = ["solid", "dashed", "dashdot"]
colors = ["red", "green", "blue"]
symbols = [[r"x", r"y", r"z"], [r"x", r"y", r"z"]]
units = ["[m]", "[m]", "[m/s]", "[m/s]"]
titles = [r"$\mathbf{r}-\hat{\mathbf{r}}$ LPF", r"$\mathbf{r}-\hat{\mathbf{r}}$ LUMIO", r"$\mathbf{v}-\hat{\mathbf{v}}$ LPF", r"$\mathbf{v}-\hat{\mathbf{v}}$ LUMIO"]
for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):

    color = color_cycle[int(type_index%len(color_cycle))]
    for case_index, window_case in enumerate(navigation_outputs_cases):

        line_style = line_style_cycle[int(case_index%len(line_style_cycle))]
        full_estimation_error_histories = []
        for run_index, (run, navigation_output) in enumerate(window_case.items()):

            # Extracting the relevant objects
            navigation_results = navigation_output.navigation_results
            navigation_simulator = navigation_output.navigation_simulator

            # Extract relevant data from the objects
            full_estimation_error_epochs = navigation_results[0][0]
            full_estimation_error_history = navigation_results[0][1]
            full_propagated_formal_errors_history = navigation_results[3][1]
            relative_epochs = full_estimation_error_epochs - navigation_simulator.mission_start_epoch
            full_estimation_error_histories.append(full_estimation_error_history)

            for n in range(axs.shape[1]):

                if run_index==0:
                    for i, gap in enumerate(navigation_simulator.observation_windows):
                        axs[type_index][n].axvspan(
                            xmin=gap[0]-navigation_simulator.mission_start_epoch,
                            xmax=gap[1]-navigation_simulator.mission_start_epoch,
                            color="gray",
                            alpha=0.1,
                            label="Observation \nwindow" if i == 0 and type_index==0 else None)

                    for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):
                        station_keeping_epoch = epoch - navigation_simulator.mission_start_epoch
                        axs[type_index][n].axvline(x=station_keeping_epoch,
                                        color='black',
                                        linestyle='--',
                                        alpha=0.2,
                                        label="SKM" if i == 0 and type_index==0 else None)

                    # axs[type_index][n].set_yscale("log")
                    axs[type_index][0].set_ylim(-100, 100)
                    # axs[type_index][1].set_ylim(-100, 100)
                    axs[type_index][2].set_ylim(-0.03, 0.03)
                    # axs[type_index][3].set_ylim(-0.03, 0.03)

                    axs[-1][n].set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]", fontsize="small")
                    # axs[type_index][n].set_title(titles[n], fontsize="small")

                # if run_index==0:
                    axs[type_index][0].set_ylabel(window_type, fontsize="small")
                    axs[type_index][n].grid(alpha=0.5, linestyle='--')
                    axs[type_index][n].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    axs[type_index][n].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

            n = 0
            for k in range(2):
                for j in range(2):
                    for i in range(3):

                        sigma = 3*full_propagated_formal_errors_history[:, 3*k+6*j+i]
                        axs[type_index][n].plot(relative_epochs, full_estimation_error_history[:,3*k+6*j+i],
                                                color=colors[i],
                                                alpha=0.1,
                                                # label=f"${symbols[k][i]}-\hat{{{symbols[k][i]}}}$" \
                                                #     # if label_index in range(6) else None
                                                #     if n==3 and run_index==0 else None
                                                    )

                        if run_index==0:
                            axs[type_index][n].plot(relative_epochs, -sigma,
                                                    color=colors[i],
                                                    ls="--",
                                                    alpha=0.3,
                                                    label=f"$3\sigma_{{{symbols[k][i]}}}$" \
                                                        # if label_index in range(6) else None
                                                        if n==3 and run_index==0 else None
                                                        )

                            axs[type_index][n].plot(relative_epochs, sigma,
                                                    color=colors[i],
                                                    ls="--",
                                                    alpha=0.3)

                    n += 1

        mean_full_estimation_error_histories = np.mean(np.array(full_estimation_error_histories), axis=0)
        # print("Mean: \n", mean_full_estimation_error_histories[-1, :])

        n=0
        for k in range(2):
            for j in range(2):
                for i in range(3):
                    axs[type_index][n].plot(relative_epochs, mean_full_estimation_error_histories[:, 3*k+6*j+i],
                        label=f"${symbols[k][i]}-\hat{{{symbols[k][i]}}}$" \
                            # if label_index in range(6) else None
                            if n==3 else None,
                        color=colors[i],
                        alpha=1)

                rss_values = np.mean(np.sqrt(np.sum(np.square(mean_full_estimation_error_histories[:, 3*k+6*j:3*k+6*j+3]), axis=1)))
                if type_index == 0:
                    axs[type_index][n].set_title(titles[n]+f"\nMean RSS: {np.round(rss_values, 2)} "+units[n], fontsize="small")
                else:
                    axs[type_index][n].set_title(f"Mean RSS: {np.round(rss_values, 1)} "+units[n], fontsize="small")
                n += 1


axs[0][-1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

# fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
fig.suptitle(f"Estimaton error history \nModel: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
# fig4.suptitle("Estimation error history: range-only, $1\sigma_{\rho}$ = 102.44 [$m$], $f_{obs}$ = $1/600$ [$s^{-1}$]")
plt.tight_layout()


utils.save_figure_to_folder(figs=[fig], labels=[current_time+"_estimation_error_history"], custom_sub_folder_name=file_name)

print("Plotting done...")

plt.show()