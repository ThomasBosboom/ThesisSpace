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
        (comparison_helper_functions.get_orbit_based_arc_observation_windows(28, margin=0.1, threshold=0.2, pass_interval=6), 1),
    ],
    "Apolune": [
        # (comparison_helper_functions.get_orbit_based_arc_observation_windows(8, margin=0.1, threshold=0.2, pass_interval=4, apolune=True), 1),
    ],
    "Random": [
        # (comparison_helper_functions.get_random_arc_observation_windows(28, [2, 0.01], [2, 0.01], [2, 0.01], seed=0), 1),
    ],
    "Continuous": [
        # (comparison_helper_functions.get_constant_arc_observation_windows(28, skm_to_arc_duration=0.1, threshold=0.1, arc_duration=0.1), 1)
    ],
    "Constant": [
        # (comparison_helper_functions.get_constant_arc_observation_windows(28, skm_to_arc_duration=3, threshold=0.2, arc_duration=0.1), 1),
    ]
}

print(observation_windows_settings)


############################################################
#### Compare results of different fidelities ###############
############################################################

fig, axs = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
model_line_style_cycle = ["solid", "dashed", "dashdot"]
for model_index, model_name in enumerate(["PM", "PMSRP", "SHSRP"]):

    model_line_style = model_line_style_cycle[model_index]

    # Run the navigation routine using given settings
    navigation_outputs = comparison_helper_functions.generate_navigation_outputs(observation_windows_settings,
                                                                                model_name = model_name,
                                                                                model_name_truth = "PMSRP",
                                                                                delta_v_min=0,
                                                                                station_keeping_error=0,
                                                                                orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*10,
                                                                                initial_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3]),
                                                                                apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
                                                                                )

    # Generate results
    objective_value_results = comparison_helper_functions.generate_objective_value_results(navigation_outputs)

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    line_style_cycle = ["solid", "dashed", "dashdot"]
    ylabels = ["3D RSS OD \n position uncertainty [m]", "3D RSS OD \n velocity uncertainty [m/s]"]
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

                # Extract the relevant information from the objects
                full_propagated_formal_errors_epochs = navigation_results[3][0]
                full_propagated_formal_errors_history = navigation_results[3][1]
                propagated_covariance_epochs = navigation_results[2][0]
                relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch

                full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

                # Plot observation windows

                for k in range(2):
                    for j in range(2):

                        if run_index==0 and model_index==0:

                            for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):

                                axs[k][j].axvspan(
                                    xmin=start_epoch-navigation_simulator.mission_start_epoch,
                                    xmax=end_epoch-navigation_simulator.mission_start_epoch,
                                    color=color,
                                    alpha=0.2,
                                    label=f"Observation window" if k==0 and j==1 and window_index==0 and case_index==0 else None
                                    )

                            for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):

                                axs[k][j].axvline(x=epoch - navigation_simulator.mission_start_epoch,
                                                    color='black',
                                                    linestyle='--',
                                                    alpha=0.3,
                                                    label="SKM" if k==0 and j==1 and i==0 and case_index==0 else None
                                                    )

                        axs[k][j].plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 3*k+6*j:3*k+6*j+3], axis=1),
                                        # label=window_type if case_index==0 and run_index==0 else None,
                                        color=color,
                                        ls=line_style,
                                        alpha=alpha)

            mean_full_propagated_formal_errors_histories = np.mean(np.array(full_propagated_formal_errors_histories), axis=0)
            for k in range(2):
                for j in range(2):
                    axs[k][j].plot(relative_epochs, 3*np.linalg.norm(mean_full_propagated_formal_errors_histories[:, 3*k+6*j:3*k+6*j+3], axis=1),
                        label=f"{window_type} case {case_index+1}, \nmodel: {model_name}",
                        color=color,
                        ls=model_line_style,
                        alpha=1)

    # Plotting results
    plot_navigation_results = PlotNavigationResults.PlotNavigationResults(navigation_output)
    plot_navigation_results.plot_estimation_error_history()
    plot_navigation_results.plot_uncertainty_history()
    plot_navigation_results.plot_dispersion_history()
    plot_navigation_results.plot_full_state_history()
    plot_navigation_results.plot_formal_error_history()
    # plot_navigation_results.plot_observations()
    # plot_navigation_results.plot_observability()
    # plot_navigation_results.plot_od_error_dispersion_relation()
    # plot_navigation_results.plot_correlation_history()

for k in range(2):
    for j in range(2):
        axs[k][0].set_ylabel(ylabels[k])
        axs[k][j].grid(alpha=0.5, linestyle='--')
        axs[k][j].set_yscale("log")
        axs[k][0].set_title("LPF")
        axs[k][1].set_title("LUMIO")
        axs[-1][j].set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")

axs[0][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
fig.suptitle(f"Total 3D RSS 3$\sigma$ uncertainty \n Comparing dynamic models")
plt.tight_layout()
plt.show()

