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
current_time = datetime.now().strftime("%Y%m%d%H%M")

import helper_functions
from tests.postprocessing import ProcessNavigationResults


#################################################################
###### Define the observation windows ###########################
#################################################################

num_runs = 5
duration = 28
mission_start_epoch = 60390.0

# Collect a series of observation window sets to compare
observation_windows_settings = {
    "Perilune": [
        (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.1, threshold=0.2, pass_interval=8, mission_start_epoch=mission_start_epoch), num_runs),
    ],
    "Apolune": [
        (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.1, threshold=0.2, pass_interval=8, apolune=True), num_runs),
    ],
    "Random": [
        (helper_functions.get_random_arc_observation_windows(duration, arc_interval_vars=[3, 1], threshold_vars=[1, 0.1], arc_duration_vars=[1, 0.1], seed=0), num_runs),
    ],
    "Constant": [
        (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=3, threshold=1, arc_duration=1), num_runs),
    ]
}

# observation_windows_settings = {
#     "0.1 day": [
#         (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=3, arc_duration=0.1, mission_start_epoch=mission_start_epoch), num_runs),
#     ],
#     "0.5 day": [
#         (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=3, arc_duration=0.5, mission_start_epoch=mission_start_epoch), num_runs),
#     ],
#     "1.0 day": [
#         (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=3, arc_duration=1, mission_start_epoch=mission_start_epoch), num_runs),
#     ],
#     # "2.0 day": [
#     #     (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=3, arc_duration=2.0, mission_start_epoch=mission_start_epoch), num_runs),
#     # ]
# }

observation_windows_settings = {
    "default": [
        (helper_functions.get_constant_arc_observation_windows(28, arc_interval=3, arc_duration=1, mission_start_epoch=mission_start_epoch), 1),
    ],
    # "optimized": [
    #     (helper_functions.get_constant_arc_observation_windows(28, arc_interval=3, arc_duration=0.5, mission_start_epoch=mission_start_epoch), 1),
    # ],
}


# observation_windows_settings = {
#     "default1": [
#         ([(60390, 60390.05)], 2),
#     ],
#     "default2": [
#         ([(60390, 60390.1)], 2),
#     ],
#     # "default3": [
#     #     ([(60390, 60390.15)], 2),
#     # ],
#     # "default4": [
#     #     ([(60390, 60390.20)], 2),
#     # ],
#     # "default5": [
#     #     ([(60390, 60390.25)], 2),
#     # ],
#     # "default6": [
#     #     ([(60390, 60390.30)], 2),
#     # ],
#     # "default7": [
#     #     ([(60390, 60390.35)], 2),
#     # ],
#     # "default8": [
#     #     ([(60390, 60390.40)], 2),
#     # ],
#     # "default9": [
#     #     ([(60390, 60390.45)], 2),
#     # ],
#     # "default10": [
#     #     ([(60390, 60390.50)], 2),
#     # ],
#     # "default11": [
#     #     ([(60390, 60390.55)], 2),
#     # ],
#     # "default12": [
#     #     ([(60390, 60390.60)], 2),
#     # ],
#     # "default13": [
#     #     ([(60390, 60390.65)], 2),
#     # ],
#     # "default14": [
#     #     ([(60390, 60390.70)], 2),
#     # ],
#     # "default15": [
#     #     ([(60390, 60390.75)], 2),
#     # ],
#     # "default16": [
#     #     ([(60390, 60390.80)], 2),
#     # ],
#     # "default17": [
#     #     ([(60390, 60390.85)], 2),
#     # ],
#     # "default18": [
#     #     ([(60390, 60390.90)], 2),
#     # ],
#     # "default19": [
#     #     ([(60390, 60390.95)], 2),
#     # ],
#     # "default20": [
#     #     ([(60390, 60391)], 2),
#     # ],
# }





#######################################################
###### Generate the navigation outputs ################
#######################################################

# Run the navigation routine using given settings
auxilary_settings = {
    # "apriori_covariance": np.diag(np.array([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2),
    # "apriori_covariance": np.diag(np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])**2),
    "step_size": 0.01,
    # "observation_interval": 10000
    # "noise": 102.44,
    "run_optimization_version": False
}


navigation_outputs = helper_functions.generate_navigation_outputs(observation_windows_settings, **auxilary_settings)


############################################################
###### Plotting detailed results ###########################
############################################################

print("Plotting results...")

detailed_results = [["default"], [0], [0]]
for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):
    for case_index, window_case in enumerate(navigation_outputs_cases):
        for run_index, (run, navigation_output) in enumerate(window_case.items()):

            # Plotting detailed results for the specified models
            if window_type in detailed_results[0]:
                if case_index in detailed_results[1]:
                    if run_index in detailed_results[2]:

                        process_single_navigation_results = ProcessNavigationResults.PlotSingleNavigationResults(
                            navigation_output,
                            figure_settings={"save_figure": True,
                                            "current_time": current_time,
                                            "file_name": file_name
                            }
                        )

                        process_single_navigation_results.plot_estimation_error_history()
                        process_single_navigation_results.plot_uncertainty_history()
                        process_single_navigation_results.plot_dispersion_history()
                        process_single_navigation_results.plot_full_state_history()
                        process_single_navigation_results.plot_formal_error_history()
                        process_single_navigation_results.plot_observations()
                        process_single_navigation_results.plot_dispersion_to_estimation_error_history()
                        process_single_navigation_results.plot_correlation_history()
                        process_single_navigation_results.plot_observability_metrics()

plt.show()

process_multiple_navigation_results = ProcessNavigationResults.PlotMultipleNavigationResults(
    navigation_outputs,
    # color_cycle=['gray', "green"],
    figure_settings={"save_figure": True,
                     "current_time": current_time,
                     "file_name": file_name
    }
)

process_multiple_navigation_results.plot_uncertainty_comparison()
process_multiple_navigation_results.plot_maneuvre_costs()
process_multiple_navigation_results.plot_monte_carlo_estimation_error_history(evaluation_threshold=14)
process_multiple_navigation_results.plot_maneuvre_costs_bar_chart(evaluation_threshold=14, bar_labeler=None, maneuvre_cost_only=True)
print("Plotting done...")

plt.show()