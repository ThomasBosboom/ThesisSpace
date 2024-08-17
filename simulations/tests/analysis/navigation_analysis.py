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
###### Define test setup ########################################
#################################################################

num_runs = 5
duration = 28
mission_start_epoch = 60390.0

auxilary_settings = {"step_size": 0.001}


#################################################################
###### Define the observation windows ###########################
#################################################################

observation_windows_settings = {
    "Perilune": [
        (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.05, threshold=0.1, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, None),
        # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.06, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.06"),
        # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.07, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.07"),
        # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.08, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.08"),
        # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.09, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.09"),
        # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.10, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.10")
    ],
    "Apolune": [
        (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.05, threshold=0.1, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, None),
    #     (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.06, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, "0.06"),
    #     # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.07, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, "0.07"),
    #     # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.08, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, "0.08"),
    #     # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.09, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, "0.09"),
    #     # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.10, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, "0.10")
    ],
    # "Random": [
    #     (helper_functions.get_random_arc_observation_windows(duration, arc_interval_vars=[3, 0.000001], threshold_vars=[1, 0.000001], arc_duration_vars=[1, 0.1], seed=0), num_runs, None),
    #     # (helper_functions.get_random_arc_observation_windows(duration, arc_interval_vars=[3, 0.000001], threshold_vars=[1, 0.000001], arc_duration_vars=[1, 0.2], seed=1), num_runs, "0.2"),
    # ],
    # "Default": [
    #     (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=3, threshold=1, arc_duration=1), num_runs, None),
    # ]
}


# Varying pass_interval per orbit case
# params = range(1, 1)
# margin = 0.05
# num_runs = 1
# observation_windows_settings = {
#     "Perilune": [
#         (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=margin, threshold=2*margin, mission_start_epoch=mission_start_epoch, apolune=False, pass_interval=pass_interval), num_runs, str(pass_interval)) for pass_interval in params
#     ],
#     "Apolune": [
#         (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=margin, threshold=2*margin, mission_start_epoch=mission_start_epoch, apolune=True, pass_interval=pass_interval), num_runs, str(pass_interval)) for pass_interval in params
#     ],
# }


params = [0.1, 0.2, 0.5, 1.0, 2.0]
params2 = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
num_runs = 3
observation_windows_settings = {
    f"{param2} day": [
        (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=param2, arc_duration=param, mission_start_epoch=mission_start_epoch), num_runs, str(param)) for param in params
    ]
    for param2 in params2
}

# params = [3.0]
# params2 = [0.1, 0.5, 1.0, 2.0]
# num_runs = 3
# observation_windows_settings = {
#     f"{param2} day": [
#         (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=param, arc_duration=param2, mission_start_epoch=mission_start_epoch), num_runs, None) for param in params
#     ]
#     for param2 in params2
# }


#######################################################
###### Generate the navigation outputs ################
#######################################################

print(observation_windows_settings)

navigation_outputs = helper_functions.generate_navigation_outputs(observation_windows_settings, **auxilary_settings)




############################################################
###### Plotting detailed results ###########################
############################################################

print("Plotting results...")

detailed_results = [list(observation_windows_settings.keys()), [0], [0]]
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

                        # process_single_navigation_results.plot_estimation_error_history()
                        # process_single_navigation_results.plot_uncertainty_history()
                        # process_single_navigation_results.plot_dispersion_history()
                        # process_single_navigation_results.plot_full_state_history()
                        # process_single_navigation_results.plot_formal_error_history()
                        # process_single_navigation_results.plot_observations()
                        # process_single_navigation_results.plot_correlation_history()
                        # process_single_navigation_results.plot_observability_metrics()

# plt.show()

process_multiple_navigation_results = ProcessNavigationResults.PlotMultipleNavigationResults(
    navigation_outputs,
    # color_cycle=["salmon", "forestgreen"],
    figure_settings={"save_figure": True,
                    "current_time": current_time,
                    "file_name": file_name
    }
)

process_multiple_navigation_results.plot_uncertainty_comparison()
process_multiple_navigation_results.plot_maneuvre_costs(separate_plots=False)
process_multiple_navigation_results.plot_maneuvre_costs(separate_plots=True)
# process_multiple_navigation_results.plot_full_state_history_comparison(step_size=None)
# process_multiple_navigation_results.plot_monte_carlo_estimation_error_history(evaluation_threshold=14)
process_multiple_navigation_results.plot_maneuvre_costs_bar_chart(evaluation_threshold=14, show_annual=False, bar_labeler=None, worst_case=False, observation_windows_settings=observation_windows_settings)
process_multiple_navigation_results.plot_maneuvre_costs_bar_chart(evaluation_threshold=14, show_annual=True, bar_labeler=None, worst_case=False, observation_windows_settings=observation_windows_settings)
# process_multiple_navigation_results.plot_estimation_arc_comparison(evaluation_threshold=14, bar_labeler=None,  worst_case=False)
print("Plotting done...")

plt.show()