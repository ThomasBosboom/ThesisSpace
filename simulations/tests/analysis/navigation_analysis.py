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

num_runs = 10
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
    # "default": [
    #     (helper_functions.get_constant_arc_observation_windows(60, arc_interval=3.8, arc_duration=0.2, mission_start_epoch=mission_start_epoch), 1),
    # ],
    # "optimized": [
    #     (helper_functions.get_constant_arc_observation_windows(28, arc_interval=3, arc_duration=0.5, mission_start_epoch=mission_start_epoch), 1),
    # ],
}

observation_windows_settings = {'Default': [([(60390, 60391.0), (60394.0, 60395.0), (60398.0, 60399.0), (60402.0, 60403.0), (60406.0, 60407.0), (60410.0, 60411.0), (60414.0, 60415.0)], 1)],
                                'Optimized': [([(60390, 60390.14042208123), (60393.14042208123, 60393.762736291814), (60396.762736291814, 60397.7889041982), (60400.7889041982, 60401.46602915859), (60404.46602915859, 60405.34219283191), (60408.34219283191, 60409.099794879134), (60412.099794879134, 60413.679242665996)], 1)]}



# observation_windows_settings = {
#     "0.2": [
#         ([(60390, 60390.2)], num_runs),
#     ],
#     "0.4": [
#         ([(60390, 60390.4)], num_runs),
#     ],
#     # "0.6": [
#     #     ([(60390, 60390.6)], num_runs),
#     # ],
#     # "0.8": [
#     #     ([(60390, 60390.8)], num_runs),
#     # ],
#     # "1.0": [
#     #     ([(60390, 60391.0)], num_runs),
#     # ],
#     # "1.2": [
#     #     ([(60390, 60391.2)], num_runs),
#     # ],
#     # "1.4": [
#     #     ([(60390, 60391.4)], num_runs),
#     # ],
#     # "1.6": [
#     #     ([(60390, 60391.6)], num_runs),
#     # ],
#     # "1.8": [
#     #     ([(60390, 60391.8)], num_runs),
#     # ],
#     # "2.0": [
#     #     ([(60390, 60392.0)], num_runs),
#     # ],
# }



#######################################################
###### Objective value versus #########################
#######################################################

# from src import NavigationSimulator, ObjectiveFunctions

# evaluation_threshold = 14
# num_runs_list = [1, 2, 5, 10, 30]
# objective_values = []
# for num_runs in num_runs_list:

#     navigation_simulator = NavigationSimulator.NavigationSimulator()

#     objective_functions_settings = {"num_runs": num_runs, "evaluation_threshold": evaluation_threshold}
#     objective_functions = ObjectiveFunctions.ObjectiveFunctions(
#         navigation_simulator,
#         **objective_functions_settings
#     )

#     observation_windows = helper_functions.get_constant_arc_observation_windows(28, arc_interval=3, arc_duration=1, mission_start_epoch=mission_start_epoch)
#     objective_value = objective_functions.worst_case_station_keeping_cost(observation_windows)

#     objective_values.append(objective_value)

#     print(objective_values)

# objective_values = [0.024228704064541032, 0.024375438666188874, 0.02458634017463967, 0.02454322733541171, 0.024917275946776524]
# plt.bar(num_runs_list, objective_values)
# plt.xlabel("Number of iterations [-]")
# plt.ylabel("Objective value [m/s]")
# plt.show()







#######################################################
###### Generate the navigation outputs ################
#######################################################

lpf_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])*1
lumio_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])*1
initial_estimation_error = np.concatenate((lpf_estimation_error, lumio_estimation_error))
# apriori_covariance = np.diag(np.array([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2)
apriori_covariance = np.diag(initial_estimation_error**2)
orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0

# Run the navigation routine using given settings
auxilary_settings = {
    # "apriori_covariance": apriori_covariance,
    # "initial_estimation_error": initial_estimation_error,
    # "orbit_insertion_error": orbit_insertion_error,
    "delta_v_min": 0.02,
    "step_size": 0.5
}


navigation_outputs = helper_functions.generate_navigation_outputs(observation_windows_settings, **auxilary_settings)


############################################################
###### Plotting detailed results ###########################
############################################################

print("Plotting results...")

detailed_results = [["Default", "Optimized"], [0], [0]]
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
                        process_single_navigation_results.plot_correlation_history()
                        process_single_navigation_results.plot_observability_metrics()

# plt.show()

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
process_multiple_navigation_results.plot_maneuvre_costs_bar_chart(evaluation_threshold=14, bar_labeler=None, worst_case=True)
process_multiple_navigation_results.plot_estimation_arc_comparison(evaluation_threshold=14, bar_labeler=None,  worst_case=True)
print("Plotting done...")

plt.show()