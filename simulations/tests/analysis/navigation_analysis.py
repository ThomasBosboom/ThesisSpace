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

num_runs = 1
duration = 28
mission_start_epoch = 60390.0

auxilary_settings = {}


#################################################################
###### Define the observation windows ###########################
#################################################################

observation_windows_settings = {
    "Perilune": [
        (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.05, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.05"),
        # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.06, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.06"),
        # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.07, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.07"),
        # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.08, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.08"),
        # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.09, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.09"),
        # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.10, threshold=0, pass_interval=8, apolune=False, mission_start_epoch=mission_start_epoch), num_runs, "0.10")
    ],
    "Apolune": [
        (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.05, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, None),
    #     (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.06, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, "0.06"),
    #     # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.07, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, "0.07"),
    #     # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.08, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, "0.08"),
    #     # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.09, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, "0.09"),
    #     # (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=0.10, threshold=0, pass_interval=8, apolune=True, mission_start_epoch=mission_start_epoch), num_runs, "0.10")
    ],
    "Random": [
        (helper_functions.get_random_arc_observation_windows(duration, arc_interval_vars=[3, 0.000001], threshold_vars=[1, 0.000001], arc_duration_vars=[1, 0.1], seed=0), num_runs, None),
        # (helper_functions.get_random_arc_observation_windows(duration, arc_interval_vars=[3, 0.000001], threshold_vars=[1, 0.000001], arc_duration_vars=[1, 0.2], seed=1), num_runs, "0.2"),
    ],
    "Default": [
        (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=3, threshold=1, arc_duration=1), num_runs, None),
    ]
}


# Varying pass_interval per orbit case
params = range(1, 9)
margin = 0.05
num_runs = 3
observation_windows_settings = {
    "Perilune": [
        (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=margin, mission_start_epoch=mission_start_epoch, apolune=False, pass_interval=pass_interval), num_runs, str(pass_interval)) for pass_interval in params
    ],
    "Apolune": [
        (helper_functions.get_orbit_based_arc_observation_windows(duration, margin=margin, mission_start_epoch=mission_start_epoch, apolune=True, pass_interval=pass_interval), num_runs, str(pass_interval)) for pass_interval in params
    ],
}


params = [0.1, 0.2, 0.5, 1.0, 2.0]
params2 = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
num_runs = 3
observation_windows_settings = {
    f"{param2} day": [
        (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=param2, arc_duration=param, mission_start_epoch=mission_start_epoch), num_runs, str(param)) for param in params
    ]
    for param2 in params2
}


# observation_windows = helper_functions.get_constant_arc_observation_windows(28, arc_interval=3, threshold=1, arc_duration=1)

# final_objective_values = {}
# rng = np.random.default_rng(seed=0)
# for mc_case in [1, 2, 5, 10]:

#     final_objective_values[mc_case] = []
#     for seed_case in range(5):


#         seeds = rng.integers(0, 10000, mc_case).tolist()
#         print(seeds)

#         objective_values = []
#         for seed in seeds:

#             print("mc_case: ", mc_case, "seed_case: ", seed_case, "seed: ", seed)
#             navigation_output = helper_functions.get_navigation_output(observation_windows, seed=seed)
#             navigation_simulator = navigation_output.navigation_simulator

#             delta_v_dict = navigation_simulator.delta_v_dict
#             delta_v = sum(np.linalg.norm(value) for key, value in delta_v_dict.items() if key > navigation_simulator.mission_start_epoch+14)

#             objective_value = delta_v
#             objective_values.append(objective_value)

#         final_objective_value = 1*np.mean(objective_values)+3*np.std(objective_values)

#         final_objective_values[mc_case].append(final_objective_value)

# # navigation_results = {1: [0.024077372876699074], 2: [0.024077372876699074, 0.023823555020290488], 5: [0.024077372876699074, 0.023823555020290488, 0.02413274377485531, 0.023332857997906273, 0.023678771780047594], 10: [0.024077372876699074, 0.023823555020290488, 0.02413274377485531, 0.023332857997906273, 0.023678771780047594, 0.024047907530802035, 0.024239130525301085, 0.024374607141202875, 0.024697331448210413, 0.024070191063758413]}
# # final_objective_values = {1: 0.024077372876699074, 2: 0.02433119073310766, 5: 0.02467928418381549, 10: 0.0251171903841037}
# # # {1: {'mean': 0.024077372876699074, 'std': nan, 'obj': nan}, 2: {'mean': 0.02395046394849478, 'std': 0.0001794763274527448, 'obj': 0.024488892930853014}, 5: {'mean': 0.02380906028995975, 'std': 0.0003243132970509986, 'obj': 0.024782000181112745}, 10: {'mean': 0.024047446915907357, 'std': 0.0003758695412875942, 'obj': 0.02517505553977014}}

# print(final_objective_values)

# # Calculate mean and standard deviation for each key
# statistics = {}
# for key, values in final_objective_values.items():
#     mean = np.mean(values)
#     std = np.std(values)  # Using sample standard deviation (ddof=1)
#     statistics[key] = {'mean': mean, 'std': std, "obj": mean+3*std}

# print(final_objective_values, statistics)

# # Extract keys, means, and standard deviations
# keys = list(statistics.keys())
# means = [statistics[key]['mean']+0*statistics[key]['std'] for key in keys]
# stds = [statistics[key]['std'] for key in keys]

# # Create a bar chart with error bars
# plt.figure(figsize=(10, 6))
# plt.bar(keys, means, yerr=stds, capsize=5, color='blue', alpha=0.7)
# plt.xlabel('Keys')
# plt.ylabel('Mean')
# plt.title('Mean with Standard Deviation as Error Bars for Each Key')
# plt.xticks(keys)
# plt.grid(axis='y')

# # Show the plot
# plt.show()







#######################################################
###### Generate the navigation outputs ################
#######################################################

print(observation_windows_settings)

observation_windows_settings = {
    "Default": [
        (helper_functions.get_constant_arc_observation_windows(duration, arc_interval=3, threshold=1, arc_duration=1), 1, None),
    ],
    "Optimized": [([
        [
            60390,
            60390.76217095995
        ],
        [
            60393.76217095995,
            60394.92989843622
        ],
        [
            60397.92989843622,
            60398.78205703818
        ],
        [
            60401.78205703818,
            60402.82977960193
        ],
        [
            60405.82977960193,
            60407.68702720861
        ],
        [
            60410.68702720861,
            60410.78702720861
        ],
        [
            60413.78702720861,
            60413.887027208606
        ]
    ], 3, None),
    ],
}

auxilary_settings = {}
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
    color_cycle=["salmon", "forestgreen"],
    figure_settings={"save_figure": True,
                    "current_time": current_time,
                    "file_name": file_name
    }
)

process_multiple_navigation_results.plot_uncertainty_comparison()
process_multiple_navigation_results.plot_maneuvre_costs(separate_plots=True)
process_multiple_navigation_results.plot_full_state_history_comparison(step_size=None)
# process_multiple_navigation_results.plot_monte_carlo_estimation_error_history(evaluation_threshold=14)
process_multiple_navigation_results.plot_maneuvre_costs_bar_chart(evaluation_threshold=14, bar_labeler=None, worst_case=False, observation_windows_settings=observation_windows_settings)
# process_multiple_navigation_results.plot_estimation_arc_comparison(evaluation_threshold=14, bar_labeler=None,  worst_case=False)
print("Plotting done...")

plt.show()