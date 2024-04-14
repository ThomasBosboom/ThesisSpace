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

from tests import utils, helper_functions
from src.optimization_models import OptimizationModel
from src import NavigationSimulator, PlotNavigationResults



#################################################################
###### Fixed arc durations ######################################
#################################################################

### Run the navigation routines for varying observation windows
run = True
start_epoch = 60390
# duration = 28
# skm_to_od_duration = 3
# threshold = 7
# od_duration = 1
# num_runs = 3
# od_durations = [0.1, 0.2, 0.5, 1, 1.5, 2.0, 2.5, 3]
# skm_to_od_durations = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
# od_durations = [0.1, 1, 3]
# skm_to_od_durations = [1, 3]
# custom_target_point_epochs = [2, 3, 3.9, 4]

models = ["PMSRP"]
durations = [28]
thresholds = [7]
custom_station_keeping_errors = [1e-2]
custom_target_point_epochs = [3]
skm_to_od_durations = [1, 2, 3, 4]
od_durations = [0.1, 0.2, 0.5, 1, 1.5]

# custom_station_keeping_errors = [1e-10]
# custom_target_point_epochs = [1, 2, 3, 3.5, 4]
# skm_to_od_durations = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
# od_durations = [0.1, 0.2, 0.5, 1, 1.5, 2.0, 2.5, 3]

custom_initial_estimation_error = np.array([5e-0, 5e-0, 5e-0, 1e-5, 1e-5, 1e-5, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])*1e0
custom_apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
custom_orbit_insertion_error = np.array([1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e1, 1e1, 1e1, 1e-4, 1e-4, 1e-4])*1e2


if run:

    current_time_string = datetime.now().strftime("%d%m%H%M")

    navigation_results_dict = {}
    delta_v_dict_constant_arc = {}
    input_dict = {}

    input_dict["models"] = models
    input_dict["durations"] = durations
    input_dict["thresholds"] = thresholds
    input_dict["custom_station_keeping_errors"] = custom_station_keeping_errors
    input_dict["custom_target_point_epochs"] = custom_target_point_epochs
    input_dict["skm_to_od_durations"] = skm_to_od_durations
    input_dict["od_durations"] = od_durations
    input_dict["custom_initial_estimation_error"] = list(custom_initial_estimation_error)
    input_dict["custom_apriori_covariance"] = list(np.diagonal(custom_apriori_covariance))
    input_dict["custom_orbit_insertion_error"] = list(custom_orbit_insertion_error)
    delta_v_dict_constant_arc["inputs"] = input_dict

    print(delta_v_dict_constant_arc)

    for model in models:

        delta_v_dict_per_duration = {}
        for duration in durations:

            delta_v_dict_per_threshold = {}
            for threshold in thresholds:

                delta_v_dict_per_station_keeping_error = {}
                for custom_station_keeping_error in custom_station_keeping_errors:

                    delta_v_dict_per_target_point_epoch = {}
                    for custom_target_point_epoch in custom_target_point_epochs:

                        delta_v_dict_per_skm_to_od_duration = {}
                        for skm_to_od_duration in skm_to_od_durations:

                            delta_v_dict_per_od_duration = {}
                            for od_duration in od_durations:

                                # print(custom_target_point_epoch, od_duration)

                                # objective_value = np.random.randint(1, 100)
                                # if custom_target_point_epoch >= 3:
                                #     objective_value *= 0.01

                                # Generate a vector with OD durations
                                epoch = start_epoch + threshold + skm_to_od_duration + od_duration
                                skm_epochs = []
                                i = 1
                                while True:
                                    if epoch <= start_epoch+duration:
                                        skm_epochs.append(epoch)
                                        epoch += skm_to_od_duration+od_duration
                                    else:
                                        design_vector = od_duration*np.ones(np.shape(skm_epochs))
                                        break
                                    i += 1

                                # Extract observation windows
                                observation_windows = [(start_epoch, start_epoch+threshold)]
                                for i, skm_epoch in enumerate(skm_epochs):
                                    observation_windows.append((skm_epoch-od_duration, skm_epoch))

                                station_keeping_epochs = [windows[1] for windows in observation_windows]

                                # Start running the navigation routine
                                navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                                                ["HF", model, 0],
                                                                                                ["HF", model, 0],
                                                                                                step_size=1e-2,
                                                                                                station_keeping_epochs=station_keeping_epochs,
                                                                                                target_point_epochs=[custom_target_point_epoch],
                                                                                                custom_station_keeping_error=custom_station_keeping_error,
                                                                                                custom_initial_estimation_error=custom_initial_estimation_error,
                                                                                                custom_apriori_covariance=custom_apriori_covariance,
                                                                                                custom_orbit_insertion_error=custom_orbit_insertion_error)

                                navigation_output = navigation_simulator.perform_navigation()
                                navigation_results = navigation_output.navigation_results
                                navigation_simulator = navigation_output.navigation_simulator

                                full_reference_state_deviation_history = navigation_results[1][1]

                                reference_state_deviations = np.linalg.norm(full_reference_state_deviation_history[:,:3], axis=1)

                                navigation_results_dict[od_duration] = navigation_results

                                delta_v = navigation_results[8][1]
                                delta_v_norm = np.linalg.norm(delta_v, axis=1)
                                objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
                                print(f"Objective: \n", delta_v, objective_value)
                                print("End of objective calculation ===============")

                                delta_v_dict_per_od_duration[od_duration] = [objective_value, min(delta_v_norm), max(delta_v_norm), max(reference_state_deviations), len(navigation_simulator.station_keeping_epochs)]

                            delta_v_dict_per_skm_to_od_duration[skm_to_od_duration] = delta_v_dict_per_od_duration

                        delta_v_dict_per_target_point_epoch[custom_target_point_epoch] = delta_v_dict_per_skm_to_od_duration

                    delta_v_dict_per_station_keeping_error[custom_station_keeping_error] = delta_v_dict_per_target_point_epoch

                delta_v_dict_per_threshold[threshold] = delta_v_dict_per_station_keeping_error

            delta_v_dict_per_duration[duration] = delta_v_dict_per_threshold

        delta_v_dict_constant_arc[model] = delta_v_dict_per_duration

    print(navigation_results_dict)
    print(delta_v_dict_constant_arc)

    utils.save_dicts_to_folder(dicts=[delta_v_dict_constant_arc], labels=[current_time_string+"_delta_v_dict_constant_arc"], custom_sub_folder_name=file_name)




# ### Run the navigation routines for varying observation windows
# run=True
# dynamic_model_list = ["HF", "PM", 0]
# truth_model_list = ["HF", "PM", 0]
# duration = 28
# skm_to_od_duration = 3
# threshold = 3
# od_duration = 1
# num_runs = 10
# mean = od_duration
# std_dev = 0.3
# custom_station_keeping_error = 1e-2

# if run:
#     navigation_results_dict = {}
#     delta_v_dict = {}
#     for model in ["PM", "PMSRP"]:

#         np.random.seed(0)
#         delta_v_dict_per_run = {}
#         for num_run in range(num_runs):

#             # Generate a vector with OD durations
#             od_durations = np.random.normal(loc=mean, scale=std_dev, size=(20))
#             start_epoch = 60390
#             epoch = start_epoch + threshold + skm_to_od_duration + od_durations[0]
#             skm_epochs = []
#             i = 1
#             while True:
#                 if epoch < start_epoch+duration:
#                     skm_epochs.append(epoch)
#                     epoch += skm_to_od_duration+od_durations[i]
#                 else:
#                     design_vector = np.ones(np.shape(skm_epochs))
#                     break
#                 i += 1

#             # Extract observation windows
#             observation_windows = [(start_epoch, start_epoch+threshold)]
#             for i, skm_epoch in enumerate(skm_epochs):
#                 observation_windows.append((skm_epoch-od_durations[i], skm_epoch))

#             # Start running the navigation routine
#             navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
#                                                                             ["HF", model, 0],
#                                                                             ["HF", model, 0],
#                                                                             station_keeping_epochs=skm_epochs,
#                                                                             target_point_epochs=[3],
#                                                                             custom_station_keeping_error=custom_station_keeping_error,
#                                                                             step_size=1e-2)

#             navigation_results = navigation_simulator.perform_navigation().navigation_results

#             navigation_results_dict[num_run] = navigation_results

#             delta_v = navigation_results[8][1]
#             objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
#             print(f"Objective: \n", delta_v, objective_value)
#             print("End of objective calculation ===============")

#             delta_v_dict_per_run[num_run] = [list(od_durations[:len(design_vector)]), objective_value]

#         delta_v_dict[model] = delta_v_dict_per_run

#     print(navigation_results_dict)
#     print(delta_v_dict)

#     utils.save_dicts_to_folder(dicts=[delta_v_dict], labels=[f"delta_v_dict_variable_arc_duration_{custom_station_keeping_error}"], custom_sub_folder_name=file_name)