# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils
from src.optimization_models import OptimizationModel
from src import NavigationSimulator

#################################################################################
###### Test runs of the navigation simulator ####################################
#################################################################################

def get_optimization_result(dynamic_model_list, truth_model_list, threshold, skm_to_od_duration, duration, od_duration, bounds=(0.5, 1.5), maxiter=5, factor=2, custom_initial_design_vector=None):

    # Create OptimizationModel instance based on timing characteristics
    optimization_model = OptimizationModel.OptimizationModel(dynamic_model_list,
                                                             truth_model_list,
                                                             threshold=threshold,
                                                             skm_to_od_duration=skm_to_od_duration,
                                                             duration=duration,
                                                             od_duration=od_duration,
                                                             bounds=bounds)

    # Adjust optimization attributes
    if custom_initial_design_vector is not None:
        optimization_model.initial_design_vector = custom_initial_design_vector
    optimization_model.maxiter = maxiter
    optimization_model.factor = factor

    # Run optimization
    optimization_result = optimization_model.optimize()

    return optimization_result



def get_combined_history_dict(dict):

    history_dict = {key: value["history"] for key, value in dict.items()}
    final_result_dict = {key: value["final_result"] for key, value in dict.items()}

    # Extract the design_vector subkey for each main key
    names = ["design_vector", "objective_value"]
    results = []
    for name in names:
        results.append({key: value[name] for key, value in history_dict.items()})

    combined_history_dict = {}
    for i, data in enumerate(results):
        transformed_data = {}
        for key in data[str(0)]:
            values = []
            for subkey in data:
                values.append(data[subkey][key])

            transformed_data[key] = values
        combined_history_dict[names[i]] = transformed_data

    results = []
    names = ["approx_annual_deltav", "reduction_percentage", "run_time"]
    for name in names:
        results.append({key: value[name] for key, value in final_result_dict.items()})

    for i, result in enumerate(results):
        combined_history_dict[names[i]] = result

    return combined_history_dict


def get_monte_carlo_stats_dict(dict):

    stats = {}
    stats["num_runs"] = len(dict)

    dict = get_combined_history_dict(dict)


    mean_values = {}
    std_dev_values = {}
    new_dict = {key: dict[key] for key in ["design_vector", "objective_value"]}
    for key, value in new_dict.items():

        mean_list = []
        std_dev_list = []
        for subkey, subvalue in value.items():

            subvalue_array = np.array(subvalue)
            mean_value = np.mean(subvalue_array, axis=0)
            std_dev_value = np.std(subvalue_array, axis=0)
            if isinstance(mean_value, np.ndarray):
                mean_value = list(mean_value)
            if isinstance(std_dev_value, np.ndarray):
                std_dev_value = list(std_dev_value)

            mean_list.append(mean_value)
            std_dev_list.append(std_dev_value)

        mean_values[key] = {i: mean for i, mean in enumerate(mean_list)}
        std_dev_values[key] = {i: std_dev for i, std_dev in enumerate(std_dev_list)}


    stats["mean"] = mean_values
    stats["std_dev"] = std_dev_values

    new_dict = {key: dict[key] for key in ["approx_annual_deltav", "reduction_percentage", "run_time"]}

    mean_values = {}
    std_dev_values = {}
    for key, value in new_dict.items():
        mean_list = []
        std_dev_list = []
        for subkey, subvalue in value.items():
            mean_list.append(subvalue)
            std_dev_list.append(subvalue)

        stats["mean"][key] = np.mean(mean_list)
        stats["std_dev"][key] = np.std(std_dev_list)

    return stats


def load_json_file(file_path):

    with open(file_path, 'r') as file:
        return json.load(file)


def concatenate_json_files(folder_path, batch=None):

    concatenated_json = {}
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and 'run' in filename:
            if batch is not None and str(batch) in filename:
                file_path = os.path.join(folder_path, filename)
                file_json = load_json_file(file_path)
                concatenated_json[str(file_count)] = file_json
                file_count += 1
            elif batch is None:
                file_path = os.path.join(folder_path, filename)
                file_json = load_json_file(file_path)
                concatenated_json[str(file_count)] = file_json
                file_count += 1

    return concatenated_json


def run_monte_carlo_optimization_model(dynamic_model_list, truth_model_list, threshold, skm_to_od_duration, duration, od_duration, bounds=(0.5, 1.5), numruns=1, maxiter=5, factor=2, custom_initial_design_vector=None, label=None):

    current_time_string = datetime.now().strftime("%d%m%H%M")

    print(f"Starting MC simulation at {current_time_string} with following settings: \n")
    print(f"dynamic_model_list: {dynamic_model_list}")
    print(f"truth_model_list: {truth_model_list}")
    print(f"threshold: {threshold}")
    print(f"skm_to_od_duration: {skm_to_od_duration}")
    print(f"duration: {duration}")
    print(f"od_duration: {od_duration}")
    print(f"bounds: {bounds}")
    print(f"numruns: {numruns}")
    print(f"maxiter: {maxiter}")
    print(f"custom_initial_design_vector: {custom_initial_design_vector}")
    print(f"factor: {factor}")

    monte_carlo_results_dict = dict()
    for run in range(numruns):

        optimization_result = get_optimization_result(dynamic_model_list,
                                                      truth_model_list,
                                                      threshold,
                                                      skm_to_od_duration,
                                                      duration,
                                                      od_duration,
                                                      bounds=bounds,
                                                      maxiter=maxiter,
                                                      custom_initial_design_vector=custom_initial_design_vector,
                                                      factor=factor)

        print(f"Optimization result of run {run}: ", optimization_result)
        monte_carlo_results_dict[str(run)] = optimization_result

        # Save individual run dictionary
        utils.save_dicts_to_folder(dicts=[optimization_result], custom_sub_folder_name=label, labels=[current_time_string+"_run_"+str(run)+"_"+label])

    # Transform dictionaries and get statistics
    # combined_history_dict = get_combined_history_dict(monte_carlo_results_dict)
    monte_carlo_stats_dict = get_monte_carlo_stats_dict(monte_carlo_results_dict)

    # Save total statistics dictionary
    utils.save_dicts_to_folder(dicts=[monte_carlo_results_dict], custom_sub_folder_name=label, labels=[current_time_string+"_combined_"+label])
    utils.save_dicts_to_folder(dicts=[monte_carlo_stats_dict], custom_sub_folder_name=label, labels=[current_time_string+"_stats_"+label])

    # print("Monte Carlo results: ", monte_carlo_results_dict)
    # print("Monte Carlo statistics: ", monte_carlo_stats_dict)



def get_navigation_results(observation_windows, dynamic_model_list, truth_model_list, step_size=1e-2, station_keeping_epochs=[], target_point_epochs=[3]):

    print(observation_windows)
    print(station_keeping_epochs)
    navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                    dynamic_model_list,
                                                                    truth_model_list,
                                                                    station_keeping_epochs=station_keeping_epochs,
                                                                    target_point_epochs=target_point_epochs,
                                                                    step_size=step_size)

    navigation_results = navigation_simulator.get_navigation_results()

    navigation_simulator.plot_navigation_results(navigation_results, show_directly=False)

    delta_v = navigation_results[8][1]
    objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
    print(f"Objective: \n", delta_v, objective_value, observation_windows[-1][-1]-observation_windows[0][0], run_time)
    print("End of objective calculation ===============")

    return navigation_results



def get_custom_observation_windows(duration, skm_to_od_duration, threshold, od_duration):

    # Generate a vector with OD durations
    start_epoch = 60390
    epoch = start_epoch + threshold + skm_to_od_duration + od_duration
    skm_epochs = []
    i = 1
    while True:
        if epoch < start_epoch+duration:
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

    # # Define the starting and ending keys
    # start_key = 60390
    # end_key = start_key + duration

    # # Generate the list of tuples representing the ranges of keys
    # observation_windows = []
    # last_key = start_key + threshold
    # i = 0
    # while True:
    #     if i == 0:
    #         observation_windows.append((start_key, start_key + threshold))
    #     else:
    #         observation_windows.append((last_key+skm_to_od_duration, last_key + skm_to_od_duration + od_duration))
    #         last_key += skm_to_od_duration + od_duration
    #         if last_key > end_key:
    #             break

    #     i += 1
    return observation_windows
