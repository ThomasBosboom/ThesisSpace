# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sp

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils
from src.optimization_models import OptimizationModel

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

    stripped_dict = {key: value["history"] for key, value in dict.items()}

    # Extract the design_vector subkey for each main key
    names = ["design_vector", "objective_value"]
    results = []
    for name in names:
        results.append({key: value[name] for key, value in stripped_dict.items()})

    combined_history_dict = {}
    for i, data in enumerate(results):
        transformed_data = {}

        for key in data[str(0)]:
            values = []
            for subkey in data:
                values.append(data[subkey][key])

            transformed_data[key] = values
        combined_history_dict[names[i]] = transformed_data

    return combined_history_dict


def get_monte_carlo_stats_dict(dict):

    stats = {}
    mean_values = {}
    std_dev_values = {}
    for key, value in dict.items():

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

    return stats


def run_monte_carlo_optimization_model(dynamic_model_list, truth_model_list, threshold, skm_to_od_duration, duration, od_duration, bounds=(0.5, 1.5), numruns=1, maxiter=5, factor=2, custom_initial_design_vector=None, label=None):

    print("Starting MC simulation with following settings: \n")
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
        utils.save_dicts_to_folder(dicts=[optimization_result], custom_sub_folder_name=label, labels=["run_"+str(run)+"_"+label])

    # Transform dictionaries and get statistics
    combined_history_dict = get_combined_history_dict(monte_carlo_results_dict)
    monte_carlo_stats_dict = get_monte_carlo_stats_dict(combined_history_dict)

    # Save total statistics dictionary
    utils.save_dicts_to_folder(dicts=[monte_carlo_results_dict], custom_sub_folder_name=label, labels=["combined_"+label])
    utils.save_dicts_to_folder(dicts=[monte_carlo_stats_dict], custom_sub_folder_name=label, labels=["stats_"+label])

    # print("Monte Carlo results: ", monte_carlo_results_dict)
    # print("Monte Carlo statistics: ", monte_carlo_stats_dict)