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

# Own
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

    combined_history_dict = {}

    for key, value in dict.items():
        design_vector = []
        objective_function = []

        # Extract design vector and objective function from each dictionary
        for history_key, history_value in value["history"].items():
            design_vector.append(history_value["design_vector"])
            objective_function.append(history_value["objective_function"])

        # Add combined data to the new dictionary
        combined_history_dict[key] = {
            "design_vector": design_vector,
            "objective_function": objective_function
        }

    return combined_history_dict



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

    # label = str(dynamic_model_list[1])+"_"+str(truth_model_list[1])+"_numruns"+str(numruns)+"_maxiter"+str(maxiter)+"_threshold"+str(threshold)+"_duration"+str(duration)+"_od_duration"+str(od_duration)

    monte_carlo_simulation_results = dict()
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
        monte_carlo_simulation_results[run] = optimization_result

    # Transform dictionaries and get statistics
    combined_history_dict = get_combined_history_dict(monte_carlo_simulation_results)
    monte_carlo_stats_dict = utils.get_monte_carlo_stats_dict(data_dict=combined_history_dict)

    # Save relevant dictionaries
    utils.save_dicts_to_folder(dicts=[monte_carlo_simulation_results], labels=[label])
    utils.save_dicts_to_folder(dicts=[monte_carlo_stats_dict], labels=["stats_"+label])

    print("Monte Carlo results: ", monte_carlo_simulation_results)
    print("Monte Carlo statistics: ", monte_carlo_stats_dict)
