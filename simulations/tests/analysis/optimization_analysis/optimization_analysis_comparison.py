import os
import sys
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Define current time
current_time = datetime.now().strftime("%Y%m%d%H%M")

# Own
from src import NavigationSimulator, ObjectiveFunctions
from tests.postprocessing import ProcessOptimizationResults, OptimizationModel
from optimization_analysis_helper_functions import \
    get_optimization_results, get_process_optimization_results, \
        transform_dict, find_first_object, get_compare_time_tags


def plot_optimization_analysis_comparison(num_runs, optimization_methods, custom_tags, comparison_labels, custom_auxiliary_settings=None):

    cases = {"delta_v_min": [0.00]}
    optimization_results = get_optimization_results(cases,
        optimization_methods=optimization_methods,
        custom_tags=custom_tags
        )

    # Transforming the initial_dict
    process_optimization_results = transform_dict(optimization_results, get_process_optimization_results)

    # Select only first case run as example
    process_optimization_result = find_first_object(process_optimization_results)

    # Get the time tags associated with the optimization results
    compare_time_tags = get_compare_time_tags(process_optimization_results, comparison_labels)

    # Plot iteration history
    process_optimization_result.plot_iteration_history(
        show_design_variables=False,
        show_annual=True,
        compare_time_tags=compare_time_tags
    )

    # Compare the (annual) objectives for the compared cases
    process_optimization_result.plot_comparison_optimization_maneuvre_costs(
        process_optimization_results,
        compare_time_tags=compare_time_tags,
        show_observation_window_settings=True,
        custom_num_runs=num_runs,
        custom_auxiliary_settings=custom_auxiliary_settings
    )

    # plt.show()


if __name__ == "__main__":

    num_runs = 5

    # plot_optimization_analysis_comparison(
    #     num_runs,
    #     ["particle_swarm", "nelder_mead"],
    #     ["default28dur1len3int"],
    #     ["PSO, 28, PMSRP", "Nelder-Mead, 28, PMSRP"],
    #     custom_auxiliary_settings=None
    # )

    # plot_optimization_analysis_comparison(
    #     num_runs,
    #     ["particle_swarm"],
    #     ["default28dur1len3int", "default56dur1len3int", "default28dur1len3intSHSRP"],
    #     ["PSO, 28, PMSRP", "PSO, 56, PMSRP", "PSO, 56, SHSRP"],
    #     custom_auxiliary_settings=None
    # )

    # plot_optimization_analysis_comparison(
    #     num_runs,
    #     ["particle_swarm"],
    #     ["default28dur1len3int", "default56dur1len3int", "default28dur1len3intSHSRP"],
    #     ["PSO, 28, PMSRP", "PSO, 56, PMSRP", "PSO, 56, SHSRP"],
    #     custom_auxiliary_settings=None
    # )

    custom_auxiliary_settings = {}
    plot_optimization_analysis_comparison(
        num_runs,
        ["particle_swarm"],
        ["default28dur1len3int", "default28dur1len3intPropulsion", "default56dur1len3intPropulsion"],
        ["0.000 m/s, 28, PMSRP", "0.003 m/s, 28, PMSRP", "0.003 m/s, 56, PMSRP"],
        custom_auxiliary_settings=custom_auxiliary_settings
    )

    plt.show()