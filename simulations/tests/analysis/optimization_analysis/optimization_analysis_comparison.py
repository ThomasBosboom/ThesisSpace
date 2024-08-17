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


if __name__ == "__main__":

    # optimization_methods = ["particle_swarm", "nelder_mead"]
    optimization_methods = ["particle_swarm"]
    custom_tags = ["default28dur1len3int", "default56dur1len3int", "default28dur1len3intSHSRP"]

    # comparison_labels = ["Particle-Swarm", "Nelder-Mead"]
    comparison_labels = ["28", "56", "SHSRP"]
    # comparison_labels = ["PMSRP", "SHSRP"]

    cases = {
        "delta_v_min": [0.00],
    }

    optimization_results = get_optimization_results(cases,
        optimization_methods=optimization_methods,
        custom_tags=custom_tags
        )

    # print("optimization_results: ", optimization_results)

    # Transforming the initial_dict
    process_optimization_results = transform_dict(optimization_results, get_process_optimization_results)
    # print("process_optimization_results: ", process_optimization_results)

    # Select only first case run as example
    process_optimization_result = find_first_object(process_optimization_results)
    # print("process_optimization_result", process_optimization_result)

    # Get the time tags associated with the optimization results
    compare_time_tags = get_compare_time_tags(process_optimization_results, comparison_labels)
    # print("compare_time_tags: ", compare_time_tags)

    process_optimization_result.plot_iteration_history(
        show_design_variables=False,
        show_annual=True,
        compare_time_tags=compare_time_tags
    )

    # Compare the (annual) objectives for the compared cases
    auxilary_settings = process_optimization_result.optimization_results["kwargs"]
    auxilary_settings["run_optimization_version"] = False
    process_optimization_result.plot_comparison_optimization_maneuvre_costs(auxilary_settings, process_optimization_results,
                                    compare_time_tags=compare_time_tags,
                                    show_observation_window_settings=True,
                                    custom_num_runs=5)

    plt.show()