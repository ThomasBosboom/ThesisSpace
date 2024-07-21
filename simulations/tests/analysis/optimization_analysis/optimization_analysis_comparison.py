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

    optimization_methods = ["particle_swarm", "nelder_mead"]
    custom_tags = ["default"]

    comparison_labels = ["Particle-Swarm", "Nelder-Mead"]

    cases = {
        "delta_v_min": [0.00],
    }

    optimization_results = get_optimization_results(cases,
        optimization_methods=optimization_methods,
        custom_tags=custom_tags
        )

    # Transforming the initial_dict
    results = transform_dict(optimization_results, get_process_optimization_results)

    print(results)

    # Select only first case run as example
    final_process_optimization_results = find_first_object(results)

    # Compare cases
    compare_time_tags = get_compare_time_tags(results, comparison_labels)
    print("compare_time_tags: ", compare_time_tags)

    final_process_optimization_results.plot_iteration_history(
        show_design_variables=False,
        compare_time_tags=compare_time_tags
    )

    plt.show()