import numpy as np
import scipy as sp
import os
import sys
import copy
import json
import itertools
from datetime import datetime
import matplotlib.pyplot as plt

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(3):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Define current time
current_time = datetime.now().strftime("%Y%m%d%H%M")

# Own
from src import NavigationSimulator, ObjectiveFunctions
from tests.postprocessing import ProcessOptimizationResults, OptimizationModel


def generate_case_time_tag(case, custom_time=False, run=0):
    params_str = "_".join(f"{run}_{k}_{v:.2f}".replace('.', '_') for k, v in case.items())
    time = current_time
    if custom_time is not False:
        time = custom_time
    return f"{time}_{params_str}"


if __name__ == "__main__":

    ##############################################################
    #### Optimization settings ###################################
    ##############################################################

    run_optimization = True
    custom_input = False
    custom_time_tag = 202406251312

    run_num = 4
    cases = {
        "delta_v_min": [0.00, 0.01, 0.02]
    }

    navigation_simulator_settings = {
        "show_corrections_in_terminal": True,
        "run_optimization_version": True,
        "step_size": 0.5
    }

    objective_functions_settings = {
        "evaluation_threshold": 14,
        "num_runs": 5,
        "seed": 0
    }

    optimization_model_settings = {
        "duration": 28,
        "arc_length": 1,
        "arc_interval": 3,
        "max_iterations": 50,
        "bounds": (-0.9, 0.9),
        "design_vector_type": "arc_lengths",
        "initial_simplex_perturbation": -0.1,
    }


    ##############################################################
    #### Start of main loop ######################################
    ##############################################################

    keys, values = zip(*cases.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    time_tags = []
    for case in combinations:
        for run in range(run_num):

            time_tag = generate_case_time_tag(case, run=run)
            if custom_input:
                time_tag = generate_case_time_tag(case, custom_time=custom_time_tag, run=run)
            time_tags.append(time_tag)

            navigation_simulator_settings.update(case)
            navigation_simulator = NavigationSimulator.NavigationSimulator(
                **navigation_simulator_settings
            )

            seed = objective_functions_settings.get("seed")
            num_runs = objective_functions_settings.get("num_runs")
            objective_functions_settings.update({"seed": seed+num_runs})
            objective_functions = ObjectiveFunctions.ObjectiveFunctions(
                navigation_simulator,
                **objective_functions_settings
            )

            optimization_model_settings.update({"json_settings": {"save_dict": True, "current_time": time_tag, "file_name": file_name}})
            optimization_model = OptimizationModel.OptimizationModel(
                **optimization_model_settings,
                **navigation_simulator_settings,
                **objective_functions_settings,
            )

            if run_optimization:
                if custom_input:
                    optimization_results = optimization_model.load_from_json(time_tag=time_tag, folder_name=file_name)
                    optimization_results.update(optimization_model_settings)
                    optimization_model = OptimizationModel.OptimizationModel(custom_input=optimization_results)

                # Choose the objective function to optimize
                objective_function = objective_functions.test
                optimization_results = optimization_model.optimize(objective_function)

            if not run_optimization:
                optimization_results = optimization_model.load_from_json(time_tag=time_tag, folder_name=file_name)

            process_optimization_results = ProcessOptimizationResults.ProcessOptimizationResults(
                time_tag,
                optimization_model,
                save_settings={"save_table": True,
                            "save_figure": True,
                            "current_time": time_tag,
                            "file_name": file_name
                }
            )

            process_optimization_results.plot_iteration_history(
                show_design_variables=True,
                compare_time_tags=[])
            process_optimization_results.tabulate_optimization_results()
            # process_optimization_results.plot_optimization_result_comparison(
            #     show_observation_window_settings=False
            # )

    process_optimization_results.plot_iteration_history(
        show_design_variables=False,
        compare_time_tags=time_tags)

    plt.show()