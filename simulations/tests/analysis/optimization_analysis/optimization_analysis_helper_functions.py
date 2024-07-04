import numpy as np
import os
import sys
import json
from datetime import datetime

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


##############################################################
#### Helper functions ########################################
##############################################################

def generate_case_custom_tag(case, custom_tag, run=0):
    params_str = "_".join(f"{run}_{k}_{v:.2f}".replace('.', '_') for k, v in case.items())
    return f"{custom_tag}_{params_str}"


def check_file_exists(cases, custom_tag, num_optims, folder_name):

    directory = os.path.join(os.path.join(file_directory, "tests", "postprocessing", "dicts"), folder_name)

    count = 0
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            for key, values in cases.items():
                for value in values:
                    for run in range(num_optims):
                        case = {key: value}
                        case_custom_tag = generate_case_custom_tag(case, custom_tag, run=run)
                        if str(case_custom_tag) in filename:
                            count += 1
                            if count == num_optims:
                                return True
    return False


def process_case(case, run, navigation_simulator_settings, objective_functions_settings, optimization_model_settings,
                 run_optimization, from_file, custom_tag, file_name, test_objective=False, use_same_seed=False):


    time_tag = generate_case_custom_tag(case, custom_tag, run=run)

    # for settings in [navigation_simulator_settings, objective_functions_settings, optimization_model_settings]:
    #     for key in case.keys():
    #         if key in settings:
    #             settings.update(case)

    navigation_simulator_settings.update(case)
    navigation_simulator = NavigationSimulator.NavigationSimulator(
        **navigation_simulator_settings
    )

    if not use_same_seed:
        seed = objective_functions_settings.get("seed")
        num_runs = objective_functions_settings.get("num_runs")
        objective_functions_settings.update({"seed": seed + run*num_runs})
    objective_functions = ObjectiveFunctions.ObjectiveFunctions(
        navigation_simulator,
        **objective_functions_settings
    )

    optimization_model_settings.update({
        "json_settings": {"save_dict": True, "current_time": time_tag, "file_name": file_name}
    })
    optimization_model = OptimizationModel.OptimizationModel(
        **optimization_model_settings,
        **navigation_simulator_settings,
        **objective_functions_settings,
    )

    if run_optimization:
        if from_file:
            optimization_results = optimization_model.load_from_json(time_tag=time_tag, folder_name=file_name)
            optimization_results.update(optimization_model_settings)
            optimization_model = OptimizationModel.OptimizationModel(custom_input=optimization_results)

        # Choose the objective function to optimize
        objective_function = objective_functions.worst_case_station_keeping_cost
        if test_objective:
            objective_function = objective_functions.test
        optimization_results = optimization_model.optimize(objective_function)

    if not run_optimization:
        optimization_results = optimization_model.load_from_json(time_tag=time_tag, folder_name=file_name)

    process_optimization_results = ProcessOptimizationResults.ProcessOptimizationResults(
        time_tag,
        optimization_model,
        save_settings={
            "save_table": True,
            "save_figure": True,
            "current_time": time_tag,
            "file_name": file_name
        }
    )

    process_optimization_results.plot_iteration_history(
        show_design_variables=True,
        compare_time_tags=[]
    )
    process_optimization_results.tabulate_optimization_results()

    if not run_optimization:
        process_optimization_results.plot_optimization_result_comparisons(case, show_observation_window_settings=True)

    return process_optimization_results