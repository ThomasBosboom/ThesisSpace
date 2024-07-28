import re
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
#### Analysis runs ###########################################
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
                 run_optimization, from_file, custom_tag, file_name, test_objective=False, use_same_seed=False, plot_full_comparison_cases=True):


    time_tag = generate_case_custom_tag(case, custom_tag, run=run)

    if optimization_model_settings["duration"] <= objective_functions_settings["evaluation_threshold"]:
        objective_functions_settings["evaluation_threshold"] = optimization_model_settings["duration"]

    navigation_simulator_settings.update(case)
    navigation_simulator = NavigationSimulator.NavigationSimulator(
        **navigation_simulator_settings
    )

    if not use_same_seed:
        seed = objective_functions_settings.get("seed")
        num_runs = objective_functions_settings.get("num_runs")
        objective_functions_settings.update({"seed": seed + run*num_runs})

        # print(objective_functions_settings)
    objective_functions = ObjectiveFunctions.ObjectiveFunctions(
        navigation_simulator,
        **objective_functions_settings
    )

    optimization_model_settings.update({
        "json_settings": {"save_dict": True, "current_time": time_tag, "file_name": file_name}
    })
    kwargs = {**optimization_model_settings, **navigation_simulator_settings, **objective_functions_settings}
    optimization_model = OptimizationModel.OptimizationModel(
        **kwargs,
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
        compare_time_tags={}
    )
    process_optimization_results.tabulate_optimization_results()

    if not run_optimization:
        runs, custom_num_runs = plot_full_comparison_cases[0], plot_full_comparison_cases[1]
        if run in runs:
            auxilary_settings = {
                                 "seed": objective_functions_settings["seed"],
                                 "run_optimization_version": False
                                 }
            auxilary_settings.update(case)
            process_optimization_results.plot_optimization_result_comparisons(auxilary_settings, show_observation_window_settings=True, custom_num_runs=custom_num_runs)

    return process_optimization_results



##############################################################
#### Comparison ##############################################
##############################################################

def subtring_in_string(substring, string):
    # The pattern ensures the substring is bounded by underscores on one or both sides
    pattern = r'(^|_)' + re.escape(substring) + r'(_|$)'
    return re.search(pattern, string) is not None


def get_optimization_results(cases,
                            optimization_methods=["nelder_mead", "particle_swarm"],
                            custom_tags=["default"]
                            ):

    directory = os.path.join(file_directory, "tests", "postprocessing", "dicts")

    optimization_results = {}
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            for optimization_method in optimization_methods:
                if subtring_in_string(optimization_method, dir_name):
                    if optimization_method not in optimization_results:
                        optimization_results[optimization_method] = {}

                        for custom_tag in custom_tags:

                            # print("custom_tag: ", custom_tag)
                            subfolder_path = os.path.join(root, dir_name)
                            file_paths = []
                            for run, file_name in enumerate(os.listdir(subfolder_path)):
                                file_path = os.path.join(subfolder_path, file_name)
                                if subtring_in_string(custom_tag, file_name):
                                    # print(custom_tag, file_name)
                                    if custom_tag not in optimization_results:
                                        optimization_results[optimization_method][custom_tag] = {}
                                    file_paths.append(file_path)

                            for run, file_name in enumerate(file_paths):
                                file_path = os.path.join(subfolder_path, file_name)

                                for casekey, casevalues in cases.items():
                                    for casevalue in casevalues:
                                        casevalue_new = f"{casevalue:.2f}".replace('.', '_')

                                        if casevalue_new in file_name:

                                            if casevalue not in optimization_results[optimization_method][custom_tag]:
                                                optimization_results[optimization_method][custom_tag][casevalue] = {}

                                            with open(file_path, 'r') as file:
                                                data = json.load(file)

                                            # import numpy as np
                                            # optimization_results[optimization_method][custom_tag][casevalue][run] = {"key1": np.random.random()}
                                            optimization_results[optimization_method][custom_tag][casevalue][run] = [data]

    return optimization_results


def get_process_optimization_results(list):

    optimization_results = list[0]
    kwargs = optimization_results["kwargs"]
    file_name = optimization_results["file_name"]
    current_time = optimization_results["current_time"]

    optimization_model = OptimizationModel.OptimizationModel(
        **kwargs,
    )

    process_optimization_results = ProcessOptimizationResults.ProcessOptimizationResults(
        current_time,
        optimization_model,
        save_settings={
            "save_table": True,
            "save_figure": True,
            "current_time": current_time,
            "file_name": file_name
        }
    )

    process_optimization_results.plot_iteration_history(
        show_design_variables=True,
        compare_time_tags={}
    )
    process_optimization_results.tabulate_optimization_results()

    return process_optimization_results


def transform_dict(d, func):
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[key] = transform_dict(value ,func)
        elif isinstance(value, list):
            new_dict[key] = func(value)
        else:
            new_dict[key] = value
    return new_dict


# Recursive function to find and return the first object
def find_first_object(d):
    for key, value in d.items():
        if isinstance(value, dict):
            return find_first_object(value)
        elif isinstance(value, list):
            return value[0][0]
        else:
            return value


def get_innermost_values(d):

    innermost_values = []
    i = 0
    def _get_innermost(d):
        inner_list = []
        for key, value in d.items():

            if isinstance(value, dict):
                _get_innermost(value)
            else:
                inner_list.append(value)

        innermost_values.append(inner_list)

    _get_innermost(d)

    innermost_values = [lst for lst in innermost_values if lst]

    return innermost_values


def get_compare_time_tags(results, comparison_labels):

    results_lists = get_innermost_values(results)
    # print("results_lists:", results_lists)
    time_tags = [[result.time_tag for result in results] for results in results_lists]

    compare_time_tags = {}
    for index, key in enumerate(comparison_labels):
        compare_time_tags[key] = time_tags[index]

    return compare_time_tags