import numpy as np
import scipy as sp
import os
import sys
import copy
import json
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

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


def check_file_exists(substring, num_optims, file_name=file_name, path="tests\postprocessing\dicts"):

    directory = os.path.join(os.path.join(file_directory, "tests", "postprocessing", "dicts"), file_name)

    count = 1
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            if str(substring) in filename:
                count += 1
                if count == num_optims:
                    return True

    return False


def collect_optimization_results(cases, substring=None):

    directory = os.path.join(file_directory, "tests", "postprocessing", "dicts")
    optimization_results = {}
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            for casekey in cases.keys():
                # casekey = str(casekey)
                for casevalues in cases.values():
                    for casevalue in casevalues:
                        casevalue = f"{casevalue:.2f}".replace('.', '_')
                        if casekey and casevalue in dir_name:

                            print(f"Subfolder: {dir_name} contains '{casekey}' and {casevalue} in its name")

                            optimization_results[casekey] = []
                            subfolder_path = os.path.join(root, dir_name)
                            for file_name in os.listdir(subfolder_path):
                                if substring is None or substring in file_name:
                                    file_path = os.path.join(subfolder_path, file_name)
                                    # print(f"Subfolder: {subfolder_path} contains '{substring}' in its name")
                                    print(file_path)

                                    with open(file_path, 'r') as file:
                                        data = json.load(file)

                                    optimization_results[casekey].append(data)

    return optimization_results


def get_optimization_results_folders(cases, substring=None):

    directory = os.path.join(file_directory, "tests", "postprocessing", "dicts")

    optimization_results = {}
    dir_names = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            for casekey in cases.keys():
                # casekey = str(casekey)
                for casevalues in cases.values():
                    for casevalue in casevalues:
                        casevalue = f"{casevalue:.2f}".replace('.', '_')
                        if casekey and casevalue in dir_name:
                            dir_names.append(dir_name)


                            # print(f"Subfolder: {dir_name} contains '{casekey}' and {casevalue} in its name")

                            # optimization_results[casekey] = []
                            # subfolder_path = os.path.join(root, dir_name)
                            # for file_name in os.listdir(subfolder_path):
                            #     if substring is None or substring in file_name:
                            #         file_path = os.path.join(subfolder_path, file_name)
                            #         # print(f"Subfolder: {subfolder_path} contains '{substring}' in its name")
                            #         print(file_path)

                            #         with open(file_path, 'r') as file:
                            #             data = json.load(file)

                            #         optimization_results[casekey].append(data)

    print(dir_names)
    return dir_names



def process_case(case, run, navigation_simulator_settings, objective_functions_settings, optimization_model_settings,
                 run_optimization, from_file, custom_tag, folder_name):


    time_tag = generate_case_time_tag(case, custom_time=custom_tag, run=run)

    for settings in [navigation_simulator_settings, objective_functions_settings, optimization_model_settings]:
        for key in case.keys():
            if key in settings:
                settings.update(case)

    navigation_simulator = NavigationSimulator.NavigationSimulator(
        **navigation_simulator_settings
    )

    seed = objective_functions_settings.get("seed")
    num_runs = objective_functions_settings.get("num_runs")
    objective_functions_settings.update({"seed": seed + run*num_runs})
    objective_functions = ObjectiveFunctions.ObjectiveFunctions(
        navigation_simulator,
        **objective_functions_settings
    )

    optimization_model_settings.update({
        "json_settings": {"save_dict": True, "current_time": time_tag, "file_name": folder_name}
    })
    optimization_model = OptimizationModel.OptimizationModel(
        **optimization_model_settings,
        **navigation_simulator_settings,
        **objective_functions_settings,
    )

    if run_optimization:
        if from_file:
            optimization_results = optimization_model.load_from_json(time_tag=time_tag, folder_name=folder_name)
            optimization_results.update(optimization_model_settings)
            optimization_model = OptimizationModel.OptimizationModel(custom_input=optimization_results)

        # Choose the objective function to optimize
        objective_function = objective_functions.test
        optimization_results = optimization_model.optimize(objective_function)

    if not run_optimization:
        optimization_results = optimization_model.load_from_json(time_tag=time_tag, folder_name=folder_name)

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

    return process_optimization_results



if __name__ == "__main__":

    ##############################################################
    #### Comparison of study cases ###############################
    ##############################################################

    auto_mode = False
    custom_tag = "0.5day_arc_interval"
    num_optims = 5

    run_optimization = False
    from_file = True

    # if custom_tag is not None:
    #     current_time = custom_tag
    # if auto_mode:
    #     if custom_tag is not None:
    #         current_time = custom_tag
    #     from_file = check_file_exists(current_time, num_optims)
    #     run_optimization = True

    cases = {
        "delta_v_min": [0.00, 0.01],
    }

    optimization_results_folders = get_optimization_results_folders(cases, substring=custom_tag)
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
        "arc_length": 0.5,
        "arc_interval": 3,
        "max_iterations": 30,
        "bounds": (-0.5, 1.5),
        "design_vector_type": "arc_lengths",
        "initial_simplex_perturbation": -0.5
    }

    ##############################################################
    #### Start of main loop ######################################
    ##############################################################

    keys, values = zip(*cases.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(combinations)

    results = []
    for case in combinations:
        for run in range(num_optims):
            print(case, run)

            optimization_results_folders = get_optimization_results_folders(cases, substring=custom_tag)

            results.append(process_case(
                case, run,
                navigation_simulator_settings=navigation_simulator_settings,
                objective_functions_settings=objective_functions_settings,
                optimization_model_settings=optimization_model_settings,
                run_optimization=True,
                from_file=True,
                custom_tag=current_time,
                folder_name=folder_name
            ))

            print(results)

    # Plot final iteration history
    final_process_optimization_results = results[-1]
    final_process_optimization_results.plot_iteration_history(
        show_design_variables=False,
        compare_time_tags=[result.time_tag for result in results]
    )

    plt.plot()





    # [(case, run) for case in combinations for run in range(num_optims)]

    # # Use multiprocessing.Pool to parallelize the loop
    # num_workers = multiprocessing.cpu_count()

    # # num_workers = 40
    # with multiprocessing.Pool(processes=num_workers) as pool:
    #     partial_process_case = partial(
    #         process_case,
    #         navigation_simulator_settings=navigation_simulator_settings,
    #         objective_functions_settings=objective_functions_settings,
    #         optimization_model_settings=optimization_model_settings,
    #         run_optimization=run_optimization,
    #         from_file=from_file,
    #         custom_tag=current_time)

    #     case_runs = [(case, run) for case in combinations for run in range(num_optims)]
    #     results = pool.starmap(partial_process_case, case_runs)

    # # Plot final iteration history
    # final_process_optimization_results = results[-1]
    # final_process_optimization_results.plot_iteration_history(
    #     show_design_variables=False,
    #     compare_time_tags=[result.time_tag for result in results]
    # )

    plt.show()