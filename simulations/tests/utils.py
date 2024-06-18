# Standard
import os
import sys
import copy
import pytest_html
import numpy as np
import time
import json
import inspect

# tudatpy
from tudatpy.kernel.astro import time_conversion

# Own
from src import Interpolator, EstimationModel

parent_dir = os.path.dirname(os.path.dirname(__file__))


def get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time, custom_model_dict=None, get_only_first=False, custom_model_list=None, custom_initial_state=None, custom_propagation_time=None, **kwargs):

    if custom_model_dict is None:
        custom_model_dict = {"LF": ["CRTBP"], "HF": ["PM", "PMSRP", "SH", "SHSRP"], "FF": ["TRUTH"]}
    else:
        custom_model_dict = custom_model_dict

    dynamic_model_objects = {}
    for package_type, package_name_list in custom_model_dict.items():
        sub_dict = {package_name_list[i]: [] for i in range(len(package_name_list))}
        packages_dir = os.path.join(parent_dir, 'src', 'dynamic_models', package_type)

        package_name_counter = 0
        for package_name in custom_model_dict[package_type]:
            package_module_path = f'dynamic_models.{package_type}.{package_name}'
            package_module = __import__(package_module_path, fromlist=[package_name])
            package_files = os.listdir(os.path.join(packages_dir, package_name))

            if custom_model_list is not None:
                package_files = [package_files[i] for i in custom_model_list]

            for file_name_index, file_name in enumerate(package_files):
                if file_name.endswith('.py') and not file_name.startswith('__init__'):
                    module_path = f'{package_module_path}.{os.path.splitext(file_name)[0]}'
                    module = __import__(module_path, fromlist=[file_name])

                    if package_type == "LF":
                        DynamicModel = module.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state, custom_propagation_time=custom_propagation_time, **kwargs)
                    else:
                        # print("here")
                        DynamicModel = module.HighFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state, custom_propagation_time=custom_propagation_time, **kwargs)

                    sub_dict[package_name_list[package_name_counter]].extend([DynamicModel])
                    dynamic_model_objects[package_type] = sub_dict

                if get_only_first:
                    break

                # if file_name_index in custom_model_list:



            package_name_counter += 1

    # if get_only_first:
    #     result_dict = {}
    #     for key, inner_dict in dynamic_model_objects.items():
    #         if inner_dict and isinstance(inner_dict, dict):
    #             updated_inner_dict = {}
    #             for subkey, sublist in inner_dict.items():
    #                 if sublist and isinstance(sublist, list):
    #                     print(key, subkey, sublist)
    #                     updated_inner_dict[subkey] = [sublist[0]]
    #                 else:
    #                     updated_inner_dict[subkey] = sublist

    #             result_dict[key] = updated_inner_dict

    #     return result_dict

    # else:
    #     return dynamic_model_objects

    return dynamic_model_objects


def get_estimation_model_objects(dynamic_model_objects,
                                 custom_truth_model=None,
                                 apriori_covariance=None,
                                 initial_estimation_error=None):

    estimation_model_objects = {}
    for package_type, package_names in dynamic_model_objects.items():
        submodels_dict = {}
        for package_name, dynamic_models in package_names.items():

            # for dynamic_model in dynamic_models:
            #     print(f"VALUE IN UTILS {dynamic_model}: \n", dynamic_model.simulation_start_epoch_MJD, dynamic_model.propagation_time)

            # print("dynamic models", dynamic_models)

            # Adjust such that full-fidelity model with the correct initial state is used
            if custom_truth_model is None:
                # for dynamic_model in dynamic_models:
                #     simuat
                # print("been here", dynamic_models[0].simulation_start_epoch_MJD, dynamic_models[0].propagation_time)
                truth_model = TRUTH.HighFidelityDynamicModel(dynamic_models[0].simulation_start_epoch_MJD, dynamic_models[0].propagation_time)
            else:
                truth_model = custom_truth_model
                # print(f"VALUE IN UTILS {truth_model}: \n", truth_model.propagation_time)
                # print(f"VALUE IN UTILS {truth_model}: \n", truth_model.custom_propagation_time)

            submodels = [EstimationModel.EstimationModel(dynamic_model, truth_model, apriori_covariance=apriori_covariance, initial_estimation_error=initial_estimation_error) for dynamic_model in dynamic_models]

            submodels_dict[package_name] = submodels
        estimation_model_objects[package_type] = submodels_dict



    return estimation_model_objects


# def save_figures_to_folder(figs=[], labels=[], save_to_report=True, use_with_pytest=True):

#     extras = []
#     folder_name = inspect.currentframe().f_back.f_code.co_name

#     base_folder = "figures"
#     if not os.path.exists(base_folder):
#         os.makedirs(base_folder, exist_ok=True)

#     figure_folder = os.path.join(base_folder, folder_name)
#     if not os.path.exists(figure_folder):
#         os.makedirs(figure_folder, exist_ok=True)

#     for i, fig in enumerate(figs):
#         base_string = "_".join([str(label) for label in labels])
#         if save_to_report:
#             file_name = f"fig{i+1}_{base_string}.png"
#         else:
#             file_name = f"fig_3d{i+1}_{base_string}.png"
#         figure_path = os.path.join(figure_folder, file_name)
#         fig.savefig(figure_path)
#         if save_to_report:
#             if use_with_pytest:
#                 extras.append(pytest_html.extras.png(figure_path))


def get_dynamic_model_results(simulation_start_epoch_MJD,
                              propagation_time,
                              custom_model_dict=None,
                              get_only_first=False,
                              custom_initial_state=None,
                              step_size=0.01,
                              epoch_in_MJD=True,
                              entry_list=None,
                              solve_variational_equations=True,
                              custom_propagation_time=None,
                              specific_model_list=None,
                              return_dynamic_model_objects=False):

    dynamic_model_objects = get_dynamic_model_objects(simulation_start_epoch_MJD,
                                                      propagation_time,
                                                      custom_model_dict=custom_model_dict,
                                                      get_only_first=get_only_first,
                                                      custom_initial_state=custom_initial_state,
                                                      custom_propagation_time=custom_propagation_time)


    dynamic_model_objects_results = copy.deepcopy(dynamic_model_objects)
    for model_type, model_names in dynamic_model_objects.items():
        for model_name, dynamic_models in model_names.items():
            for i, dynamic_model in enumerate(dynamic_models):

                dynamic_model_objects_results[model_type][model_name][i] = [None]

                if specific_model_list is None:

                    start_time = time.time()
                    results_list = list(Interpolator.Interpolator(step_size=step_size, epoch_in_MJD=epoch_in_MJD).get_propagation_results(dynamic_model,
                                                                                                                    solve_variational_equations=solve_variational_equations,
                                                                                                                    custom_initial_state=custom_initial_state,
                                                                                                                    custom_propagation_time=custom_propagation_time))
                    results_list.append(time.time()-start_time)
                    dynamic_model_objects_results[model_type][model_name][i] = results_list

                if specific_model_list is not None:
                    if i in specific_model_list:

                        start_time = time.time()
                        results_list = list(Interpolator.Interpolator(step_size=step_size, epoch_in_MJD=epoch_in_MJD).get_propagation_results(dynamic_model,
                                                                                                                        solve_variational_equations=solve_variational_equations,
                                                                                                                        custom_initial_state=custom_initial_state,
                                                                                                                        custom_propagation_time=custom_propagation_time))
                        results_list.append(time.time()-start_time)
                        dynamic_model_objects_results[model_type][model_name][i] = results_list

                if return_dynamic_model_objects:

                    dynamic_model_objects_results[model_type][model_name][i].append(dynamic_model_objects[model_type][model_name][i])

    if entry_list is not None:
        for model_type, model_names in dynamic_model_objects_results.items():
            for model_name, model_results in model_names.items():
                for i, model_result in enumerate(model_results):
                    model_result_list = []
                    for entry in entry_list:
                        model_result_list.append(model_result[entry])
                    dynamic_model_objects_results[model_type][model_name][i] = model_result_list

    return dynamic_model_objects_results


def get_estimation_model_results(dynamic_model_objects,
                                 custom_estimation_model_objects=None,
                                 custom_truth_model=None,
                                 get_only_first=False,
                                 entry_list=None,
                                 apriori_covariance=None,
                                 initial_estimation_error=None,
                                 custom_range_noise=None,
                                 custom_observation_interval=None):


    if custom_estimation_model_objects is None:
        estimation_model_objects = get_estimation_model_objects(dynamic_model_objects,
                                                                custom_truth_model=custom_truth_model,
                                                                apriori_covariance=apriori_covariance,
                                                                initial_estimation_error=initial_estimation_error)

    else:
        estimation_model_objects = custom_estimation_model_objects

    # Create dictiontary to save results to of first estimation result per dynamic model type
    if get_only_first:
        result_dict = {}
        for key, inner_dict in estimation_model_objects.items():
            if inner_dict and isinstance(inner_dict, dict):
                updated_inner_dict = {}
                for subkey, sublist in inner_dict.items():
                    if sublist and isinstance(sublist, list):
                        updated_inner_dict[subkey] = [sublist[0]]
                    else:
                        updated_inner_dict[subkey] = sublist
                result_dict[key] = updated_inner_dict
        estimation_model_objects = result_dict

    # Assign estimation results to a nested dictionary
    estimation_model_objects_results = estimation_model_objects.copy()
    for model_type, model_names in estimation_model_objects.items():
        for model_name, estimation_models in model_names.items():
            for i, estimation_model in enumerate(estimation_models):

                # Adjust the attributes if wanted
                if custom_range_noise is not None:
                    estimation_model.noise = custom_range_noise
                if custom_observation_interval is not None:
                    estimation_model.observation_interval = custom_observation_interval

                # print("esimationmode", estimation_model)
                # Solve the results of the estimation arc and save to dictionary
                # start_time = time.time()
                # results_list = list(estimation_model.get_estimation_results())
                # results_list.append(time.time()-start_time)
                # estimation_model_objects_results[model_type][model_name][i] = results_list

                # print("random number utisls", np.random.randint(1,1000))
                results_list = estimation_model.get_estimation_results()
                estimation_model_objects_results[model_type][model_name][i] = results_list

    # Selectic only specific estimation model outputs to save to the dictionary
    if entry_list is not None:
        for model_type, model_names in estimation_model_objects_results.items():
            for model_name, model_results in model_names.items():
                for i, model_result in enumerate(model_results):
                    model_result_list = []
                    for entry in entry_list:
                        model_result_list.append(model_result[entry])
                    estimation_model_objects_results[model_type][model_name][i] = model_result_list

    return estimation_model_objects_results


def get_first_of_model_types(model_objects):

    result_dict = {}
    for key, inner_dict in model_objects.items():
        if inner_dict and isinstance(inner_dict, dict):
            first_elements = inner_dict.get(next(iter(inner_dict), None), [])[:1]
            result_dict[key] = {list(inner_dict.keys())[0]: first_elements}

    return result_dict


def convert_model_objects_to_list(model_objects, specific_model_type=None):

    model_type_list = []
    model_names_list = []
    model_objects_list = []
    for model_type, model_names in model_objects.items():
        if specific_model_type is not None:
            if model_type == specific_model_type:
                model_type_list.append(specific_model_type)
                model_names_list.append(model_names)
                for model_name, models in model_names.items():
                    for model in models:
                        model_objects_list.append(model)

        else:
            model_type_list.append(model_type)
            model_names_list.append(model_names)
            for model_name, models in model_names.items():
                for model in models:
                    model_objects_list.append(model)

    return model_objects_list


def convert_dictionary_to_array(dictionary):

    keys = np.stack(list(dictionary.keys()), axis=0)
    values = np.stack(list(dictionary.values()), axis=0)

    return keys, values


def convert_epochs_to_MJD(epochs, full_array=True):

    if full_array:
        return np.array([time_conversion.julian_day_to_modified_julian_day(\
                    time_conversion.seconds_since_epoch_to_julian_day(epoch)) for epoch in epochs])
    else:
        return time_conversion.julian_day_to_modified_julian_day(\
                    time_conversion.seconds_since_epoch_to_julian_day(epochs))


def convert_MJD_to_epoch(epochs, full_array=True):

    if full_array:
        return np.array([time_conversion.julian_day_to_seconds_since_epoch(\
                    time_conversion.modified_julian_day_to_julian_day(epoch)) for epoch in epochs])
    else:
        return time_conversion.julian_day_to_seconds_since_epoch(\
                    time_conversion.modified_julian_day_to_julian_day(epochs))


def get_max_depth(dictionary):
    if isinstance(dictionary, dict):
        return 1 + max((get_max_depth(value) for value in dictionary.values()), default=0)
    else:
        return 0


def save_dict_to_folder(dicts=[], labels=[], custom_sub_folder_name=None, folder_name='dicts'):

    # Get the frame of the caller
    caller_frame = inspect.stack()[1]
    file_path = caller_frame.filename
    file_path = os.path.dirname(file_path)

    dict_folder = os.path.join(file_path, folder_name)
    if not os.path.exists(dict_folder):
        os.makedirs(dict_folder, exist_ok=True)

    if custom_sub_folder_name is None:
        sub_folder_name = inspect.currentframe().f_back.f_code.co_name
    else:
        sub_folder_name = custom_sub_folder_name

    sub_folder = os.path.join(dict_folder, sub_folder_name)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder, exist_ok=True)

    for i, _dict in enumerate(dicts):
        if len(dicts) != len(labels):
            file_name = f"dict_{i}.json"
        else:
            file_name = f"{labels[i]}.json"
        path = os.path.join(sub_folder, file_name)

        # Convert keys to int
        def convert_keys_to_int(d):
            if isinstance(d, dict):
                return {int(k) if isinstance(k, (np.int32, np.int64)) else k: convert_keys_to_int(v) for k, v in d.items()}
            return d

        _dict = convert_keys_to_int(_dict)

        with open(path, 'w') as json_file:
            json.dump(_dict, json_file, indent=get_max_depth(_dict))



def save_figure_to_folder(figs=[], labels=[], custom_sub_folder_name=None, folder_name='figures'):

    # Get the frame of the caller
    caller_frame = inspect.stack()[1]
    file_path = caller_frame.filename
    file_path = os.path.dirname(file_path)

    dict_folder = os.path.join(file_path, folder_name)
    if not os.path.exists(dict_folder):
        os.makedirs(dict_folder, exist_ok=True)

    if custom_sub_folder_name is None:
        sub_folder_name = inspect.currentframe().f_back.f_code.co_name

    else:
        sub_folder_name = custom_sub_folder_name

    sub_folder = os.path.join(dict_folder, sub_folder_name)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder, exist_ok=True)

    for i, fig in enumerate(figs):
        if len(figs) != len(labels):
            file_name = f"fig_{i}.png"
        else:
            file_name = f"{labels[i]}.png"
        figure_path = os.path.join(sub_folder, file_name)
        fig.savefig(figure_path)



def save_table_to_folder(tables=[], labels=[], custom_sub_folder_name=None, folder_name='tables'):

    # Get the frame of the caller
    caller_frame = inspect.stack()[1]
    file_path = caller_frame.filename
    file_path = os.path.dirname(file_path)

    dict_folder = os.path.join(file_path, folder_name)
    if not os.path.exists(dict_folder):
        os.makedirs(dict_folder, exist_ok=True)

    if custom_sub_folder_name is None:
        sub_folder_name = inspect.currentframe().f_back.f_code.co_name
    else:
        sub_folder_name = custom_sub_folder_name

    sub_folder = os.path.join(dict_folder, sub_folder_name)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder, exist_ok=True)

    for i, table in enumerate(tables):
        if len(tables) != len(labels):
            file_name = f"table_{i}.tex"
        else:
            file_name = f"{labels[i]}.tex"
        file_path = os.path.join(sub_folder, file_name)
        with open(file_path, 'w') as file:
                file.write(table)




def get_monte_carlo_stats_dict(data_dict):

    # Initialize dictionary to store mean and standard deviation for each value
    stats = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            stats[key] = get_monte_carlo_stats_dict(value)
        elif isinstance(value, list):
            mean_value = np.mean(value, axis=0)
            std_dev_value = np.std(value, axis=0)
            if isinstance(mean_value, np.ndarray):
                mean_value = list(mean_value)
            if isinstance(std_dev_value, np.ndarray):
                std_dev_value = list(std_dev_value)
            stats[key] = {'mean': mean_value, 'std_dev': std_dev_value}
        else:
            stats[key] = value

    return stats