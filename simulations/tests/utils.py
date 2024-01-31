# Standard
import os
import sys
import pytest_html
import numpy as np
import time
import inspect

# tudatpy
from tudatpy.kernel.astro import time_conversion

# Own
from src.dynamic_models import Interpolator
from src.dynamic_models.full_fidelity import *

parent_dir = os.path.dirname(os.path.dirname(__file__))


def get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time, package_dict=None, get_only_first=False, custom_initial_state=None):

    if package_dict is None:
        package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]}
    else:
        package_dict = package_dict

    dynamic_model_objects = {}
    for package_type, package_name_list in package_dict.items():
        sub_dict = {package_name_list[i]: [] for i in range(len(package_name_list))}
        packages_dir = os.path.join(parent_dir, 'src', 'dynamic_models', package_type)

        package_name_counter = 0
        for package_name in package_dict[package_type]:
            package_module_path = f'dynamic_models.{package_type}.{package_name}'
            package_module = __import__(package_module_path, fromlist=[package_name])
            package_files = os.listdir(os.path.join(packages_dir, package_name))

            for file_name in package_files:
                if file_name.endswith('.py') and not file_name.startswith('__init__'):
                    module_path = f'{package_module_path}.{os.path.splitext(file_name)[0]}'
                    module = __import__(module_path, fromlist=[file_name])

                    if package_type == "low_fidelity":
                        DynamicModel = module.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state)
                    else:
                        DynamicModel = module.HighFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state)

                    sub_dict[package_name_list[package_name_counter]].extend([DynamicModel])
                    dynamic_model_objects[package_type] = sub_dict

            package_name_counter += 1

    if get_only_first:
        result_dict = {}
        for key, inner_dict in dynamic_model_objects.items():
            if inner_dict and isinstance(inner_dict, dict):
                updated_inner_dict = {}
                for subkey, sublist in inner_dict.items():
                    if sublist and isinstance(sublist, list):
                        updated_inner_dict[subkey] = [sublist[0]]
                    else:
                        updated_inner_dict[subkey] = sublist

                result_dict[key] = updated_inner_dict

        return result_dict

    else:
        return dynamic_model_objects


def get_estimation_model_objects(estimation_model, dynamic_model_objects, custom_truth_model=None, apriori_covariance=None):

    estimation_model_objects = {}
    for package_type, package_names in dynamic_model_objects.items():
        submodels_dict = {}
        for package_name, dynamic_models in package_names.items():

            # Adjust such that full-fidelity model with the correct initial state is used
            if custom_truth_model is None:
                truth_model = full_fidelity.HighFidelityDynamicModel(dynamic_models[0].simulation_start_epoch, dynamic_models[0].propagation_time)
            else:
                truth_model = custom_truth_model
            # print(package_type, package_name)
            submodels = [estimation_model.EstimationModel(dynamic_model, truth_model, apriori_covariance=apriori_covariance) for dynamic_model in dynamic_models]

            submodels_dict[package_name] = submodels
        estimation_model_objects[package_type] = submodels_dict

    return estimation_model_objects


def save_figures_to_folder(figs=[], labels=[], save_to_report=True):

    extras = []
    folder_name = inspect.currentframe().f_back.f_code.co_name

    base_folder = "figures"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)

    figure_folder = os.path.join(base_folder, folder_name)
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder, exist_ok=True)

    for i, fig in enumerate(figs):
        base_string = "_".join([str(label) for label in labels])
        if save_to_report:
            file_name = f"fig{i+1}_{base_string}.png"
        else:
            file_name = f"fig_3d{i+1}_{base_string}.png"
        figure_path = os.path.join(figure_folder, file_name)
        fig.savefig(figure_path)
        if save_to_report:
            extras.append(pytest_html.extras.png(figure_path))


def get_dynamic_model_results(simulation_start_epoch_MJD, propagation_time, package_dict=None, get_only_first=False, custom_initial_state=None, step_size=0.01, epoch_in_MJD=True, entry_list=None):

    dynamic_model_objects = get_dynamic_model_objects(simulation_start_epoch_MJD,
                                                      propagation_time,
                                                      package_dict=package_dict,
                                                      get_only_first=get_only_first,
                                                      custom_initial_state=custom_initial_state)

    dynamic_model_objects_results = dynamic_model_objects.copy()
    for model_type, model_names in dynamic_model_objects.items():
        for model_name, dynamic_models in model_names.items():
            for i, dynamic_model in enumerate(dynamic_models):

                start_time = time.time()
                results_list = list(Interpolator.Interpolator(step_size=step_size, epoch_in_MJD=epoch_in_MJD).get_propagation_results(dynamic_model))
                results_list.append(time.time()-start_time)
                dynamic_model_objects_results[model_type][model_name][i] = results_list

    if entry_list is not None:
        for model_type, model_names in dynamic_model_objects_results.items():
            for model_name, model_results in model_names.items():
                for i, model_result in enumerate(model_results):
                    model_result_list = []
                    for entry in entry_list:
                        model_result_list.append(model_result[entry])
                    dynamic_model_objects_results[model_type][model_name][i] = model_result_list

    return dynamic_model_objects_results


def get_estimation_model_results(dynamic_model_objects, estimation_model, custom_truth_model=None, get_only_first=False, entry_list=None, apriori_covariance=None):

    estimation_model_objects = get_estimation_model_objects(estimation_model, dynamic_model_objects, custom_truth_model=custom_truth_model, apriori_covariance=apriori_covariance)

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

    estimation_model_objects_results = estimation_model_objects.copy()
    for model_type, model_names in estimation_model_objects.items():
        for model_name, estimation_models in model_names.items():
            for i, estimation_model in enumerate(estimation_models):

                start_time = time.time()
                results_list = list(estimation_model.get_estimation_results())
                results_list.append(time.time()-start_time)
                estimation_model_objects_results[model_type][model_name][i] = results_list

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


# def get_model_result_for_given_entry(dynamic_model_objects_results, entry_list):

#     for model_type, model_names in dynamic_model_objects_results.items():
#         for model_name, model_results in model_names.items():
#             result_list_model_name = []
#             for model_result in model_results:
#                 for entry in entry_list:
#                     result_list_model_name.append(model_result[entry])
#             dynamic_model_objects_results[model_type][model_name] = result_list_model_name

#     return dynamic_model_objects_results


def convert_dictionary_to_array(dictionary):

    keys = np.stack(list(dictionary.keys()), axis=0)
    values = np.stack(list(dictionary.values()), axis=0)

    return keys, values


def convert_epochs_to_MJD(epochs):

    epochs_MJD = np.array([time_conversion.julian_day_to_modified_julian_day(\
        time_conversion.seconds_since_epoch_to_julian_day(epoch)) for epoch in epochs])

    return epochs_MJD


# def get_first_of_model_types(dynamic_model_objects):

#     dynamic_models = []
#     for model_type, model_names in dynamic_model_objects.items():
#         for model_name, models in model_names.items():
#             dynamic_models.append(models[0])

#     return dynamic_models


# Commonly used parameters
synodic_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
                                  1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])