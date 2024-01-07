# Standard
import os
import sys
import pytest_html
import numpy as np

# tudatpy
from tudatpy.kernel.astro import time_conversion

parent_dir = os.path.dirname(os.path.dirname(__file__))

package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]}
def get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time, package_dict=package_dict):

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
                        DynamicModel = module.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time)
                    else:
                        DynamicModel = module.HighFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time)

                    sub_dict[package_name_list[package_name_counter]].extend([DynamicModel])
                    dynamic_model_objects[package_type] = sub_dict

            package_name_counter += 1

    return dynamic_model_objects


def get_estimation_model_objects(estimation_model, dynamic_model_objects):

    estimation_model_objects = {}
    for package_type, package_names in dynamic_model_objects.items():
        submodels_dict = {}
        for package_name, dynamic_models in package_names.items():
            submodels = [estimation_model.EstimationModel(dynamic_model) for dynamic_model in dynamic_models]
            submodels_dict[package_name] = submodels
        estimation_model_objects[package_type] = submodels_dict

    return estimation_model_objects


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


# def convert_model_objects_to_list_1(model_objects, specific_model_type=None):

#     models_nested = []
#     for value in model_objects.values():
#         if isinstance(value, list):
#             models_nested.append(value)
#         elif isinstance(value, dict):
#             models_nested.append(convert_model_objects_to_list_1(value))

#     return models_nested


def convert_dictionary_to_array(dictionary):

    keys = np.stack(list(dictionary.keys()), axis=0)
    values = np.stack(list(dictionary.values()), axis=0)

    return keys, values


def save_figures_to_folder(folder_name, extras, figs=[], labels=[], save_to_report=True):

    # Save the figure to designated folder belong to the respective test method
    os.makedirs(folder_name, exist_ok=True)
    for i, fig in enumerate(figs):
        base_string = "_".join([str(label) for label in labels])
        if save_to_report:
            file_name = f"fig{i+1}_{base_string}.png"
        else:
            file_name = f"fig_3d{i+1}_{base_string}.png"
        figure_path = os.path.join(folder_name, file_name)
        fig.savefig(figure_path)
        if save_to_report:
            extras.append(pytest_html.extras.png(figure_path))



def convert_epochs_to_MJD(epochs):

    epochs_MJD = np.array([time_conversion.julian_day_to_modified_julian_day(\
        time_conversion.seconds_since_epoch_to_julian_day(epoch)) for epoch in epochs])

    return epochs_MJD


def get_first_of_model_types(dynamic_model_objects):

    dynamic_models = []
    for model_type, model_names in dynamic_model_objects.items():
        for model_name, models in model_names.items():
            dynamic_models.append(models[0])

    return dynamic_models



# def iterate_model_objects(model_objects):

#     for i, (model_type, model_names) in enumerate(model_objects.items()):

#         model_names = model_names.keys()
#         for k, model_name in enumerate(model_names):
#             for dynamic_model in dynamic_model_objects[model_type][model_name]:







synodic_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
                                  1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])





# Define a dictionary with configurations for LPF and LUMIO
plot_config_LPF = {
    'label': 'Example Label',
    'color': 'blue',
    'linestyle': '--',
    'linewidth': 2,
    'marker': 'o'
}

plot_config_LUMIO = {
    'label': 'Example Label',
    'color': 'blue',
    'linestyle': '--',
    'linewidth': 2,
    'marker': 'o'
}

plot_config_primary = {
    'label': 'Example Label',
    'color': 'blue',
    'linestyle': '--',
    'linewidth': 2,
    'marker': 'o'
}

plot_config_secondary = {
    'label': 'Example Label',
    'color': 'blue',
    'linestyle': '--',
    'linewidth': 2,
    'marker': 'o'
}