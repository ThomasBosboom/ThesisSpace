# Standard
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(__file__))


def get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time, package_dict={"low_fidelity": ["integration_settings"], "high_fidelity": ["point_mass"]}):

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


def loop_through_model_objects(model_objects):

    model_objects_list = []
    for package_type, package_names in model_objects.items():
        for package_name, models in package_names.items():
            for model in models:
                model_objects_list.append(model)

    return model_objects_list