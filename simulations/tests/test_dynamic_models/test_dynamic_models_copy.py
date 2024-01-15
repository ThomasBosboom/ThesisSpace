# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Third party
import pytest
import pytest_html
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Own
from tests import utils
from src.dynamic_models import validation_LUMIO
from src.dynamic_models import Interpolator, FrameConverter, TraditionalLowFidelity
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model


# Fixture function to set up multiple sets of sample data
@pytest.fixture(params=[("simulation_start_epoch_MJD", "propagation_time", "package_dict", "get_first_only", "custom_initial_state")],
                scope="module")
def dynamic_model_objects_results(request):
    return utils.get_dynamic_model_objects_results(*request.param, step_size=0.1)

@pytest.fixture(params=[("simulation_start_epoch_MJD", "propagation_time", "package_dict", "get_first_only", "custom_initial_state")],
                scope="module")
def estimation_model_objects_results(request):
    dynamic_model_objects = utils.get_dynamic_model_objects(*request.param)
    return utils.get_estimation_model_objects_results(dynamic_model_objects, estimation_model, request.param[-1])

# class Test:

#     package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}
#     @pytest.mark.parametrize("dynamic_model_objects_results", [(60390,1, package_dict, False, None)], indirect=True)
#     def test_dynamic_model_objects_results(self, dynamic_model_objects_results):

#         # print(dynamic_model_objects_results["high_fidelity"]["point_mass"][0][-1])

#         # Iterate through the top-level keys
#         key_count = 0
#         for top_level_key in dynamic_model_objects_results:
#             if isinstance(dynamic_model_objects_results[top_level_key], dict):
#                 key_count += len(dynamic_model_objects_results[top_level_key])

#         run_times_dict = utils.get_model_result_for_given_entry(dynamic_model_objects_results, -1)


#         ### Plot run times for each model
#         fig, axs = plt.subplots(1, key_count, figsize=(10, 3*key_count), sharey=True)
#         for j, (model_types, model_names) in enumerate(run_times_dict.items()):
#             for i, (key, values) in enumerate(model_names.items()):
#                 axs[i+j].bar(range(1, len(values)+1), values, label=key)
#                 axs[i+j].set_xlabel(key)
#                 axs[i+j].set_xticks(range(1, 1+max([len(value) for value in  values])))
#                 axs[i+j].set_yscale("log")

#         axs[0].set_ylabel('Run time [s]')
#         # fig.suptitle(f"Run time dynamic models, {simulation_start_epoch_MJD} MJD, {propagation_time} days")
#         utils.save_figures_to_folder([fig], [])

#     @pytest.mark.parametrize("estimation_model_objects_results", [(60390,1, package_dict, True, None)], indirect=True)
#     def test_estimation_model_objects_results(self, estimation_model_objects_results):

#         fig1, axs1 = plt.subplots(1, 3, figsize=(14, 3*1), layout="constrained")
#         utils.save_figures_to_folder([fig1], [])



class Test2:


    def test_monte_carlo(self):

        package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]}
        get_only_first=False

        # Initialize dictionaries to store accumulated values
        accumulator_dict = {
            fidelity_key: {subkey: [[] for value in values]
                for subkey, values in sub_dict.items()
            }
            for fidelity_key, sub_dict in utils.get_dynamic_model_objects(60390, 1, package_dict=package_dict, get_only_first=get_only_first).items()
        }

        for start_epoch in range(60390, 60414, 1):
            params = (start_epoch, 1, package_dict, get_only_first, None)
            dynamic_model_objects_results = utils.get_dynamic_model_objects_results(*params, step_size=0.1)
            run_times_dict = utils.get_model_result_for_given_entry(dynamic_model_objects_results, -1)

            # Accumulate values during the loop
            for fidelity_key, sub_dict in run_times_dict.items():
                for i, (subkey, subvalue_list) in enumerate(sub_dict.items()):
                    for j, subvalue in enumerate(subvalue_list):
                        accumulator_dict[fidelity_key][subkey][j].append(subvalue)

        # Calculate averages and standard deviations
        import statistics
        result_dict = {
            fidelity_key: {
                subkey: [
                    {
                        "average": statistics.mean(sublist),
                        "std_dev": statistics.stdev(sublist)
                    }
                    for sublist in sublists
                ]
                for subkey, sublists in sub_dict.items()
            }
            for fidelity_key, sub_dict in accumulator_dict.items()
        }


        ### Plot run times for each model
        keys_list = [["CRTBP"], ["PM", "PM SRP", "SH", "SH SRP"]]
        key_count = sum(len(sublist) for sublist in package_dict.values())
        fig, axs = plt.subplots(1, key_count, figsize=(8, 0.75*key_count), sharey=True)
        for i, (model_types, model_names) in enumerate(result_dict.items()):
            for j, (key, values) in enumerate(model_names.items()):
                print(values)
                averages = []
                std_devs = []
                for subvalue in values:
                    averages.append(subvalue["average"])
                    std_devs.append(subvalue["std_dev"])
                axs[i+j].grid(alpha=0.5, linestyle='--')
                axs[i+j].bar(range(1, len(values)+1), averages, yerr=std_devs, ecolor="black", capsize=4, label=key)
                axs[i+j].set_xlabel(keys_list[i][j])
                axs[i+j].set_xticks(range(1, 1+max([len(value) for value in model_names.values()])))
                axs[i+j].set_yscale("log")

        axs[0].set_ylabel('Run time [s]')
        legend_handles = [plt.Line2D([0], [0], color='black', markersize=5, label=r'1$\sigma$ Std Dev')]
        fig.legend(handles=[legend_handles[0]], loc='upper right')
        fig.suptitle(f"Mean run time dynamic models, varying start epoch, 1 day")
        utils.save_figures_to_folder([fig], [])




    # package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}
    # params = "simulation_start_epoch_MJD, propagation_time, package_dict, get_first_only, custom_initial_state"
    # @pytest.mark.parametrize(params, [(60390,1, package_dict, False, None)])
    # def test_dynamic_model_objects_results(self, dynamic_model_objects_results):

    #     dynamic_model_objects_results = utils.get_dynamic_model_objects_results(*request.param, step_size=0.1)

    #     # Iterate through the top-level keys
    #     key_count = 0
    #     for top_level_key in dynamic_model_objects_results:
    #         if isinstance(dynamic_model_objects_results[top_level_key], dict):
    #             key_count += len(dynamic_model_objects_results[top_level_key])

    #     print(key_count)

    #     run_times_dict = utils.get_model_result_for_given_entry(dynamic_model_objects_results, -1)
    #     # print(run_times)

    #     ### Plot run times for each model
    #     fig, axs = plt.subplots(1, key_count, figsize=(10, 3*key_count), sharey=True)
    #     for j, (model_types, model_names) in enumerate(run_times_dict.items()):
    #         for i, (key, values) in enumerate(model_names.items()):
    #             axs[i+j].bar(range(1, len(values)+1), values, label=key)
    #             axs[i+j].set_xlabel(key)
    #             axs[i+j].set_xticks(range(1, 1+max([len(value) for value in run_times_dict.values()])))
    #             axs[i+j].set_yscale("log")

    #     axs[0].set_ylabel('Run time [s]')
    #     # fig.suptitle(f"Run time dynamic models, {simulation_start_epoch_MJD} MJD, {propagation_time} days")
    #     utils.save_figures_to_folder([fig], [])

    # @pytest.mark.parametrize(params, [(60390,1, package_dict, True, None)])
    # def test_estimation_model_objects_results(self, *params):

    #     print(estimation_model_objects_results)
    #     print(estimation_model_objects_results["high_fidelity"]["point_mass"][0][-1])

    #     fig1, axs1 = plt.subplots(1, 3, figsize=(14, 3*1), layout="constrained")
    #     utils.save_figures_to_folder([fig1], [])








# package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}

# @pytest.fixture(scope="module")
# def generated_test_data():
#     return utils.get_interpolated_dynamic_model_objects_results(60390,50, package_dict=package_dict, step_size=0.1)

# @pytest.fixture(scope="module")
# def generated_test_data_estimation():
#     return utils.get_interpolated_estimation_model_objects_results(60390,50, estimation_model, package_dict=package_dict)["high_fidelity"]["point_mass"][-1]

# def test_sample_data_with_indirect(generated_test_data):
#     print(generated_test_data)
#     assert isinstance(generated_test_data, dict)

# def test_sample_data_with_indirect_1(generated_test_data):
#     print(generated_test_data)
#     assert isinstance(generated_test_data, dict)

# def test_sample_data_with_indirect_2(generated_test_data):
#     print(generated_test_data)
#     assert isinstance(generated_test_data, dict)

# def test_sample_data_with_indirect_3(generated_test_data_estimation):
#     print(generated_test_data_estimation)
#     assert isinstance(generated_test_data_estimation, dict)
