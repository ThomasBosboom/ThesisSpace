# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import statistics
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
from src.dynamic_models import validation
from src.dynamic_models import Interpolator, FrameConverter, TraditionalLowFidelity
from src.dynamic_models.full_fidelity.full_fidelity import *
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model


### Defining global fixtures of 1 day estimation models
@pytest.fixture(scope="module")
def estimation_model_objects_results():

    # Argument settings for dynamic models to be used in estimation
    simulation_start_epoch = 60390
    propagation_time = 1
    # package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}
    package_dict = None
    get_only_first = True
    custom_initial_state = None
    params = (simulation_start_epoch, propagation_time, package_dict, get_only_first, custom_initial_state)

    # Argument settings for estimation outputs
    get_only_first = False
    entry_list = None

    dynamic_model_objects = utils.get_dynamic_model_objects(*params)
    truth_model = full_fidelity.HighFidelityDynamicModel(*params[:2])

    return utils.get_estimation_model_results(dynamic_model_objects, custom_truth_model=truth_model, get_only_first=get_only_first, entry_list=entry_list)


### Define adjustable fixture for custom generations
@pytest.fixture(params=[("simulation_start_epoch_MJD", "propagation_time", "package_dict", "get_only_fist", "custom_initial_state")],
                scope="module")
def custom_estimation_model_objects_results(request):
    dynamic_model_objects = utils.get_dynamic_model_objects(*request.param)
    truth_model = low_fidelity.LowFidelityDynamicModel(*request.param[:2], custom_initial_state=request.param[-1])
    return utils.get_estimation_model_results(dynamic_model_objects, custom_truth_model=truth_model, get_only_first=False, entry_list=None)


class TestObservability:

    package_dict = {"low_fidelity": ["three_body_problem"]}
    @pytest.mark.parametrize("custom_estimation_model_objects_results", [(60400, 14, package_dict, True, None)], indirect=True)
    def test_observability_history(self, custom_estimation_model_objects_results):

        model_type = "low_fidelity"
        model_name = "three_body_problem"
        model_entry = 0
        single_information_dict = custom_estimation_model_objects_results[model_type][model_name][model_entry][1]

        fig, axs = plt.subplots(2, 1, figsize=(8.3, 5.7), sharex=True)
        for i, (observable_type, information_sets) in enumerate(single_information_dict.items()):

            for j, information_set in enumerate(information_sets.values()):
                for k, single_information_set in enumerate(information_set):
                    information_dict = single_information_dict[observable_type][j][k]
                    epochs = utils.convert_epochs_to_MJD(np.array(list(information_dict.keys())))
                    information_matrix_history = np.array(list(information_dict.values()))

                    for m in range(2):
                        observability_lpf = np.sqrt(np.stack([np.diagonal(matrix) for matrix in information_matrix_history[:,0+3*m:3+3*m,0+3*m:3+3*m]]))
                        observability_lumio = np.sqrt(np.stack([np.diagonal(matrix) for matrix in information_matrix_history[:,6+3*m:9+3*m,6+3*m:9+3*m]]))
                        observability_lpf_total = np.sqrt(np.max(np.linalg.eigvals(information_matrix_history[:,0+3*m:3+3*m,0+3*m:3+3*m]), axis=1, keepdims=True))
                        observability_lumio_total = np.sqrt(np.max(np.linalg.eigvals(information_matrix_history[:,6+3*m:9+3*m,6+3*m:9+3*m]), axis=1, keepdims=True))

                        axs[2*i+m].plot(epochs, observability_lpf_total, label="total", color="darkred")
                        axs[2*i+m].plot(epochs, observability_lumio_total, label="total", color="darkblue")

                        ls = ["dashdot", "dashed", "dotted"]
                        label = [[r"$\mathbf{x}$", r"$\mathbf{y}$", r"$\mathbf{z}$"],[r"$\mathbf{\dot{x}}$", r"$\mathbf{\dot{y}}$", r"$\mathbf{\dot{z}}$"]]
                        ylabels = [r"$\sqrt{\mathbf{\Lambda_{r}}}$ [-]", r"$\sqrt{\mathbf{\Lambda_{v}}}$ [-]"]
                        observable_types = ["two-way range", "two-way doppler"]
                        for l in range(3):
                            alpha=0.3
                            axs[2*i+m].plot(epochs, observability_lpf[:,l], label=label[m][l], color="red", ls=ls[l], alpha=alpha)
                            axs[2*i+m].plot(epochs, observability_lumio[:,l], label=label[m][l], color="blue", ls=ls[l], alpha=alpha)

                        axs[2*i+m].set_ylabel(ylabels[m])
                        axs[2*i+m].set_yscale("log")
                        axs[2*i+m].grid(alpha=0.5, linestyle='--')

                        if i == 0:
                            axs[2*i+m].legend()

                    axs[2*i].set_title(observable_types[i])
                    axs[-1].set_xlabel(r"Time since start propagation")

            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.suptitle("Observability effectiveness ")
        # plt.show()

        utils.save_figures_to_folder([fig], [])


class TestEstimationOutput:

    def test_correlations(self, estimation_model_objects_results):

        fig, axs = plt.subplots(1, 5, figsize=(10, 4), sharey=True, sharex=True)
        keys_list = [["CRTBP"], ["PM", "PM SRP", "SH", "SH SRP"]]
        images = []

        for i, (model_type, model_names_dict) in enumerate(estimation_model_objects_results.items()):
            for j, (model_name, models) in enumerate(model_names_dict.items()):
                for k, (model) in enumerate(models):
                    for l, (value) in enumerate([0]):

                        images.append(axs[i+j].imshow(np.abs(estimation_model_objects_results[model_type][model_name][k][l].correlations), aspect='auto', interpolation='none'))
                        axs[i+j].set_xlabel(keys_list[i][j])

        fig.colorbar(images[-1], ax=axs[-1], orientation='vertical')
        fig.suptitle("Correlation Matrix")
        axs[0].set_ylabel("Index - Estimated Parameter")
        plt.tight_layout()
        # plt.show()

        utils.save_figures_to_folder([fig], [])


    # def test_formal_errors(self, estimation_model_objects_results):




class TestMonteCarlo:

    def test_estimation_model_run_times(self):

        simulation_start_epoch = 60390
        propagation_time = 1
        package_dict = {"low_fidelity": ["three_body_problem"]}
        get_only_first = True
        custom_initial_state = None

        # Initialize dictionaries to store accumulated values
        accumulator_dict = {
            fidelity_key: {subkey: [[] for value in values]
                for subkey, values in sub_dict.items()
            }
            for fidelity_key, sub_dict in utils.get_dynamic_model_objects(60390, 1, package_dict=package_dict, get_only_first=get_only_first).items()
        }

        # Start the monte carlo simulation with 14 1-day estimations with different starting epochs
        for start_epoch in range(60390, 60404, 1):

            params = (start_epoch, propagation_time, package_dict, get_only_first, custom_initial_state)
            print(params)
            dynamic_model_objects = utils.get_dynamic_model_objects(*params)
            truth_model = full_fidelity.HighFidelityDynamicModel(*params[:2], custom_initial_state=params[-1])
            run_times_dict = utils.get_estimation_model_results(dynamic_model_objects, custom_truth_model=truth_model, get_only_first=False, entry_list=[-1])

            # Accumulate values during the loop
            for fidelity_key, sub_dict in run_times_dict.items():
                for i, (subkey, subvalue_list) in enumerate(sub_dict.items()):
                    for j, subvalue in enumerate(subvalue_list):
                        for k, entry in enumerate(subvalue):
                            accumulator_dict[fidelity_key][subkey][j].append(entry)

        # Calculate averages and standard deviations
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
        fig.suptitle(f"Mean run time estimation models, varying start epoch, 1 day")

        utils.save_figures_to_folder([fig], [60390, 1])







# class Test:


#     def test_observation_residuals(self, estimation_model_objects_results):


#         print("estimation_model_objects_results", estimation_model_objects_results)
#         print("test", estimation_model_objects_results["high_fidelity"]["point_mass"][0])

#         fig, axs = plt.subplots(3, 1, figsize=(10, 5))
#         model_count = 0
#         for model_type, model_names in estimation_model_objects_results.items():
#             for model_name in model_names:
#                 print(estimation_model_objects_results[model_type][model_name][0])
#                 residual_history = estimation_model_objects_results[model_type][model_name][0][0].residual_history[:,:]
#                 # concatenated_times = estimation_model_objects_results[model_type][model_name][0][-2].concatenated_times
#                 # print(concatenated_times)
#                 axs[model_count].plot(residual_history, marker="o")
#                 model_count += 1
#         plt.show()

#         fig, axs = plt.subplots(3, 3, figsize=(10, 5))
#         model_count = 0
#         for model_type, model_names in estimation_model_objects_results.items():
#             plane_count = 0
#             for model_name in model_names:
#                 state_history = estimation_model_objects_results[model_type][model_name][0][0].residual_history[:,:]
#                 axs[model_count][model_count].plot(residual_history, marker="o")
#                 plane_count += 1
#                 model_count += 1

#         plt.show()


#         utils.save_figures_to_folder([fig], [])








# Test().test_observation_residuals()




    # def test_estimation_errors(self, params):

    #     dynamic_model_objects = utils.get_dynamic_model_objects(*params)

    #     truth_model = full_fidelity.HighFidelityDynamicModel(*params[:2])
    #     estimation_model_objects_results = utils.get_estimation_model_results(dynamic_model_objects, truth_model, get_only_first=False)

    #     sorted_observation_sets = estimation_model_objects_results["high_fidelity"]["point_mass"][0][-2]

    #     print("sorted_observation_sets: ", sorted_observation_sets)

    #     fig = plt.figure()
    #     for observable_type, observation_sets in sorted_observation_sets.items():
    #         for observation_set in observation_sets.values():
    #             for single_observation_set in observation_set:
    #                 plt.plot(single_observation_set.observation_times, single_observation_set.concatenated_observations)
    #     # plt.show()

    #     utils.save_figures_to_folder([fig],[])




