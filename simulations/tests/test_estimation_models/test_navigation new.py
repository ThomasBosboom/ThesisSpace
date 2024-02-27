# Standard
import os
import sys
import copy
import numpy as np
import time
import pytest
import pytest_html
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from tests import utils
from src.dynamic_models import Interpolator, NavigationSimulator, PlotNavigationResults

class TestNavigation():

    # batch_start_times = np.array([60390, 60394.7, 60401.5, 60406.5])
    # batch_end_times = np.array([60392.5, 60397.2, 60404, 60409])

    # observation_windows = list(zip(batch_start_times, batch_end_times))

    # @pytest.mark.parametrize(
    # "package_dict, truth_model_list, get_only_first, include_station_keeping, observation_windows",
    # [
    #     ({"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass"]},
    #     ["low_fidelity", "three_body_problem", 0],
    #     True,
    #     True,
    #     observation_windows)
    # ])


    def get_navigation_results(self, package_dict, truth_model_list, get_only_first, include_station_keeping, observation_windows):

        # Start the simulation
        dynamic_model_objects = utils.get_dynamic_model_objects(60390, 1, get_only_first=True, package_dict=package_dict)

        # Save histories of the navigation simulations
        results_dict = copy.deepcopy(dynamic_model_objects)
        for i, (model_type, model_names) in enumerate(results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, model in enumerate(models):

                    navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows, [model_type, model_name, k], truth_model_list)

                    results_dict[model_type][model_name][k] = []
                    for result_dict in navigation_simulator.perform_navigation(include_station_keeping=include_station_keeping):
                        results_dict[model_type][model_name][k].append(utils.convert_dictionary_to_array(result_dict))

        return results_dict



batch_start_times = np.array([60392, 60394, 60408, 60410])
batch_end_times = np.array([60393, 60397, 60409, 60413])

observation_windows = list(zip(batch_start_times, batch_end_times))

# mission_time = 9
# mission_start_epoch = 60390
# mission_end_epoch = mission_start_epoch + mission_time
# mission_epoch = mission_start_epoch

# # Initial batch timing settings
# propagation_time = 1
# batch_start_times = np.arange(mission_start_epoch, mission_end_epoch, propagation_time)
# batch_end_times = np.arange(propagation_time+mission_start_epoch, propagation_time+mission_end_epoch, propagation_time)
# observation_windows = list(zip(batch_start_times, batch_end_times))

{"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}
params = ({"high_fidelity": ["point_mass"]},
            ["high_fidelity", "point_mass", 0],
            True,
            True,
            observation_windows)

params = ({"low_fidelity": ["three_body_problem"]},
["low_fidelity", "three_body_problem", 0],
True,
True,
observation_windows)

results_dict = TestNavigation().get_navigation_results(*params)

shape = (len(results_dict), len(next(iter(results_dict.values()))))
# print(results_dict)
print(shape)

PlotNavigationResults.PlotNavigationResults(results_dict, observation_windows).plot_formal_error_history()
PlotNavigationResults.PlotNavigationResults(results_dict, observation_windows).plot_uncertainty_history()
PlotNavigationResults.PlotNavigationResults(results_dict, observation_windows).plot_reference_deviation_history()
PlotNavigationResults.PlotNavigationResults(results_dict, observation_windows).plot_estimation_error_history()
PlotNavigationResults.PlotNavigationResults(results_dict, observation_windows).plot_full_state_history()

plt.show()