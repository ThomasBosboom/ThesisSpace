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
    # "custom_model_dict, truth_model_list, get_only_first, include_station_keeping, observation_windows",
    # [
    #     ({"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass"]},
    #     ["low_fidelity", "three_body_problem", 0],
    #     True,
    #     True,
    #     observation_windows)
    # ])


    def get_navigation_results(self, custom_model_dict, truth_model_list, get_only_first, observation_windows, include_station_keeping, exclude_first_manouvre, custom_station_keeping_epochs):

        # Start the simulation
        dynamic_model_objects = utils.get_dynamic_model_objects(60390, 14, get_only_first=True, custom_model_dict=custom_model_dict)

        # Save histories of the navigation simulations
        results_dict = copy.deepcopy(dynamic_model_objects)
        for i, (model_type, model_names) in enumerate(results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, model in enumerate(models):

                    navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows, [model_type, model_name, k], truth_model_list,
                                                                                    include_station_keeping=include_station_keeping, exclude_first_manouvre=exclude_first_manouvre,
                                                                                    custom_station_keeping_epochs=custom_station_keeping_epochs
                                                                                    )

                    results_dict[model_type][model_name][k] = []
                    for result_dict in navigation_simulator.perform_navigation():
                        if result_dict:
                            results_dict[model_type][model_name][k].append(utils.convert_dictionary_to_array(result_dict))
                        else:
                            results_dict[model_type][model_name][k].append(([],[]))
                    results_dict[model_type][model_name][k].extend((model, navigation_simulator))

        return results_dict



dictionary = dict()
for i in [1]:

    observation_windows =   [(60390, 60394), (60397+i, 60399)]

    model_type = "low_fidelity"
    model_name = "three_body_problem"
    model_type = "high_fidelity"
    model_name = "point_mass"
    params = ({model_type: [model_name]},
                [model_type, model_name, 0],
                True,
                observation_windows,
                True,
                True,
                [60399])

    results_dict = TestNavigation().get_navigation_results(*params)
    delta_v = results_dict[model_type][model_name][0][8][1]
    objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
    print("Window: ", observation_windows)
    print("Objective: ", objective_value)

    dictionary[str(60397+i)] = [objective_value]

    PlotNavigationResults.PlotNavigationResults(results_dict).plot_estimation_error_history()
    PlotNavigationResults.PlotNavigationResults(results_dict).plot_uncertainty_history()
    PlotNavigationResults.PlotNavigationResults(results_dict).plot_formal_error_history()
    PlotNavigationResults.PlotNavigationResults(results_dict).plot_reference_deviation_history()
    PlotNavigationResults.PlotNavigationResults(results_dict).plot_full_state_history()

print(dictionary)

plt.show()