# Standard
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils
import helper_functions
from src.optimization_models import OptimizationModel
from src import NavigationSimulator, PlotNavigationResults



#################################################################
###### Monte carlo continuous observation windows ###############
#################################################################

### Run the navigation routines for varying observation windows
dynamic_model_list = ["HF", "PM", 0]
truth_model_list = ["HF", "PM", 0]
duration = 28
skm_to_od_duration = 3
threshold = 3
od_duration = 1
custom_station_keeping_error = 1e-2
num_runs = 10

observation_windows = helper_functions.get_custom_observation_windows(duration, skm_to_od_duration, threshold, od_duration)
station_keeping_epochs = [windows[1] for windows in observation_windows]

print(observation_windows)

monte_carlo_results = {}
history = {}
np.random.seed(1)
for run in range(num_runs):

    print(f"Start of run {run}")

    navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                    dynamic_model_list,
                                                                    truth_model_list,
                                                                    station_keeping_epochs=station_keeping_epochs,
                                                                    custom_station_keeping_error=custom_station_keeping_error)

    navigation_results = navigation_simulator.get_navigation_results()

    objective_value = np.sum(np.linalg.norm(navigation_results[8][1], axis=1))

    history[run] = objective_value

    print(f"End of run {run}\n Delta V: {objective_value}")

monte_carlo_results["station_keeping_error"] = custom_station_keeping_error
monte_carlo_results["history"] = history
monte_carlo_results["stats"] = helper_functions.get_monte_carlo_stats(history)

utils.save_dicts_to_folder(dicts=[monte_carlo_results], labels=[f"skm{skm_to_od_duration}_num{num_runs}_err{custom_station_keeping_error}"], custom_sub_folder_name=file_name)





dynamic_model_list = ["HF", "PM", 0]
truth_model_list = ["HF", "PM", 0]
duration = 28
skm_to_od_duration = 3
threshold = 3
od_duration = 1
custom_station_keeping_error = 1e-10
num_runs = 10

observation_windows = helper_functions.get_custom_observation_windows(duration, skm_to_od_duration, threshold, od_duration)
station_keeping_epochs = [windows[1] for windows in observation_windows]

print(observation_windows)

monte_carlo_results = {}
history = {}
np.random.seed(1)
for run in range(num_runs):

    print(f"Start of run {run}")

    navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                    dynamic_model_list,
                                                                    truth_model_list,
                                                                    station_keeping_epochs=station_keeping_epochs,
                                                                    custom_station_keeping_error=custom_station_keeping_error)

    navigation_results = navigation_simulator.get_navigation_results()

    objective_value = np.sum(np.linalg.norm(navigation_results[8][1], axis=1))

    history[run] = objective_value

    print(f"End of run {run}\n Delta V: {objective_value}")

monte_carlo_results["station_keeping_error"] = custom_station_keeping_error
monte_carlo_results["history"] = history
monte_carlo_results["stats"] = helper_functions.get_monte_carlo_stats(history)

utils.save_dicts_to_folder(dicts=[monte_carlo_results], labels=[f"skm{skm_to_od_duration}_num{num_runs}_err{custom_station_keeping_error}"], custom_sub_folder_name=file_name)



dynamic_model_list = ["HF", "PM", 0]
truth_model_list = ["HF", "PM", 0]
duration = 28
skm_to_od_duration = 3
threshold = 3
od_duration = 1
custom_station_keeping_error = 2e-2
num_runs = 10

observation_windows = helper_functions.get_custom_observation_windows(duration, skm_to_od_duration, threshold, od_duration)
station_keeping_epochs = [windows[1] for windows in observation_windows]

print(observation_windows)

monte_carlo_results = {}
history = {}
np.random.seed(1)
for run in range(num_runs):

    print(f"Start of run {run}")

    navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                    dynamic_model_list,
                                                                    truth_model_list,
                                                                    station_keeping_epochs=station_keeping_epochs,
                                                                    custom_station_keeping_error=custom_station_keeping_error)

    navigation_results = navigation_simulator.get_navigation_results()

    objective_value = np.sum(np.linalg.norm(navigation_results[8][1], axis=1))

    history[run] = objective_value

    print(f"End of run {run}\n Delta V: {objective_value}")

monte_carlo_results["station_keeping_error"] = custom_station_keeping_error
monte_carlo_results["history"] = history
monte_carlo_results["stats"] = helper_functions.get_monte_carlo_stats(history)

utils.save_dicts_to_folder(dicts=[monte_carlo_results], labels=[f"skm{skm_to_od_duration}_num{num_runs}_err{custom_station_keeping_error}"], custom_sub_folder_name=file_name)