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
###### Fixed arc durations ######################################
#################################################################

### Run the navigation routines for varying observation windows
run = True
dynamic_model_list = ["HF", "PM", 0]
truth_model_list = ["HF", "PM", 0]
duration = 28
skm_to_od_duration = 3
threshold = 3
od_duration = 1
num_runs = 3
od_durations = [0.2, 0.5, 1, 1.5, 1.8]
skm_to_od_durations = [1, 1.5, 2, 2.5, 3, 3.5, 4]
custom_station_keeping_error = 1e-2

if run:
    navigation_results_dict = {}
    delta_v_dict = {}
    for model in ["PM", "PMSRP"]:

        delta_v_dict_per_skm_to_od_duration = {}
        for skm_to_od_duration in skm_to_od_durations:

            delta_v_dict_per_od_duration = {}
            for od_duration in od_durations:

                # Generate a vector with OD durations
                start_epoch = 60390
                epoch = start_epoch + threshold + skm_to_od_duration + od_duration
                skm_epochs = []
                i = 1
                while True:
                    if epoch < start_epoch+duration:
                        skm_epochs.append(epoch)
                        epoch += skm_to_od_duration+od_duration
                    else:
                        design_vector = od_duration*np.ones(np.shape(skm_epochs))
                        break
                    i += 1

                # Extract observation windows
                observation_windows = [(start_epoch, start_epoch+threshold)]
                for i, skm_epoch in enumerate(skm_epochs):
                    observation_windows.append((skm_epoch-od_duration, skm_epoch))

                # Start running the navigation routine
                navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                                ["HF", model, 0],
                                                                                ["HF", model, 0],
                                                                                station_keeping_epochs=skm_epochs,
                                                                                target_point_epochs=[3],
                                                                                custom_station_keeping_error=custom_station_keeping_error,
                                                                                step_size=1e-2)

                navigation_results = navigation_simulator.get_navigation_results()

                navigation_results_dict[od_duration] = navigation_results

                delta_v = navigation_results[8][1]
                objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
                print(f"Objective: \n", delta_v, objective_value)
                print("End of objective calculation ===============")

                delta_v_dict_per_od_duration[od_duration] = [list(design_vector), objective_value]

            delta_v_dict_per_skm_to_od_duration[skm_to_od_duration] = delta_v_dict_per_od_duration

        delta_v_dict[model] = delta_v_dict_per_skm_to_od_duration

    print(navigation_results_dict)
    print(delta_v_dict)

    utils.save_dicts_to_folder(dicts=[delta_v_dict], labels=["delta_v_dict_constant_arc_duration"], custom_sub_folder_name=file_name)


### Run the navigation routines for varying observation windows
run=True
dynamic_model_list = ["HF", "PM", 0]
truth_model_list = ["HF", "PM", 0]
duration = 28
skm_to_od_duration = 3
threshold = 3
od_duration = 1
num_runs = 10
mean = od_duration
std_dev = 0.3
custom_station_keeping_error = 1e-2

if run:
    navigation_results_dict = {}
    delta_v_dict = {}
    for model in ["PM", "PMSRP"]:

        np.random.seed(0)
        delta_v_dict_per_run = {}
        for num_run in range(num_runs):

            # Generate a vector with OD durations
            od_durations = np.random.normal(loc=mean, scale=std_dev, size=(20))
            start_epoch = 60390
            epoch = start_epoch + threshold + skm_to_od_duration + od_durations[0]
            skm_epochs = []
            i = 1
            while True:
                if epoch < start_epoch+duration:
                    skm_epochs.append(epoch)
                    epoch += skm_to_od_duration+od_durations[i]
                else:
                    design_vector = np.ones(np.shape(skm_epochs))
                    break
                i += 1

            # Extract observation windows
            observation_windows = [(start_epoch, start_epoch+threshold)]
            for i, skm_epoch in enumerate(skm_epochs):
                observation_windows.append((skm_epoch-od_durations[i], skm_epoch))

            # Start running the navigation routine
            navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                            ["HF", model, 0],
                                                                            ["HF", model, 0],
                                                                            station_keeping_epochs=skm_epochs,
                                                                            target_point_epochs=[3],
                                                                            custom_station_keeping_error=custom_station_keeping_error,
                                                                            step_size=1e-2)

            navigation_results = navigation_simulator.get_navigation_results()

            navigation_results_dict[num_run] = navigation_results

            delta_v = navigation_results[8][1]
            objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
            print(f"Objective: \n", delta_v, objective_value)
            print("End of objective calculation ===============")

            delta_v_dict_per_run[num_run] = [list(od_durations[:len(design_vector)]), objective_value]

        delta_v_dict[model] = delta_v_dict_per_run

    print(navigation_results_dict)
    print(delta_v_dict)

    utils.save_dicts_to_folder(dicts=[delta_v_dict], labels=["delta_v_dict_variable_arc_duration"], custom_sub_folder_name=file_name)