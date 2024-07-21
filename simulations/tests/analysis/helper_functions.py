# Standard
import os
import sys
import numpy as np
import copy
from matplotlib import pyplot as plt
from collections import defaultdict

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from src import NavigationSimulator



#################################################################
###### Generate observation windows #############################
#################################################################


def get_random_arc_observation_windows(duration=28, arc_interval_vars=[3.5, 0.1], threshold_vars=[0.5, 0.001], arc_duration_vars=[0.5, 0.1], seed=0, mission_start_epoch=60390):

    rng = np.random.default_rng(seed=seed)

    # arc_interval = np.random.normal(loc=arc_interval_vars[0], scale=arc_interval_vars[1], size=100)
    # arc_duration = np.random.normal(loc=arc_duration_vars[0], scale=arc_duration_vars[1], size=100)
    # threshold = np.random.normal(loc=threshold_vars[0], scale=threshold_vars[1], size=100)

    arc_interval = rng.uniform(
        low=arc_interval_vars[0] - arc_interval_vars[1],
        high=arc_interval_vars[0] + arc_interval_vars[1],
        size=100
    )

    arc_duration = rng.uniform(
        low=arc_duration_vars[0] - arc_duration_vars[1],
        high=arc_duration_vars[0] + arc_duration_vars[1],
        size=100
    )

    threshold = rng.uniform(
        low=threshold_vars[0] - threshold_vars[1],
        high=threshold_vars[0] + threshold_vars[1],
        size=100
    )

    # Generate a vector with OD durations
    epoch = mission_start_epoch + threshold[0] + arc_interval[0] + arc_duration[0]
    skm_epochs = []
    i = 1
    while True:
        if epoch <= mission_start_epoch+duration:
            skm_epochs.append(epoch)
            epoch += arc_interval[i]+arc_duration[i]
        else:
            design_vector = arc_duration[i]*np.ones(np.shape(skm_epochs))
            break
        i += 1

    # Extract observation windows
    observation_windows = [(mission_start_epoch, mission_start_epoch+threshold[0])]
    for i, skm_epoch in enumerate(skm_epochs):
        observation_windows.append((skm_epoch-arc_duration[i+1], skm_epoch))

    return observation_windows


def get_constant_arc_observation_windows(duration=28, arc_interval=3, threshold=1, arc_duration=1, mission_start_epoch=60390):

    threshold=arc_duration
    # Generate a vector with OD durations
    epoch = mission_start_epoch + threshold + arc_interval + arc_duration
    skm_epochs = []
    i = 1
    while True:
        if epoch <= mission_start_epoch+duration:
            skm_epochs.append(epoch)
            epoch += arc_interval+arc_duration
        else:
            design_vector = arc_duration*np.ones(np.shape(skm_epochs))
            break
        i += 1

    # Extract observation windows
    observation_windows = [(mission_start_epoch, mission_start_epoch+threshold)]
    for i, skm_epoch in enumerate(skm_epochs):
        observation_windows.append((skm_epoch-arc_duration, skm_epoch))

    return observation_windows


def get_constant_arc_with_subarcs_observation_windows(duration=28, arc_interval=3.5, threshold=[1, 0.1], arc_duration=[0.5, 0.1], mission_start_epoch=60390):

    threshold_subarcs = np.linspace(0, threshold[0], int(threshold[0]/threshold[1]+1))
    arc_subarcs = np.linspace(0, arc_duration[0], int(arc_duration[0]/arc_duration[1]+1))

    # Extract threshold observation windows
    threshold_observation_windows = [(threshold_subarcs[i], threshold_subarcs[i + 1]) for i in range(len(threshold_subarcs) - 1)]
    arc_observation_windows = [(arc_subarcs[i], arc_subarcs[i + 1]) for i in range(len(arc_subarcs) - 1)]

    observation_windows = threshold_observation_windows
    while observation_windows[-1][-1]+arc_interval+arc_subarcs[-1]<duration:
        off_set = observation_windows[-1][-1]+arc_interval
        observation_windows.extend([(tup[0] + off_set, tup[1] + off_set) for tup in arc_observation_windows])

    observation_windows = mission_start_epoch + np.array(observation_windows)
    observation_windows = [tuple(window) for window in observation_windows]

    return observation_windows

# print(get_constant_arc_with_subarcs_observation_windows(duration=28, arc_interval=3.5, threshold=[1, 0.1], arc_duration=[0.5, 0.1], mission_start_epoch=60390))


def get_orbit_based_arc_observation_windows(duration=28, period=0.4597, step_size=0.01, mission_start_epoch=60390, margin=0.05, apolune=True, pass_interval=7, threshold=0):

    ### Constant arc, around perilune
    epochs = np.arange(0, duration, step_size) + mission_start_epoch
    total_indices = len(epochs)
    pass_interval += 1
    pass_interval_index = int(pass_interval*period/step_size)
    period = int(period/step_size)
    margin = int(margin/step_size)
    threshold_index = int(threshold/step_size)

    if apolune:
        indices = np.arange(0, total_indices, period)
    else:
        indices = np.arange(0+int(period/2), total_indices, period)

    if pass_interval == 0:
        pass_interval = None

    ranges = []
    if threshold > 0:
        ranges = [(0, threshold_index)]
    ranges.extend([(index-margin, index+margin) for index in indices[indices > threshold_index+pass_interval_index][::pass_interval]])

    observation_windows = []
    for start_index, end_index in ranges:
        values = epochs[start_index:end_index+1]
        observation_windows.append((min(values), max(values)))

    return observation_windows

# print(get_orbit_based_arc_observation_windows(duration=28, period=0.4597, step_size=0.01, mission_start_epoch=60390, margin=0.05,  apolune=False, pass_interval=7, threshold=0))


#################################################################
###### Generate NavigationOutput objects ########################
#################################################################

def get_navigation_output(observation_windows, seed=0, **kwargs):

    navigation_simulator = NavigationSimulator.NavigationSimulator(**kwargs)
    navigation_output = navigation_simulator.perform_navigation(observation_windows, seed=seed)

    return navigation_output


# def generate_navigation_outputs(observation_windows_settings, **kwargs):

#     # Run the navigation routine using given settings
#     navigation_outputs = {}
#     for window_type in observation_windows_settings.keys():

#         navigation_output_per_type = []
#         for (observation_windows, num_runs, label) in observation_windows_settings[window_type]:

#             navigation_output_per_run = {}
#             for run in range(num_runs):

#                 print(f"Run {run+1} of {num_runs}, seed {run}")
#                 navigation_output_per_run[run] = get_navigation_output(observation_windows, seed=run, **kwargs)

#             navigation_output_per_type.append(navigation_output_per_run)

#         navigation_outputs[window_type] = navigation_output_per_type

#     return navigation_outputs


def generate_navigation_outputs(observation_windows_settings, **kwargs):

    kwargs_copy = kwargs.copy()
    # Run the navigation routine using given settings
    navigation_outputs = {}
    for window_type in observation_windows_settings.keys():

        navigation_output_per_type = []
        for (observation_windows, num_runs, label) in observation_windows_settings[window_type]:

            navigation_output_per_run = {}

            seed = 0
            seed_copy = 0
            if "seed" in kwargs.keys():
                seed = kwargs["seed"]
                seed_copy = seed
                kwargs.pop("seed")

            for run, seed in enumerate(range(seed, seed+num_runs)):

                print(f"Run {run+1} of {num_runs}, seed {seed}")
                navigation_output_per_run[run] = get_navigation_output(observation_windows, seed=seed, **kwargs)

            kwargs.update({"seed": seed_copy})

            navigation_output_per_type.append(navigation_output_per_run)

        navigation_outputs[window_type] = navigation_output_per_type

    return navigation_outputs



def generate_navigation_outputs_sensitivity_analysis(num_runs, sensitivity_settings, default_window_inputs, **kwargs):

    observation_windows_settings = {
        "Constant": [
            (get_constant_arc_observation_windows(**default_window_inputs), num_runs, None),
        ]
    }

    navigation_outputs_sensitivity = {}

    for arg_name, arg_values in sensitivity_settings.items():
        for arg_index, arg_value in enumerate(arg_values):

            print("Input: \n", {arg_name: arg_value})

            for key, value in kwargs.items():
                if key==arg_name:
                    kwargs.pop(key)

            if arg_name in ["threshold", "arc_interval", "arc_duration", "mission_start_epoch"]:

                window_inputs = copy.deepcopy(default_window_inputs)
                window_inputs[arg_name] = arg_value
                observation_windows_sensitivity_settings = {}
                for window_type in observation_windows_settings.keys():

                    observation_windows_sensitivity_settings[window_type] = [(get_constant_arc_observation_windows(**window_inputs), num_runs, None)]

                navigation_outputs = generate_navigation_outputs(observation_windows_sensitivity_settings, **{arg_name: arg_value}, **kwargs)

            else:
                navigation_outputs = generate_navigation_outputs(observation_windows_settings, **{arg_name: arg_value}, **kwargs)


            for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):
                for case_index, window_case in enumerate(navigation_outputs_cases):

                    if window_type not in navigation_outputs_sensitivity:
                        navigation_outputs_sensitivity[window_type] = {}

                    if arg_name not in navigation_outputs_sensitivity[window_type]:
                        navigation_outputs_sensitivity[window_type][arg_name] = [window_case]

                    else:
                        navigation_outputs_sensitivity[window_type][arg_name].append(window_case)

    return navigation_outputs_sensitivity


def generate_total_observation_time(observation_windows):
    return np.sum([window[1]-window[0] for window in observation_windows])