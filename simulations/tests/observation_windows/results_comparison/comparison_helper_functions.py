# Standard
import os
import sys
import numpy as np

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


def get_constant_arc_observation_windows(duration, skm_to_od_duration, threshold, od_duration, simulation_start_epoch=60390):

    # Generate a vector with OD durations
    epoch = simulation_start_epoch + threshold + skm_to_od_duration + od_duration
    skm_epochs = []
    i = 1
    while True:
        if epoch <= simulation_start_epoch+duration:
            skm_epochs.append(epoch)
            epoch += skm_to_od_duration+od_duration
        else:
            design_vector = od_duration*np.ones(np.shape(skm_epochs))
            break
        i += 1

    # Extract observation windows
    observation_windows = [(simulation_start_epoch, simulation_start_epoch+threshold)]
    for i, skm_epoch in enumerate(skm_epochs):
        observation_windows.append((skm_epoch-od_duration, skm_epoch))

    return observation_windows


def get_orbit_based_arc_observation_windows(duration, period=0.4597, margin=0.05, step_size=0.01, simulation_start_epoch=60390, threshold=0, apolune=False, pass_interval=2):

    ### Constant arc, around perilune
    epochs = np.arange(0, duration, step_size) + simulation_start_epoch
    total_indices = len(epochs)
    pass_interval += 1
    pass_interval_index = int(pass_interval*period/step_size)
    period = int(period/step_size)
    margin = int(margin/step_size)
    threshold_index = int(threshold/step_size)

    if apolune:
        indices = np.arange(0+int(period/2), total_indices, period)
    else:
        indices = np.arange(0, total_indices, period)


    if pass_interval == 0:
        pass_interval = None

    ranges = []
    if threshold > 0:
        ranges = [(0, threshold_index)]
    ranges.extend([(index-margin, index+margin) for index in indices[indices > threshold_index+pass_interval_index][::pass_interval]])

    observation_windows = []
    for start_index, end_index in ranges:
        values = epochs[start_index:end_index]
        observation_windows.append((min(values), max(values)))

    return observation_windows


# print(get_orbit_based_arc_observation_windows(5, margin=0.05, step_size=0.01, threshold=0.5, pass_interval=1))



#################################################################
###### Generate NavigationOutput objects ########################
#################################################################

def generate_navigation_outputs(observation_windows_settings, seed=0, **kwargs):

    np.random.seed(seed)

    # Run the navigation routine using given settings
    navigation_outputs = {}
    for window_type in observation_windows_settings.keys():

        navigation_output_per_type = []
        for (observation_windows, num_runs) in observation_windows_settings[window_type]:

            navigation_output_per_run = {}
            for run in range(num_runs):

                navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows)

                navigation_simulator.configure(**kwargs)

                navigation_output_per_run[run] = navigation_simulator.perform_navigation()

            navigation_output_per_type.append(navigation_output_per_run)

        navigation_outputs[window_type] = navigation_output_per_type

    return navigation_outputs
