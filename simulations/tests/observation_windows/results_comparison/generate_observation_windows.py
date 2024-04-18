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

from tests import utils, helper_functions
from src import NavigationSimulator, PlotNavigationResults


#################################################################
###### Generate observation windows #############################
#################################################################

simulation_start_epoch = 60390

### Constant arc, constant distance
observation_windows_list = []
observation_windows_list.append(helper_functions.get_custom_observation_windows(28, 3, 1, 1, simulation_start_epoch=simulation_start_epoch))
# observation_windows_list.append([(60390, 60390.54000000004), (60390.799999999814, 60390.99000000022), (60391.25, 60391.439999999944), (60391.700000000186, 60391.89000000013), (60392.14999999991, 60392.33999999985), (60392.60000000009, 60392.79000000004)])


### Constant arc, around perilune
observation_windows_list = []

step_size = 0.01
start_epoch = 60390
margins = [0.07]
epochs = np.arange(0, 28, step_size) + start_epoch
total_indices = len(epochs)
period = int(0.4597/step_size)

for margin in margins:

    margin = int(margin/step_size)

    ranges = []
    index = 0
    while index < total_indices:
        if index-margin>0:
            index_range = (index-margin, index+margin)
            ranges.append(index_range)
        index += period
    print(ranges)

    observation_windows = []
    for start_index, end_index in ranges:
        values = epochs[start_index:end_index]
        observation_windows.append((min(values), max(values)))

    print(observation_windows)

    observation_windows_list.append(observation_windows)
