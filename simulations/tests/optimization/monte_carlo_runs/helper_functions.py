# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils
from src.optimization_models import OptimizationModel
from src import NavigationSimulator


#################################################################################
###### Helper functions for monte carlo #########################################
#################################################################################


def get_custom_observation_windows(duration, skm_to_od_duration, threshold, od_duration):

    # Generate a vector with OD durations
    start_epoch = 60390
    epoch = start_epoch + threshold + skm_to_od_duration + od_duration
    skm_epochs = []
    i = 1
    while True:
        if epoch <= start_epoch+duration:
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

    return observation_windows


def get_monte_carlo_stats(dict):

    values = []
    for key, value in dict.items():
        values.append(value)

    return {"mean": np.mean(values), "std_dev": np.std(values)}

