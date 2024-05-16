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

from tests import utils, helper_functions


#################################################################
###### Monte Carlo test case ####################################
#################################################################

dynamic_model_list = ["HF", "PM", 0]
truth_model_list = ["HF", "PM", 0]
threshold = 1
skm_to_arc_duration = 3
duration = 28
arc_duration = 0.5
numruns = 3
maxiter = 20
factor = 1
bounds = (0.1, 0.9)
custom_station_keeping_error=1e-10
helper_functions.run_monte_carlo_optimization_model(dynamic_model_list,
                                                    truth_model_list,
                                                    threshold,
                                                    skm_to_arc_duration,
                                                    duration,
                                                    arc_duration,
                                                    bounds=bounds,
                                                    numruns=numruns,
                                                    maxiter=maxiter,
                                                    factor=factor,
                                                    custom_station_keeping_error=custom_station_keeping_error,
                                                    label=file_name)