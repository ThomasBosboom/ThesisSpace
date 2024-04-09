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

from tests import utils
import helper_functions


#################################################################
###### Monte Carlo test case ####################################
#################################################################

dynamic_model_list = ["HF", "PM",0]
truth_model_list = ["HF", "PM",0]
threshold = 3
skm_to_od_duration = 2.5
duration = 28
od_duration = 1
numruns = 1
maxiter = 10
factor = 1
bounds = (0.5, 1.5)
helper_functions.run_monte_carlo_optimization_model(dynamic_model_list,
                                                    truth_model_list,
                                                    threshold,
                                                    skm_to_od_duration,
                                                    duration,
                                                    od_duration,
                                                    bounds=bounds,
                                                    numruns=numruns,
                                                    maxiter=maxiter,
                                                    factor=factor,
                                                    label=file_name)