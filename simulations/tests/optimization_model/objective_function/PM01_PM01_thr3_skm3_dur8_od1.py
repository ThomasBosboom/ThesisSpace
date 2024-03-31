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

#################################################################
###### Monte Carlo test case ####################################
#################################################################

dynamic_model_list = ["high_fidelity", "point_mass",0]
truth_model_list = ["high_fidelity", "point_mass",0]
threshold = 3
skm_to_od_duration = 3
duration = 8
od_duration = 1
bounds = (0.5, 1.5)

# Create OptimizationModel instance based on timing characteristics
optimization_model = OptimizationModel.OptimizationModel(dynamic_model_list,
                                                        truth_model_list,
                                                        threshold=threshold,
                                                        skm_to_od_duration=skm_to_od_duration,
                                                        duration=duration,
                                                        od_duration=od_duration,
                                                        bounds=bounds)

# optimization_model.xk = [0.975, 0.925]
x = optimization_model.initial_design_vector
optimization_model.objective_function(x, plot_results=True)

plt.show()