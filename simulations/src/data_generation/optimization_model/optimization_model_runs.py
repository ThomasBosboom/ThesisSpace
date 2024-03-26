# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sp

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(parent_dir)

from tests import utils

# Own
from src.optimization_models import OptimizationModel
from src.dynamic_models import PlotNavigationResults, NavigationSimulator


#################################################################################
###### Test runs of the navigation simulator ####################################
#################################################################################


def run_optimization_model(model, threshold, skm_to_od_duration, duration, od_duration, maxiter=5):

    optimization_model = OptimizationModel.OptimizationModel(["high_fidelity", model, 0], ["high_fidelity", model, 0], threshold=threshold, skm_to_od_duration=skm_to_od_duration, duration=duration, od_duration=od_duration)
    optimization_model.maxiter = maxiter
    result_dict = optimization_model.optimize()

    return result_dict


def monte_carlo_optimization_runs(model, threshold, skm_to_od_duration, duration, od_duration, num_runs=1, maxiter=5):

    # model = "point_mass"
    # threshold = 7
    # skm_to_od_duration = 3
    # duration = 28
    # od_duration = 1

    monte_carlo_optimization_runs_total_dict = dict()
    for run in range(num_runs):

        result_dict = run_optimization_model(model, threshold, skm_to_od_duration, duration, od_duration, maxiter=maxiter)

        # utils.save_dicts_to_folder(dicts=[result_dict], labels=["run"+str(run)+"_"+str(model)+"_threshold"+str(threshold)+"_duration"+str(duration)])

        print(result_dict)
        monte_carlo_optimization_runs_total_dict[run] = result_dict

    utils.save_dicts_to_folder(dicts=[monte_carlo_optimization_runs_total_dict])




model = "point_mass"
threshold = 7
skm_to_od_duration = 3
duration = 28
od_duration = 1
monte_carlo_optimization_runs(model, threshold, skm_to_od_duration, duration, od_duration, num_runs=2, maxiter=6)






# def test_objective_function():

#     threshold = 3
#     duration = 10
#     od_duration = 1
#     skm_to_od_duration = 2
#     model = "three_body_problem"
#     optimization_model = OptimizationModel.OptimizationModel(["low_fidelity", "three_body_problem", 0],
#                                                              ["low_fidelity", model, 0],
#                                                              threshold=threshold,
#                                                              skm_to_od_duration=skm_to_od_duration,
#                                                              duration=duration,
#                                                              od_duration=od_duration)

#     x = optimization_model.initial_design_vector
#     optimization_model.objective_function(x, show_directly=False)

#     plt.show()

# test_objective_function()