# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sp

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils

# Own
from src.optimization_models import OptimizationModel


#################################################################################
###### Test runs of the navigation simulator ####################################
#################################################################################


def get_optimization_result(model, threshold, skm_to_od_duration, duration, od_duration, bounds=(0.5, 1.5), maxiter=5, factor=2):

    # Create OptimizationModel instance based on timing characteristics
    optimization_model = OptimizationModel.OptimizationModel(["high_fidelity", model, 0],
                                                             ["high_fidelity", model, 0],
                                                             threshold=threshold,
                                                             skm_to_od_duration=skm_to_od_duration,
                                                             duration=duration,
                                                             od_duration=od_duration,
                                                             bounds=bounds)

    # Adjust optimization attributes
    optimization_model.maxiter = maxiter
    optimization_model.factor = factor

    # Run optimization
    optimization_result = optimization_model.optimize()

    return optimization_result



def get_combined_history_dict(dict):

    combined_history_dict = {}

    for key, value in dict.items():
        design_vector = []
        objective_function = []

        # Extract design vector and objective function from each dictionary
        for history_key, history_value in value["history"].items():
            design_vector.append(history_value["design_vector"])
            objective_function.append(history_value["objective_function"])

        # Add combined data to the new dictionary
        combined_history_dict[key] = {
            "design_vector": design_vector,
            "objective_function": objective_function
        }

    print(combined_history_dict)

    return combined_history_dict





def run_monte_carlo_simulation(model, threshold, skm_to_od_duration, duration, od_duration, bounds=(0.5, 1.5), numruns=1, maxiter=5, factor=2):

    monte_carlo_simulation_results = dict()
    for run in range(numruns):

        optimization_result = get_optimization_result(model, threshold, skm_to_od_duration, duration, od_duration, bounds=bounds, maxiter=maxiter, factor=factor)

        print("optimization_result: ", optimization_result)
        monte_carlo_simulation_results[run] = optimization_result

    print("monte_carlo_simulation_results: ", monte_carlo_simulation_results)
    utils.save_dicts_to_folder(dicts=[monte_carlo_simulation_results])

    combined_history_dict = get_combined_history_dict(monte_carlo_simulation_results)
    monte_carlo_stats_dict = utils.get_monte_carlo_stats_dict(data_dict=combined_history_dict)
    print("monte_carlo_stats_dict", monte_carlo_stats_dict)
    utils.save_dicts_to_folder(dicts=[monte_carlo_stats_dict], labels=["numruns"+str(numruns)+"_"+str(model)+"_threshold"+str(threshold)+"_duration"+str(duration)])



#############################
###### Run setup ############
#############################


# model = "point_mass"
# threshold = 7
# skm_to_od_duration = 3
# duration = 17
# od_duration = 1
# run_monte_carlo_simulation(model, threshold, skm_to_od_duration, duration, od_duration, numruns=2, maxiter=6)






def test_objective_function(custom_x=None):

    threshold = 3
    duration = 10
    od_duration = 1
    skm_to_od_duration = 2
    model = "point_mass"
    optimization_model = OptimizationModel.OptimizationModel(["high_fidelity", "point_mass", 0],
                                                             ["high_fidelity", model, 0],
                                                             threshold=threshold,
                                                             skm_to_od_duration=skm_to_od_duration,
                                                             duration=duration,
                                                             od_duration=od_duration)

    x = optimization_model.initial_design_vector
    objective_value = optimization_model.objective_function(custom_x, show_directly=False)

    return objective_value
    # plt.show()

xs = np.arange(0.9, 1.1, 0.025)
ys = np.arange(0.9, 1.1, 0.025)

# Create the 2D mesh
# X, Y = np.meshgrid(x_values, y_values)

# xs = [[1, 1], [1,  1.0375], [1.025,  1], [0.9875 , 1.01875]]
res = []
a = []
for x in xs:
    for y in ys:
        res.append(test_objective_function(custom_x=[x, y]))
        a.append([x, y])
print(res)
print(a)

plt.show()