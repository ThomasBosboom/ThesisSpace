import numpy as np
import os
import sys
import copy
import scipy as sp
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(3):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Define current time
current_time = datetime.now().strftime("%Y%m%d%H%M")

# Own
from src import NavigationSimulator, ObjectiveFunctions
from tests.postprocessing import ProcessOptimizationResults, OptimizationModel


run_optimization = True
if __name__ == "__main__":

    if not run_optimization:
        current_time = str(202406041632)

    else:

        navigation_simulator = NavigationSimulator.NavigationSimulator(
            # step_size=0.01,
            # noise_range=1,
            # margin=0,
            # orbit_insertion_error=np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0,
            # initial_estimation_error=np.array([5e1, 5e1, 5e1, 1e-4, 1e-4, 1e-4, 5e3, 5e3, 5e3, 1e-2, 1e-2, 1e-2]),
            # apriori_covariance=np.diag([5e1, 5e1, 5e1, 1e-4, 1e-4, 1e-4, 5e3, 5e3, 5e3, 1e-2, 1e-2, 1e-2])**2
        )

        optimization_model = OptimizationModel.OptimizationModel(
            json_settings={"save_json": True, "current_time": current_time, "file_name": file_name},
            duration=28,
            arc_length=1,
            arc_interval=3,
            max_iterations=100,
            bounds=(-0.9, 0.9),
            optimization_method="Nelder-Mead",
            design_vector_type="arc_lengths",
            initial_simplex_perturbation = 0.3,
            # custom_initial_guess=[]
        )

        objective_functions = ObjectiveFunctions.ObjectiveFunctions(
            navigation_simulator,
            evaluation_threshold=14,
            num_runs=1
        )

        # Chose the objective function to optimize
        # optimization_results = optimization_model.optimize(objective_functions.test)
        optimization_results = optimization_model.optimize(objective_functions.station_keeping_cost)
        # optimization_results = optimization_model.optimize(objective_functions.overall_uncertainty)

        # Compare before and after optimization
        observation_windows = optimization_model.generate_observation_windows(optimization_results.final_solution)
        cost_initial = objective_functions.station_keeping_cost(observation_windows)
        observation_windows = optimization_model.generate_observation_windows(optimization_results.initial_guess)
        cost_optimized = objective_functions.station_keeping_cost(observation_windows)

    plot_optimization_results = ProcessOptimizationResults.PlotOptimizationResults(time_tag=current_time)
    plot_optimization_results.plot_iteration_history()
    plt.show()