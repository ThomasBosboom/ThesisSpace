import numpy as np
import os
import sys
import copy
import scipy as sp
import json
from datetime import datetime
import matplotlib.pyplot as plt
import tracemalloc
from memory_profiler import profile

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

    # tracemalloc.start()

    navigation_simulator_settings = {
        "show_corrections_in_terminal": True,
        "run_optimization_version": True,
        "step_size": 0.05,
        "delta_v_min": 0.00,
    }
    navigation_simulator = NavigationSimulator.NavigationSimulator(
        **navigation_simulator_settings
    )


    objective_functions_settings = {
        "evaluation_threshold": 14,
        "num_runs": 1,
        "seed": 0
    }
    objective_functions = ObjectiveFunctions.ObjectiveFunctions(
        navigation_simulator,
        **objective_functions_settings
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
        initial_simplex_perturbation = -0.3,
        **navigation_simulator_settings,
        **objective_functions_settings
        # custom_initial_design_vector= [
        #             0.9142857142857144,
        #             0.9142857142857144,
        #             0.9142857142857144,
        #             1.3,
        #             0.9142857142857144,
        #             0.9142857142857144,
        #             0.9142857142857144
        #         ]
    )

    if not run_optimization:
        current_time = str(202406112013)

    else:

        # Chose the objective function to optimize
        # optimization_results = optimization_model.optimize(objective_functions.test)
        optimization_results = optimization_model.optimize(objective_functions.station_keeping_cost)
        # optimization_results = optimization_model.optimize(objective_functions.overall_uncertainty)

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('traceback')
    # largest_allocation = max(top_stats, key=lambda stat: stat.size)

    # # Display information about the largest allocation
    # for line in largest_allocation.traceback.format():
    #     print(line)


    process_optimization_results = ProcessOptimizationResults.ProcessOptimizationResults(
        current_time,
        optimization_model,
        save_settings={"save_table": True,
                       "save_figure": True,
                       "current_time": current_time,
                       "file_name": file_name
        }
    )

    process_optimization_results.plot_iteration_history()
    process_optimization_results.plot_optimization_result_comparison(
        show_observation_window_settings=False,
    )

    plt.show()


