import os
import sys
import numpy as np
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Define current time
current_time = datetime.now().strftime("%Y%m%d%H%M")

# Own
from src import NavigationSimulator, ObjectiveFunctions
from tests.postprocessing import ProcessOptimizationResults, OptimizationModel
from optimization_analysis_helper_functions import check_file_exists, process_case

from tests.postprocessing import OptimizationModelPSO


if __name__ == "__main__":

    # function we are attempting to optimize (minimize)
    def func1(x):
        # total=0
        # for i in range(len(x)):
        #     total+=x[i]**2
        # return total

        return np.sum([tup[-1]-tup[0] for tup in x])

    bounds = [0.5, 2]

    num_particles = 50
    maxiter = 40
    time_tag = 1111

    # mission_start_epoch = 60390
    # arc_interval = 3
    # arc_length = 1
    # duration = 28

    kwargs = {
        "num_particles": 50,
        "maxiter": 40
    }

    optimization_model = OptimizationModelPSO.OptimizationModel(**kwargs)
    # optimization_results = optimization_model.load_from_json(time_tag=time_tag, folder_name="optimization_analysis_pso")
    # optimization_model = OptimizationModelPSO.OptimizationModel(custom_input=optimization_results, **kwargs)
    optimization_model.optimize(func1)












# if __name__ == "__main__":

#     ##############################################################
#     #### Optimization settings ###################################
#     ##############################################################

#     cases = {
#         "delta_v_min": [0.00],
#     }

#     auto_mode = True
#     custom_tag = "default_10timesIO"
#     num_optims = 5

#     duration = 28
#     arc_length = 1
#     arc_interval = 3

#     test_objective = False
#     use_same_seed = False
#     run_optimization = False
#     plot_full_comparison = False
#     from_file = True

#     if custom_tag is not None:
#         current_time = custom_tag
#     if auto_mode:
#         if custom_tag is not None:
#             current_time = custom_tag
#         from_file = check_file_exists(cases, current_time, num_optims, file_name)
#         run_optimization = True


#     ##############################################################
#     #### Default optimization settings ###########################
#     ##############################################################

#     navigation_simulator_settings = {
#         "show_corrections_in_terminal": True,
#         "run_optimization_version": True,
#         "step_size_optimization_version": 0.5,
#         "orbit_insertion_error": np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*10
#     }

#     objective_functions_settings = {
#         "evaluation_threshold": 14,
#         "num_runs": 1,
#         "seed": 0
#     }

#     optimization_model_settings = {
#         "duration": duration,
#         "arc_length": arc_length,
#         "arc_interval": arc_interval,
#         "max_iterations": 10,
#         "bounds": (0.1, 2.0),
#         "design_vector_type": "arc_lengths",
#         "initial_simplex_perturbation": -arc_length/2,
#         "results_in_terminal": True
#     }


#     ##############################################################
#     #### Start of main loop ######################################
#     ##############################################################

#     keys, values = zip(*cases.items())
#     case_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     # Use multiprocessing.Pool to parallelize the loop
#     num_workers = multiprocessing.cpu_count()
#     with multiprocessing.Pool(processes=num_workers) as pool:
#         partial_process_case = partial(
#             process_case,
#             navigation_simulator_settings=navigation_simulator_settings,
#             objective_functions_settings=objective_functions_settings,
#             optimization_model_settings=optimization_model_settings,
#             run_optimization=run_optimization,
#             from_file=from_file,
#             custom_tag=current_time,
#             file_name=file_name,
#             test_objective=test_objective,
#             use_same_seed=use_same_seed,
#             plot_full_comparison=plot_full_comparison)

#         case_runs = [(case, run) for case in case_combinations for run in range(num_optims)]
#         results = pool.starmap(partial_process_case, case_runs)

#     # Select only first case run as example
#     final_process_optimization_results = results[0]
#     final_process_optimization_results.plot_iteration_history(
#         show_design_variables=False,
#         compare_time_tags=[result.time_tag for result in results]
#     )

#     final_process_optimization_results.tabulate_optimization_results(
#         compare_time_tags=[result.time_tag for result in results]
#     )

#     plt.show()