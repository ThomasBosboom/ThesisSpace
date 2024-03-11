# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import scipy as sp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Own
from dynamic_models import NavigationSimulator, PlotNavigationResults
from OptimizationModelBase import OptimizationModelBase
from tests import utils


class OptimizationModel(OptimizationModelBase):

    def __init__(self, dynamic_model_list, truth_model_list, threshold=8, duration=14, options=None):
        super().__init__(threshold, duration, options)

        # Specify the dynamic and truth model used for the estimation arcs
        self.model_type, self.model_name, self.model_number = dynamic_model_list[0], dynamic_model_list[1], dynamic_model_list[2]
        self.model_type_truth, self.model_name_truth, self.model_number_truth = truth_model_list[0], truth_model_list[1], truth_model_list[2]


    def objective_function(self, x):

        print("Design vector: ", x)

        # observation_windows = self.get_initial_observation_windows()
        station_keeping_epochs = self.get_initial_design_vector_dict()["t_skm"]
        if self.threshold in station_keeping_epochs:
            station_keeping_epochs = station_keeping_epochs[1:]

        observation_windows = [(self.mission_start_time, self.threshold)]
        observation_windows.extend([(x[i], station_keeping_epochs[i]) for i in range(len(station_keeping_epochs))])
        station_keeping_epochs = self.get_initial_design_vector_dict()["t_skm"]
        print("observation windows in objective: \n", observation_windows)
        print("custom station keeping windows: \n", station_keeping_epochs)

        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                       [self.model_type, self.model_name, self.model_number],
                                                                       [self.model_type_truth, self.model_name_truth, self.model_number_truth],
                                                                       custom_station_keeping_epochs=station_keeping_epochs
                                                                       )

        # print("station keeping epochs: ", navigation_simulator.station_keeping_epochs)
        results = []
        for result_dict in navigation_simulator.perform_navigation():
            if result_dict:
                results.append(utils.convert_dictionary_to_array(result_dict))
            else: # if there is no station keeping, so empty delta_v_dict (last entry)
                results.append(([],[]))
        results.append((navigation_simulator))

        results_dict = {self.model_type: {self.model_name: [results]}}
        # print("RESULTS_DICT: ", results_dict)
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_formal_error_history()
        PlotNavigationResults.PlotNavigationResults(results_dict).plot_uncertainty_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_reference_deviation_history()
        PlotNavigationResults.PlotNavigationResults(results_dict).plot_estimation_error_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_full_state_history()
        # plt.show()

        delta_v = results[8][1]
        objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
        print(f"objective value at {x}: ", objective_value)

        return objective_value




    def optimize(self):

        # Initial guess for the design vector
        x0 = self.get_initial_design_vector()
        print("Initial state: \n", x0)

        # Define boundaries for the design vector entries
        # xl = x0 + a*np.ones(len(x0))
        # xu = x0 + b*np.ones(len(x0))
        # print("Lower bound: \n", xl)
        # print("Upper bound: \n", xu)

        # Define constraints
        # constraints = (
            # {'type': 'ineq', 'fun': self.inequality_constraint1},
            # {'type': 'ineq', 'fun': self.inequality_constraint2},
            # {'type': 'ineq', 'fun': self.inequality_constraint3}
        # )

        # Define bounds
        bounds_dict = self.get_design_vector_bounds()
        bounds_list = []
        for key, value in bounds_dict.items():
            bounds_list.extend(value)
        print(bounds_list)

        # Define a callback function to record iteration history
        iteration_history = []
        design_vectors = []
        def callback(xk):
            iteration_history.append(self.objective_function(xk))
            design_vectors.append(xk)

        # Minimize the objective function subject to constraints
        result = sp.optimize.minimize(self.objective_function, x0,
                                    # constraints=constraints,
                                    bounds=bounds_list,
                                    method='Nelder-Mead',
                                    options={'maxiter': 3, "xatol": 1e-2, 'disp': True},
                                    callback=callback)

        # Extract optimized start times
        x_optim = result.x
        print("x_optim:", x_optim)
        print("iteration history: \n", iteration_history)
        print("design vector history: \n", np.array(design_vectors))

        plt.plot(iteration_history)
        plt.yscale("log")


        return iteration_history, np.array(design_vectors)


estimation_model


dynamic_model_list = ["low_fidelity", "three_body_problem", 0]
truth_model_list = ["low_fidelity", "three_body_problem", 0]
# dynamic_model_list = ["high_fidelity", "point_mass", 0]
# truth_model_list = ["high_fidelity", "point_mass", 0]



optimization_model = OptimizationModel(dynamic_model_list, truth_model_list,
                                threshold=8, duration=14, options=None)


print(optimization_model.get_initial_observation_windows())
print(optimization_model.get_initial_design_vector())
print(optimization_model.get_design_vector_bounds())

# iteration_history, design_vectors = optimization_model.optimize()
# print(iteration_history, design_vectors)

# plt.show()


