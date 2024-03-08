# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import scipy as sp

# Tudatpy imports

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Own
from dynamic_models import NavigationSimulator, PlotNavigationResults
from tests import utils


class WindowOptimizer():

    def __init__(self, dynamic_model_list, truth_model_list):

        # Managing the dynamic model specifications
        self.model_type, self.model_name, self.model_number = dynamic_model_list[0], dynamic_model_list[1], dynamic_model_list[2]
        self.model_type_truth, self.model_name_truth, self.model_number_truth = truth_model_list[0], truth_model_list[1], truth_model_list[2]


    # Constraint function: ensure that time between start times is at least 1 day
    # def inequality_constraint1(self, x):
    #     l = len(x)
    #     differences = [x[4] - x[0], x[5] - x[1], x[6] - x[2], x[7] - x[3]]
    #     print([1 - diff for diff in differences] + [diff - 0.5 for diff in differences])
    #     return [1 - diff for diff in differences] + [diff - 0.5 for diff in differences]


    # def inequality_constraint2(self, x):
    #     differences = [x[5] - x[4], x[6] - x[5], x[7] - x[6]]
    #     print([5 - diff for diff in differences] + [diff - 2 for diff in differences])
    #     return [5 - diff for diff in differences] + [diff - 2 for diff in differences]


    def inequality_constraint3(self, x):
        differences = x[1] - x[0]
        # print(differences)
        return differences

    # def inequality_constraint4(self, x):
    #     return x[0] - 60390

    def penalty_function(self, x):
        penalty = self.inequality_constraint3(x)
        print("penalty_function:", penalty)
        if penalty < 0:
            penalty = max(0, 1000*abs(self.inequality_constraint3(x)))  # Quadratic penalty
        else:
            penalty = 0
        return penalty


    def objective_function(self, x):

        l = int(len(x)/2)
        # print("x: ", x, l)
        custom_station_keeping_epochs = x[l:]
        observation_windows = [(x[i], x[i+l]) for i in range(l)]
        print("observation windows: \n", observation_windows)
        # print("custom station keeping windows: \n", custom_station_keeping_epochs)

        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                       [self.model_type, self.model_name, self.model_number],
                                                                       [self.model_type_truth, self.model_name_truth, self.model_number_truth],
                                                                       exclude_first_manouvre=False,
                                                                       custom_station_keeping_epochs=custom_station_keeping_epochs
                                                                       )

        print("station keeping epochs: ", navigation_simulator.station_keeping_epochs)
        results = []
        for result_dict in navigation_simulator.perform_navigation():
            if result_dict:
                results.append(utils.convert_dictionary_to_array(result_dict))
            else: # if there is no station keeping, so empty delta_v_dict (last entry)
                results.append(([],[]))
        results.append((navigation_simulator))

        # print("RESULTS:", results)

        results_dict = {self.model_type: {self.model_name: [results]}}
        # print("RESULTS_DICT: ", results_dict)
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_formal_error_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_uncertainty_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_reference_deviation_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_estimation_error_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_full_state_history()
        # plt.show()

        delta_v = results[8][1]
        # print(delta_v)
        # print(results)
        # print("deltav: ", np.sum(np.linalg.norm(delta_v, axis=1)))
        print("delta_v: ", delta_v)
        print(f"objective value at {x}: ", np.sum(np.linalg.norm(delta_v, axis=1))+self.penalty_function(x))

        return np.sum(np.linalg.norm(delta_v, axis=1))+self.penalty_function(x)




    def optimize(self):

        # Design vector
        mission_duration = 14
        mission_start_time = 60390
        set_up_phase_time = mission_start_time + 4
        skm_obs = 4


        t_od_array = np.arange(3, mission_duration+1, skm_obs)  + mission_start_time
        for i, t in enumerate(t_od_array):
            if t < set_up_phase_time:
                t_od_array = np.delete(t_od_array, i)
        print(t_od_array)
        # t_skm_array = np.arange(4, mission_duration+1, skm_obs)
        # len_od_array = len(t_od_array)
        # len_skm_array = len(t_skm_array)
        # x0 = np.concatenate((t_od_array, t_skm_array)) + mission_start_time
        x0 = t_od_array
        print("Initial state: \n", x0)



        # Initial guess for the design vector
        # x0 = np.array([60393, 60397, 60401, 60405, 60394, 60398, 60402, 60406])
        # print(x0)

        # Define boundaries for the design vector entries
        # xl = np.array([60390, 60394, 60398, 60402, 60393, 60397, 60401, 60405])
        # xu = np.array([60396, 60400, 60404, 60408, 60395, 60399, 60403, 60407])
        # xl = x0 + np.concatenate((-0.5*np.ones(len_od_array),-1*np.ones(len_skm_array)))
        # xu = x0 + np.concatenate((0.5*np.ones(len_od_array),1*np.ones(len_skm_array)))
        xl = x0 - 0.5*np.ones(len(t_od_array))
        xu = x0 + 0.5*np.ones(len(t_od_array))
        print("Lower bound: \n", xl)
        print("Upper bound: \n", xu)
        # xl = np.array([0, 4, 8, 12, -1, -1, -1, -1])
        # xu = np.array([6, 10, 14, 18, 1, 1, 1, 1])
        # xl = [0, 7, 11, 15, -1, -1, -1, -1]
        # xu = [7, 11, 15, 18, 1, 1, 1, 1]

        # Define constraints
        constraints = (
            # {'type': 'ineq', 'fun': self.inequality_constraint1},
            # {'type': 'ineq', 'fun': self.inequality_constraint2},
            {'type': 'ineq', 'fun': self.inequality_constraint3}
        )

        # Define a callback function to record iteration history
        iteration_history = []
        def callback(xk):
            iteration_history.append(self.objective_function(xk))

        # Minimize the objective function subject to constraints
        result = sp.optimize.minimize(self.objective_function, x0,
                                    constraints=constraints,
                                    bounds=list(zip(xl, xu)),
                                    method='Nelder-Mead',
                                    options={'maxiter': 10, 'gtol': 1e-1, 'disp': True},
                                    callback=callback)

        # Extract optimized start times
        x_optim = result.x

        print("x_optim:", x_optim)

        print("iteration history: \n", iteration_history)

        plt.plot(iteration_history)
        plt.show()

        return optimized_start_times, optimized_end_times, iteration_history








dynamic_model_list = ["low_fidelity", "three_body_problem", 0]
truth_model_list = ["low_fidelity", "three_body_problem", 0]
# dynamic_model_list = ["high_fidelity", "point_mass", 0]
# truth_model_list = ["high_fidelity", "point_mass", 0]
# observation_windows = [(60393, 60394), (60397, 60398), (60401, 60402)]
# observation_windows = [(60398, 60400), (60402, 60406)]


window_optimizer = WindowOptimizer(dynamic_model_list, truth_model_list)
window_optimizer.optimize()

# window_optimizer = window_optimizer.objective_function(observation_windows)
