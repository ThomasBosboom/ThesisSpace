# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import scipy as sp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Own
from dynamic_models import NavigationSimulator, PlotNavigationResults
from tests import utils


class OptimizationModel():

    def __init__(self, dynamic_model_list, truth_model_list, set_up_phase_time=1, mission_duration=14, skm_frquency=4):

        # Managing the dynamic model specifications
        self.model_type, self.model_name, self.model_number = dynamic_model_list[0], dynamic_model_list[1], dynamic_model_list[2]
        self.model_type_truth, self.model_name_truth, self.model_number_truth = truth_model_list[0], truth_model_list[1], truth_model_list[2]

        # Timing parameters
        self.mission_start_time = 60390
        self.set_up_phase_time = set_up_phase_time + self.mission_start_time
        self.mission_duration = mission_duration
        self.skm_frquency = skm_frquency

        self.t_od_array = np.arange(3, self.mission_duration+1, self.skm_frquency)  + self.mission_start_time
        print(self.t_od_array)
        self.custom_station_keeping_epochs = np.arange(skm_frquency, mission_duration, skm_frquency) + self.mission_start_time
        print(self.custom_station_keeping_epochs)
        for i, t in enumerate(self.t_od_array):
            if t < self.set_up_phase_time:
                print(t, self.set_up_phase_time)
                self.t_od_array = np.delete(self.t_od_array, 0)
            if self.custom_station_keeping_epochs[i] < self.set_up_phase_time:
                print("skm: ", self.custom_station_keeping_epochs[i])
                self.custom_station_keeping_epochs = np.delete(self.custom_station_keeping_epochs, 0)

        print(self.t_od_array, self.custom_station_keeping_epochs)

        if len(self.t_od_array) >= 1:
            if self.custom_station_keeping_epochs[0] < self.t_od_array[0]:
                self.custom_station_keeping_epochs = self.custom_station_keeping_epochs[1:]
            if self.custom_station_keeping_epochs[-1] > self.mission_duration + self.mission_start_time:
                self.custom_station_keeping_epochs = self.custom_station_keeping_epochs[:-1]
        else:


        print(self.t_od_array)
        print(self.custom_station_keeping_epochs)


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


    # def inequality_constraint3(self, x):
    #     differences = x[1] - x[0]
    #     # print(differences)
    #     return differences


    # def penalty_function(self, x):
    #     penalty = self.inequality_constraint3(x)
    #     print("penalty_function:", penalty)
    #     if penalty < 0:
    #         penalty = max(0, 1000*abs(self.inequality_constraint3(x)))  # Quadratic penalty
    #     else:
    #         penalty = 0
    #     return penalty


    def objective_function(self, x):

        observation_windows = [(self.mission_start_time, self.set_up_phase_time)]
        observation_windows.extend([(x[i], self.custom_station_keeping_epochs[i]) for i in range(len(x))])
        print("observation windows in objective: \n", observation_windows)
        # print("custom station keeping windows: \n", custom_station_keeping_epochs)

        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                       [self.model_type, self.model_name, self.model_number],
                                                                       [self.model_type_truth, self.model_name_truth, self.model_number_truth],
                                                                       custom_station_keeping_epochs=self.custom_station_keeping_epochs
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


    def objective_function1(self, x):

        print("--------------")
        print(x)

        return (x - 60393.141234234)**2




    def optimize(self, a, b):

        # Initial guess for the design vector
        x0 = self.t_od_array
        print("Initial state: \n", x0)

        # Define boundaries for the design vector entries
        xl = x0 + a*np.ones(len(x0))
        xu = x0 + b*np.ones(len(x0))
        print("Lower bound: \n", xl)
        print("Upper bound: \n", xu)

        # Define constraints
        # constraints = (
            # {'type': 'ineq', 'fun': self.inequality_constraint1},
            # {'type': 'ineq', 'fun': self.inequality_constraint2},
            # {'type': 'ineq', 'fun': self.inequality_constraint3}
        # )

        # Define a callback function to record iteration history
        iteration_history = []
        design_vectors = []
        def callback(xk):
            iteration_history.append(self.objective_function(xk))
            design_vectors.append(xk)

        # Minimize the objective function subject to constraints
        result = sp.optimize.minimize(self.objective_function, x0,
                                    # constraints=constraints,
                                    bounds=list(zip(xl, xu)),
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
        # plt.show()

        return iteration_history, np.array(design_vectors)


dynamic_model_list = ["low_fidelity", "three_body_problem", 0]
truth_model_list = ["low_fidelity", "three_body_problem", 0]
# dynamic_model_list = ["high_fidelity", "point_mass", 0]
# truth_model_list = ["high_fidelity", "point_mass", 0]

optimization_model = OptimizationModel(dynamic_model_list, truth_model_list,
                                set_up_phase_time=4, mission_duration=6, skm_frquency=4)

iteration_history, design_vectors = optimization_model.optimize(-1, 0.5)
print(iteration_history, design_vectors)

dictionary = dict()
for a in [-1, -0.5, -0.25]:
    for b in [0.25, 0.5, 0.75]:

        iteration_history, design_vectors = optimization_model.optimize(a, b)
        dictionary[f'({a}, {b})'] = [iteration_history, design_vectors]


print(dictionary)
plt.show()