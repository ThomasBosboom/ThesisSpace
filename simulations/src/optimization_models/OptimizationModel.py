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

    def __init__(self, dynamic_model_list, truth_model_list, threshold=8, duration=14, skm_to_od_duration=3, od_duration=1, bounds=(0.5, 1.5)):

        # Specify the dynamic and truth model used for the estimation arcs
        self.model_type, self.model_name, self.model_number = dynamic_model_list[0], dynamic_model_list[1], dynamic_model_list[2]
        self.model_type_truth, self.model_name_truth, self.model_number_truth = truth_model_list[0], truth_model_list[1], truth_model_list[2]

        # Timing parameters
        self.mission_start_time = 60390
        self.threshold = threshold + self.mission_start_time
        self.duration = duration + self.mission_start_time

        self.skm_to_od_duration = skm_to_od_duration
        self.od_duration = od_duration

        skm_epochs = [self.threshold]
        while skm_epochs[-1]<self.duration:
            skm_epochs.append(skm_epochs[-1] + self.skm_to_od_duration + self.od_duration)
        self.skm_epochs = skm_epochs[:-1]

        self.observation_windows = []
        for i, skm_epoch in enumerate(self.skm_epochs):
            window = (skm_epoch-self.od_duration, skm_epoch)
            if i == 0:
                window = (self.mission_start_time, self.threshold)

            self.observation_windows.append(window)

        self.vec_len = len(self.skm_epochs)-1

        self.initial_design_vector = self.od_duration*np.ones(self.vec_len)

        self.design_vector_bounds = list(zip(bounds[0]*np.ones(self.vec_len), bounds[-1]*np.ones(self.vec_len)))

        # if od_duration > skm_to_od_duration:
        #     print("Warning: OD duration is larger than time between SKMs")


    def get_updated_skm_epochs(self, x):

        new_skm_epochs = [self.threshold]
        for i, epoch in enumerate(self.skm_epochs[1:]):
            new_skm_epochs.append(new_skm_epochs[-1]+self.skm_to_od_duration+x[i])

        return new_skm_epochs


    def get_updated_observation_windows(self, x):

        new_skm_epochs = self.get_updated_skm_epochs(x)
        new_observation_windows = [(self.mission_start_time, self.threshold)]
        for i, skm_epoch in enumerate(new_skm_epochs[1:]):
            new_observation_windows.append((skm_epoch-x[i], skm_epoch))

        return new_observation_windows


    def objective_function(self, x):

        station_keeping_epochs = self.get_updated_skm_epochs(x)
        observation_windows = self.get_updated_observation_windows(x)
        target_point_epochs = [self.skm_to_od_duration+self.od_duration]
        # target_point_epochs = [7, 14]

        print("design vector in objective: \n", x)
        print("observation windows in objective: \n", observation_windows)
        print("station keeping windows in objective: \n", station_keeping_epochs)
        print("target_point_epochs in objective: \n", target_point_epochs)

        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                    [self.model_type, self.model_name, self.model_number],
                                                                    [self.model_type_truth, self.model_name_truth, self.model_number_truth],
                                                                    custom_station_keeping_epochs=station_keeping_epochs,
                                                                    target_point_epochs=target_point_epochs)

        results = []
        for result_dict in navigation_simulator.perform_navigation():
            if result_dict:
                results.append(utils.convert_dictionary_to_array(result_dict))
            else: # if there is no station keeping, so empty delta_v_dict (last entry)
                results.append(([],[]))
        results.append((navigation_simulator))

        results_dict = {self.model_type: {self.model_name: [results]}}
        # print("RESULTS_DICT: ", results_dict)
        PlotNavigationResults.PlotNavigationResults(results_dict).plot_estimation_error_history()
        PlotNavigationResults.PlotNavigationResults(results_dict).plot_uncertainty_history()
        PlotNavigationResults.PlotNavigationResults(results_dict).plot_reference_deviation_history()
        PlotNavigationResults.PlotNavigationResults(results_dict).plot_full_state_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_formal_error_history()

        # plt.show()

        delta_v = results[8][1]
        objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
        print(f"objective value at {x}: \n", delta_v, objective_value)

        return objective_value


    def optimize(self):

        # Initial guess for the design vector
        x0 = self.initial_design_vector
        print("Initial design vector: \n", x0)

        # Define a callback function to record iteration history
        iteration_history = []
        design_vectors = []

        def callback(xk):
            iteration_history.append(self.objective_function(xk))
            design_vectors.append(xk)
            callback.iteration += 1
            print("Iteration:", callback.iteration, xk)

        callback.iteration = 0

        # Minimize the objective function subject to constraints
        result = sp.optimize.minimize(self.objective_function, x0,
                                        bounds=self.design_vector_bounds,
                                        method='Nelder-Mead',
                                        options={'maxiter': 10, "return_all": True, 'disp': True},
                                        callback=callback)

        # Extract optimized start times
        x_optim = result.x
        # print("x_optim:", x_optim)
        # print("iteration history: \n", iteration_history)
        # print("design vector history: \n", np.array(design_vectors))

        plt.plot(iteration_history)
        plt.yscale("log")

        print(dict(zip(iteration_history, np.array(design_vectors))))

        return iteration_history




dynamic_model_list = ["low_fidelity", "three_body_problem", 0]
truth_model_list = ["low_fidelity", "three_body_problem", 0]
dynamic_model_list = ["high_fidelity", "point_mass", 0]
truth_model_list = ["high_fidelity", "point_mass", 0]
# dynamic_model_list = ["high_fidelity", "point_mass", 0]
# truth_model_list = ["high_fidelity", "point_mass_srp", 0]

optimization_model = OptimizationModel(dynamic_model_list, truth_model_list, duration=28, od_duration=0.5)

threshold = 7
duration = 28
od_duration = 1
delta_v_list = []
for i in [1,2,3,4]:

    optimization_model = OptimizationModel(dynamic_model_list, truth_model_list, threshold=threshold, skm_to_od_duration=i, duration=duration, od_duration=od_duration)
    x = optimization_model.initial_design_vector
    # print("initial_design_vector: ", optimization_model.initial_design_vector)
    # print("design_vector_bounds: ", optimization_model.design_vector_bounds)
    # print("observation_windows: ", optimization_model.observation_windows)

    delta_v = optimization_model.objective_function(x)
    delta_v_list.append(delta_v)
    print("ITERATION: ", x, i, delta_v)

print(delta_v_list)

plt.show()


# design_vector_bounds = optimization_model.design_vector_bounds
# design_vector = optimization_model.initial_design_vector
# print(design_vector)
# print(optimization_model.get_updated_skm_epochs(1.5*np.ones(len(design_vector_bounds))))
# print(optimization_model.get_updated_observation_windows(1.5*np.ones(len(design_vector_bounds))))
