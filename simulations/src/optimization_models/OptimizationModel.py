# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import scipy as sp
import time

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

        self.xk = self.initial_design_vector

        # optimization parmameters
        self.factor = 2
        self.maxiter = 10


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


    def objective_function(self, x, show_directly=False):

        print("before adjustment: ", x)
        x_old = self.xk
        diff = np.array(x) - np.array(x_old)
        x = x + diff*(self.factor-1)
        print("x_old: ", x_old)
        print("diff: ", diff)
        observation_windows = self.get_updated_observation_windows(x)
        station_keeping_epochs = self.get_updated_skm_epochs(x)
        target_point_epochs = [self.skm_to_od_duration]

        print("Objective, design vector: \n", x)
        print("Objective, observation windows: \n", observation_windows)
        # print("Objective, station keeping windows: \n", station_keeping_epochs)
        # print("Objective, target_point_epochs: \n", target_point_epochs)

        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                       [self.model_type, self.model_name, self.model_number],
                                                                       [self.model_type_truth, self.model_name_truth, self.model_number_truth],
                                                                       custom_station_keeping_epochs=station_keeping_epochs,
                                                                       target_point_epochs=target_point_epochs,
                                                                       step_size=1e-2)

        navigation_results = navigation_simulator.get_navigation_results()

        # navigation_simulator.plot_navigation_results(navigation_results, show_directly=show_directly)

        delta_v = navigation_results[8][1]
        objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
        print(f"objective value at {x}: \n", delta_v, objective_value)

        return objective_value


    def optimize(self):

        # Initial guess for the design vector
        x0 = self.initial_design_vector
        print("Initial design vector: \n", x0)

        # Define a callback function to record iteration history
        iterations = []
        design_vectors = []
        objective_values = []
        def callback(xk):
            self.xk = xk
            print("Iteration:", callback.iteration, xk, self.objective_function(xk))
            iterations.append(callback.iteration)
            design_vectors.append(xk)
            objective_values.append(self.objective_function(xk))
            callback.iteration += 1

        callback.iteration = 0
        initial_simplex = x0

        print("Design vector bounds: ", self.design_vector_bounds)

        # Minimize the objective function subject to constraints
        result = sp.optimize.minimize(self.objective_function, x0,
                                        bounds=[(1-0.5/self.factor, 1+0.5/self.factor) for i, bound in enumerate(self.design_vector_bounds)],
                                        method='Nelder-Mead',
                                        options={
                                                 'maxiter': self.maxiter,
                                                 "return_all": True,
                                                 'disp': True,
                                                #  "initial_simplex": initial_simplex,
                                                 "xatol": 0.01,
                                                 "adaptive": True
                                                 },
                                        callback=callback)

        # Extract optimized start times
        x_optim = result.x
        print("x_optim:", x_optim)

        # plt.plot(objective_values)
        # plt.show()

        print(iterations, design_vectors, objective_values)

        result_dict =  {"threshold": self.threshold,
                        "skm_to_od_duration": self.skm_to_od_duration,
                        "duration": self.duration-self.mission_start_time,
                        "model":
                        {"dynamic":
                            {"model_type": self.model_type,
                             "model_name": self.model_name,
                             "model_number": self.model_number},
                         "truth":
                            {"model_type": self.model_type_truth,
                             "model_name": self.model_name_truth,
                             "model_number": self.model_number_truth}
                        },
                        "history": {iteration:
                                        {"design_vector": list(design_vectors[iteration]),
                                         "objective_function": objective_values[iteration]}
                                    for iteration in iterations},
                        "optim": {"x_optim": list(x_optim),
                                  "x_observation_windows": self.get_updated_observation_windows(x_optim),
                                  "x_skm_epochs": self.get_updated_skm_epochs(x_optim)}
                        }

        return result_dict