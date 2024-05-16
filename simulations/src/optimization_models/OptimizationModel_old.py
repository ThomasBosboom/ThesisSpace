# General imports
import numpy as np
import os
import sys
import scipy as sp
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Own
import NavigationSimulator
from tests import utils

class OptimizationModel():

    def __init__(self, dynamic_model_list,
                       truth_model_list,
                       threshold=8,
                       duration=14,
                       skm_to_arc_duration=3,
                       arc_duration=1,
                       bounds=(0.5, 1.5),
                       custom_station_keeping_error=None,
                       custom_initial_estimation_error=None,
                       custom_apriori_covariance=None,
                       custom_orbit_insertion_error=None,
                       mission_start_epoch=60390):

        # Specify the dynamic and truth model used for the estimation arcs
        self.model_type, self.model_name, self.model_number = dynamic_model_list[0], dynamic_model_list[1], dynamic_model_list[2]
        self.model_type_truth, self.model_name_truth, self.model_number_truth = truth_model_list[0], truth_model_list[1], truth_model_list[2]

        # Timing parameters
        self.mission_start_epoch = mission_start_epoch
        self.threshold = threshold + self.mission_start_epoch
        self.duration = duration + self.mission_start_epoch

        self.skm_to_arc_duration = skm_to_arc_duration
        self.arc_duration = arc_duration

        skm_epochs = [self.threshold]
        while skm_epochs[-1]<self.duration:
            skm_epochs.append(skm_epochs[-1] + self.skm_to_arc_duration + self.arc_duration)
        self.skm_epochs = skm_epochs[:-1]

        self.observation_windows = []
        for i, skm_epoch in enumerate(self.skm_epochs):
            window = (skm_epoch-self.arc_duration, skm_epoch)
            if i == 0:
                window = (self.mission_start_epoch, self.threshold)

            self.observation_windows.append(window)

        self.vec_len = len(self.skm_epochs)-1
        self.initial_design_vector = self.arc_duration*np.ones(self.vec_len)
        self.design_vector_bounds = list(zip(bounds[0]*np.ones(self.vec_len), bounds[-1]*np.ones(self.vec_len)))
        self.xk = self.initial_design_vector

        # optimization parmameters
        self.factor = 1
        self.maxiter = 10

        # Error parameters for SKMs
        self.custom_station_keeping_error = custom_station_keeping_error
        self.custom_initial_estimation_error = custom_initial_estimation_error
        self.custom_apriori_covariance = custom_apriori_covariance
        self.custom_orbit_insertion_error = custom_orbit_insertion_error


    def get_updated_skm_epochs(self, x):

        new_skm_epochs = [self.threshold]
        for i, epoch in enumerate(self.skm_epochs[1:]):
            new_skm_epochs.append(new_skm_epochs[-1]+self.skm_to_arc_duration+x[i])

        return new_skm_epochs


    def get_updated_observation_windows(self, x):

        new_skm_epochs = self.get_updated_skm_epochs(x)
        new_observation_windows = [(self.mission_start_epoch, self.threshold)]
        for i, skm_epoch in enumerate(new_skm_epochs[1:]):
            new_observation_windows.append((skm_epoch-x[i], skm_epoch))

        return new_observation_windows


    # def get_adjusted_design_vector(self, x):

    #     diff = np.array(x) - np.array(self.xk)
    #     x = x + diff*(self.factor-1)

    #     return x


    def objective_function(self, x, plot_results=False, show_directly=False):

        print("Start of objective calculation ===============")

        # x = self.get_adjusted_design_vector(x)
        observation_windows = self.get_updated_observation_windows(x)
        station_keeping_epochs = self.get_updated_skm_epochs(x)
        target_point_epochs = [self.skm_to_arc_duration]

        print("Objective, design vector: \n", x)
        print("Objective, observation windows: \n", observation_windows)
        print("Objective, station keeping windows: \n", station_keeping_epochs)
        print("Objective, target_point_epochs: \n", target_point_epochs)

        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                       [self.model_type, self.model_name, self.model_number],
                                                                       [self.model_type_truth, self.model_name_truth, self.model_number_truth],
                                                                       step_size=1e-2,
                                                                       station_keeping_epochs=station_keeping_epochs,
                                                                       target_point_epochs=target_point_epochs,
                                                                       custom_station_keeping_error=self.custom_station_keeping_error,
                                                                       custom_initial_estimation_error=self.custom_initial_estimation_error,
                                                                       custom_apriori_covariance=self.custom_apriori_covariance,
                                                                       custom_orbit_insertion_error=self.custom_orbit_insertion_error,
                                                                       mission_start_epoch=self.mission_start_epoch,
                                                                       )

        navigation_results = navigation_simulator.perform_navigation().navigation_results

        delta_v = navigation_results[8][1]
        objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
        print(f"Objective: \n", delta_v, objective_value, observation_windows[-1][-1]-observation_windows[0][0])
        print("End of objective calculation ===============")

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
        callback(x0)

        print("Design vector bounds: ", self.design_vector_bounds)

        # Minimize the objective function subject to constraints
        start_time = time.time()
        result = sp.optimize.minimize(self.objective_function, x0,
                                        bounds=[(1-0.5/self.factor, 1+0.5/self.factor) for i, bound in enumerate(self.design_vector_bounds)],
                                        method='Nelder-Mead',
                                        options={
                                                 'maxiter': self.maxiter,
                                                 "return_all": True,
                                                 'disp': True,
                                                 "adaptive": True
                                                 },
                                        callback=callback)
        run_time = time.time()-start_time

        x_optim = result.x

        result_dict =  {"threshold": self.threshold,
                        "skm_to_arc_duration": self.skm_to_arc_duration,
                        "duration": self.duration-self.mission_start_epoch,
                        "factor": self.factor,
                        "maxiter": self.maxiter,
                        "initial_design_vector": list(self.initial_design_vector),
                        "station_keeping_noise": self.custom_station_keeping_error,
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
                        "history": {"design_vector": {iteration: list(design_vectors[iteration]) for iteration in iterations},
                                    "objective_value": {iteration: objective_values[iteration] for iteration in iterations}},
                        "final_result": {"x_optim": list(x_optim),
                                        "observation_windows": self.get_updated_observation_windows(x_optim),
                                        "skm_epochs": self.get_updated_skm_epochs(x_optim),
                                        "approx_annual_deltav": objective_values[-1]*365/(self.duration-self.mission_start_epoch),
                                        "reduction_percentage": (objective_values[-1]-objective_values[0])/objective_values[0]*100,
                                        "run_time": run_time}
                        }

        return result_dict