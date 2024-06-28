import numpy as np
import os
import sys
import copy
import scipy as sp
import json
import psutil

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(3):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils

class OptimizationModel:

    def __init__(self, json_settings={"save_dict": True, "current_time": float, "file_name": str}, custom_input={}, **kwargs):

        self.duration = 28
        self.arc_length = 1
        self.arc_interval = 3
        self.mission_start_epoch = 60390
        self.optimization_method = "Nelder-Mead"
        self.max_iterations = 50
        self.bounds = (-0.9, 0.9)
        self.design_vector_type = 'arc_lengths'

        self.custom_initial_design_vector = None
        self.custom_initial_simplex = None
        self.initial_simplex_perturbation = 0.5
        self.iteration = 0
        self.iteration_history = {}
        self.intermediate_iteration_history = {}
        self.initial_objective_value = None
        self.best_objective_value = None
        self.latest_objective_value = None
        self.best_simplex = []
        self.run_counter = 0
        self.total_run_counter = 0
        self.num_runs = 1
        self.evaluation_threshold = 14

        for key, value in json_settings.items():
            setattr(self, key, value)

        self.use_custom_input = False
        if custom_input:
            for key, value in custom_input.items():
                setattr(self, key, value)
            self.use_custom_input = True

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.options = {'maxiter': self.max_iterations+1, 'disp': False, "adaptive": True}


    def load_from_json(self, time_tag, folder_name="optimization_analysis"):

        folder = os.path.join(os.path.dirname(__file__), "dicts")
        folder = os.path.join(folder, folder_name)
        filename=f'{time_tag}_optimization_analysis.json'

        file_path = os.path.join(folder, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)

        return data


    def convert_ndarray(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_ndarray(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarray(elem) for elem in obj]
        else:
            return obj


    def save_to_json(self):
        if self.save_dict:
            converted_vars = self.convert_ndarray(vars(self))
            utils.save_dict_to_folder(dicts=[converted_vars], labels=[f"{self.current_time}_optimization_analysis"], custom_sub_folder_name=self.file_name)


    def generate_observation_windows(self, design_vector):

        observation_windows = []
        current_time = self.mission_start_epoch

        if self.design_vector_type == 'arc_lengths':
            arc_interval = self.arc_interval
            for arc_length in design_vector:
                if current_time + arc_length > self.mission_start_epoch + self.duration:
                    arc_length = self.mission_start_epoch + self.duration - current_time
                observation_windows.append((current_time, current_time + arc_length))
                current_time += arc_length + arc_interval

                if current_time >= self.mission_start_epoch + self.duration:
                    break

        elif self.design_vector_type == 'arc_intervals':
            arc_length = self.arc_length
            for arc_interval in design_vector:
                end_time = current_time + arc_length
                if end_time > self.mission_start_epoch + self.duration:
                    end_time = self.mission_start_epoch + self.duration
                observation_windows.append((current_time, end_time))
                current_time = end_time + arc_interval

                if current_time >= self.mission_start_epoch + self.duration:
                    break

        else:
            raise ValueError("Invalid design_vector_type. Must be 'arc_lengths' or 'arc_intervals'.")

        return observation_windows


    def generate_initial_design_vector(self):

        initial_observation_windows = []
        current_time = 0
        while current_time < self.duration:
            initial_observation_windows.append((current_time, current_time + self.arc_length))
            current_time += self.arc_length + self.arc_interval

        for arc_set in initial_observation_windows:
            if arc_set[1]+self.bounds[1] >= self.duration:
                initial_observation_windows.remove(arc_set)
                break

        initial_design_vector = np.ones(len(initial_observation_windows))
        if self.design_vector_type == 'arc_lengths':
            initial_design_vector *= self.arc_length
        if self.design_vector_type == 'arc_intervals':
            initial_design_vector *= self.arc_interval

        return initial_design_vector


    def generate_initial_simplex(self, initial_design_vector):

        n = len(initial_design_vector)
        perturbations = np.eye(n) * self.initial_simplex_perturbation

        initial_simplex = [initial_design_vector]
        for i in range(n):
            vertex = initial_design_vector + perturbations[i]
            initial_simplex.append(vertex)
        initial_simplex = np.array(initial_simplex)

        return initial_simplex


    def generate_iteration_history_entry(self, design_vector, objective_value, initial_objective_value):
        return {
                    'design_vector': design_vector,
                    'objective_value': objective_value,
                    'reduction': (objective_value-initial_objective_value)/initial_objective_value*100
                }


    def optimize(self, objective_function):

        # def constraints(design_vector):

        #     observation_windows = self.generate_observation_windows(design_vector)
        #     if observation_windows[-1][-1]-observation_windows[0][0] > self.duration:
        #         return np.inf
        #     else:
        #         return 0

        def objective(design_vector):

            print("==============")

            observation_windows = self.generate_observation_windows(design_vector)
            objective_value = objective_function(observation_windows)

            # Initialize initial objective value
            if self.initial_objective_value is None:
                self.initial_objective_value = objective_value

            self.latest_objective_value = objective_value

            # Save all intermediate function evaluations between iterations
            if self.iteration not in self.intermediate_iteration_history:
                self.intermediate_iteration_history[self.iteration] = {}

            # self.intermediate_iteration_history[self.iteration][self.run_counter] = {
            #         'design_vector': design_vector.tolist(),
            #         'objective_value': objective_value,
            #         'reduction': (objective_value-self.initial_objective_value)/self.initial_objective_value*100
            #     }
            self.intermediate_iteration_history[self.iteration][self.run_counter] = self.generate_iteration_history_entry(design_vector, objective_value, self.initial_objective_value)

            if self.iteration == 0 and self.run_counter == 0:
                # self.iteration_history[self.iteration] = {
                #     'design_vector': design_vector.tolist(),
                #     'objective_value': objective_value,
                #     'reduction': (objective_value-self.initial_objective_value)/self.initial_objective_value*100
                # }
                self.iteration_history[self.iteration] = self.generate_iteration_history_entry(design_vector, objective_value, self.initial_objective_value)


            # Update the best objective value and arc lengths for the current iteration
            if self.best_objective_value is None or objective_value < self.best_objective_value:
                self.best_objective_value = objective_value
                self.best_design_vector = design_vector
                self.best_observation_windows = observation_windows

            print(f"Function summary: \nDesign vector: {design_vector} \nObjective: {objective_value} \nObservation windows: \n {observation_windows}")
            print("==============")

            self.run_counter += 1
            self.best_simplex.append(design_vector)
            self.best_simplex = self.best_simplex[-len(design_vector)-1:]

            self.save_to_json()

            print(psutil.virtual_memory())

            return objective_value


        def callback_function(x):

            self.iteration += 1
            self.total_run_counter += self.run_counter
            self.run_counter = 0

            print(f"Callback iteration {self.iteration} =================")
            print(f"Design vector: \n", self.best_design_vector)
            print(f"Objective value: \n", self.best_objective_value)
            print(f"Reduction: \n", (self.best_objective_value-self.initial_objective_value)/self.initial_objective_value*100)
            print("===========================")

            # Only save the final result of each iteration
            # self.iteration_history[self.iteration] = {
            #         'design_vector': self.best_design_vector,
            #         'objective_value': self.best_objective_value,
            #         'reduction': (self.best_objective_value-self.initial_objective_value)/self.initial_objective_value*100
            #     }
            self.iteration_history[self.iteration] = self.generate_iteration_history_entry(self.best_design_vector, self.best_objective_value, self.initial_objective_value)

        # Initialize the design vector with the maximum number of arcs
        initial_design_vector = self.generate_initial_design_vector()
        self.initial_design_vector = initial_design_vector.copy()
        if self.use_custom_input:
            initial_design_vector = self.best_design_vector

        # Define bounds for the design vector entries
        self.bounds_vector = [(state+self.bounds[0], state+self.bounds[1]) for state in self.generate_initial_design_vector()]

        # Adjust the initial simplex for better convergence
        self.initial_simplex = self.generate_initial_simplex(initial_design_vector)
        if self.use_custom_input:
            self.initial_simplex = self.best_simplex
        self.options.update({"initial_simplex": self.initial_simplex})

        # Define the initial observation windows
        self.initial_observation_windows = self.generate_observation_windows(initial_design_vector)

        # Plotting preliminary details
        print("===========")
        print("Current time: \n", self.current_time)
        print("Design vector type: \n", self.design_vector_type)
        print("Initial design vector: \n", initial_design_vector)
        print("Initial simplex: \n", self.initial_simplex)
        print("Initial observation windows: \n", self.initial_observation_windows)
        print("Bounds: \n", self.bounds)
        print("===========")

        # Performing the optimization itself
        result = sp.optimize.minimize(
            fun=objective,
            callback=callback_function,
            x0=initial_design_vector,
            method=self.optimization_method,
            bounds=self.bounds_vector,
            options=self.options,
        )

        self.final_solution = result.x.tolist()

        print(f"Optimization Result: {result}")

        self.save_to_json()

        return self
