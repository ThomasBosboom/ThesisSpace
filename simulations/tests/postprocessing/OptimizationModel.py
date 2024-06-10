import numpy as np
import os
import sys
import copy
import scipy as sp
import json
import tracemalloc
from memory_profiler import profile

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(3):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)


class OptimizationModel:

    def __init__(self, json_settings={"save_json": True, "current_time": float, "file_name": str}, **kwargs):

        self.duration = 28
        self.arc_length = 1
        self.arc_interval = 3
        self.mission_start_epoch = 60390
        self.optimization_method = "Nelder-Mead"
        self.max_iterations = 50
        self.bounds = (-0.9, 0.9)
        self.design_vector_type = 'arc_lengths'
        self.custom_initial_design_vector = None
        self.initial_simplex_perturbation = 0.5

        self.iteration = 0
        self.iteration_history = {}
        self.intermediate_iteration_history = {}
        self.initial_objective_value = None
        self.best_objective_value = None
        self.latest_objective_value = None
        self.run_counter = 0

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.options = {'maxiter': self.max_iterations, 'disp': False, "adaptive": True}

        for key, value in json_settings.items():
            if json_settings["save_json"]:
                setattr(self, key, value)


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

        return initial_design_vector.tolist()


    def generate_initial_simplex(self, initial_design_vector):

        n = len(initial_design_vector)
        perturbations = np.eye(n) * self.initial_simplex_perturbation

        initial_simplex = [initial_design_vector]
        for i in range(n):
            vertex = initial_design_vector + perturbations[i]
            initial_simplex.append(vertex)
        initial_simplex = np.array(initial_simplex)

        return initial_simplex.tolist()


    def optimize(self, objective_function):

        def wrapped_objective(design_vector):

            # tracemalloc.start()

            observation_windows = self.generate_observation_windows(design_vector)
            objective_value = objective_function(observation_windows)

            # Initialize initial objective value
            if self.initial_objective_value is None:
                self.initial_objective_value = objective_value

            self.latest_objective_value = objective_value

            # Save all intermediate function evaluations between iterations
            if self.iteration not in self.intermediate_iteration_history:
                self.intermediate_iteration_history[self.iteration] = {}

            self.intermediate_iteration_history[self.iteration][self.run_counter] = {
                    'design_vector': design_vector.tolist(),
                    'objective_value': objective_value,
                    'reduction': (objective_value-self.initial_objective_value)/self.initial_objective_value*100
                }

            if self.iteration == 0 and self.run_counter == 0:
                self.iteration_history[self.iteration] = {
                    'design_vector': design_vector.tolist(),
                    'objective_value': objective_value,
                    'reduction': (objective_value-self.initial_objective_value)/self.initial_objective_value*100
                }

            # Update the best objective value and arc lengths for the current iteration
            if self.best_objective_value is None or objective_value < self.best_objective_value:
                self.best_objective_value = objective_value
                self.best_design_vector = np.copy(design_vector).tolist()
                self.best_observation_windows = observation_windows

            self.save_to_json()

            print(f"Function summary: \nDesign vector: {design_vector} \nObjective: {objective_value}")

            self.run_counter += 1

            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('lineno')
            # for stat in top_stats[:10]:
            #     print(stat)
            # total_memory = sum(stat.size for stat in top_stats)
            # print(f"Total memory used after iteration: {total_memory / (1024 ** 2):.2f} MB")

            return objective_value


        def callback_function(x):

            self.iteration += 1
            self.run_counter = 0

            print(f"Callback iteration {self.iteration} =================")
            print(f"Design vector: \n", self.best_design_vector)
            print(f"Objective value: \n", self.best_objective_value)
            print(f"Reduction: \n", (self.best_objective_value-self.initial_objective_value)/self.initial_objective_value*100)
            print("===========================")

            # Only save the final result of each iteration
            self.iteration_history[self.iteration] = {
                    'design_vector': self.best_design_vector,
                    'objective_value': self.best_objective_value,
                    'reduction': (self.best_objective_value-self.initial_objective_value)/self.initial_objective_value*100
                }


        # Initialize the design vector with the maximum number of arcs
        if self.custom_initial_design_vector is None:
            self.initial_design_vector = self.generate_initial_design_vector()
        else:
            self.initial_design_vector = self.custom_initial_design_vector

        # Define bounds for the design vector entries
        self.bounds_vector = [(state+self.bounds[0], state+self.bounds[1]) for state in self.generate_initial_design_vector()]

        # Adjust the initial simplex for better convergence
        initial_simplex = self.generate_initial_simplex(self.initial_design_vector)
        self.options.update({"initial_simplex": initial_simplex})

        # Define the initial observation windows
        self.initial_observation_windows = self.generate_observation_windows(self.initial_design_vector)

        # Plotting preliminary details
        print("Current time: ", self.current_time)
        print("Design vector type: \n", self.design_vector_type)
        print("Initial guess: \n", self.initial_design_vector)
        print("Initial simplex: \n", initial_simplex)
        print("Initial observation windows: \n", self.generate_observation_windows(self.initial_design_vector))
        print("Bounds: \n", self.bounds)

        result = sp.optimize.minimize(
            fun=wrapped_objective,
            callback=callback_function,
            x0=self.initial_design_vector,
            method=self.optimization_method,
            bounds=self.bounds_vector,
            options=self.options,
        )

        self.final_solution = result.x.tolist()

        print(f"Optimization Result: {result}")

        self.save_to_json()

        return self

    def save_to_json(self):

        filename=f'{self.current_time}_optimization_results.json'

        folder = os.path.join(os.path.dirname(__file__), "optimization_results")
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = os.path.join(folder, filename)
        with open(file_path, 'w') as file:
            json.dump(vars(self), file, indent=4)