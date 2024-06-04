import numpy as np
import os
import sys
import copy
import scipy as sp
import json

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
        self.custom_initial_guess = None
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


    def generate_initial_guess(self):

        initial_observation_windows = []
        current_time = 0
        while current_time < self.duration:
            initial_observation_windows.append((current_time, current_time + self.arc_length))
            current_time += self.arc_length + self.arc_interval

        for arc_set in initial_observation_windows:
            if arc_set[1]+self.bounds[1] >= self.duration:
                initial_observation_windows.remove(arc_set)
                break

        initial_guess = np.ones(len(initial_observation_windows))
        if self.design_vector_type == 'arc_lengths':
            initial_guess *= self.arc_length
        if self.design_vector_type == 'arc_intervals':
            initial_guess *= self.arc_interval

        return initial_guess.tolist()


    def generate_initial_simplex(self, initial_guess):

        n = len(initial_guess)
        perturbations = np.eye(n) * self.initial_simplex_perturbation

        initial_simplex = [initial_guess]
        for i in range(n):
            vertex = initial_guess + perturbations[i]
            initial_simplex.append(vertex)
        initial_simplex = np.array(initial_simplex)

        return initial_simplex.tolist()


    def optimize(self, objective_function):

        def wrapped_objective(design_vector):

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

            self.save_to_json()

            print(f"Function summary: \nDesign vector: {design_vector} \nObjective: {objective_value}")

            self.run_counter += 1

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
        if self.custom_initial_guess is None:
            self.initial_guess = self.generate_initial_guess()
        else:
            self.initial_guess = self.custom_initial_guess

        # Define bounds for the design vector entries
        self.bounds_vector = [(state+self.bounds[0], state+self.bounds[1]) for state in self.generate_initial_guess()]

        # Adjust the initial simplex for better convergence
        initial_simplex = self.generate_initial_simplex(self.initial_guess)
        self.options.update({"initial_simplex": initial_simplex})

        # Plotting preliminary details
        print("Current time: ", self.current_time)
        print("Design vector type: \n", self.design_vector_type)
        print("Initial guess: \n", self.initial_guess)
        print("Initial simplex: \n", initial_simplex)
        print("Initial observation windows: \n", self.generate_observation_windows(self.initial_guess))
        print("Bounds: \n", self.bounds)

        result = sp.optimize.minimize(
            fun=wrapped_objective,
            callback=callback_function,
            x0=self.initial_guess,
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






# class ObjectiveFunctions():

#     def __init__(self, navigation_simulator, **kwargs):

#         self.navigation_simulator = navigation_simulator
#         self.default_navigation_simulator = copy.deepcopy(navigation_simulator)
#         self.evaluation_threshold = 14
#         self.num_runs = 2
#         self.seed = 0

#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)


#     def reset_navigation_simulator(self):
#         self.navigation_simulator.__dict__ = copy.deepcopy(self.default_navigation_simulator.__dict__)


#     def test(self, observation_windows):

#         costs = []
#         for run in range(1):
#             noise = np.random.normal(0, 0.0000001)
#             cost = np.sum([tup[-1]-tup[0] for tup in observation_windows]) + noise
#             costs.append(cost)
#         mean_cost = np.mean(costs)
#         return mean_cost


#     def station_keeping_cost(self, observation_windows):

#         cost_list = []
#         for run in range(self.num_runs):

#             print(f"Run {run+1} of {self.num_runs}")

#             navigation_output = self.navigation_simulator.perform_navigation(observation_windows, seed=self.seed)
#             navigation_results = navigation_output.navigation_results
#             navigation_simulator = navigation_output.navigation_simulator

#             delta_v_dict = navigation_simulator.delta_v_dict
#             delta_v_epochs = np.stack(list(delta_v_dict.keys()))
#             delta_v_history = np.stack(list(delta_v_dict.values()))
#             delta_v = sum(np.linalg.norm(value) for key, value in delta_v_dict.items() if key > navigation_simulator.mission_start_epoch+self.evaluation_threshold)

#             cost_list.append(delta_v)

#             self.reset_navigation_simulator()

#         total_cost = np.mean(cost_list)

#         return total_cost


#     def overall_uncertainty(self, observation_windows):

#         navigation_simulator = self.navigation_simulator.perform_navigation(observation_windows, seed=self.seed).navigation_simulator
#         covariance_dict = navigation_simulator.full_propagated_covariance_dict
#         covariance_epochs = np.stack(list(covariance_dict.keys()))
#         covariance_history = np.stack(list(covariance_dict.values()))

#         beta_aves = []
#         for i in range(2):

#             beta_bar_1 = np.mean(3*np.max(np.sqrt(np.abs(np.linalg.eigvals(covariance_history[:, 3*i+0:3*i+3, 3*i+0:3*i+3]))), axis=1))
#             beta_bar_2 = np.mean(3*np.max(np.sqrt(np.abs(np.linalg.eigvals(covariance_history[:, 3*i+6:3*i+3+6, 3*i+6:3*i+3+6]))), axis=1))
#             beta_ave = 1/2*(beta_bar_1+beta_bar_2)

#             beta_aves.append(beta_ave)

#         self.reset_navigation_simulator()

#         return beta_aves[0]

