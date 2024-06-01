import numpy as np
import os
import sys
import scipy as sp
import json
from datetime import datetime

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(3):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Define current time
current_time = datetime.now().strftime("%Y%m%d%H%M")

# Own
from src import NavigationSimulator

class OptimizationModel:
    def __init__(self, **kwargs):

        self.duration = 28
        self.arc_length = 1
        self.arc_interval = 2
        self.mission_start_epoch = 60390
        self.optimization_method = "Nelder-Mead"
        self.max_iterations = 50
        self.bounds_values = (-0.9, 0.9)

        self.iteration = 0
        self.iteration_history = {}
        self.intermediate_iteration_history = {}
        self.initial_objective_value = None

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def generate_observation_windows(self, design_vector):

        observation_windows = []
        current_time = self.mission_start_epoch
        for arc_length in design_vector:
            if current_time + arc_length > current_time + self.duration:
                arc_length = self.duration - current_time
            observation_windows.append((current_time, current_time + arc_length))
            current_time += arc_length + self.arc_interval

        return observation_windows


    def optimize(self, objective_function):

        self.options = {'maxiter': self.max_iterations, 'disp': False}
        self.best_objective_value = None
        self.latest_objective_value = None

        self.run_counter = 0
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

            # Update the best objective value and arc lengths for the current iteration
            if self.best_objective_value is None or objective_value < self.best_objective_value:
                self.best_objective_value = objective_value.tolist()
                self.best_design_vector = np.copy(design_vector).tolist()

            # print(self.iteration_history[self.iteration])
            self.save_to_json()

            print(f"Function summary: \nDesign vector: {design_vector} \nObjective: {objective_value}")

            self.run_counter += 1

            return objective_value


        def callback_function(x):

            print(f"Final result iteration {self.iteration}: ", x)

            # Only save the final result of each iteration
            self.iteration_history[self.iteration] = {
                    'design_vector': self.best_design_vector,
                    'objective_value': self.best_objective_value,
                    'reduction': (self.best_objective_value-self.initial_objective_value)/self.initial_objective_value*100
                }

            self.iteration += 1
            self.run_counter = 0

        # Define bounds for the design vector entries
        lower_bounds = [self.arc_length + self.bounds_values[0]] * int(self.duration / (self.arc_length + self.arc_interval))
        upper_bounds = [self.arc_length + self.bounds_values[1]] * int(self.duration / (self.arc_length + self.arc_interval))
        self.bounds = list(zip(lower_bounds, upper_bounds))

        # Initialize the design vector with the maximum number of arcs
        self.initial_guess = np.full(len(self.bounds), self.arc_length).tolist()
        print("Initial observation windows: ", self.generate_observation_windows(self.initial_guess))

        result = sp.optimize.minimize(
            fun=wrapped_objective,
            x0=self.initial_guess,
            method=self.optimization_method,
            bounds=self.bounds,
            options=self.options,
            callback=callback_function
        )

        self.optimization_result = result.x.tolist()

        print(f"Optimization Result: {result}")

        self.save_to_json()

    def save_to_json(self, filename=f'{current_time}_optimization_results.json'):
        folder = os.path.join(os.path.dirname(__file__), "optimization_results")
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = os.path.join(folder, filename)
        with open(file_path, 'w') as file:
            json.dump(vars(self), file, indent=4)


class ObjectiveFunctions():

    def __init__(self, navigation_simulator, **kwargs):

        self.navigation_simulator = navigation_simulator
        self.num_runs = 2
        self.seed = 0

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def test(self, observation_windows):
        return observation_windows[-1][-1]

    def station_keeping_cost(self, observation_windows):

        cost_list = []
        for run in range(self.num_runs):

            print(f"Run {run+1} of {self.num_runs}")

            # Extracting the relevant objects
            navigation_output = self.navigation_simulator.perform_navigation(observation_windows, seed=self.seed)
            navigation_results = navigation_output.navigation_results
            navigation_simulator = navigation_output.navigation_simulator

            delta_v_dict = navigation_simulator.delta_v_dict
            delta_v_epochs = np.stack(list(delta_v_dict.keys()))
            delta_v_history = np.stack(list(delta_v_dict.values()))
            delta_v = sum(np.linalg.norm(value) for key, value in delta_v_dict.items() if key > navigation_simulator.mission_start_epoch+0)
            # delta_v = np.sum(np.linalg.norm(navigation_results[8][1], axis=1)[::])
            cost_list.append(delta_v)

            self.navigation_simulator.__init__()

        total_cost = np.mean(cost_list)

        return total_cost

    def overall_uncertainty(self, observation_windows):
        navigation_simulator = self.navigation_simulator.perform_navigation(observation_windows, seed=self.seed).navigation_simulator

        covariance_dict = navigation_simulator.full_propagated_covariance_dict
        covariance_epochs = np.stack(list(covariance_dict.keys()))
        covariance_history = np.stack(list(covariance_dict.values()))

        beta_aves = []
        for i in range(2):

            beta_bar_1 = np.mean(3*np.max(np.sqrt(np.abs(np.linalg.eigvals(covariance_history[:, 3*i+0:3*i+3, 3*i+0:3*i+3]))), axis=1))
            beta_bar_2 = np.mean(3*np.max(np.sqrt(np.abs(np.linalg.eigvals(covariance_history[:, 3*i+6:3*i+3+6, 3*i+6:3*i+3+6]))), axis=1))
            beta_ave = 1/2*(beta_bar_1+beta_bar_2)

            beta_aves.append(beta_ave)

        return beta_aves[0]




# Example usage:
if __name__ == "__main__":

    auxilary_settings = {"step_size": 0.01}
    navigation_simulator = NavigationSimulator.NavigationSimulator(**auxilary_settings)

    optimization_model = OptimizationModel(duration=28, arc_length=1, arc_interval=3, max_iterations=100, optimization_method="Nelder-Mead")

    objective_functions = ObjectiveFunctions(navigation_simulator, num_runs=1)
    # optimization_model.optimize(objective_functions.test)
    optimization_model.optimize(objective_functions.station_keeping_cost)
    # optimization_model.optimize(objective_functions.overall_uncertainty)

    optimization_result = optimization_model.optimization_result

    # optimization_result = np.array([
    #             0.09999999999999998,
    #             0.6684685763424225,
    #             1.021500082597472,
    #             1.7750015326330297,
    #             0.7839964044252619,
    #             1.9,
    #             1.4526243195019104
    #         ])

    # Evaluate the results related to the final result
    observation_windows = optimization_model.generate_observation_windows(optimization_result)
    print(observation_windows)

    cost = objective_functions.station_keeping_cost(observation_windows)
    print(cost)

