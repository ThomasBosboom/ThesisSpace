import numpy as np
import os
import sys
import scipy as sp
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Define current time
current_time = datetime.now().strftime("%d%m%H%M")

# Own
import NavigationSimulator

class OptimizationModel:
    def __init__(self, duration=28, arc_length=1, arc_interval=3, mission_start_epoch=60390, max_iterations=20, bounds_values=(0.9, 0.9), optimization_method="Nelder-Mead"):

        self.duration = duration
        self.arc_length = arc_length
        self.arc_interval = arc_interval
        self.mission_start_epoch = mission_start_epoch
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.bounds_values = bounds_values

        self.iteration = 0
        self.iteration_history = {}
        self.intermediate_iteration_history = {}
        self.initial_objective_value = None


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

        self.options = {'maxiter': self.max_iterations, 'disp': True}
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

            print(f"Function evaluation: \nDesign vector: {design_vector} \nObjective: {objective_value}")

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
        lower_bounds = [self.arc_length - self.bounds_values[0]] * int(self.duration / (self.arc_length + self.arc_interval))
        upper_bounds = [self.arc_length + self.bounds_values[1]] * int(self.duration / (self.arc_length + self.arc_interval))
        self.bounds = list(zip(lower_bounds, upper_bounds))

        # Initialize the design vector with the maximum number of arcs
        self.initial_guess = np.full(len(lower_bounds), self.arc_length).tolist()

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

    def get_optimization_result(self):
        return self.optimization_result

    def save_to_json(self, filename=f'{current_time}_optimization_results.json'):
        folder = os.path.join(os.path.dirname(__file__), "optimization_results")
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = os.path.join(folder, filename)
        with open(file_path, 'w') as file:
            json.dump(vars(self), file, indent=4)


# Example usage:
if __name__ == "__main__":

    def test(observation_windows):
        return observation_windows[-1][-1]

    def maneuvre_cost(observation_windows, numruns=3, **kwargs):
        cost_list = []
        for run in range(numruns):
            print(f"Run {run} of {numruns}")
            navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                        orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*1)
            navigation_results = navigation_simulator.perform_navigation(seed=0).navigation_results
            delta_v = np.sum(np.linalg.norm(navigation_results[8][1], axis=1)[:-2])
            cost_list.append(delta_v)

        total_cost = np.mean(cost_list)

        return total_cost

    def overall_uncertainty(observation_windows):
        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows)
        navigation_simulator = navigation_simulator.perform_navigation(seed=0).navigation_simulator

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



    duration = 28 # Example total duration
    arc_length = 1  # Example arc length
    arc_interval = 3  # Example arc interval

    optimizer = OptimizationModel(duration, arc_length, arc_interval, max_iterations=30)
    # optimizer.optimize(maneuvre_cost)
    # # optimizer.optimize(overall_uncertainty)
    # # optimizer.optimize(test)

    # optimization_result = optimizer.get_optimization_result()
    # print(f"Optimized Arc Lengths: {optimization_result}")

    optimizer.save_to_json()

    # optimization_result = np.array([0.17596644, 1.13618278, 1.15153068, 1.01679524, 1.17737341, 1.03086586, 1.38582027])
    optimization_result = np.array([
                1.0177113702623903,
                0.9165243648479793,
                0.9889733444398163,
                0.9749791753436066,
                1.0379529362765512,
                1.0379529362765512,
                1.0379529362765512
            ])
    # print(optimization_result)

    # Evaluate the results related to the final result
    observation_windows = optimizer.generate_observation_windows(optimization_result)
    print(observation_windows)
    # cost = overall_uncertainty(observation_windows
    cost = maneuvre_cost(observation_windows, orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*1)
    print(cost)

    # 0.05929281641011152
