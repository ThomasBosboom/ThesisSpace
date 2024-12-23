# Standard
import os
import sys
import numpy as np
import copy
import tracemalloc

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)


class ObjectiveFunctions():

    def __init__(self, navigation_simulator, **kwargs):

        self.navigation_simulator = navigation_simulator
        self.evaluation_threshold = 14
        self.num_runs = 1
        self.seed = 0

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def test(self, observation_windows):

        objective_values = []
        noises = []
        for run, seed in enumerate(range(self.seed, self.seed+self.num_runs)):
            rng = np.random.default_rng(seed=seed)
            noise = rng.normal(0, 0.1)
            noises.append(noise)
            objective_value = np.sum([tup[-1]-tup[0] for tup in observation_windows]) + noise
            objective_values.append(objective_value)
        # import time
        # time.sleep(1)
        mean_objective_value = np.mean(objective_values) + 3*np.std(objective_values)

        return mean_objective_value, noises


    def worst_case_station_keeping_cost(self, observation_windows):

        objective_values = []
        for run, seed in enumerate(range(self.seed, self.seed+self.num_runs)):

            print(f"Run {run+1} of {self.num_runs}, seed {seed}")

            navigation_output = self.navigation_simulator.perform_navigation(observation_windows, seed=seed)
            navigation_simulator = navigation_output.navigation_simulator

            delta_v_dict = navigation_simulator.delta_v_dict
            delta_v = sum(np.linalg.norm(value) for key, value in delta_v_dict.items() if key > navigation_simulator.mission_start_epoch+self.evaluation_threshold)

            objective_values.append(delta_v)
            navigation_simulator.reset_attributes()

        final_objective_value = 1*np.mean(objective_values)+3*np.std(objective_values)

        print("Final: ", final_objective_value, "Mean: ", np.mean(objective_values), "Std: ", np.std(objective_values))

        delta_v_epochs = np.stack(list(delta_v_dict.keys()))
        delta_v_history = np.stack(list(delta_v_dict.values()))
        individual_corrections = np.linalg.norm(delta_v_history, axis=1)

        return final_objective_value, individual_corrections


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

        navigation_simulator.reset_attributes()

        return beta_aves[0]


if __name__ == "__main__":

    from src import NavigationSimulator

    navigation_simulator_settings = {
        "show_corrections_in_terminal": True,
        "run_optimization_version": True
    }
    navigation_simulator = NavigationSimulator.NavigationSimulator(**navigation_simulator_settings)

    objective_functions = ObjectiveFunctions(navigation_simulator, num_runs=1, seed=0)

    observation_windows = [(60390, 60391), (60394, 60395)]

    final_objective_value, individual_corrections = objective_functions.worst_case_station_keeping_cost(observation_windows)


    print(final_objective_value, individual_corrections)



