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
        self.num_runs = 2
        self.seed = 0

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def test(self, observation_windows):

        costs = []
        for run, seed in enumerate(range(self.seed, self.seed+self.num_runs)):
            noise = np.random.normal(0, 0.0000000001)
            cost = np.sum([tup[-1]-tup[0] for tup in observation_windows]) + noise
            costs.append(cost)
        mean_cost = np.mean(costs)
        return mean_cost


    def mean_station_keeping_cost(self, observation_windows):

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        cost_list = []
        for run, seed in enumerate(range(self.seed, self.seed+self.num_runs)):


            print(f"Run {run+1} of {self.num_runs}, seed {seed}")

            navigation_output = self.navigation_simulator.perform_navigation(observation_windows, seed=seed)
            navigation_simulator = navigation_output.navigation_simulator

            delta_v_dict = navigation_simulator.delta_v_dict
            delta_v_epochs = np.stack(list(delta_v_dict.keys()))
            delta_v_history = np.stack(list(delta_v_dict.values()))
            delta_v = sum(np.linalg.norm(value) for key, value in delta_v_dict.items() if key > navigation_simulator.mission_start_epoch+self.evaluation_threshold)

            cost_list.append(delta_v)
            navigation_simulator.reset_attributes()

            # # Take another snapshot after the function call
            snapshot2 = tracemalloc.take_snapshot()
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            total_memory = sum(stat.size for stat in top_stats)
            print(f"Total memory used after iteration: {total_memory / (1024 ** 2):.2f} MB")

        total_cost = np.mean(cost_list)

        return total_cost


    def worst_case_station_keeping_cost(self, observation_windows):

        # tracemalloc.start()
        # snapshot1 = tracemalloc.take_snapshot()

        cost_list = []
        for run, seed in enumerate(range(self.seed, self.seed+self.num_runs)):

            print(f"Run {run+1} of {self.num_runs}, seed {seed}")

            navigation_output = self.navigation_simulator.perform_navigation(observation_windows, seed=seed)
            navigation_simulator = navigation_output.navigation_simulator

            delta_v_dict = navigation_simulator.delta_v_dict
            delta_v_epochs = np.stack(list(delta_v_dict.keys()))
            delta_v_history = np.stack(list(delta_v_dict.values()))
            delta_v = sum(np.linalg.norm(value) for key, value in delta_v_dict.items() if key > navigation_simulator.mission_start_epoch+self.evaluation_threshold)

            cost_list.append(delta_v)
            navigation_simulator.reset_attributes()

            # # Take another snapshot after the function call
            # snapshot2 = tracemalloc.take_snapshot()
            # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            # total_memory = sum(stat.size for stat in top_stats)
            # print(f"Total memory used after iteration: {total_memory / (1024 ** 2):.2f} MB")

            import psutil
            print(psutil.virtual_memory())

        total_cost = np.mean(cost_list)*1+3*np.std(cost_list)

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

        navigation_simulator.reset_attributes()

        return beta_aves[0]