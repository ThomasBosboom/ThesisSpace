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
        self.default_navigation_simulator = copy.deepcopy(navigation_simulator)
        self.evaluation_threshold = 14
        self.num_runs = 2
        self.seed = 0

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def reset_navigation_simulator(self):
        self.navigation_simulator.__dict__ = copy.deepcopy(self.default_navigation_simulator.__dict__)


    def test(self, observation_windows):

        costs = []
        for run in range(1):
            noise = np.random.normal(0, 0.0000001)
            cost = np.sum([tup[-1]-tup[0] for tup in observation_windows]) + noise
            costs.append(cost)
        mean_cost = np.mean(costs)
        return mean_cost

    def station_keeping_cost(self, observation_windows):

        cost_list = []
        for run in range(self.num_runs):

            # tracemalloc.start()

            print(f"Run {run+1} of {self.num_runs}, seed {run}")

            navigation_output = self.navigation_simulator.perform_navigation(observation_windows, seed=run)
            navigation_results = navigation_output.navigation_results
            navigation_simulator = navigation_output.navigation_simulator

            delta_v_dict = navigation_simulator.delta_v_dict
            delta_v_epochs = np.stack(list(delta_v_dict.keys()))
            delta_v_history = np.stack(list(delta_v_dict.values()))
            delta_v = sum(np.linalg.norm(value) for key, value in delta_v_dict.items() if key > navigation_simulator.mission_start_epoch+self.evaluation_threshold)

            cost_list.append(delta_v)

            self.reset_navigation_simulator()

            # del navigation_output, navigation_results, navigation_simulator

            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('lineno')
            # for stat in top_stats[:10]:
            #     print(stat)
            # total_memory = sum(stat.size for stat in top_stats)
            # print(f"Total memory used after iteration: {total_memory / (1024 ** 2):.2f} MB")

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

        self.reset_navigation_simulator()

        return beta_aves[0]