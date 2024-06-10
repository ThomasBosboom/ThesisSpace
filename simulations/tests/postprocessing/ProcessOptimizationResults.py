# Standard
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)


class ProcessOptimizationResults():

    def __init__(self, time_tag, **kwargs):

        self.time_tag = str(time_tag)


    def load_from_json(self):
        folder = os.path.join(os.path.dirname(__file__), "optimization_results")
        filename=f'{self.time_tag}_optimization_results.json'

        file_path = os.path.join(folder, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)

        return data


    def plot_iteration_history(self):

        optimization_results = self.load_from_json()
        iteration_history = optimization_results["iteration_history"]

        iterations = list(map(str, iteration_history.keys()))
        design_vectors = np.array([iteration_history[key]["design_vector"] for key in iterations])
        objective_values = np.array([iteration_history[key]["objective_value"] for key in iterations])
        reduction = np.array([iteration_history[key]["reduction"] for key in iterations])

        # Plot the objective values over the iterations
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axs_twin = axs[0].twinx()
        marker = None

        axs[0].plot(iterations, objective_values, marker=marker, color='b')
        # axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel(r"||$\Delta V$|| [m/s]")
        # axs[0].set_title('Objective values')
        axs[0].grid(alpha=0.5, linestyle='--', which='both')
        # axs[0].set_ylim((min(objective_values), max(objective_values)))

        axs_twin.plot(iterations, reduction, marker=marker, color='b')
        # axs[0].set_xlabel('Iteration')
        axs_twin.set_ylabel("Reduction [%]")
        # axs_twin.set_ylim((min(reduction), max(reduction)))

        for i in range(design_vectors.shape[1]):
            axs[1].plot(iterations, design_vectors[:, i], marker=marker, label=f'$T_{i+1}$')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel("Design variables [days]")
        # axs[1].set_title('Design vector history')
        axs[1].grid(alpha=0.5, linestyle='--', which='both')
        axs[1].set_xticks(iterations[::5])

        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize="small", title="Design variables")
        plt.tight_layout()
        # plt.show()

    # def