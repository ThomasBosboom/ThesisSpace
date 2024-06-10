# Standard
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# # Define current time
# current_time = datetime.now().strftime("%Y%m%d%H%M")

from tests.postprocessing import TableGenerator

class ProcessOptimizationResults():

    def __init__(self, time_tag, save_settings={"save_table": True, "save_figure": True, "current_time": float, "file_name": str}, **kwargs):

        self.time_tag = str(time_tag)
        self.optimization_results = self.load_from_json()

        for key, value in save_settings.items():
            if save_settings["save_table"] or save_settings["save_figure"]:
                setattr(self, key, value)


    def load_from_json(self):
        folder = os.path.join(os.path.dirname(__file__), "optimization_results")
        filename=f'{self.time_tag}_optimization_results.json'

        file_path = os.path.join(folder, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)

        return data


    def plot_iteration_history(self):

        iteration_history = self.optimization_results["iteration_history"]

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


    def plot_improved_design(self):

        table_generator = TableGenerator.TableGenerator(
            # table_settings={
            #     "save_table": True,
            #     "current_time": self.current_time,
            #     "file_name": self.file_name
            # }
        )

        current_time = self.optimization_results["current_time"]
        table_generator.generate_optimization_analysis_table(
            self.optimization_results,
            file_name=f"{current_time}_optimization_analysis.tex"
        )


if __name__ == "__main__":
    # Example usage
    data = {
        "initial": {
            "values": [1, 1, 1, 1, 1, 1],
            "cost": 10
        },
        "final": {
            "values": [2, 5, 1, 5, 2, 6],
            "cost": 8
        }
    }

    # Generate the Overleaf table with custom caption, label, and decimals
    file_name = "design_vector_table.tex"
    overleaf_table = save_optimization_results_table(data, caption="Design vector comparison before and after optimization", label="tab:DesignVectorOptimization", file_name=file_name, decimals=4)

    # Print the Overleaf table
    print(overleaf_table)


