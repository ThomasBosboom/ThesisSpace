# Standard
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# # Define current time
# current_time = datetime.now().strftime("%Y%m%d%H%M")

from tests.postprocessing import TableGenerator, ProcessNavigationResults
from tests.analysis import helper_functions

class ProcessOptimizationResults():

    def __init__(self, time_tag, optimization_model, save_settings={"save_table": True, "save_figure": False, "current_time": float, "file_name": str}, **kwargs):

        self.time_tag = str(time_tag)
        self.optimization_model = optimization_model
        self.optimization_results = self.optimization_model.load_from_json(time_tag)

        for key, value in save_settings.items():
            if save_settings["save_table"] or save_settings["save_figure"]:
                setattr(self, key, value)


    def plot_iteration_history(self, compare_time_tags=[]):

        # Plot the objective values over the iterations
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axs_twin = axs[0].twinx()
        marker = None

        iteration_histories = [self.optimization_results["iteration_history"]]
        for time_tag in compare_time_tags:
            optimization_results = self.optimization_model.load_from_json(time_tag)
            iteration_histories.append(optimization_results["iteration_history"])

        for iteration_history in iteration_histories:
            iterations = list(map(str, iteration_history.keys()))
            design_vectors = np.array([iteration_history[key]["design_vector"] for key in iterations])
            objective_values = np.array([iteration_history[key]["objective_value"] for key in iterations])
            reduction = np.array([iteration_history[key]["reduction"] for key in iterations])

            axs[0].plot(iterations, objective_values, marker=marker)
            axs_twin.plot(iterations, reduction, marker=marker)


        axs[0].set_ylabel(r"||$\Delta V$|| [m/s]")
        axs[0].grid(alpha=0.5, linestyle='--', which='both')
        axs_twin.set_ylabel("Reduction [%]")

        for i in range(design_vectors.shape[1]):
            axs[1].plot(iterations, design_vectors[:, i], marker=marker, label=f'$T_{i+1}$')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel("Design variables [days]")
        axs[1].grid(alpha=0.5, linestyle='--', which='both')
        axs[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize="small", title="Design variables")
        plt.tight_layout()


    def plot_optimization_result_comparison(self, show_observation_window_settings=False):

        observation_windows_settings = {
            "Default": [
                (self.optimization_model.generate_observation_windows(self.optimization_results["initial_design_vector"]), self.optimization_results["num_runs"]),
            ],
            "Optimized": [
                (self.optimization_model.generate_observation_windows(self.optimization_results["best_design_vector"]), self.optimization_results["num_runs"])
            ],
        }

        if show_observation_window_settings:
            print("Observation window settings \n:", observation_windows_settings)

        # Run the navigation routine using given settings
        auxilary_settings = {}
        navigation_outputs = helper_functions.generate_navigation_outputs(
            observation_windows_settings,
            **auxilary_settings)

        process_multiple_navigation_results = ProcessNavigationResults.PlotMultipleNavigationResults(
            navigation_outputs,
            color_cycle=["grey", "green"],
            figure_settings={"save_figure": self.save_figure,
                            "current_time": self.current_time,
                            "file_name": self.file_name
            }
        )

        evaluation_threshold = self.optimization_results["evaluation_threshold"]
        process_multiple_navigation_results.plot_uncertainty_comparison()
        process_multiple_navigation_results.plot_maneuvre_costs()
        process_multiple_navigation_results.plot_monte_carlo_estimation_error_history(evaluation_threshold=evaluation_threshold)
        process_multiple_navigation_results.plot_maneuvre_costs_bar_chart(evaluation_threshold=evaluation_threshold, bar_labeler=None)

        if self.save_table:
            table_generator = TableGenerator.TableGenerator(
                table_settings={"save_table": self.save_table,
                                "current_time": self.current_time,
                                "file_name": self.file_name})
            current_time = self.optimization_results["current_time"]
            table_generator.generate_optimization_analysis_table(
                self.optimization_results,
                file_name=f"{current_time}_optimization_analysis.tex"
            )
