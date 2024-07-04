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
from tests import utils

class ProcessOptimizationResults():

    def __init__(self, time_tag, optimization_model, save_settings={"save_table": True, "save_figure": False, "current_time": float, "file_name": str}, **kwargs):

        for key, value in save_settings.items():
            if save_settings["save_table"] or save_settings["save_figure"]:
                setattr(self, key, value)

        self.time_tag = str(time_tag)
        self.optimization_model = optimization_model
        self.optimization_results = self.optimization_model.load_from_json(time_tag, folder_name=self.file_name)



    def plot_iteration_history(self, show_design_variables=True, compare_time_tags=[], highlight_mean_only=True):

        if show_design_variables:
            fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        else:
            fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

        labels = [self.optimization_results["current_time"]]
        iteration_histories = [self.optimization_results["iteration_history"]]
        initial_design_vector = self.optimization_results["initial_design_vector"]
        if compare_time_tags:
            labels = []
            iteration_histories = []
            initial_design_vector = []
        for time_tag in compare_time_tags:
            optimization_results = self.optimization_model.load_from_json(time_tag, folder_name=self.file_name)
            iteration_histories.append(optimization_results["iteration_history"])
            labels.append(optimization_results["current_time"])
            initial_design_vector = self.optimization_results["initial_design_vector"]

        objective_values_total = []
        reduction_total = []
        iterations_total = []
        for index, iteration_history in enumerate(iteration_histories):
            iterations = list(map(str, iteration_history.keys()))
            design_vectors = np.array([iteration_history[key]["design_vector"] for key in iterations])
            objective_values = np.array([iteration_history[key]["objective_value"] for key in iterations])
            reduction = np.array([iteration_history[key]["reduction"] for key in iterations])

            objective_values_total.append(objective_values)
            reduction_total.append(reduction)
            iterations_total.append(iterations)

            marker = None
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][int(index%10)]
            label=labels[index]
            if highlight_mean_only:
                color = "lightgray"
                label = None
            axs[0].plot(iterations, objective_values, marker=marker, label=label, color=color)
            axs[1].plot(iterations, reduction, marker=marker, label=label, color=color)


        means, stds = [], []
        for histories in [iterations_total, objective_values_total, reduction_total]:

            # Determine the maximum length among all sublists
            max_length = max(len(sublist) for sublist in histories)
            array_nan = np.full((len(histories), max_length), np.nan)
            for i, sublist in enumerate(histories):
                array_nan[i, :len(sublist)] = sublist

            # Calculate column-wise mean ignoring NaNs
            means.append(np.nanmean(array_nan, axis=0))
            stds.append(np.nanstd(array_nan, axis=0))

        means = np.array(means)
        stds = np.array(stds)

        if highlight_mean_only:
            axs[0].plot(means[0], means[1], marker=marker, label="Mean", color="red")
            axs[1].plot(means[0], means[2], marker=marker, label="Mean", color="red")

        # min_length = min(len(sublist) for sublist in iterations_total)
        # objective_values_total = np.array([sublist[:min_length] for sublist in objective_values_total], dtype=np.float64)
        # reduction_total = np.array([sublist[:min_length] for sublist in reduction_total], dtype=np.float64)
        # iterations_total = np.array([sublist[:min_length] for sublist in iterations_total], dtype=np.float64)[0]



        # mean_objective_values = np.mean(objective_values_total, axis=0)
        # mean_reduction = np.mean(reduction_total, axis=0)
        # if highlight_mean_only:
        #     axs[0].plot(iterations_total, mean_objective_values, marker=marker, label="Mean", color="red")
        #     axs[1].plot(iterations_total, mean_reduction, marker=marker, label="Mean", color="red")

        axs[0].legend()
        axs[0].set_ylabel(r"||$\Delta V$|| [m/s]")
        axs[0].grid(alpha=0.5, linestyle='--', which='both')

        # axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(iteration_histories), fontsize="small", title="Design variables")
        axs[1].legend()
        axs[1].set_ylabel("Reduction [%]")
        axs[1].grid(alpha=0.5, linestyle='--', which='both')

        if show_design_variables:
            for i in range(design_vectors.shape[1]):
                axs[2].plot(iterations, design_vectors[:, i], marker=marker, label=f'$T_{i+1}$')
            axs[2].set_xlabel('Iteration')
            axs[2].set_ylabel("Design variables [days]")
            axs[2].grid(alpha=0.5, linestyle='--', which='both')
            axs[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
            axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(initial_design_vector), fontsize="small", title="Design variables")

        else:
            axs[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
            axs[1].set_xlabel('Iteration')

        plt.tight_layout()

        if self.save_figure:
            if not compare_time_tags:
                utils.save_figure_to_folder(figs=[fig], labels=[f"{self.current_time}_iteration_history"], custom_sub_folder_name=self.file_name)
            if compare_time_tags:
                utils.save_figure_to_folder(figs=[fig], labels=[f"combined_{self.current_time}_iteration_history"], custom_sub_folder_name=self.file_name)


    def plot_optimization_result_comparisons(self, case, show_observation_window_settings=False):

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
        auxilary_settings = case
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
        process_multiple_navigation_results.plot_full_state_history_comparison()
        process_multiple_navigation_results.plot_uncertainty_comparison()
        process_multiple_navigation_results.plot_maneuvre_costs()
        process_multiple_navigation_results.plot_monte_carlo_estimation_error_history(evaluation_threshold=evaluation_threshold)
        process_multiple_navigation_results.plot_maneuvre_costs_bar_chart(evaluation_threshold=evaluation_threshold, worst_case=True, bar_labeler=None)


    def tabulate_optimization_results(self):

        if self.save_table:
            table_generator = TableGenerator.TableGenerator(
                table_settings={"save_table": self.save_table,
                                "current_time": self.current_time,
                                "file_name": self.file_name})
            current_time = self.optimization_results["current_time"]
            table_generator.generate_optimization_analysis_table(
                self.optimization_results,
                file_name=f"{current_time}.tex"
            )
