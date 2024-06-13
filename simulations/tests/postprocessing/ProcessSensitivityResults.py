# Standard
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(3):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Own
from tests import utils
from tests.postprocessing import TableGenerator

class PlotSensitivityResults():

    def __init__(self, navigation_outputs_sensitivity, figure_settings={"save_figure": False, "current_time": float, "file_name": str}):

        self.navigation_outputs_sensitivity = navigation_outputs_sensitivity
        for key, value in figure_settings.items():
            if figure_settings["save_figure"]:
                setattr(self, key, value)


    def convert_key(self, key):
        words = key.split('_')
        words = [word.capitalize() for word in words]
        return ' '.join(words)


    def calculate_sensitivity_statistics(self, data, mission_start_epoch, custom_mission_start_epoch=False, evaluation_threshold=14):

        result_dict = {}

        # Iterate through each test case
        for case_type, epochs in data.items():

            if custom_mission_start_epoch:
                mission_start_epoch = float(case_type)

            # Iterate through each epoch
            epoch_stats = {}
            combined_per_run = {}
            combined_per_run_with_threshold = {}
            for epoch, runs in epochs.items():
                keys = list(runs.keys())
                values = list(runs.values())
                epoch_stats[epoch] = {'mean': np.mean(values), 'std': np.std(values)}

                for key in keys:
                    if key not in combined_per_run:
                        combined_per_run[key] = []
                        combined_per_run_with_threshold[key] = []

                    combined_per_run[key].append(runs[key])
                    if epoch >= mission_start_epoch + evaluation_threshold:
                        combined_per_run_with_threshold[key].append(runs[key])

            total = []
            total_with_threshold = []
            for run, combined in combined_per_run.items():
                total.append(np.sum(combined))
            for run, combined in combined_per_run_with_threshold.items():
                total_with_threshold.append(np.sum(combined))

            total_stats = {'mean': np.mean(total), 'std': np.std(total)}
            total_stats_with_threshold = {'mean': np.mean(total_with_threshold), 'std': np.std(total_with_threshold)}

            # Store statistics in the result dictionary
            result_dict[case_type] = {'epoch_stats': epoch_stats, 'total_stats': total_stats, 'total_stats_with_threshold': total_stats_with_threshold}

        return result_dict




    def plot_sensitivity_analysis_results(self, sensitivity_settings, evaluation_threshold=14, save_figure=True, save_table=True):

        self.save_figure = save_figure
        units = {
            "arc_duration": "[days]",
            "arc_interval": "[days]",
            "mission_start_epoch": "[MJD]",
            "initial_estimation_error": "[m]/[m/s]",
            "orbit_insertion_error": "[m]/[m/s]",
            "observation_interval": "[s]",
            "noise": "[m]",
            "target_point_epochs": "[days]",
            "delta_v_min": "[m/s]",
            "station_keeping_error": "[%]"
        }

        nrows = len(list(sensitivity_settings.items()))
        fig1, axs1 = plt.subplots(nrows, 1, figsize=(7, min(3*nrows, 10)), sharex=True)

        if not isinstance(axs1, np.ndarray):
            axs1 = np.array([axs1])
        axs1 = axs1.flatten()

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        sensitivity_statistics = {}
        for type_index, (window_type, navigation_outputs_sensitivity_types) in enumerate(self.navigation_outputs_sensitivity.items()):

            for sensitivity_type_index, (sensitivity_type, navigation_outputs_sensitivity_cases) in enumerate(navigation_outputs_sensitivity_types.items()):

                fig, axs = plt.subplots(4, 1, figsize=(7, 10), sharex=False)
                for i in range(1, 3):
                    axs[i].sharex(axs[0])

                delta_v_runs_dict_sensitivity_case = {}
                mission_start_epochs = []
                for index, navigation_outputs_sensitivity_case in enumerate(navigation_outputs_sensitivity_cases):

                    if isinstance(sensitivity_settings[sensitivity_type][index], np.ndarray):
                        sensitivity_settings[sensitivity_type][index] = f"{sensitivity_settings[sensitivity_type][index][6]}/\n{sensitivity_settings[sensitivity_type][index][9]}"

                    full_propagated_formal_errors_histories = []
                    full_reference_state_deviation_histories = []
                    full_estimation_error_histories = []
                    delta_v_runs_dict = {}
                    for run_index, (run, navigation_output) in enumerate(navigation_outputs_sensitivity_case.items()):

                        # Extracting the relevant objects
                        # navigation_results = navigation_output.navigation_results
                        navigation_simulator = navigation_output.navigation_simulator

                        # Extracting the relevant results from objects
                        for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):
                            if end_epoch in navigation_simulator.delta_v_dict.keys():

                                delta_v = np.linalg.norm(navigation_simulator.delta_v_dict[end_epoch])

                                if end_epoch not in delta_v_runs_dict:
                                    delta_v_runs_dict[end_epoch] = {}

                                delta_v_runs_dict[end_epoch][run_index] = delta_v


                            if run_index==0:

                                alpha = 0.1
                                for j in range(3):
                                    axs[j].axvspan(
                                        xmin=start_epoch-navigation_simulator.mission_start_epoch,
                                        xmax=end_epoch-navigation_simulator.mission_start_epoch,
                                        # color=color_cycle[index],
                                        color="lightgray",
                                        alpha=alpha
                                        )

                        if run_index == 0:

                            for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):
                                station_keeping_epoch = epoch - navigation_simulator.mission_start_epoch

                                alpha=0.3
                                for j in range(3):
                                    axs[j].axvline(x=station_keeping_epoch,
                                                        color='black',
                                                        linestyle='--',
                                                        alpha=alpha,
                                                        zorder=0,
                                                        label="SKM" if i==0 and index==0 else None)

                        full_propagated_formal_errors_epochs = np.stack(list(navigation_simulator.full_propagated_formal_errors_dict.keys()))
                        full_propagated_formal_errors_history = np.stack(list(navigation_simulator.full_propagated_formal_errors_dict.values()))
                        relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch
                        full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

                        full_estimation_error_epochs = np.stack(list(navigation_simulator.full_estimation_error_dict.keys()))
                        full_estimation_error_history = np.stack(list(navigation_simulator.full_estimation_error_dict.values()))
                        full_estimation_error_histories.append(full_estimation_error_history)

                        full_reference_state_deviation_epochs = np.stack(list(navigation_simulator.full_reference_state_deviation_dict.keys()))
                        full_reference_state_deviation_history = np.stack(list(navigation_simulator.full_reference_state_deviation_dict.values()))
                        full_reference_state_deviation_histories.append(full_reference_state_deviation_history)

                        axs[0].plot(relative_epochs, np.linalg.norm(full_estimation_error_history[:, 6:9], axis=1),
                                        color=color_cycle[index],
                                        alpha=0.05
                                        )

                        axs[1].plot(relative_epochs, np.linalg.norm(full_reference_state_deviation_history[:, 6:9], axis=1),
                                        color=color_cycle[index],
                                        alpha=0.05
                                        )

                    mean_full_estimation_error_histories = np.mean(np.array(full_estimation_error_histories), axis=0)
                    axs[0].plot(relative_epochs, np.linalg.norm(mean_full_estimation_error_histories[:, 6:9], axis=1),
                                    color=color_cycle[index],
                                    alpha=0.8
                                    )

                    mean_full_reference_state_deviation_histories = np.mean(np.array(full_reference_state_deviation_histories), axis=0)
                    axs[1].plot(relative_epochs, np.linalg.norm(mean_full_reference_state_deviation_histories[:, 6:9], axis=1),
                                    color=color_cycle[index],
                                    alpha=0.8
                                    )

                    delta_v_runs_dict_sensitivity_case[str(sensitivity_settings[sensitivity_type][index])] = delta_v_runs_dict

                if sensitivity_type == "mission_start_epoch":
                    custom_mission_start_epoch=True
                else:
                    custom_mission_start_epoch=False

                # Plot the station keeping costs standard deviations
                sensitivity_case_delta_v_stats = self.calculate_sensitivity_statistics(
                                                            delta_v_runs_dict_sensitivity_case,
                                                            mission_start_epoch=navigation_simulator.mission_start_epoch,
                                                            evaluation_threshold=evaluation_threshold,
                                                            custom_mission_start_epoch=custom_mission_start_epoch)
                for case_index, (sensitivity_case, delta_v_statistics) in enumerate(sensitivity_case_delta_v_stats.items()):

                    for epoch, statistics in delta_v_statistics["epoch_stats"].items():
                        relative_epoch = epoch-navigation_simulator.mission_start_epoch
                        if sensitivity_type == "mission_start_epoch":
                            relative_epoch = epoch-float(sensitivity_case)
                        axs[2].bar(relative_epoch, statistics["mean"],
                                color=color_cycle[case_index],
                                alpha=0.6,
                                width=0.2,
                                yerr=statistics["std"],
                                capsize=4,
                                label=f"{window_type}" if navigation_outputs_sensitivity_case==0 and delta_v_runs_dict_index==0 else None)

                    axs[3].barh(sensitivity_case, delta_v_statistics["total_stats"]["mean"],
                        color=color_cycle[case_index],
                        xerr=delta_v_statistics["total_stats"]["std"],
                        capsize=4,
                        label=f"{sensitivity_settings[sensitivity_type][case_index]}"
                        )

                    axs[3].barh(sensitivity_case, delta_v_statistics["total_stats_with_threshold"]["mean"],
                        color="white", hatch='/', edgecolor='black', alpha=0.6, height=0.6,
                        xerr=delta_v_statistics["total_stats_with_threshold"]["std"],
                        capsize=4,
                        label=f"After {evaluation_threshold} days" if sensitivity_case==list(sensitivity_case_delta_v_stats.keys())[-1] else None,
                        )

                    axs1[sensitivity_type_index].barh(sensitivity_case, delta_v_statistics["total_stats"]["mean"],
                        color=color_cycle[case_index],
                        xerr=delta_v_statistics["total_stats"]["std"],
                        capsize=4,
                        # label=f"{sensitivity_settings[sensitivity_type][case_index]}"
                        )

                    axs1[sensitivity_type_index].barh(sensitivity_case, delta_v_statistics["total_stats_with_threshold"]["mean"],
                        color="white", hatch='/', edgecolor='black', alpha=0.6, height=0.6,
                        xerr=delta_v_statistics["total_stats_with_threshold"]["std"],
                        capsize=4,
                        # label=f"After {evaluation_threshold} days" if sensitivity_type==list(sensitivity_settings.keys())[-1] else None,
                        label=f"After {evaluation_threshold} days" if case_index==0 else None,
                        )

                    ylabel = self.convert_key(sensitivity_type)
                    if sensitivity_type == "initial_estimation_error":
                        ylabel = "Estimation Error"
                    if sensitivity_type == "orbit_inseration_error":
                        ylabel = "Insertation Error"
                    if sensitivity_type == "observation_interval":
                        ylabel = "Obs. Interval"
                    if sensitivity_type == "target_point_epochs":
                            ylabel = "Target Points"
                    if sensitivity_type == "station_keeping_error":
                            ylabel = "SKM Error"

                    axs1[sensitivity_type_index].grid(alpha=0.5, linestyle='--', zorder=0)
                    axs1[sensitivity_type_index].set_ylabel(f"{ylabel} \n{units[sensitivity_type]}")
                    axs1[-1].set_xlabel(r"Total $||\Delta V||$ [m/s]", fontsize="small")
                    # axs1[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize="small")

                sensitivity_statistics[sensitivity_type] = sensitivity_case_delta_v_stats

                # Plot the total delta v results per sensitivity case
                axs[0].grid(alpha=0.5, linestyle='--', zorder=0)
                axs[0].set_ylabel(r"||$\hat{\mathbf{r}}-\mathbf{r}_{true}$|| [m]")
                axs[0].set_yscale("log")

                axs[1].grid(alpha=0.5, linestyle='--', zorder=0)
                axs[1].set_ylabel(r"||$\mathbf{r}_{true}-\mathbf{r}_{ref}$|| [m]")

                axs[2].set_ylabel(r"$||\Delta V||$ [m/s]")
                axs[2].grid(alpha=0.5, linestyle='--', zorder=0)
                axs[2].set_xlabel(f"Time since MJD start epoch [days]", fontsize="small")

                axs[3].grid(alpha=0.5, linestyle='--', zorder=0)
                axs[3].set_ylabel("Parameter value")
                axs[3].set_xlabel(r"Total $||\Delta V||$ [m/s]", fontsize="small")
                axs[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=len(navigation_outputs_sensitivity_cases)+1, fontsize="small")

                for ax in axs1:
                    ax.yaxis.set_label_position("left")
                    ax.yaxis.tick_right()
                    ax.legend(loc='best', fontsize="small")

                plt.tight_layout()

                if self.save_figure:
                    utils.save_figure_to_folder(figs=[fig], labels=[f"{self.current_time}_sensitivity_analysis_{sensitivity_type}"], custom_sub_folder_name=self.file_name)
                    utils.save_figure_to_folder(figs=[fig1], labels=[f"{self.current_time}_sensitivity_analysis"], custom_sub_folder_name=self.file_name)

        if save_table:

            sensitivity_statistics = {self.convert_key(key): value for key, value in sensitivity_statistics.items()}
            table_generator = TableGenerator.TableGenerator()
            table_generator.generate_sensitivity_analysis_table(
                sensitivity_statistics,
                file_name=f"{self.current_time}_sensitivity_analysis.tex"
            )
















