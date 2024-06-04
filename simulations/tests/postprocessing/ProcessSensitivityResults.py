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

    def __init__(self, navigation_outputs_sensitivity, figure_settings={"save_figure": True, "current_time": float, "file_name": str}):

        self.navigation_outputs_sensitivity = navigation_outputs_sensitivity

        for key, value in figure_settings.items():
            if figure_settings["save_figure"]:
                setattr(self, key, value)


    def plot_sensitivity_analysis_results(self, sensitivity_settings, evaluation_threshold=14, save_figure=True, save_table=True):

        self.save_figure = save_figure

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        line_style_cycle = ["solid", "dashed", "dashdot"]
        sensitivity_statistics = {}
        for type_index, (window_type, navigation_outputs_sensitivity_types) in enumerate(self.navigation_outputs_sensitivity.items()):

            # sensitivity_statistics = {}
            color = color_cycle[int(type_index%len(color_cycle))]
            for sensitivity_type_index, (sensitivity_type, navigation_outputs_sensitivity_cases) in enumerate(navigation_outputs_sensitivity_types.items()):

                fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=False)
                # axs_twin = axs[0].twinx()
                for i in range(1, 3):
                    axs[i].sharex(axs[0])

                shades = color_cycle
                # shades = [mcolors.to_rgb(color)[:-1] + (l,) for l in np.linspace(0.1, 0.9, len(navigation_outputs_sensitivity_cases))]

                # shapes = np.linspace(1, 0.5, len(navigation_outputs_sensitivity_cases))
                delta_v_runs_dict_sensitivity_case = {}
                for index, navigation_outputs_sensitivity_case in enumerate(navigation_outputs_sensitivity_cases):

                    # line_style = line_style_cycle[int(case_index%len(line_style_cycle))]
                    full_propagated_formal_errors_histories = []
                    full_reference_state_deviation_histories = []
                    full_estimation_error_histories = []
                    delta_v_runs_dict = {}
                    for run_index, (run, navigation_output) in enumerate(navigation_outputs_sensitivity_case.items()):

                        # Extracting the relevant objects
                        navigation_results = navigation_output.navigation_results
                        navigation_simulator = navigation_output.navigation_simulator

                        # Extracting the relevant results from objects
                        for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):
                            if end_epoch in navigation_simulator.delta_v_dict.keys():

                                delta_v = np.linalg.norm(navigation_simulator.delta_v_dict[end_epoch])

                                if end_epoch in delta_v_runs_dict:
                                    delta_v_runs_dict[end_epoch].append(delta_v)
                                else:
                                    delta_v_runs_dict[end_epoch] = [delta_v]

                            if run_index==0:

                                alpha = 0.1
                                for j in range(3):
                                    axs[j].axvspan(
                                        xmin=start_epoch-navigation_simulator.mission_start_epoch,
                                        xmax=end_epoch-navigation_simulator.mission_start_epoch,
                                        # color=shades[index],
                                        color="lightgray",
                                        alpha=alpha,
                                        # label=f"Observation window" if window_index==0 and case_index==0 else None
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

                        full_propagated_formal_errors_epochs = navigation_results[3][0]
                        full_propagated_formal_errors_history = navigation_results[3][1]
                        relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch
                        full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

                        full_estimation_error_epochs = navigation_results[0][0]
                        full_estimation_error_history = navigation_results[0][1]
                        full_estimation_error_histories.append(full_estimation_error_history)

                        full_reference_state_deviation_epochs = navigation_results[1][0]
                        full_reference_state_deviation_history = navigation_results[1][1]
                        full_reference_state_deviation_histories.append(full_reference_state_deviation_history)



                        # axs_twin.plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 6:9], axis=1),
                        #                 color=shades[index],
                        #                 # ls=line_style,
                        #                 alpha=0.2,
                        #                 # label=f"{sensitivity_settings[sensitivity_type][index]}"
                        #                 )

                        axs[0].plot(relative_epochs, np.linalg.norm(full_estimation_error_history[:, 6:9], axis=1),
                                        color=shades[index],
                                        # ls='--',
                                        alpha=0.05
                                        )

                        axs[1].plot(relative_epochs, np.linalg.norm(full_reference_state_deviation_history[:, 6:9], axis=1),
                                        color=shades[index],
                                        # ls='--',
                                        alpha=0.05
                                        )

                    # mean_full_propagated_formal_errors_histories = np.mean(np.array(full_propagated_formal_errors_histories), axis=0)
                    # axs[2].plot(relative_epochs, 3*np.linalg.norm(mean_full_propagated_formal_errors_histories[:, 6:9], axis=1),
                    #                 color=shades[index],
                    #                 # ls=line_style,
                    #                 alpha=0.8,
                    #                 # label=f"{sensitivity_settings[sensitivity_type][index]}"
                    #                 )

                    mean_full_estimation_error_histories = np.mean(np.array(full_estimation_error_histories), axis=0)
                    axs[0].plot(relative_epochs, np.linalg.norm(mean_full_estimation_error_histories[:, 6:9], axis=1),
                                    color=shades[index],
                                    # ls=line_style,
                                    alpha=0.8,
                                    # label=f"{sensitivity_settings[sensitivity_type][index]}"
                                    )

                    mean_full_reference_state_deviation_histories = np.mean(np.array(full_reference_state_deviation_histories), axis=0)
                    axs[1].plot(relative_epochs, np.linalg.norm(mean_full_reference_state_deviation_histories[:, 6:9], axis=1),
                                    color=shades[index],
                                    # ls=line_style,
                                    alpha=0.8,
                                    # label=f"{sensitivity_settings[sensitivity_type][index]}"
                                    )

                    # Plot the station keeping costs standard deviations
                    for delta_v_runs_dict_index, (end_epoch, delta_v_runs) in enumerate(delta_v_runs_dict.items()):
                        mean_delta_v = np.mean(delta_v_runs)
                        std_delta_v = np.std(delta_v_runs)
                        axs[2].bar(end_epoch-navigation_simulator.mission_start_epoch, mean_delta_v,
                                color=shades[index],
                                alpha=0.6,
                                width=0.2,
                                yerr=std_delta_v,
                                capsize=4,
                                label=f"{window_type}" if navigation_outputs_sensitivity_case==0 and delta_v_runs_dict_index==0 else None)

                    delta_v_runs_dict_sensitivity_case[str(sensitivity_settings[sensitivity_type][index])] = delta_v_runs_dict

                print("delta_v_runs_dict_sensitivity_case", delta_v_runs_dict_sensitivity_case)
                print("condition", navigation_simulator.mission_start_epoch + evaluation_threshold)

                stats_types = {}
                stats_types_14 = {}
                for index, (type_key, type_value) in enumerate(delta_v_runs_dict_sensitivity_case.items()):

                    print("delta_v_runs_dict_sensitivity_case[type_key]", delta_v_runs_dict_sensitivity_case[type_key])

                    runs = len(list(type_value.values())[0])
                    sums = {}
                    sums_14 = {}
                    for run in range(runs):
                        value_list = []
                        value_list_14 = []
                        for key, value in delta_v_runs_dict_sensitivity_case[type_key].items():
                            value_list.append(value[run])
                            if key >= navigation_simulator.mission_start_epoch + evaluation_threshold:
                                value_list_14.append(value[run])
                                print("Appending: ", value[run])
                        sums[run] = value_list
                        sums_14[run] = value_list_14

                    for key, value in sums.items():
                        sums[key] = np.sum(value)
                    for key, value in sums_14.items():
                        sums_14[key] = np.sum(value)

                    all_values = list(sums.values())
                    stats_types[type_key] = [np.mean(all_values), np.std(all_values)]

                    all_values_14 = list(sums_14.values())
                    stats_types_14[type_key] = [np.mean(all_values_14), np.std(all_values_14)]

                    axs[3].barh(type_key, np.mean(all_values),
                        color=shades[index],
                        # width=0.2,
                        xerr=np.std(all_values),
                        capsize=4,
                        label=f"{sensitivity_settings[sensitivity_type][index]}",
                        # left=np.mean(all_values)
                        )

                    print("allvalues14:", type_key, all_values_14)
                    axs[3].barh(type_key, np.mean(all_values_14),
                        # color=shades[index],
                        color="white", hatch='/', edgecolor='black', alpha=0.6, height=0.6,
                        # width=0.2,
                        xerr=np.std(all_values_14),
                        capsize=4,
                        label=f"Last 14 days" if type_key==list(delta_v_runs_dict_sensitivity_case.keys())[-1] else None,
                        )





                sensitivity_statistics[sensitivity_type] = stats_types

                # Plot the total delta v results per sensitivity case
                # axs[0].set_ylabel("3D RSS OD error [m]")
                axs[0].grid(alpha=0.5, linestyle='--', zorder=0)
                # axs[0].set_title("3D RSS estimation error")
                axs[0].set_ylabel(r"||$\hat{\mathbf{r}}-\mathbf{r}_{true}$|| [m]")
                # axs[0].set_xlabel(f"Time since MJD start epoch [days]", fontsize="small")
                axs[0].set_yscale("log")


                axs[1].grid(alpha=0.5, linestyle='--', zorder=0)
                # axs[1].set_title("Dispersion from reference orbit")
                axs[1].set_ylabel(r"||$\mathbf{r}_{true}-\mathbf{r}_{ref}$|| [m]")
                # axs[1].set_xlabel(f"Time since MJD start epoch [days]", fontsize="small")
                # axs[1].set_yscale("log")


                axs[2].set_ylabel(r"$||\Delta V||$ [m/s]")
                axs[2].grid(alpha=0.5, linestyle='--', zorder=0)
                # axs[2].set_title("Station keeping costs")
                axs[2].set_xlabel(f"Time since MJD start epoch [days]", fontsize="small")
                # axs[2].set_yscale("log")
                # axs[3].set_yscale("log")

                axs[3].grid(alpha=0.5, linestyle='--', zorder=0)
                # axs[3].set_title("Parameter sensitivity")
                axs[3].set_ylabel("Parameter value")
                axs[3].set_xlabel(r"Total $||\Delta V||$ [m/s]", fontsize="small")
                axs[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=len(navigation_outputs_sensitivity_cases)+1, fontsize="small")

                plt.tight_layout()

                if self.save_figure:
                    utils.save_figure_to_folder(figs=[fig], labels=[f"{self.current_time}_sensitivity_analysis_{sensitivity_type}"], custom_sub_folder_name=self.file_name)

        if save_table:

            table_generator = TableGenerator.TableGenerator()
            table_generator.generate_sensitivity_analysis_table(
                sensitivity_statistics,
                file_name=f"{self.current_time}_sensitivity_analysis.tex"
                # caption="Statistical results of Monte Carlo sensitivity analysis",
                # label="tab:ResultsSensitivityAnalysis",
            )
        print(sensitivity_statistics)