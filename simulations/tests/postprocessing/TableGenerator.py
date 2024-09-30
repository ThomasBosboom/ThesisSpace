# Standard
import os
import sys
import numpy as np
import re

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils

class TableGenerator():

    def __init__(self, table_settings={"save_table": True, "current_time": float, "file_name": str}):

        for key, value in table_settings.items():
            setattr(self, key, value)


    def escape_tex_symbols(self, string):
        escape_chars = {'%': '\\%', '&': '\\&', '_': '\\_', '#': '\\#', '$': '\\$', '{': '\\{', '}': '\\}'}
        return re.sub(r'[%&_#${}]', lambda match: escape_chars[match.group(0)], string)


    def generate_sensitivity_analysis_table(self, sensitivity_statistics, caption="Statistical results of Monte Carlo sensitivity analysis", label="tab:SensitivityAnalysis", file_name="sensitivity_analysis.tex", decimals=4, include_worst_case=True):

        table_str = ''
        table_str += r'\begin{table}[H]' + '\n'
        table_str += r'\centering' + '\n'
        table_str += r'\begin{tabular}{lllllllll}' + '\n'  # Added an extra column
        table_str += r' &  & \cellcolor[HTML]{EFEFEF}\textbf{Total} &  & \cellcolor[HTML]{EFEFEF}\textbf{After 14} & & \cellcolor[HTML]{EFEFEF}\textbf{Annual} & &\\' + '\n'
        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\textbf{Case} & \textbf{Value} & \textbf{$\mu_{\Delta \boldsymbol{V}}$} & \textbf{$\sigma_{\Delta \boldsymbol{V}}$} & \textbf{$\mu_{\Delta \boldsymbol{V}}$} & \textbf{$\sigma_{\Delta \boldsymbol{V}}$} & \textbf{$\mu_{\Delta \boldsymbol{V}}$} & \textbf{$\sigma_{\Delta \boldsymbol{V}}$} & \textbf{Worst} \\ ' + '\n'

        for case, case_data in sensitivity_statistics.items():
            is_first_row = True
            for subkey, stats in case_data.items():
                full_stats = stats.get('total_stats', {})
                threshold_stats = stats.get('total_stats_with_threshold', {})
                annual_stats = stats.get('total_annual_stats_with_threshold', {})

                if is_first_row:
                    table_str += f'{case} & '
                    is_first_row = False
                else:
                    table_str += ' & '

                table_str += f"{subkey} & " + \
                            f"{full_stats.get('mean', 0):.{decimals}f} & " + \
                            f"{full_stats.get('std', 0):.{decimals}f} & " + \
                            f"{threshold_stats.get('mean', 0):.{decimals}f} & " + \
                            f"{threshold_stats.get('std', 0):.{decimals}f} & " + \
                            f"{annual_stats.get('mean', 0):.{decimals}f} & " + \
                            f"{annual_stats.get('std', 0):.{decimals}f} & " + \
                            f"{annual_stats.get('mean', 0)+3*annual_stats.get('std', 0):.{decimals}f}" + r' \\ ' + '\n'

                print(f"{annual_stats.get('mean', 0):.{decimals}f} & ", f"{annual_stats.get('std', 0):.{decimals}f} & ")

        table_str += r'\end{tabular}' + '\n'
        table_str += r'\caption{' + caption + '}' + '\n'
        table_str += r'\label{' + label + '}' + '\n'
        table_str += r'\end{table}'

        if self.save_table:
            utils.save_table_to_folder(tables=[table_str], labels=[f"{self.current_time}_sensitivity_analysis"], custom_sub_folder_name=self.file_name)

        # print(table_str, sensitivity_statistics)

        print(f"LaTeX table code has been written")


    def generate_optimization_analysis_table(self, optimization_results, caption="Results of optimization", label="tab:OptimizationAnalysis", file_name='optimization_analysis.tex', decimals=4):

        table_str = ""
        table_str += r'\begin{table}[H]' + '\n'
        table_str += r'\centering' + '\n'
        table_str += r'\begin{tabular}{llll}' + '\n'
        table_str += r'\textbf{}      & \cellcolor[HTML]{EFEFEF}\textbf{Vectors} & \textbf{} & \textbf{}         \\' + '\n'
        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\textbf{Entry} & \textbf{Initial} & \textbf{Optimized} & \textbf{Difference} \\' + '\n'

        states = [f"T_{i+1}" for i in range(len(optimization_results["initial_design_vector"]))]
        initial_values = optimization_results["initial_design_vector"]
        final_values = optimization_results["best_design_vector"]

        for state, initial, final in zip(states, initial_values, final_values):
            if initial != 0:
                percentage_diff = ((final - initial) / initial) * 100
            else:
                percentage_diff = 0  # Handle division by zero case
            table_str += f"${state}$ & {initial:.{decimals}f} & {final:.{decimals}f} & {round(percentage_diff, 2)}\%" + r' \\ ' + '\n'

        initial_cost = optimization_results["initial_objective_value"]
        final_cost = optimization_results["best_objective_value"]

        initial_cost = optimization_results["initial_objective_value"]*365/(optimization_results["duration"]-optimization_results["evaluation_threshold"])
        final_cost = optimization_results["best_objective_value"]*365/(optimization_results["duration"]-optimization_results["evaluation_threshold"])

        initial_tracking_time = sum([tup[1] - tup[0] for tup in optimization_results["initial_observation_windows"]])
        tracking_time = sum([tup[1] - tup[0] for tup in optimization_results["best_observation_windows"]])
        difference_tracking_time = (tracking_time - initial_tracking_time) / initial_tracking_time * 100

        if initial_cost != 0:
            cost_percentage_diff = ((final_cost - initial_cost) / initial_cost) * 100
        else:
            cost_percentage_diff = 0  # Handle division by zero case

        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += f"\\textbf{{Annual SKM cost (m/s)}}  & {initial_cost:.{decimals}f} & {final_cost:.{decimals}f} & {round(cost_percentage_diff, 2)}\%" + r' \\ ' + '\n'

        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += f"\\textbf{{Tracking time (days)}}  & {initial_tracking_time:.{decimals}f} & {tracking_time:.{decimals}f} & {round(difference_tracking_time, 2)}\%" + r' \\ ' + '\n'

        table_str += r'\end{tabular}' + '\n'
        table_str += r'\caption{' + caption + '}' + '\n'
        table_str += r'\label{' + label + '}' + '\n'
        table_str += r'\end{table}'

        if self.save_table:
            utils.save_table_to_folder(tables=[table_str], labels=[f"{self.current_time}_optimization_analysis"], custom_sub_folder_name=self.file_name)


    def generate_statistics_table(self, optimization_results_list, caption="Statistics of optimization results", label="tab:StatisticsOptimizationAnalysis", file_name='optimization_analysis.tex', decimals=4):

        table_str = ""
        table_str += r'\begin{table}[H]' + '\n'
        table_str += r'\centering' + '\n'
        table_str += r'\begin{tabular}{llllllllll}' + '\n'
        # table_str += r'\textbf{} & & \cellcolor[HTML]{EFEFEF}\textbf{Optimized} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF} \\' + '\n'
        table_str += r'\textbf{} & \textbf{} & \cellcolor[HTML]{EFEFEF}\textbf{Results} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF}\textbf{\% Diff} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF} & \cellcolor[HTML]{EFEFEF} \\' + '\n'
        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\textbf{Entry} & \textbf{Initial} & \textbf{Min} & \textbf{Max} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} & \textbf{Mean} & \textbf{Std} \\' + '\n'

        num_entries = len(optimization_results_list[0]["initial_design_vector"])

        caption = caption + f". n={len(optimization_results_list)}"

        for i in range(num_entries):
            state = "$T_"+str({i+1})+"$"
            initial_value = optimization_results_list[0]["initial_design_vector"][i]
            optimized_values = [results["best_design_vector"][i] for results in optimization_results_list]
            mean_value = np.mean(optimized_values)
            std_value = np.std(optimized_values)
            min_value = min(optimized_values)
            max_value = max(optimized_values)
            differences = [(final - initial_value) / initial_value * 100 if initial_value != 0 else 0 for final in optimized_values]
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            min_diff = min(differences)
            max_diff = max(differences)

            table_str += f"{state} & {initial_value:.{decimals}f} & {min_value:.{decimals}f} & {max_value:.{decimals}f} & {mean_value:.{decimals}f} & {std_value:.{decimals}f} & "
            table_str += f"{min_diff:.{decimals}f} & {max_diff:.{decimals}f} & {mean_diff:.{decimals}f} & {std_diff:.{decimals}f}" + r' \\ ' + '\n'

        initial_cost = optimization_results_list[0]["initial_objective_value"]
        final_costs = [results["best_objective_value"] for results in optimization_results_list]
        cost_differences = [(final - initial_cost) / initial_cost * 100 if initial_cost != 0 else 0 for final in final_costs]

        initial_cost = initial_cost*365/(optimization_results_list[0]["duration"]-optimization_results_list[0]["evaluation_threshold"])
        final_costs = [results["best_objective_value"]*365/(results["duration"]-results["evaluation_threshold"]) for results in optimization_results_list]
        cost_differences = [(final - initial_cost) / initial_cost * 100 if initial_cost != 0 else 0 for final in final_costs]

        mean_cost_diff = np.mean(cost_differences)
        std_cost_diff = np.std(cost_differences)
        min_cost_diff = min(cost_differences)
        max_cost_diff = max(cost_differences)

        mean_cost = np.mean(final_costs)
        std_cost = np.std(final_costs)
        min_cost = min(final_costs)
        max_cost = max(final_costs)

        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\textbf{Annual SKM cost (m/s)} & ' + f"{initial_cost:.{decimals}f} & {min_cost:.{decimals}f} & {max_cost:.{decimals}f} & {mean_cost:.{decimals}f} & {std_cost:.{decimals}f} & "
        table_str += f"{min_cost_diff:.{decimals}f} & {max_cost_diff:.{decimals}f} & {mean_cost_diff:.{decimals}f} & {std_cost_diff:.{decimals}f}" + r' \\ ' + '\n'

        table_str += r'\end{tabular}' + '\n'
        table_str += r'\caption{' + caption + '}' + '\n'
        table_str += r'\label{' + label + '}' + '\n'
        table_str += r'\end{table}'

        if self.save_table:
            utils.save_table_to_folder(tables=[table_str], labels=[file_name], custom_sub_folder_name=self.file_name)


    def generate_design_vector_table(self, optimization_results_list, caption="Design vector entries", label="tab:DesignVectorEntries", file_name='design_vector_entries.tex', decimals=4):
        table_str = ""
        table_str += r'\begin{table}[H]' + '\n'
        table_str += r'\centering' + '\n'

        # Determine the number of runs
        num_runs = len(optimization_results_list)

        # Header row for Vectors
        header_row1 = r'\textbf{} & \cellcolor[HTML]{EFEFEF}\textbf{Vectors}'
        header_row1 += r' & \textbf{}' * (num_runs - 1) + r' & \textbf{} \\' + '\n'

        # Header row for Entries
        header_row2 = r'\rowcolor[HTML]{EFEFEF} ' + r'\cellcolor[HTML]{EFEFEF}\textbf{Entry} & \cellcolor[HTML]{EFEFEF}\textbf{Initial}'
        for i in range(1, num_runs + 1):
            header_row2 += r' & \cellcolor[HTML]{EFEFEF}\textbf{Run ' + f'{i}' + r'}'
        header_row2 += r' \\' + '\n'

        # Combine header rows
        table_str += r'\begin{tabular}{l' + 'l' * (num_runs + 1) + '}' + '\n'
        table_str += header_row1 + header_row2

        num_entries = len(optimization_results_list[0]["initial_design_vector"])

        for i in range(num_entries):
            state = "$T_"+str({i+1})+"$"
            initial_value = optimization_results_list[0]["initial_design_vector"][i]
            run_values = [optimization_results_list[j]["best_design_vector"][i] for j in range(num_runs)]

            # Add row for each entry in design vector
            table_str += f"{state} & {initial_value:.{decimals}f} & " + " & ".join([f"{value:.{decimals}f}" for value in run_values]) + r' \\' + '\n'

        initial_cost = optimization_results_list[0]["initial_objective_value"]
        initial_cost_annual = initial_cost*365/(optimization_results_list[0]["duration"]-optimization_results_list[0]["evaluation_threshold"])
        # initial_cost = optimization_results_list[0]["initial_objective_value"]

        final_costs = [results["best_objective_value"] for results in optimization_results_list]
        final_costs_annual = [results["best_objective_value"]*365/(results["duration"]-results["evaluation_threshold"]) for results in optimization_results_list]

        cost_differences = [(final - initial_cost) / initial_cost * 100 if initial_cost != 0 else 0 for final in final_costs]
        cost_differences_annual = [(final - initial_cost_annual) / initial_cost_annual * 100 if initial_cost_annual != 0 else 0 for final in final_costs_annual]

        initial_tracking_times = [sum([tup[1] - tup[0] for tup in results["initial_observation_windows"]]) for results in optimization_results_list]
        tracking_times = [sum([tup[1] - tup[0] for tup in results["best_observation_windows"]]) for results in optimization_results_list]
        difference_tracking_times = [(tracking_times[i] - initial_tracking_time) / initial_tracking_time * 100 if initial_tracking_time != 0 else 0 for i, initial_tracking_time in enumerate(initial_tracking_times)]


        # # Add cost row
        # table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        # simulations\tests\analysis\optimization_analysis\optimization_analysis_comparison.pytable_str += r'\cellcolor[HTML]{EFEFEF}\textbf{Annual SKM cost (m/s)} & \cellcolor[HTML]{EFEFEF}' + f"{initial_cost:.{decimals}f} & " + " & ".join([f"{cost:.{decimals}f}" for cost in final_costs]) + r' \\' + '\n'

        # # Add percentage difference row
        # table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        # table_str += r'\cellcolor[HTML]{EFEFEF}\textbf{\%Diff} & \cellcolor[HTML]{EFEFEF} 00.00\% &' + " & ".join([f"{diff:.2f}\\%" for diff in cost_differences]) + r' \\' + '\n'

        # Add cost row
        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\cellcolor[HTML]{EFEFEF}\textbf{Annual $\Delta \boldsymbol{V}$ (m/s)} & \cellcolor[HTML]{EFEFEF}' + f"{initial_cost_annual:.{decimals}f} & " + " & ".join([f"{cost:.{decimals}f}" for cost in final_costs_annual]) + r' \\' + '\n'

        # Add percentage difference row
        # table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\% Change & 00.00\% &' + " & ".join([f"{diff:.2f}\\%" for diff in cost_differences]) + r' \\' + '\n'

        # Add tracking time row
        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\cellcolor[HTML]{EFEFEF}\textbf{Tracking time (days)} & \cellcolor[HTML]{EFEFEF}' + f"{initial_tracking_times[0]:.{decimals}f} & " + " & ".join([f"{tracking_time:.{decimals}f}" for tracking_time in tracking_times]) + r' \\' + '\n'

        # Add percentage difference row
        # table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\% Change & 00.00\% &' + " & ".join([f"{diff:.2f}\\%" for diff in difference_tracking_times]) + r' \\' + '\n'


        # Calculate the average power consumption for thruster and transponder
        def calculate_energy_power(x_days, powers, fractions):

            # Total time in seconds per day and total tracking time
            seconds_per_day = 86400  # seconds in one day
            total_time_seconds = x_days * seconds_per_day

            # Energy consumed in one 300-second cycle
            energies = []
            for i, power in enumerate(powers):
                energies.append(power*fractions[i]*total_time_seconds)

            # Average power consumption over total tracking time
            total_energy = sum(energies)
            average_power = total_energy/total_time_seconds

            return total_energy, average_power

        # Defining the power related settings
        initial_average_power_transponder_list = []
        initial_average_power_thruster_list = []
        initial_average_power_obc_list = []
        initial_total_average_power_list = []
        average_power_transponder_list = []
        average_power_thruster_list = []
        average_power_obc_list = []
        total_average_power_list = []
        power_difference_list = []
        for results in optimization_results_list:

            duration = results["duration"]
            initial_tracking_time = sum([tup[1] - tup[0] for tup in results["initial_observation_windows"]])
            tracking_time = sum([tup[1] - tup[0] for tup in results["best_observation_windows"]])
            total_heatups = len(results["best_observation_windows"])

            # Settings
            signal_time = 6
            signal_interval = 300
            seconds_per_day = 86400
            heatup_time = 2*60*60

            # Initial results
            initial_total_signals = initial_tracking_time/(signal_interval/seconds_per_day)
            initial_total_signal_time = initial_total_signals*signal_time/seconds_per_day
            fraction_off = 1-initial_tracking_time/duration
            fraction_tracking = initial_tracking_time/duration
            fraction_signals = initial_total_signal_time/duration

            initial_total_energy_transponder, initial_average_power_transponder = calculate_energy_power(duration, [0, 7.4, 94.4], [fraction_off, fraction_tracking, fraction_signals])
            initial_average_power_transponder_list.append(initial_average_power_transponder)

            total_heatup_time = total_heatups*heatup_time/seconds_per_day
            initial_total_energy_thruster, initial_average_power_thruster = calculate_energy_power(duration, [0, 10], [1-total_heatup_time/duration, total_heatup_time/duration])
            initial_average_power_thruster_list.append(initial_average_power_thruster)

            initial_total_energy_obc, initial_average_power_obc = calculate_energy_power(duration, [0, 1.3], [0, 1])
            initial_average_power_obc_list.append(initial_average_power_obc)

            initial_total_average_power = initial_average_power_transponder+initial_average_power_thruster+initial_average_power_obc
            initial_total_average_power_list.append(initial_total_average_power)

            # Updated version
            total_signals = tracking_time/(signal_interval/seconds_per_day)
            total_signal_time = total_signals*signal_time/seconds_per_day
            fraction_off = 1-tracking_time/duration
            fraction_tracking = tracking_time/duration
            fraction_signals = total_signal_time/duration

            total_energy_transponder, average_power_transponder = calculate_energy_power(duration, [0, 7.4, 94.4], [fraction_off, fraction_tracking, fraction_signals])
            average_power_transponder_list.append(average_power_transponder)

            total_energy_thruster, average_power_thruster = calculate_energy_power(duration, [0, 10], [1-total_heatup_time/duration, total_heatup_time/duration])
            average_power_thruster_list.append(average_power_thruster)

            average_energy_obc, average_power_obc = calculate_energy_power(duration, [0, 1.3], [0, 1])
            average_power_obc_list.append(average_power_obc)

            # print(fraction_off, fraction_tracking, fraction_signals)
            # print(total_energy_transponder, average_power_transponder)
            # print(total_energy_thruster, average_power_thruster)

            total_average_power = average_power_transponder+average_power_thruster+average_power_obc
            total_average_power_list.append(total_average_power)

            # Store percentage differences
            power_difference = (total_average_power - initial_total_average_power) / initial_total_average_power * 100
            power_difference_list.append(power_difference)


        # print(initial_tracking_times, tracking_times, difference_tracking_times)
        # print(initial_average_power_transponder_list, initial_average_power_thruster_list)
        # print(average_power_transponder_list, average_power_thruster_list)

        # Add power consumption row
        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\cellcolor[HTML]{EFEFEF}\textbf{Average Power (W)} & \cellcolor[HTML]{EFEFEF}' + f"{initial_total_average_power_list[0]:.{decimals}f} & " + " & ".join([f"{total_average_power:.{decimals}f}" for total_average_power in total_average_power_list]) + r' \\' + '\n'

        table_str += r'Transponder & ' + f"{initial_average_power_transponder_list[0]:.{decimals}f} & " + " & ".join([f"{average_power_transponder:.{decimals}f}" for average_power_transponder in average_power_transponder_list]) + r' \\' + '\n'
        table_str += r'Thruster & ' + f"{initial_average_power_thruster_list[0]:.{decimals}f} & " + " & ".join([f"{average_power_thruster:.{decimals}f}" for average_power_thruster in average_power_thruster_list]) + r' \\' + '\n'
        table_str += r'OBC & ' + f"{initial_average_power_obc_list[0]:.{decimals}f} & " + " & ".join([f"{average_power_obc:.{decimals}f}" for average_power_obc in average_power_obc_list]) + r' \\' + '\n'
        table_str += r'\% Change & 00.00\% &' + " & ".join([f"{diff:.2f}\\%" for diff in power_difference_list]) + r' \\' + '\n'

        # Final table endings
        label = label+"_"+optimization_results_list[0]["current_time"].split('_')[0]
        table_str += r'\end{tabular}' + '\n'
        table_str += r'\caption{' + caption + '}' + '\n'
        table_str += r'\label{' + label + '}' + '\n'
        table_str += r'\end{table}'

        if self.save_table:
            utils.save_table_to_folder(tables=[table_str], labels=[file_name], custom_sub_folder_name=self.file_name)