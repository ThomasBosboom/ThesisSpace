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

        if initial_cost != 0:
            cost_percentage_diff = ((final_cost - initial_cost) / initial_cost) * 100
        else:
            cost_percentage_diff = 0  # Handle division by zero case

        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += f"\\textbf{{Cost}}  & {initial_cost:.{decimals}f} & {final_cost:.{decimals}f} & {round(cost_percentage_diff, 2)}\%" + r' \\ ' + '\n'

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
            state = f"$T_{i+1}$"
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
        mean_cost_diff = np.mean(cost_differences)
        std_cost_diff = np.std(cost_differences)
        min_cost_diff = min(cost_differences)
        max_cost_diff = max(cost_differences)

        mean_cost = np.mean(final_costs)
        std_cost = np.std(final_costs)
        min_cost = min(final_costs)
        max_cost = max(final_costs)

        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\textbf{Objective} & ' + f"{initial_cost:.{decimals}f} & {min_cost:.{decimals}f} & {max_cost:.{decimals}f} & {mean_cost:.{decimals}f} & {std_cost:.{decimals}f} & "
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
            state = f"$T_{i+1}$"
            initial_value = optimization_results_list[0]["initial_design_vector"][i]
            run_values = [optimization_results_list[j]["best_design_vector"][i] for j in range(num_runs)]

            # Add row for each entry in design vector
            table_str += f"{state} & {initial_value:.{decimals}f} & " + " & ".join([f"{value:.{decimals}f}" for value in run_values]) + r' \\' + '\n'

        initial_cost = optimization_results_list[0]["initial_objective_value"]
        final_costs = [results["best_objective_value"] for results in optimization_results_list]
        cost_differences = [(final - initial_cost) / initial_cost * 100 if initial_cost != 0 else 0 for final in final_costs]

        # Add cost row
        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\cellcolor[HTML]{EFEFEF}\textbf{Cost} & \cellcolor[HTML]{EFEFEF}' + f"{initial_cost:.{decimals}f} & " + " & ".join([f"{cost:.{decimals}f}" for cost in final_costs]) + r' \\' + '\n'

        # Add percentage difference row
        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\cellcolor[HTML]{EFEFEF}\textbf{\%Diff} & \cellcolor[HTML]{EFEFEF} 00.00\% &' + " & ".join([f"{diff:.2f}\\%" for diff in cost_differences]) + r' \\' + '\n'

        table_str += r'\end{tabular}' + '\n'
        table_str += r'\caption{' + caption + '}' + '\n'
        table_str += r'\label{' + label + '}' + '\n'
        table_str += r'\end{table}'

        if self.save_table:
            utils.save_table_to_folder(tables=[table_str], labels=[file_name], custom_sub_folder_name=self.file_name)
