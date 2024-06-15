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

class TableGenerator():

    def __init__(self, table_settings={"save_table": True, "current_time": float, "file_name": str}):

        for key, value in table_settings.items():
            if table_settings["save_table"]:
                setattr(self, key, value)


    def escape_tex_symbols(self, string):
        escape_chars = {'%': '\\%', '&': '\\&', '_': '\\_', '#': '\\#', '$': '\\$', '{': '\\{', '}': '\\}'}
        return re.sub(r'[%&_#${}]', lambda match: escape_chars[match.group(0)], string)


    def save_table_to_folder(self, table_str, file_name):

        # Define the path to the tables folder
        tables_folder = os.path.join(os.path.dirname(__file__), "tables")

        # Create the tables folder if it doesn't exist
        if not os.path.exists(tables_folder):
            os.makedirs(tables_folder)

        # Define the file path for the LaTeX table
        file_path = os.path.join(tables_folder, file_name)

        print(file_path)

        if file_name:
            with open(file_path, 'w') as file:
                file.write(table_str)


    def generate_sensitivity_analysis_table(self, sensitivity_statistics, caption="Statistical results of Monte Carlo sensitivity analysis", label="tab:SensitivityAnalysis", file_name="sensitivity_analysis.tex", decimals=4, include_worst_case=True):

        table_str = ''
        table_str += r'\begin{table}[h!]' + '\n'
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

        table_str += r'\end{tabular}' + '\n'
        table_str += r'\caption{' + caption + '}' + '\n'
        table_str += r'\label{' + label + '}' + '\n'
        table_str += r'\end{table}'

        if self.save_table:
            self.save_table_to_folder(table_str, file_name)

        print(table_str, sensitivity_statistics)

        print(f"LaTeX table code has been written")


    def generate_optimization_analysis_table(self, optimization_results, caption="Results of optimization", label="tab:OptimizationAnalysis", file_name='optimization_analysis.tex', decimals=4):

        table_str = ""
        table_str += r'\begin{table}[h!]' + '\n'
        table_str += r'\centering' + '\n'
        table_str += r'\begin{tabular}{llll}' + '\n'
        table_str += r'\textbf{}      & \cellcolor[HTML]{EFEFEF}\textbf{Vectors} & \textbf{} & \textbf{}         \\' + '\n'
        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\textbf{Entry} & \textbf{Initial} & \textbf{Optimized} & \textbf{Difference} \\' + '\n'

        states = [f"T_{i}" for i in range(len(optimization_results["initial_design_vector"]))]
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
            self.save_table_to_folder(table_str, file_name)