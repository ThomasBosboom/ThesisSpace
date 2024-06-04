# Standard
import os
import sys
import numpy as np
import re

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(3):
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


    def generate_sensitivity_analysis_table(self, sensitivity_statistics, caption="Statistical results of Monte Carlo sensitivity analysis", label="tab:SensitivityAnalysis", file_name="sensitivity_analysis.tex", decimals=4):

        # Define the path to the tables folder
        tables_folder = os.path.join(os.path.dirname(__file__), "tables")

        # Create the tables folder if it doesn't exist
        if not os.path.exists(tables_folder):
            os.makedirs(tables_folder)

        # Define the file path for the LaTeX table
        file_path = os.path.join(tables_folder, file_name)

        # Initialize the LaTeX table string
        latex_table = f"""
                    \\begin{{table}}[]
                    \\centering
                    \\begin{{tabular}}{{l l l l}}
                    \\rowcolor[HTML]{{EFEFEF}} \\textbf{{Parameter}} & \\textbf{{Value}} & \\textbf{{$\\mu_{{\\Delta V}}$}} & \\textbf{{$\\sigma_{{\\Delta V}}$}} \\\\
                    """

        # Iterate through the dictionary to populate the table
        for main_key, sub_dict in sensitivity_statistics.items():
            main_key_formatted = self.escape_tex_symbols(main_key)  # Escape special TeX symbols in main key
            for idx, (sub_key, values) in enumerate(sub_dict.items()):
                mean = round(values[0], decimals)
                std_dev = round(values[1], decimals)
                sub_key_formatted = self.escape_tex_symbols(sub_key)  # Escape special TeX symbols in subkey
                if idx == 0:
                    latex_table += f"\\textit{{{main_key_formatted}}} & {sub_key_formatted} & {mean} & {std_dev} \\\\"
                    latex_table += "\n"  # No hline here
                else:
                    latex_table += f" & {sub_key_formatted} & {mean} & {std_dev} \\\\"
                    latex_table += "\n"  # No hline here

        # End the LaTeX table
        latex_table += f"""
                    \\end{{tabular}}
                    \\caption{{{self.escape_tex_symbols(caption)}}}
                    \\label{{{self.escape_tex_symbols(label)}}}
                    \\end{{table}}
                    """

        # Write the LaTeX table to a .tex file
        with open(file_path, 'w') as f:
            f.write(latex_table)

        print(f"LaTeX table code has been written")