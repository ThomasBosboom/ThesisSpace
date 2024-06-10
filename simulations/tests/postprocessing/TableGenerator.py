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

        table_str = ''
        table_str += r'\begin{table}[h!]' + '\n'
        table_str += r'\centering' + '\n'
        table_str += r'\begin{tabular}{llllll}' + '\n'
        table_str += r' &  & \cellcolor[HTML]{EFEFEF}\textbf{Total} &  & \cellcolor[HTML]{EFEFEF}\textbf{Last 14 days} &  \\' + '\n'
        table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
        table_str += r'\textbf{Case} & \textbf{Value} & \textbf{$\mu_{\Delta \boldsymbol{V}}$} & \textbf{$\sigma_{\Delta \boldsymbol{V}}$} & \textbf{$\mu_{\Delta \boldsymbol{V}}$} & \textbf{$\sigma_{\Delta \boldsymbol{V}}$} \\ ' + '\n'

        for case, case_data in sensitivity_statistics.items():
            is_first_row = True
            for subkey, stats in case_data.items():
                full_stats = stats.get('total_stats', {})
                threshold_stats = stats.get('total_stats_with_threshold', {})

                if is_first_row:
                    table_str += f'{case} & '
                    is_first_row = False
                else:
                    table_str += ' & '

                table_str += f"{subkey} & " + \
                            f"{full_stats.get('mean', ''):.{decimals}f} & " + \
                            f"{full_stats.get('std', ''):.{decimals}f} & " + \
                            f"{threshold_stats.get('mean', ''):.{decimals}f} & " + \
                            f"{threshold_stats.get('std', ''):.{decimals}f}" + r' \\ ' + '\n'

        table_str += r'\end{tabular}' + '\n'
        table_str += r'\caption{' + caption + '}' + '\n'
        table_str += r'\label{' + label + '}' + '\n'
        table_str += r'\end{table}'

        # Define the path to the tables folder
        tables_folder = os.path.join(os.path.dirname(__file__), "tables")

        # Create the tables folder if it doesn't exist
        if not os.path.exists(tables_folder):
            os.makedirs(tables_folder)

        # Define the file path for the LaTeX table
        file_path = os.path.join(tables_folder, file_name)

        if file_name:
            with open(file_path, 'w') as file:
                file.write(table_str)

        print(table_str, sensitivity_statistics)

        print(f"LaTeX table code has been written")




# sensitivity_statistics = {'Mission Start Epoch': {'60400': {'epoch_stats': {60401: {'mean': 0.0009584622699008587, 'std': 0.00012090392308297616}, 60405: {'mean': 0.005817788376310036, 'std': 0.00037029311626137146}, 60409: {'mean': 0.008644247621561418, 'std': 0.0005053710034636791}, 60413: {'mean': 0.005080131839618855, 'std': 0.00023168028875643863}, 60417: {'mean': 0.004809221532270387, 'std': 3.9893292452597776e-05}, 60421: {'mean': 0.006285460633615646, 'std': 8.38717396107417e-06}, 60425: {'mean': 0.004344988578237152, 'std': 1.8757136821656323e-05}}, 'total_stats': {'mean': 0.035940300851514353, 'std': 0.0008319253572869126}, 'total_stats_with_threshold': {'mean': 0.010630449211852798, 'std': 2.7144310782730927e-05}}, '60405': {'epoch_stats': {60406: {'mean': 0.0015351618242127139, 'std': 0.00013187301205383895}, 60410: {'mean':
# 0.00799192741303232, 'std': 0.0002248332180725481}, 60414: {'mean': 0.005974354267140028, 'std': 0.000644619795165433}, 60418: {'mean': 0.0018000426173089507, 'std': 5.893201985299472e-05}, 60422: {'mean': 0.006652248384311513, 'std': 7.734345273977027e-05}, 60426: {'mean': 0.0036958802153875215, 'std': 4.1799903744081006e-05},
# 60430: {'mean': 0.006595939931383789, 'std': 0.00010541291879686415}}, 'total_stats': {'mean': 0.03424555465277684, 'std': 0.0006336840370862809}, 'total_stats_with_threshold': {'mean': 0.016944068531082823, 'std': 0.00014095646779255233}}}}
# # sensitivity_statistics = sensitivity_statistics['Mission Start Epoch'][""]['total_stats']
# table_generator = TableGenerator()
# table_generator.generate_sensitivity_analysis_table(sensitivity_statistics)