# Standard
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils
import helper_functions
from src.optimization_models import OptimizationModel
from src import NavigationSimulator, PlotNavigationResults



#################################################################
###### Show constant OD solutions ###############################
#################################################################

### Extract information from specific folder and run
folder_name = "get_data_initial_navigation_routines"
folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", folder_name)

file_name = "delta_v_dict_constant_arc_duration.json"
file_path = os.path.join(folder_path, file_name)
data = helper_functions.load_json_file(file_path)

model = "PM"
# Extracting data for plotting
subkeys = ["0.2", "0.5", "1", "1.5", "1.8"]
values = {subkey: [data[model][key][subkey][1] for key in data[model]] for subkey in subkeys}

# Plotting
fig = plt.figure(figsize=(10, 3))
for i, subkey in enumerate(subkeys):
    plt.bar([j + i * 0.1 for j in range(len(values[subkey]))], values[subkey], width=0.1, label=subkey)

plt.xlabel('Arc separation interval [days]')
plt.ylabel(r'$\Delta V$ [m/s]')
plt.title(f'Station keeping costs, simulation of {28} [days]')
plt.xticks([i + 0.2 for i in range(len(data[model]))], data[model].keys())
plt.yscale('log')
plt.legend(title='Arc durations', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(alpha=0.3)
plt.show()


utils.save_dicts_to_folder(figs=[fig], labels=["fig_constant_arc_duration"], custom_sub_folder_name=file_name)





# navigation_results = navigation_results_dict[0]

# results_dict = {"HF": {"PM": [navigation_results]}}

# results_plotter = PlotNavigationResults.PlotNavigationResults(results_dict)
# results_plotter.plot_uncertainty_history()
# results_plotter.plot_full_state_history()
# results_plotter.plot_observations()
# results_plotter.plot_estimation_error_history()


# plt.plot()


# navigation_results = navigation_results_dict[1]


# results_dict = {"HF": {"PM": [navigation_results]}}

# results_plotter = PlotNavigationResults.PlotNavigationResults(results_dict)
# results_plotter.plot_uncertainty_history()
# results_plotter.plot_full_state_history()
# results_plotter.plot_observations()
# results_plotter.plot_estimation_error_history()

# plt.plot()