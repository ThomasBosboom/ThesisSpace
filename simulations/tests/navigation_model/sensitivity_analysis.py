# Standard
import os
import sys
import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Define current time
current_time = datetime.now().strftime("%Y%m%d%H%M")

# Own
import helper_functions
from tests.postprocessing import ProcessNavigationResults, ProcessSensitivityResults


#################################################################
###### Sensitivity analysis #####################################
#################################################################

num_runs = 5

default_window_inputs = {
    "duration": 28,
    "arc_interval": 3,
    "threshold": 1,
    "arc_duration": 1
}

sensitivity_settings = {
    # "threshold": [0.1, 0.5, 1.0, 2.0],
    # "orbit_insertion_error": [np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0
    # "arc_duration": [0.1, 0.5, 1.0, 2.0],
    # "arc_interval": [1.0, 2.0, 3.0, 4.0],
    "mission_start_epoch": [60390, 60395, 60400],
    # "noise_range": [1, 5, 10, 50],
    # "target_point_epochs": [[2], [3], [4]],
    # "delta_v_min": [0.00, 0.01, 0.02, 0.03],
    # "station_keeping_error": [0.00, 0.01, 0.05, 0.1],
}

auxilary_settings = {
    # "orbit_insertion_error": np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0
}

navigation_outputs_sensitivity = helper_functions.generate_navigation_outputs_sensitivity_analysis(num_runs, sensitivity_settings, default_window_inputs, **auxilary_settings)

print(navigation_outputs_sensitivity)


#################################################################
###### Plot results of sensitivity analysis #####################
#################################################################

process_sensitivity_results = ProcessSensitivityResults.PlotSensitivityResults(
    navigation_outputs_sensitivity,
    figure_settings={"save_figure": True, "current_time": current_time, "file_name": file_name}
)

print("Plotting results...")
process_sensitivity_results.plot_sensitivity_analysis_results(
    sensitivity_settings
)
print("Plotting done...")

plt.show()



