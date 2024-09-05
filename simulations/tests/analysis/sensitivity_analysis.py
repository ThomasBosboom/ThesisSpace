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

num_runs = 2
default_window_inputs = {
    "duration": 28,
    "arc_interval": 3,
    "threshold": 1,
    "arc_duration": 1
}

combined_sensitivity_settings = {
    # "sensitivity_settings_windows": {
    #     "arc_duration": [0.1, 0.5, 1.0, 2.0],
    #     "arc_interval": [1.0, 2.0, 3.0, 4.0],
    #     "mission_start_epoch": [60390, 60395, 60400, 60405],
    # },
    "sensitivity_settings_auxiliary": {
        # "initial_estimation_error": [
        #     np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])/100,
        #     np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])/10,
        #     np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3]),
        #     np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])*10
        #     ],
        # "orbit_insertion_error": [np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0,
        #                           np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0.5,
        #                           np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*1,
        #                           np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*2
        #     ],
        # "observation_interval": [100, 500, 1000, 5000],
        # "noise": [0.1, 1, 10, 100],
        # "target_point_epochs": [[2], [3], [4], [5]],
        "delta_v_min": [0.00, 0.003, 0.01, 0.02, 0.03],
        # "station_keeping_error": [0.00, 0.01, 0.05, 0.1],
    }
}

auxilary_settings = {
    "show_corrections_in_terminal": True,
    "step_size": 0.05
}

custom_observation_windows_settings = {
    "Optimized": [
        ([
        [
            60390,
            60390.76217095995
        ],
        [
            60393.76217095995,
            60394.92989843622
        ],
        [
            60397.92989843622,
            60398.78205703818
        ],
        [
            60401.78205703818,
            60402.82977960193
        ],
        [
            60405.82977960193,
            60407.68702720861
        ],
        [
            60410.68702720861,
            60410.78702720861
        ],
        [
            60413.78702720861,
            60413.887027208606
        ]
    ], num_runs, None),
    ]
}

for sensitivity_name, sensitivity_settings in combined_sensitivity_settings.items():

    print("Sensitivity settings \n", sensitivity_settings)
    navigation_outputs_sensitivity = helper_functions.generate_navigation_outputs_sensitivity_analysis(
        num_runs,
        sensitivity_settings,
        default_window_inputs,
        custom_observation_windows_settings=custom_observation_windows_settings,
        **auxilary_settings
    )

    #################################################################
    ###### Plot results of sensitivity analysis #####################
    #################################################################

    process_sensitivity_results = ProcessSensitivityResults.PlotSensitivityResults(
        navigation_outputs_sensitivity,
        figure_settings={"save_figure": True,
                         "save_table": True,
                         "save_dict": True,
                         "current_time": f"{current_time}_{sensitivity_name}",
                         "file_name": file_name
        }
    )

    print("Plotting results...")
    process_sensitivity_results.plot_sensitivity_analysis_results(
        sensitivity_settings,
        evaluation_threshold=14
    )

    process_sensitivity_results.plot_sensitivity_analysis_results(
        sensitivity_settings,
        evaluation_threshold=14,
        show_annual=True,
        custom_color_cycle=["gray", "red", "gray", "gray", "gray"]
    )
    print("Plotting done...")

plt.show()