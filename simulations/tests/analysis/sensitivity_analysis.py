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
        # "arc_duration": [0.1, 0.5, 1.0, 2.0],
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
        "delta_v_min": [0.00, 0.003, 0.03],
        # "station_keeping_error": [0.00, 0.01, 0.05, 0.1],
    }
}

auxilary_settings = {
    "show_corrections_in_terminal": True,
    "step_size": 0.05,
    # "apriori_covariance": np.diag(np.array([5e2, 5e2, 5e2, 1e-2, 1e-2, 1e-2, 5e2, 5e2, 5e2, 1e-2, 1e-2, 1e-2])**2)
}

custom_observation_windows_settings = {
    "Optimized": [
        ([
        [
            60390,
            60391
        ],
        [
            60394,
            60395
        ],
        [
            60398,
            60399
        ],
        [
            60402,
            60403
        ],
        [
            60406,
            60407
        ],
        [
            60410,
            60411
        ],
        [
            60414,
            60415
        ]
    ], num_runs, None),
    ]
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


# custom_observation_windows_settings = {
#     "Optimized": [
#         ([
#         [
#             60390,
#             60391.618612525024
#         ],
#         [
#             60394.618612525024,
#             60395.756028792246
#         ],
#         [
#             60398.756028792246,
#             60399.922544256435
#         ],
#         [
#             60402.922544256435,
#             60403.02254425643
#         ],
#         [
#             60406.02254425643,
#             60406.24953812655
#         ],
#         [
#             60409.24953812655,
#             60409.349538126546
#         ],
#         [
#             60412.349538126546,
#             60412.449538126544
#         ]
#     ], num_runs, None),
#     ]
# }




custom_observation_windows_settings = {
    "Optimized": [
        ([
        [
            60390,
            60391.04069549083
        ],
        [
            60394.04069549083,
            60395.195556475635
        ],
        [
            60398.195556475635,
            60399.67613808943
        ],
        [
            60402.67613808943,
            60403.183957148256
        ],
        [
            60406.183957148256,
            60407.337046560395
        ],
        [
            60410.337046560395,
            60411.05394740322
        ],
        [
            60414.05394740322,
            60415.1967917898
        ],
        [
            60418.1967917898,
            60418.907927269436
        ],
        [
            60421.907927269436,
            60423.03926250301
        ],
        [
            60426.03926250301,
            60427.00222169031
        ],
        [
            60430.00222169031,
            60430.81009296409
        ],
        [
            60433.81009296409,
            60434.68216766112
        ],
        [
            60437.68216766112,
            60438.60251526832
        ],
        [
            60441.60251526832,
            60441.711549175234
        ]
    ], num_runs, None),
    ]
}


# custom_observation_windows_settings = None


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