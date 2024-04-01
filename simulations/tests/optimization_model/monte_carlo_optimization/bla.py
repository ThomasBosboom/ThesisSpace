# Standard
import os
import sys
import numpy as np

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils
import helper_functions



#################################################################
###### Monte Carlo test case ####################################
#################################################################

dynamic_model_list = ["HF", "PM",0]
truth_model_list = ["HF", "PM",0]


dict = {
    "0": {
        "threshold": 60397,
        "skm_to_od_duration": 3,
        "duration": 28,
        "model": {
            "dynamic": {
                "model_type": "HF",
                "model_name": "PM",
                "model_number": 0
            },
            "truth": {
                "model_type": "HF",
                "model_name": "PM",
                "model_number": 0
            }
        },
        "history": {
            "0": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_value": 2.777506078331983
            },
            "1": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_value": 2.5585026597041516
            },
            "2": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_value": 2.227164016916117
            },
            "3": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_value": 3.4332514990063494
            },
            "4": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_value": 1.962426342790788
            }
        },
        "optim": {
            "x_optim": [
                1.0,
                1.0,
                1.0,
                1.05,
                1.0
            ],
            "x_observation_windows": [
                [
                    60390,
                    60397
                ],
                [
                    60400.0,
                    60401.0
                ],
                [
                    60404.0,
                    60405.0
                ],
                [
                    60408.0,
                    60409.0
                ],
                [
                    60412.0,
                    60413.05
                ],
                [
                    60416.05,
                    60417.05
                ]
            ],
            "x_skm_epochs": [
                60397,
                60401.0,
                60405.0,
                60409.0,
                60413.05,
                60417.05
            ]
        }
    },
    "1": {
        "threshold": 60397,
        "skm_to_od_duration": 3,
        "duration": 28,
        "model": {
            "dynamic": {
                "model_type": "HF",
                "model_name": "PM",
                "model_number": 0
            },
            "truth": {
                "model_type": "HF",
                "model_name": "PM",
                "model_number": 0
            }
        },
        "history": {
            "0": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.05
                ],
                "objective_value": 2.9259387444537714
            },
            "1": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.05
                ],
                "objective_value": 3.1401649808592444
            },
            "2": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.05
                ],
                "objective_value": 2.8428506132683404
            },
            "3": {
                "design_vector": [
                    1.003136,
                    1.003136,
                    1.0216960000000002,
                    1.003136,
                    1.021136
                ],
                "objective_value": 2.6440271790945915
            },
            "4": {
                "design_vector": [
                    1.003136,
                    1.003136,
                    1.0216960000000002,
                    1.003136,
                    1.021136
                ],
                "objective_value": 3.1106456716851056
            }
        },
        "optim": {
            "x_optim": [
                1.003136,
                1.003136,
                1.0216960000000002,
                1.003136,
                1.021136
            ],
            "x_observation_windows": [
                [
                    60390,
                    60397
                ],
                [
                    60400.0,
                    60401.003136
                ],
                [
                    60404.003136,
                    60405.006272
                ],
                [
                    60408.006272,
                    60409.027968
                ],
                [
                    60412.027968,
                    60413.031104
                ],
                [
                    60416.031104,
                    60417.052240000005
                ]
            ],
            "x_skm_epochs": [
                60397,
                60401.003136,
                60405.006272,
                60409.027968,
                60413.031104,
                60417.052240000005
            ]
        }
    }
}


import numpy as np
import matplotlib.pyplot as plt

# Data
x_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
y_arrays = [
    np.array([1, 1, 1, 1, 1, 1]),
    np.array([1.01944444, 1.01944444, 1.01944444, 0.93333333, 1.01944444, 1.01944444]),
    np.array([1.01944444, 1.01944444, 1.01944444, 0.93333333, 1.01944444, 1.01944444]),
    np.array([0.94989712, 1.03600823, 0.99063786, 0.9654321 , 1.03600823, 1.03600823]),
    np.array([0.99707933, 1.05001143, 0.98699703, 0.95198903, 1.05001143, 0.96390032]),
    np.array([0.99594352, 1.06946032, 0.98194032, 0.93331809, 1.06946032, 1.01652822]),
    np.array([0.994366  , 1.09647267, 0.97491711, 0.90738624, 1.01036156, 1.02295586]),
    np.array([0.994366  , 1.09647267, 0.97491711, 0.90738624, 1.01036156, 1.02295586]),
    np.array([0.9497004 , 1.09341646, 1.02849876, 0.86676464, 1.0682107 , 1.0026446 ]),
    np.array([0.92751599, 1.12712163, 0.94806926, 0.90013607, 1.09211363, 1.00104961]),
    np.array([1.01154199, 1.14046957, 0.96992358, 0.83194482, 1.09184735, 0.96536954]),
    np.array([0.95425674, 1.15697681, 0.96813835, 0.80318506, 1.08944594, 1.06208478]),
    np.array([0.93955963, 1.16507914, 0.96951321, 0.77747235, 1.07128627, 1.00963068]),
    np.array([0.9221976 , 1.23891067, 0.98500422, 0.79190732, 1.07196584, 0.98834798]),
    np.array([0.89270126, 1.22991533, 0.98255673, 0.72347789, 1.17474504, 0.98071942]),
    np.array([0.89270126, 1.22991533, 0.98255673, 0.72347789, 1.17474504, 0.98071942]),
    np.array([0.89270126, 1.22991533, 0.98255673, 0.72347789, 1.17474504, 0.98071942]),
    np.array([0.82846111, 1.32607481, 0.96191508, 0.63894442, 1.13173729, 1.06229682]),
    np.array([0.82846111, 1.32607481, 0.96191508, 0.63894442, 1.13173729, 1.06229682]),
    np.array([0.82846111, 1.32607481, 0.96191508, 0.63894442, 1.13173729, 1.06229682])
]

objective = np.array([2.0701149836078447, 2.064904307916726, 2.009791182295538, 2.0310131834072354, 1.944314276858265, 1.877554546204298, 1.863537747830019, 1.8758357711222557, 1.7660631116290022, 1.7671636503331385, 1.6746877143775092, 1.69110548248101, 1.6551851632747379, 1.6545915241618974, 1.5248477384453178, 1.5145962064869063, 1.5226043680247603, 1.5112858594402474, 1.4650022661934639])
y_arrays1 = [np.array([1, 1, 1, 1, 1, 1]), np.array([1.01944444, 1.01944444, 1.01944444, 0.93333333, 1.01944444,
       1.01944444]), np.array([1.01944444, 1.01944444, 1.01944444, 0.93333333, 1.01944444,
       1.01944444]), np.array([0.94989712, 1.03600823, 1.03600823, 0.9654321 , 1.03600823,
       0.99063786]), np.array([0.94989712, 1.03600823, 1.03600823, 0.9654321 , 1.03600823,
       0.99063786]), np.array([0.99610578, 1.06668191, 1.02131154, 0.93598537, 1.06668191,
       0.9826627 ]), np.array([0.99459136, 1.09261376, 1.02959936, 0.91109079, 1.00650265,
       0.97592042]), np.array([0.99459136, 1.09261376, 1.02959936, 0.91109079, 1.00650265,
       0.97592042]), np.array([0.99459136, 1.09261376, 1.02959936, 0.91109079, 1.00650265,
       0.97592042]), np.array([0.93215287, 1.11368967, 1.01859238, 0.91545051, 1.08008199,
       0.94654562]), np.array([0.93600526, 1.11000185, 1.09273709, 0.8645546 , 1.06332451,
       0.9708786 ]), np.array([0.99406881, 1.14792209, 1.03505437, 0.81654532, 1.08309246,
       0.96081676]), np.array([0.99406881, 1.14792209, 1.03505437, 0.81654532, 1.08309246,
       0.96081676]), np.array([0.91456732, 1.20236888, 1.04890538, 0.83918981, 1.04563705,
       0.97589613]), np.array([0.94353486, 1.18772613, 1.07700004, 0.83625586, 1.05428582,
       0.9121394 ]), np.array([0.87545598, 1.22264531, 1.08760657, 0.76832952, 1.13816821,
       0.927902  ]), np.array([0.87545598, 1.22264531, 1.08760657, 0.76832952, 1.13816821,
       0.927902  ]), np.array([0.87545598, 1.22264531, 1.08760657, 0.76832952, 1.13816821,
       0.927902  ]), np.array([0.87545598, 1.22264531, 1.08760657, 0.76832952, 1.13816821,
       0.927902  ]), np.array([0.82362816, 1.33799252, 1.12997585, 0.72909111, 1.0800967 ,
       0.90654394])]

fig, ax = plt.subplots(1, 1)

# Plot each column
# ax.plot(x_values, y_arrays, color='blue')
# ax.plot(x_values, y_arrays1, color="red")
ax.plot(x_values, y_arrays)
ax.plot(x_values, y_arrays1)
ax.plot(x_values[1:], objective)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot for Each Column')
plt.legend()
plt.grid(True)
# plt.show()


# utils.save_dicts_to_folder(dicts=[dict], custom_sub_folder_name=str(file_name), labels=["test_dict"])
# utils.save_figure_to_folder(figs=[fig], custom_sub_folder_name=str(file_name), labels=["test_figure"])

# new_dict = {'threshold': 60393, 'skm_to_od_duration': 3, 'duration': 9, 'factor': 1, 'initial_design_vector': list(np.array([1.])), 'model': {'dynamic': {'model_type': 'HF', 'model_name': 'PM', 'model_number': 0}, 'truth': {'model_type': 'HF', 'model_name': 'PM', 'model_number': 0}}, 'history': {0: {'design_vector': [1.0], 'objective_function': 0.1320727346835261}}, 'optim': {'x_optim': [1.0], 'x_observation_windows': [(60390, 60393), (60396.0, 60397.0)], 'x_skm_epochs': [60393, 60397.0]}}

# utils.save_dicts_to_folder(dicts=[new_dict], custom_sub_folder_name=str(file_name), labels=["newtest_dict"])



data = {
    "0": {
        "history": {
            "design_vector": {
                "0": [1, 1],
                "1": [1.05, 0.95]
            },
            "objective_value": {
                "0": 0.1,
                "1": 0.1
            }
        }
    },
    "1": {
        "history": {
            "design_vector": {
                "0": [1.1, 1.1],
                "1": [2.2, 2.2]
            },
            "objective_value": {
                "0": 0.1,
                "1": 0.1
            }
        }
    }
}


data = {
    "0": {
        "threshold": 60393,
        "skm_to_od_duration": 3,
        "duration": 14,
        "factor": 1,
        "maxiter": 2,
        "initial_design_vector": [
            1.0,
            1.0
        ],
        "model": {
            "dynamic": {
                "model_type": "HF",
                "model_name": "PM",
                "model_number": 0
            },
            "truth": {
                "model_type": "HF",
                "model_name": "PM",
                "model_number": 0
            }
        },
        "history": {
            "design_vector": {
                "0": [
                    1.0,
                    1.0
                ],
                "1": [
                    1.0499999999999998,
                    0.95
                ]
            },
            "objective_value": {
                "0": [
                    1.0,
                    1.0
                ],
                "1": [
                    1.0499999999999998,
                    0.95
                ]
            }
        },
        "final_result": {
            "x_optim": [
                1.0499999999999998,
                0.95
            ],
            "observation_windows": [
                [
                    60390,
                    60393
                ],
                [
                    60396.0,
                    60397.05
                ],
                [
                    60400.05,
                    60401.0
                ]
            ],
            "skm_epochs": [
                60393,
                60397.05,
                60401.0
            ],
            "approx_annual_deltav": 5.319783779089685,
            "reduction_percentage": -1.7619115743050866
        }
    },
    "1": {
        "threshold": 60393,
        "skm_to_od_duration": 3,
        "duration": 14,
        "factor": 1,
        "maxiter": 2,
        "initial_design_vector": [
            1.0,
            1.0
        ],
        "model": {
            "dynamic": {
                "model_type": "HF",
                "model_name": "PM",
                "model_number": 0
            },
            "truth": {
                "model_type": "HF",
                "model_name": "PM",
                "model_number": 0
            }
        },
        "history": {
            "design_vector": {
                "0": [
                    1.0,
                    1.0
                ],
                "1": [
                    1.0499999999999998,
                    0.95
                ]
            },
            "objective_value": {
                "0": [
                    1.0,
                    1.0
                ],
                "1": [
                    1.0499999999999998,
                    0.95
                ]
            }
        },
        "final_result": {
            "x_optim": [
                1.0499999999999998,
                0.95
            ],
            "observation_windows": [
                [
                    60390,
                    60393
                ],
                [
                    60396.0,
                    60397.05
                ],
                [
                    60400.05,
                    60401.0
                ]
            ],
            "skm_epochs": [
                60393,
                60397.05,
                60401.0
            ],
            "approx_annual_deltav": 5.332089557939623,
            "reduction_percentage": -1.8265379083877387
        }
    }
}
# data = {key: value["history"] for key, value in data.items()}
# print(data)
# data = {'threshold': 60393, 'skm_to_od_duration': 3, 'duration': 14, 'factor': 1, 'maxiter': 2, 'initial_design_vector': [1.0, 1.0], 'model': {'dynamic': {'model_type': 'HF', 'model_name': 'PM', 'model_number': 0},
# 'truth': {'model_type': 'HF', 'model_name': 'PM', 'model_number': 0}}, 'history': {'design_vector': {0: [1.0,
# 1.0], 1: [1.0499999999999998, 0.95]}, 'objective_value': {0: [1.0, 1.0], 1: [1.0499999999999998, 0.95]}}, 'final_result': {'x_optim': [1.0499999999999998, 0.95], 'x_observation_windows': [(60390, 60393), (60396.0, 60397.05), (60400.05, 60401.0)], 'x_skm_epochs': [60393, 60397.05, 60401.0], 'approx_annual_deltav': 5.303798236051521, 'reduction_percentage': -2.42935004949849}}


combined_history_dict = helper_functions.get_combined_history_dict(data)
monte_carlo_stats_dict = helper_functions.get_monte_carlo_stats_dict(combined_history_dict)

label="bla"

utils.save_dicts_to_folder(dicts=[combined_history_dict], custom_sub_folder_name=label, labels=["combined1_"+label])
utils.save_dicts_to_folder(dicts=[monte_carlo_stats_dict], custom_sub_folder_name=label, labels=["stats1_"+label])
