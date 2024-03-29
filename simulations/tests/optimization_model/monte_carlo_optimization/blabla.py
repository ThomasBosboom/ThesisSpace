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

dynamic_model_list = ["high_fidelity", "point_mass",0]
truth_model_list = ["high_fidelity", "point_mass",0]


dict = {
    "0": {
        "threshold": 60397,
        "skm_to_od_duration": 3,
        "duration": 28,
        "model": {
            "dynamic": {
                "model_type": "high_fidelity",
                "model_name": "point_mass",
                "model_number": 0
            },
            "truth": {
                "model_type": "high_fidelity",
                "model_name": "point_mass",
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
                "objective_function": 2.777506078331983
            },
            "1": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_function": 2.5585026597041516
            },
            "2": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_function": 2.227164016916117
            },
            "3": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_function": 3.4332514990063494
            },
            "4": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_function": 1.962426342790788
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
                "model_type": "high_fidelity",
                "model_name": "point_mass",
                "model_number": 0
            },
            "truth": {
                "model_type": "high_fidelity",
                "model_name": "point_mass",
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
                "objective_function": 2.9259387444537714
            },
            "1": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.05
                ],
                "objective_function": 3.1401649808592444
            },
            "2": {
                "design_vector": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.05
                ],
                "objective_function": 2.8428506132683404
            },
            "3": {
                "design_vector": [
                    1.003136,
                    1.003136,
                    1.0216960000000002,
                    1.003136,
                    1.021136
                ],
                "objective_function": 2.6440271790945915
            },
            "4": {
                "design_vector": [
                    1.003136,
                    1.003136,
                    1.0216960000000002,
                    1.003136,
                    1.021136
                ],
                "objective_function": 3.1106456716851056
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


# import numpy as np
# import matplotlib.pyplot as plt

# # Data
# x_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
# y_arrays = [
#     np.array([1, 1, 1, 1, 1, 1]),
#     np.array([1.01944444, 1.01944444, 1.01944444, 0.93333333, 1.01944444, 1.01944444]),
#     np.array([1.01944444, 1.01944444, 1.01944444, 0.93333333, 1.01944444, 1.01944444]),
#     np.array([0.94989712, 1.03600823, 0.99063786, 0.9654321 , 1.03600823, 1.03600823]),
#     np.array([0.99707933, 1.05001143, 0.98699703, 0.95198903, 1.05001143, 0.96390032]),
#     np.array([0.99594352, 1.06946032, 0.98194032, 0.93331809, 1.06946032, 1.01652822]),
#     np.array([0.994366  , 1.09647267, 0.97491711, 0.90738624, 1.01036156, 1.02295586]),
#     np.array([0.994366  , 1.09647267, 0.97491711, 0.90738624, 1.01036156, 1.02295586]),
#     np.array([0.9497004 , 1.09341646, 1.02849876, 0.86676464, 1.0682107 , 1.0026446 ]),
#     np.array([0.92751599, 1.12712163, 0.94806926, 0.90013607, 1.09211363, 1.00104961]),
#     np.array([1.01154199, 1.14046957, 0.96992358, 0.83194482, 1.09184735, 0.96536954]),
#     np.array([0.95425674, 1.15697681, 0.96813835, 0.80318506, 1.08944594, 1.06208478]),
#     np.array([0.93955963, 1.16507914, 0.96951321, 0.77747235, 1.07128627, 1.00963068]),
#     np.array([0.9221976 , 1.23891067, 0.98500422, 0.79190732, 1.07196584, 0.98834798]),
#     np.array([0.89270126, 1.22991533, 0.98255673, 0.72347789, 1.17474504, 0.98071942]),
#     np.array([0.89270126, 1.22991533, 0.98255673, 0.72347789, 1.17474504, 0.98071942]),
#     np.array([0.89270126, 1.22991533, 0.98255673, 0.72347789, 1.17474504, 0.98071942]),
#     np.array([0.82846111, 1.32607481, 0.96191508, 0.63894442, 1.13173729, 1.06229682]),
#     np.array([0.82846111, 1.32607481, 0.96191508, 0.63894442, 1.13173729, 1.06229682]),
#     np.array([0.82846111, 1.32607481, 0.96191508, 0.63894442, 1.13173729, 1.06229682])
# ]

# fig, ax = plt.subplots(1, 1)

# # Plot each column
# ax.plot(x_values, y_arrays)

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Plot for Each Column')
# plt.legend()
# plt.grid(True)
# # plt.show()


# utils.save_dicts_to_folder(dicts=[dict], custom_sub_folder_name=str(file_name), labels=["test_dict"])
# utils.save_figure_to_folder(figs=[fig], custom_sub_folder_name=str(file_name), labels=["test_figure"])

# new_dict = {'threshold': 60393, 'skm_to_od_duration': 3, 'duration': 9, 'factor': 1, 'initial_design_vector': list(np.array([1.])), 'model': {'dynamic': {'model_type': 'high_fidelity', 'model_name': 'point_mass', 'model_number': 0}, 'truth': {'model_type': 'high_fidelity', 'model_name': 'point_mass', 'model_number': 0}}, 'history': {0: {'design_vector': [1.0], 'objective_function': 0.1320727346835261}}, 'optim': {'x_optim': [1.0], 'x_observation_windows': [(60390, 60393), (60396.0, 60397.0)], 'x_skm_epochs': [60393, 60397.0]}}

# utils.save_dicts_to_folder(dicts=[new_dict], custom_sub_folder_name=str(file_name), labels=["newtest_dict"])



data = {
    "0": {
        "threshold": 60393,
        "skm_to_od_duration": 3,
        "duration": 16,
        "model": {
            "dynamic": {
                "model_type": "high_fidelity",
                "model_name": "point_mass",
                "model_number": 0
            },
            "truth": {
                "model_type": "high_fidelity",
                "model_name": "point_mass",
                "model_number": 0
            }
        },
        "history": {
            "0": {
                "design_vector": [
                    0.9166666666666665,
                    1.0444444444444443,
                    1.0444444444444443
                ],
                "objective_function": 0.6399434623536319
            },
            "1": {
                "design_vector": [
                    0.9166666666666665,
                    1.0444444444444443,
                    1.0444444444444443
                ],
                "objective_function": 0.6383965334278083
            },
            "2": {
                "design_vector": [
                    0.9166666666666665,
                    1.0444444444444443,
                    1.0444444444444443
                ],
                "objective_function": 0.6358554562692473
            },
            "3": {
                "design_vector": [
                    0.8333333333333334,
                    1.1054183813443066,
                    1.1146776406035657
                ],
                "objective_function": 0.6124651335550197
            },
            "4": {
                "design_vector": [
                    0.8333333333333334,
                    1.1054183813443066,
                    1.1146776406035657
                ],
                "objective_function": 0.6065221041903059
            },
            "5": {
                "design_vector": [
                    0.8333333333333334,
                    1.1054183813443066,
                    1.1146776406035657
                ],
                "objective_function": 0.6082620180890888
            },
            "6": {
                "design_vector": [
                    0.8333333333333334,
                    1.1054183813443066,
                    1.1146776406035657
                ],
                "objective_function": 0.6061566009548761
            },
            "7": {
                "design_vector": [
                    0.8333333333333334,
                    1.1054183813443066,
                    1.1146776406035657
                ],
                "objective_function": 0.5996176493568932
            },
            "8": {
                "design_vector": [
                    0.8333333333333334,
                    1.1054183813443066,
                    1.1146776406035657
                ],
                "objective_function": 0.6070729550708338
            },
            "9": {
                "design_vector": [
                    0.8333333333333334,
                    1.1054183813443066,
                    1.1146776406035657
                ],
                "objective_function": 0.6098541868940068
            }
        },
        "optim": {
            "x_optim": [
                0.8333333333333334,
                1.1054183813443066,
                1.1146776406035657
            ],
            "x_observation_windows": [
                [
                    60390,
                    60393
                ],
                [
                    60396.0,
                    60396.833333333336
                ],
                [
                    60399.833333333336,
                    60400.93875171468
                ],
                [
                    60403.93875171468,
                    60405.053429355285
                ]
            ],
            "x_skm_epochs": [
                60393,
                60396.833333333336,
                60400.93875171468,
                60405.053429355285
            ]
        }
    },
    "1": {
        "threshold": 60393,
        "skm_to_od_duration": 3,
        "duration": 16,
        "model": {
            "dynamic": {
                "model_type": "high_fidelity",
                "model_name": "point_mass",
                "model_number": 0
            },
            "truth": {
                "model_type": "high_fidelity",
                "model_name": "point_mass",
                "model_number": 0
            }
        },
        "history": {
            "0": {
                "design_vector": [
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_function": 0.6409457827831796
            },
            "1": {
                "design_vector": [
                    0.9555555555555557,
                    1.074074074074074,
                    0.9462962962962962
                ],
                "objective_function": 0.6355462061866141
            },
            "2": {
                "design_vector": [
                    0.9555555555555557,
                    1.074074074074074,
                    0.9462962962962962
                ],
                "objective_function": 0.6385209865037603
            },
            "3": {
                "design_vector": [
                    0.9555555555555557,
                    1.074074074074074,
                    0.9462962962962962
                ],
                "objective_function": 0.6314155163770052
            },
            "4": {
                "design_vector": [
                    0.8853223593964343,
                    1.1666666666666667,
                    0.8706904435299498
                ],
                "objective_function": 0.6030126348184148
            },
            "5": {
                "design_vector": [
                    0.8853223593964343,
                    1.1666666666666667,
                    0.8706904435299498
                ],
                "objective_function": 0.5939420579915518
            },
            "6": {
                "design_vector": [
                    0.8853223593964343,
                    1.1666666666666667,
                    0.8706904435299498
                ],
                "objective_function": 0.5921603367073226
            },
            "7": {
                "design_vector": [
                    0.8853223593964343,
                    1.1666666666666667,
                    0.8706904435299498
                ],
                "objective_function": 0.5921113888021745
            },
            "8": {
                "design_vector": [
                    0.8853223593964343,
                    1.1666666666666667,
                    0.8706904435299498
                ],
                "objective_function": 0.5838396158855719
            },
            "9": {
                "design_vector": [
                    0.8853223593964343,
                    1.1666666666666667,
                    0.8706904435299498
                ],
                "objective_function": 0.598889113610605
            }
        },
        "optim": {
            "x_optim": [
                0.8853223593964343,
                1.1666666666666667,
                0.8706904435299498
            ],
            "x_observation_windows": [
                [
                    60390,
                    60393
                ],
                [
                    60396.0,
                    60396.8853223594
                ],
                [
                    60399.8853223594,
                    60401.05198902606
                ],
                [
                    60404.05198902606,
                    60404.92267946959
                ]
            ],
            "x_skm_epochs": [
                60393,
                60396.8853223594,
                60401.05198902606,
                60404.92267946959
            ]
        }
    },
    "2": {
        "threshold": 60393,
        "skm_to_od_duration": 3,
        "duration": 16,
        "model": {
            "dynamic": {
                "model_type": "high_fidelity",
                "model_name": "point_mass",
                "model_number": 0
            },
            "truth": {
                "model_type": "high_fidelity",
                "model_name": "point_mass",
                "model_number": 0
            }
        },
        "history": {
            "0": {
                "design_vector": [
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_function": 0.6498453954257764
            },
            "1": {
                "design_vector": [
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_function": 0.6304244477644483
            },
            "2": {
                "design_vector": [
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_function": 0.6642961290998252
            },
            "3": {
                "design_vector": [
                    1.0,
                    1.05,
                    1.0
                ],
                "objective_function": 0.6841665198647512
            },
            "4": {
                "design_vector": [
                    0.9528120713305896,
                    1.1011659807956105,
                    0.9967078189300413
                ],
                "objective_function": 0.6344332801629385
            },
            "5": {
                "design_vector": [
                    0.9528120713305896,
                    1.1011659807956105,
                    0.9967078189300413
                ],
                "objective_function": 0.6245737871268641
            },
            "6": {
                "design_vector": [
                    0.9528120713305896,
                    1.1011659807956105,
                    0.9967078189300413
                ],
                "objective_function": 0.6244195108826376
            },
            "7": {
                "design_vector": [
                    0.9528120713305896,
                    1.1011659807956105,
                    0.9967078189300413
                ],
                "objective_function": 0.6285027083179426
            },
            "8": {
                "design_vector": [
                    0.9528120713305896,
                    1.1011659807956105,
                    0.9967078189300413
                ],
                "objective_function": 0.627754852352925
            },
            "9": {
                "design_vector": [
                    0.9059078994657919,
                    1.1489176597213988,
                    0.9940670742377804
                ],
                "objective_function": 0.6186927613161652
            }
        },
        "optim": {
            "x_optim": [
                0.9059078994657919,
                1.1489176597213988,
                0.9940670742377804
            ],
            "x_observation_windows": [
                [
                    60390,
                    60393
                ],
                [
                    60396.0,
                    60396.90590789947
                ],
                [
                    60399.90590789947,
                    60401.05482555919
                ],
                [
                    60404.05482555919,
                    60405.04889263343
                ]
            ],
            "x_skm_epochs": [
                60393,
                60396.90590789947,
                60401.05482555919,
                60405.04889263343
            ]
        }
    }
}


# Transform dictionaries and get statistics
combined_history_dict = helper_functions.get_combined_history_dict(data)
monte_carlo_stats_dict = utils.get_monte_carlo_stats_dict(combined_history_dict)

# Save total statistics dictionary
label="blablabla"
utils.save_dicts_to_folder(dicts=[monte_carlo_stats_dict], custom_sub_folder_name=label, labels=["stats_"+label])
