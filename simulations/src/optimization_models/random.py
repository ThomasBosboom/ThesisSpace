# # General imports
# import numpy as np
# from matplotlib import pyplot as plt
# import os
# import sys
# import scipy as sp
# import time

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# # Own
# from dynamic_models import NavigationSimulator, PlotNavigationResults
# from tests import utils


# data = {
#     "0": {
#         "threshold": 60391,
#         "skm_to_od_duration": 3,
#         "duration": 6,
#         "model": {
#             "dynamic": {
#                 "model_type": "high_fidelity",
#                 "model_name": "point_mass",
#                 "model_number": 0
#             },
#             "truth": {
#                 "model_type": "high_fidelity",
#                 "model_name": "point_mass",
#                 "model_number": 0
#             }
#         },
#         "history": {
#             "0": {
#                 "design_vector": [
#                     1, 2
#                 ],
#                 "objective_function": 0.09064449209356919
#             },
#             "1": {
#                 "design_vector": [
#                     3, 4
#                 ],
#                 "objective_function": 0.99064449209356919
#             }
#         },
#         "optim": {
#             "x_optim": [
#                 0.8499999999999996
#             ],
#             "x_observation_windows": [
#                 [
#                     60390,
#                     60391
#                 ],
#                 [
#                     60394.0,
#                     60394.85
#                 ]
#             ],
#             "x_skm_epochs": [
#                 60391,
#                 60394.85
#             ]
#         }
#     },
#     "1": {
#         "threshold": 60391,
#         "skm_to_od_duration": 3,
#         "duration": 6,
#         "model": {
#             "dynamic": {
#                 "model_type": "high_fidelity",
#                 "model_name": "point_mass",
#                 "model_number": 0
#             },
#             "truth": {
#                 "model_type": "high_fidelity",
#                 "model_name": "point_mass",
#                 "model_number": 0
#             }
#         },
#         "history": {
#             "0": {
#                 "design_vector": [
#                     5, 6
#                 ],
#                 "objective_function": 0.09767742658730819
#             },
#             "1": {
#                 "design_vector": [
#                     7, 8
#                 ],
#                 "objective_function": 0.09664449209356919
#             }
#         },
#         "optim": {
#             "x_optim": [
#                 1.0
#             ],
#             "x_observation_windows": [
#                 [
#                     60390,
#                     60391
#                 ],
#                 [
#                     60394.0,
#                     60395.0
#                 ]
#             ],
#             "x_skm_epochs": [
#                 60391,
#                 60395.0
#             ]
#         }
#     }
# }

# # print(history_values)

# def test():

#     combined_data = {}

#     for key, value in data.items():
#         design_vector = []
#         objective_function = []

#         # Extract design vector and objective function from each dictionary
#         for history_key, history_value in value["history"].items():
#             design_vector.append(history_value["design_vector"])
#             objective_function.append(history_value["objective_function"])

#         # Add combined data to the new dictionary
#         combined_data[key] = {
#             "design_vector": design_vector,
#             "objective_function": objective_function
#         }

#     print(combined_data)

#     utils.save_dicts_to_folder(dicts=[combined_data], labels=["test123"])
#     mc_test = utils.get_monte_carlo_stats_dict(data_dict=combined_data)
#     print(mc_test)
#     utils.save_dicts_to_folder(dicts=[mc_test], labels=["mc_test123"])

# test()



# import numpy as np

# data = {
#     "0": {
#         "design_vector": [
#             [1, 2, 5],
#             [3, 4, 7]
#         ],
#         "objective_function": [
#             0.09064449209356919,
#             0.9906444920935692,
#             0.9906444920935692
#         ]
#     },
#     "1": {
#         "design_vector": [
#             [2, 3],
#             [4, 5]
#         ],
#         "objective_function": [
#             0.09123456789012345,
#             0.9912345678901234
#         ]
#     }
# }

# result = {}

# for key, value in data.items():
#     result[key] = {
#         "design_vector": {
#             "mean": np.mean(value["design_vector"], axis=0).tolist(),
#             "std_dev": np.std(value["design_vector"], axis=0).tolist()
#         },
#         "objective_function": {
#             "mean": np.mean(value["objective_function"]),
#             "std_dev": np.std(value["objective_function"])
#         }
#     }

# print(result)



# result = utils.get_monte_carlo_stats_dict(data)

# print(result)







import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data
data = [[1, 1], [2, 3]]

# Extract x and y values
x_values = [point[0] for point in data]
y_values = [point[1] for point in data]
z_values = np.zeros_like(x_values)  # Z-values are set to 0

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create 3D bars
ax.bar3d(x_values, y_values, z_values, 1, 1, z_values)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Bar Chart')

# Show plot
plt.show()
