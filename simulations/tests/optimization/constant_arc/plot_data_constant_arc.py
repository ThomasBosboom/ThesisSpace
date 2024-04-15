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

from tests import utils, helper_functions
from src.optimization_models import OptimizationModel
from src import NavigationSimulator, PlotNavigationResults



#################################################################
###### Show constant OD solutions ###############################
#################################################################

### Plot heatmaps of the total delta_v for different scenarios

# Extract information from specific folder and run
folder_name = "get_data_constant_arc"
folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", folder_name)
file_name = f"14042327_delta_v_dict_constant_arc.json"
file_path = os.path.join(folder_path, file_name)
data = helper_functions.load_json_file(file_path)

input_keys = data["inputs"].keys()
input_values = data["inputs"].values()
print(input_keys)
print(input_values)

# print(data)

case_lists = [['PM', '28', '1', '0.01', '3']]

for i, (key, value) in enumerate(data['inputs'].items()):
   if key == "skm_to_od_durations":
      parameters1 = value
   if key == "od_durations":
      parameters2 = value

print(parameters1, parameters2)

# Create a heatmap_data of values for the heatmap
heatmap_data = np.zeros((len(parameters1), len(parameters2)))

heatmap_data_list = []
for case_list in case_lists:
    for i, parameter1 in enumerate(parameters1):
        for j, parameter2 in enumerate(parameters2):

            current_dict = data
            for key in case_list:
                if key in current_dict:
                    current_dict = current_dict[key]

            # print(current_dict)

            heatmap_data[i, j] = current_dict[str(parameter1)][str(parameter2)][0]
    heatmap_data_list.append(heatmap_data)

print(heatmap_data)

# Plot the heatmap
fig, axs = plt.subplots(1, 1, figsize=(12, 4), sharey=True)
for heatmap_data in heatmap_data_list:
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.yticks(np.arange(len(parameters1)), parameters1)
    plt.xticks(np.arange(len(parameters2)), parameters2)
    plt.xlabel("Arc duration")
    plt.ylabel("Arc separation interval")
    plt.title(r'$\Delta V$ [m/s] for 28 days')
    # plt.show()


# model = "PMSRP"
# Extracting data for plotting
keys = data["inputs"]["skm_to_od_durations"]
subkeys = data["inputs"]["od_durations"]
values = {subkey: [current_dict[str(key)][str(subkey)][0] for key in current_dict] for subkey in subkeys}

# Plotting
fig = plt.figure(figsize=(10, 3))
for i, subkey in enumerate(subkeys):
    plt.bar([j + i * 0.1 for j in range(len(values[subkey]))], values[subkey], width=0.1, label=subkey)

plt.xlabel('Arc separation interval [days]')
plt.ylabel(r'$\Delta V$ [m/s]')
plt.title(f'Station keeping costs, simulation of {28} [days]')
plt.xticks([i + 0.2 for i in range(len(keys))], keys)
plt.yscale('log')
plt.legend(title='Arc durations', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(alpha=0.3)
plt.show()


# # utils.save_dicts_to_folder(figs=[fig], labels=["fig_constant_arc_duration"], custom_sub_folder_name=file_name)




# ### Extract information from specific folder and run
# folder_name = "get_data_initial_navigation_routines"
# file_name = "delta_v_dict_constant_arc_duration_1e-10.json"
# # folder_name = "plot_navigation_routines"
# # file_name = "delta_v_dict_constant.json"
# # folder_name = "get_data_constant_arc"
# # file_name = "13041644_delta_v_dict_constant_arc.json"
# folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", folder_name)
# file_path = os.path.join(folder_path, file_name)
# data = helper_functions.load_json_file(file_path)

# model = "PMSRP"
# # Extracting data for plotting
# subkeys = ["0.2", "0.5", "1", "1.5", "1.8"]
# values = {subkey: [data[model][key][subkey][1] for key in data[model]] for subkey in subkeys}
# print(values)

# # Plotting
# fig = plt.figure(figsize=(10, 3))
# for i, subkey in enumerate(subkeys):
#     plt.bar([j + i * 0.1 for j in range(len(values[subkey]))], values[subkey], width=0.1, label=subkey)

# plt.xlabel('Arc separation interval [days]')
# plt.ylabel(r'||$\Delta V$|| [m/s]')
# plt.title(f'Station keeping costs, simulation of {28} [days]')
# plt.xticks([i + 0.2 for i in range(len(data[model]))], data[model].keys())
# plt.yscale('log')
# plt.legend(title='Arc durations', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(alpha=0.3)
# plt.show()


# # utils.save_dicts_to_folder(figs=[fig], labels=["fig_constant_arc_duration"], custom_sub_folder_name=file_name)
