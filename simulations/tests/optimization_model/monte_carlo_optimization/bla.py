# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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


# Extract information from specific folder
# folder_name = "PM01_PM01_thr3_skm3_dur28_od1_num10_max20"
folder_name = "PM01_PM01_thr3_skm3_dur28_od1_num10_max20"
folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", folder_name)
print(folder_path)

concatenated_json = helper_functions.concatenate_json_files(folder_path)
# combined_history_dict = helper_functions.get_combined_history_dict(concatenated_json)
monte_carlo_stats_dict = helper_functions.get_monte_carlo_stats_dict(concatenated_json)
print(concatenated_json)
# print(monte_carlo_stats_dict)
# Extract information directly
# file_name = "stats1_bla.json"
# file_path = os.path.join(folder_path, file_name)
# monte_carlo_stats_dict = helper_functions.load_json_file(file_path)



fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
epochs = np.array(list(monte_carlo_stats_dict["mean"]["design_vector"].keys()))
mean_history = np.array(list(monte_carlo_stats_dict["mean"]["design_vector"].values()))
std_history = np.array(list(monte_carlo_stats_dict["std_dev"]["design_vector"].values()))
mean_history_obj = np.array(list(monte_carlo_stats_dict["mean"]["objective_value"].values()))

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(len(mean_history[0])):
    axs[0].plot(epochs, mean_history[:, i],  label=f"State {i}", color=color_cycle[i])
    # axs[0].errorbar(epochs, mean_history[:, i], yerr=std_history[:, i], fmt='o', capsize=5, color=color_cycle[i])

for j in range(len(concatenated_json)):
    original_string = "PM01_PM01_thr3_skm3_dur28_od1_num10_max20"
    run_dict = helper_functions.load_json_file(os.path.join(folder_path, f"01042300_run_{j}_{original_string}.json"))
    design_vectors = np.array(list(run_dict["history"]["design_vector"].values()))
    objective_values = np.array(list(run_dict["history"]["objective_value"].values()))
    for i in range(len(mean_history[0])):
        # plt.plot(epochs, design_vectors[:, i], color=color_cycle[i])
        axs[0].plot(epochs, design_vectors[:, i], color=color_cycle[i], alpha=0.2)
    axs[1].plot(epochs, objective_values, color=color_cycle[i], alpha=0.2)
axs[1].plot(epochs, mean_history_obj, color=color_cycle[i], label="Mean")

for ax in axs:
    ax.grid(alpha=0.3)
    ax.legend()
axs[0].set_ylabel("State entry value [days]")
axs[1].set_ylabel(r"||$\Delta$V||")
axs[1].set_xlabel("Iteration [-]")
axs[1].set_xticks(range(len(epochs)))

# concatenated_json["0"]["model"]["dynamic"]["model_name"]
fig.suptitle(f"Iteration histories for optimization runs \n n={len(concatenated_json)}, model: {1}")

plt.legend()
plt.show()
