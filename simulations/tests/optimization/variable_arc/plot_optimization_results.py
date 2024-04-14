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

from tests import utils, helper_functions


#################################################################
###### Monte Carlo test case ####################################
#################################################################

# Extract information from specific folder
folder_name = "PM01_PM01_thr3_skm3_dur28_od1_num5_max20_err1e-10"
batch_name = "09042312"
folder_name = "PM01_PM01_thr3_skm3_dur28_od1_num5_max20_err1e-2"
batch_name = "09042312"
folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", folder_name)

concatenated_json = helper_functions.concatenate_json_files(folder_path, batch=batch_name)
monte_carlo_stats_dict = helper_functions.get_monte_carlo_stats_dict(concatenated_json)

# Extract information directly
# file_name = f"{batch_name}_stats_{folder_name}.json"
# file_path = os.path.join(folder_path, file_name)
# monte_carlo_stats_dict = helper_functions.load_json_file(file_path)

fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
epochs = np.array(list(monte_carlo_stats_dict["mean"]["design_vector"].keys()))
mean_history = np.array(list(monte_carlo_stats_dict["mean"]["design_vector"].values()))
std_history = np.array(list(monte_carlo_stats_dict["std_dev"]["design_vector"].values()))
mean_history_obj = np.array(list(monte_carlo_stats_dict["mean"]["objective_value"].values()))
std_history_obj = np.array(list(monte_carlo_stats_dict["std_dev"]["objective_value"].values()))

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(len(mean_history[0])):
    axs[0].plot(epochs, mean_history[:, i],  label=f"State {i+1}", color=color_cycle[i+1])
    axs[0].errorbar(epochs, mean_history[:, i], yerr=std_history[:, i], fmt='o', capsize=2, color=color_cycle[i+1], alpha=0.3)

for j in range(monte_carlo_stats_dict["num_runs"]):
    run_dict = helper_functions.load_json_file(os.path.join(folder_path, f"{batch_name}_run_{j}_{folder_name}.json"))
    design_vectors = np.array(list(run_dict["history"]["design_vector"].values()))
    objective_values = np.array(list(run_dict["history"]["objective_value"].values()))
    for i in range(len(mean_history[0])):
        axs[0].plot(epochs, design_vectors[:, i], color=color_cycle[i+1], alpha=0.2)

    axs[1].plot(epochs, objective_values, color="gray", alpha=0.2)
    axs[1].errorbar(epochs, mean_history_obj, yerr=std_history_obj, fmt='o', capsize=2, color="gray", alpha=0.3)

axs[1].plot(epochs, mean_history_obj, color="gray")

for ax in axs:
    ax.grid(alpha=0.3)
axs[0].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
axs[1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
axs[0].set_ylabel("OD durations [days]")
axs[1].set_ylabel(r"Objective ||$\Delta$V|| [m/s]")
axs[1].set_xlabel("Iteration [-]")
axs[1].set_xticks(range(len(epochs)+1))
fig.suptitle(f"Iteration histories for optimization runs \n ")
plt.tight_layout()
plt.legend()
plt.show()



# n={monte_carlo_stats_dict["num_runs"]}, model: {1}

