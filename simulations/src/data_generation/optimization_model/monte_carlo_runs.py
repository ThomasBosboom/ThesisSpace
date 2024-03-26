# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sp

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(parent_dir)

from tests import utils

# Own
from src.optimization_models import OptimizationModel
from src.dynamic_models import PlotNavigationResults, NavigationSimulator


#################################################################################
###### Monte Carlo run on optimal delta_v based on different on-board models ####
#################################################################################


def monte_carlo_skm_interval():

    ## Varying the skm_to_od_duration
    threshold = 7
    duration = 28
    od_duration = 1
    # skm_to_od_duration = 3
    num_runs = 2
    delta_v_per_skm_interval_dict = dict()
    for model in ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]:
    # for model in ["point_mass"]:
        delta_v_dict = dict()
        for skm_to_od_duration in np.arange(1, 4.5, 0.5):

            delta_v_list = []
            for run in range(num_runs):

                optimization_model = OptimizationModel.OptimizationModel(["high_fidelity", model, 0], ["high_fidelity", model, 0], threshold=threshold, skm_to_od_duration=skm_to_od_duration, duration=duration, od_duration=od_duration)
                # start_time = time.time()
                delta_v = optimization_model.objective_function(optimization_model.initial_design_vector, show_directly=False)
                # run_time = time.time() - start_time
                delta_v_list.append(delta_v)
                print(delta_v)

            delta_v_dict[skm_to_od_duration] = delta_v_list

        delta_v_per_skm_interval_dict[model] = delta_v_dict



    stats = utils.get_monte_carlo_statistics(delta_v_per_skm_interval_dict)

    print(delta_v_per_skm_interval_dict)
    print(stats)

    utils.save_dicts_to_folder([delta_v_per_skm_interval_dict, stats], labels=['delta_v_per_skm_interval_dict', "monte_carlo_delta_v_per_skm_interval_dict"])

    # Plot bar chart of delta v per model per skm frequency
    data = delta_v_per_skm_interval_dict
    groups = list(data.keys())
    inner_keys = list(data[groups[0]].keys())
    num_groups = len(groups)

    fig, ax = plt.subplots(figsize=(8, 3))
    index = np.arange(len(inner_keys))
    bar_width = 0.2

    ax.set_xlabel('SKM interval [days]')
    ax.set_ylabel(r'$\Delta V$ [m/s]')
    ax.set_title(f'Station keeping costs, OD of {od_duration} [day], simulation of {duration} [days]')

    # Center the bars around each xtick
    bar_offsets = np.arange(-(num_groups-1)/2, (num_groups-1)/2 + 1, 1) * bar_width
    for i in range(num_groups):
        values = [data[groups[i]][inner_key][0] for inner_key in inner_keys]
        ax.bar(index + bar_offsets[i], values, bar_width, label=str(groups[i]))

    ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)
    ax.set_xticks(index)
    ax.set_xticklabels([key + od_duration for key in inner_keys])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")
    plt.tight_layout()

    utils.save_figure_to_folder(figs=[fig], labels=["monte_carlo_skm_interval"])

    # plt.show()


monte_carlo_skm_interval()




def monte_carlo_skm_interval_truth():

    ## Varying the skm_to_od_duration
    threshold = 7
    duration = 28
    od_duration = 1
    # skm_to_od_duration = 3
    num_runs = 2
    delta_v_per_skm_interval_dict = dict()
    for model in ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]:
    # for model in ["point_mass"]:
        delta_v_dict = dict()
        for skm_to_od_duration in np.arange(1, 4.5, 0.5):

            delta_v_list = []
            for run in range(num_runs):

                optimization_model = OptimizationModel.OptimizationModel(["high_fidelity", model, 0], ["high_fidelity", "spherical_harmonics_srp", 0], threshold=threshold, skm_to_od_duration=skm_to_od_duration, duration=duration, od_duration=od_duration)
                delta_v = optimization_model.objective_function(optimization_model.initial_design_vector)
                delta_v_list.append(delta_v)
                print(delta_v)

            delta_v_dict[skm_to_od_duration] = delta_v_list

        delta_v_per_skm_interval_dict[model] = delta_v_dict



    stats = utils.get_monte_carlo_statistics(delta_v_per_skm_interval_dict)

    print(delta_v_per_skm_interval_dict)
    print(stats)

    utils.save_dicts_to_folder([delta_v_per_skm_interval_dict, stats], labels=[])

    # Plot bar chart of delta v per model per skm frequency
    data = delta_v_per_skm_interval_dict
    groups = list(data.keys())
    inner_keys = list(data[groups[0]].keys())
    num_groups = len(groups)

    fig, ax = plt.subplots(figsize=(8, 3))
    index = np.arange(len(inner_keys))
    bar_width = 0.2

    ax.set_xlabel('SKM interval [days]')
    ax.set_ylabel(r'$\Delta V$ [m/s]')
    ax.set_title(f'Station keeping costs, OD of {od_duration} [day], simulation of {duration} [days]')

    # Center the bars around each xtick
    bar_offsets = np.arange(-(num_groups-1)/2, (num_groups-1)/2 + 1, 1) * bar_width
    for i in range(num_groups):
        values = [data[groups[i]][inner_key][0] for inner_key in inner_keys]
        ax.bar(index + bar_offsets[i], values, bar_width, label=str(groups[i]))

    ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)
    ax.set_xticks(index)
    ax.set_xticklabels([key + od_duration for key in inner_keys])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")
    plt.tight_layout()

    utils.save_figure_to_folder(figs=[fig], labels=[])

    # plt.show()


monte_carlo_skm_interval_truth()

plt.show()