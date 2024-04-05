# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import time

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Own
from tests import utils
import reference_data, Interpolator, FrameConverter
from src.dynamic_models.LF.CRTBP import *
from src.dynamic_models.HF.PM import *
from src.dynamic_models.HF.PMSRP import *
from src.dynamic_models.HF.SH import *
from src.dynamic_models.HF.SHSRP import *
from src.dynamic_models.FF.TRUTH import *



#################################
#### Dynamic model run times ####
#################################

def run_times():

    custom_model_dict = {"LF": ["CRTBP"], "HF": ["PM", "PMSRP", "SH", "SHSRP"], "FF": ["FF"]}
    # custom_model_dict = {"LF": ["CRTBP"], "HF": ["PM"]}
    propagation_time = 1
    get_only_first = False
    start_epoch = 60390
    end_epoch = 60404
    n = 5
    run_cases = np.linspace(start_epoch, end_epoch, n)

    # Initialize dictionaries to store accumulated values
    dynamic_model_objects = utils.get_dynamic_model_objects(start_epoch, propagation_time, custom_model_dict=custom_model_dict, get_only_first=get_only_first)
    accumulator_dict = {
        fidelity_key: {subkey: {i: [] for i, value in enumerate(values)}
            for subkey, values in sub_dict.items()
        }
        for fidelity_key, sub_dict in dynamic_model_objects.items()
    }

    # Start collecting the simulation results
    for run_case in run_cases:
        params = (run_case, propagation_time, custom_model_dict, get_only_first)
        run_times_dict = utils.get_dynamic_model_results(*params, step_size=0.1, entry_list=[-1])

        # Accumulate values during the loop
        for fidelity_key, sub_dict in run_times_dict.items():
            for i, (subkey, subvalue_list) in enumerate(sub_dict.items()):
                for j, subvalue in enumerate(subvalue_list):
                    for k, entry in enumerate(subvalue):
                        accumulator_dict[fidelity_key][subkey][j].append(entry)

    result_dict = utils.get_monte_carlo_stats_dict(accumulator_dict)

    ### Plot run times for each model
    keys_list = [["CRTBP"], ["PM", "PMSRP", "SH", "SHSRP"], ["FF"]]
    key_count = sum(len(sublist) for sublist in custom_model_dict.values()) #0.75*key_count
    fig, axs = plt.subplots(1, key_count, figsize=(6.4, 0.75*5), sharey=True)
    index = 0
    for i, (model_types, model_names) in enumerate(result_dict.items()):
        for j, (key, values) in enumerate(model_names.items()):
            averages = []
            std_devs = []
            for num, subvalue in values.items():
                averages.append(subvalue["mean"])
                std_devs.append(subvalue["std_dev"])
            axs[index].grid(alpha=0.5, linestyle='--')
            axs[index].bar(range(1, len(averages)+1), averages, yerr=std_devs, ecolor="black", capsize=4, label=key)
            axs[index].set_xlabel(keys_list[i][j])
            axs[index].set_xticks(range(1, 1+max([len(value) for value in model_names.values()])))
            axs[index].set_yscale("log")
            index += 1

    axs[0].set_ylabel('Run time [s]')
    legend_handles = [plt.Line2D([0], [0], color='black', markersize=1, label=r'1$\sigma$ Std Dev')]
    fig.legend(handles=[legend_handles[0]], loc='upper right', fontsize="x-small")
    fig.suptitle(f"Run time dynamic models for {propagation_time} day, n={len(run_cases)*10}, varying MJD ([{start_epoch}, {end_epoch}]) \nProcessor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz, 2208 Mhz", fontsize=8)

    utils.save_dicts_to_folder([accumulator_dict, result_dict], labels=["run_times", "run_times_monte_carlo_statistics"])
    utils.save_figure_to_folder(figs=[fig], labels=["run_times"])

    # plt.show()


run_times()