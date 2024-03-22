# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import scipy as sp
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Own
from optimization_models import OptimizationModel
from dynamic_models import PlotNavigationResults, NavigationSimulator
from tests import utils


# dynamic_model_list = ["low_fidelity", "three_body_problem", 0]
# truth_model_list = ["low_fidelity", "three_body_problem", 0]
# dynamic_model_list = ["high_fidelity", "point_mass", 0]
# truth_model_list = ["high_fidelity", "point_mass", 0]
# dynamic_model_list = ["high_fidelity", "point_mass_srp", 0]
# truth_model_list = ["high_fidelity", "point_mass_srp", 0]
# dynamic_model_list = ["high_fidelity", "spherical_harmonics_srp", 1]
# truth_model_list = ["high_fidelity", "spherical_harmonics_srp", 1]

#################################################################################
###### Test runs ####
#################################################################################


observation_windows = [(60390, 60390.5), (60391, 60391.5), (60395, 60397), (60400, 60401)]
# observation_windows = [(60390, 60390.5), (60390.5, 60391), (60395, 60397), (60400, 60401)]

observation_windows = [(60390, 60392), (60392, 60394), (60394, 60396), (60398, 60400)]
# observation_windows = [(60390, 60391), (60392, 60393), (60394, 60395), (60398, 60400)]
# observation_windows = [(60390, 60390.1), (60390.1, 60390.2), (60390.2, 60390.3), (60390.3, 60390.4), (60390.4, 60390.5), (60392, 60393), (60394, 60395),
# (60395, 60395.1), (60395.1, 60395.2), (60395.2, 60395.3), (60395.3, 60395.4), (60395.4, 60395.5)]

# observation_windows = [(60390, 60390.5), (60390.5, 60391), (60391, 60391.5), (60391.5, 60392), (60392, 60392.5), (60394, 60395)]

# navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
#                                                                 ["high_fidelity", "point_mass_srp", 0],
#                                                                 ["high_fidelity", "point_mass_srp", 0],
#                                                                 include_station_keeping=True,
#                                                                 step_size=1e-2)

# navigation_results = navigation_simulator.get_navigation_results()

# navigation_simulator.plot_navigation_results(navigation_results, show_directly=True)
# threshold = 7
# duration = 20
# od_duration = 1
# skm_to_od_duration = 2
# optimization_model = OptimizationModel.OptimizationModel(["high_fidelity", "point_mass_srp", 0], ["high_fidelity", "point_mass_srp", 0], threshold=threshold, skm_to_od_duration=skm_to_od_duration, duration=duration, od_duration=od_duration)
# delta_v = optimization_model.objective_function(optimization_model.initial_design_vector)
# delta_v_list.append(delta_v)


threshold = 7
duration = 20
od_duration = 1
skm_to_od_duration = 2
delta_v_list = []
for threshold in [7]:
    for i in range(1):

        optimization_model = OptimizationModel.OptimizationModel(["high_fidelity", "point_mass_srp", 0], ["high_fidelity", "point_mass_srp", 0], threshold=threshold, skm_to_od_duration=skm_to_od_duration, duration=duration, od_duration=od_duration)
        delta_v = optimization_model.objective_function(optimization_model.initial_design_vector)
        delta_v_list.append(delta_v)

    print("delta_v_list", delta_v_list)
plt.show()


#1 PMSRP, SHSRP
#2 PM, PMSRP
#3 PMSRP, SHSRP

#################################################################################
###### Monte Carlo run on optimal delta_v based on different on-board models ####
#################################################################################

## Varying the skm_to_od_duration
threshold = 7
duration = 14
od_duration = 1
num_runs = 2
delta_v_per_skm_to_od_duration_dict = dict()
# for model in ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]:
for model in ["point_mass"]:
    delta_v_dict = dict()
    for skm_to_od_duration in [0.5, 1.5]:

        delta_v_list = []
        for run in range(num_runs):

            optimization_model = OptimizationModel.OptimizationModel(["high_fidelity", model, 0], ["high_fidelity", model, 0], threshold=threshold, skm_to_od_duration=skm_to_od_duration, duration=duration, od_duration=od_duration)
            x = optimization_model.initial_design_vector

            # start_time = time.time()
            delta_v = optimization_model.objective_function(x)
            # run_time = time.time() - start_time
            delta_v_list.append(delta_v)

        delta_v_dict[skm_to_od_duration] = delta_v_list

    delta_v_per_skm_to_od_duration_dict[model] = delta_v_dict

print(delta_v_per_skm_to_od_duration_dict)



data = delta_v_per_skm_to_od_duration_dict
utils.save_nested_dict_to_json(data, file_name='delta_v_per_skm_to_od_duration_dict.json')

stats = utils.get_monte_carlo_statistics(data)

print(stats)
utils.save_nested_dict_to_json(stats, file_name='monte_carlo_delta_v_per_skm_to_od_duration_dict.json')









# data = delta_v_per_skm_to_od_duration_dict


# Plot bar chart of delta v per model per skm frequency
groups = list(data.keys())
inner_keys = list(data[groups[0]].keys())

print(inner_keys)
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
plt.show()




model = "point_mass"
threshold = 7
skm_to_od_duration = 3
duration = 28
od_duration = 1

optimization_model = OptimizationModel.OptimizationModel(["high_fidelity", model, 0], ["high_fidelity", model, 0], threshold=threshold, skm_to_od_duration=skm_to_od_duration, duration=duration, od_duration=od_duration)
result_dict = optimization_model.optimize()
print(result_dict)



