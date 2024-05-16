# Standard
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

# Define path to import src files
file_name = os.path.splitext(os.path.basename(__file__))[0]
file_directory = os.path.realpath(__file__)
for _ in range(5):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from src import NavigationSimulator



#################################################################
###### Generate observation windows #############################
#################################################################


def get_random_arc_observation_windows(duration=28, skm_to_arc_duration_vars=[3.5, 0.1], threshold_vars=[0.5, 0.001], arc_duration_vars=[0.5, 0.1], seed=0, simulation_start_epoch=60390):

    np.random.seed(seed)

    skm_to_arc_duration = np.random.normal(loc=skm_to_arc_duration_vars[0], scale=skm_to_arc_duration_vars[1], size=100)
    arc_duration = np.random.normal(loc=arc_duration_vars[0], scale=arc_duration_vars[1], size=100)
    threshold = np.random.normal(loc=threshold_vars[0], scale=threshold_vars[1], size=100)

    # Generate a vector with OD durations
    epoch = simulation_start_epoch + threshold[0] + skm_to_arc_duration[0] + arc_duration[0]
    skm_epochs = []
    i = 1
    while True:
        if epoch <= simulation_start_epoch+duration:
            skm_epochs.append(epoch)
            epoch += skm_to_arc_duration[i]+arc_duration[i]
        else:
            design_vector = arc_duration[i]*np.ones(np.shape(skm_epochs))
            break
        i += 1

    # Extract observation windows
    observation_windows = [(simulation_start_epoch, simulation_start_epoch+threshold[0])]
    for i, skm_epoch in enumerate(skm_epochs):
        observation_windows.append((skm_epoch-arc_duration[i+1], skm_epoch))

    return observation_windows


def get_constant_arc_observation_windows(duration=28, skm_to_arc_duration=3.5, threshold=0.5, arc_duration=0.5, simulation_start_epoch=60390):

    # Generate a vector with OD durations
    epoch = simulation_start_epoch + threshold + skm_to_arc_duration + arc_duration
    skm_epochs = []
    i = 1
    while True:
        if epoch <= simulation_start_epoch+duration:
            skm_epochs.append(epoch)
            epoch += skm_to_arc_duration+arc_duration
        else:
            design_vector = arc_duration*np.ones(np.shape(skm_epochs))
            break
        i += 1

    # Extract observation windows
    observation_windows = [(simulation_start_epoch, simulation_start_epoch+threshold)]
    for i, skm_epoch in enumerate(skm_epochs):
        observation_windows.append((skm_epoch-arc_duration, skm_epoch))

    return observation_windows


def get_orbit_based_arc_observation_windows(duration=28, period=0.4597, step_size=0.01, simulation_start_epoch=60390, margin=0.05,  apolune=False, pass_interval=2, threshold=0):

    ### Constant arc, around perilune
    epochs = np.arange(0, duration, step_size) + simulation_start_epoch
    total_indices = len(epochs)
    pass_interval += 1
    pass_interval_index = int(pass_interval*period/step_size)
    period = int(period/step_size)
    margin = int(margin/step_size)
    threshold_index = int(threshold/step_size)

    if apolune:
        indices = np.arange(0+int(period/2), total_indices, period)
    else:
        indices = np.arange(0, total_indices, period)


    if pass_interval == 0:
        pass_interval = None

    ranges = []
    if threshold > 0:
        ranges = [(0, threshold_index)]
    ranges.extend([(index-margin, index+margin) for index in indices[indices > threshold_index+pass_interval_index][::pass_interval]])

    observation_windows = []
    for start_index, end_index in ranges:
        values = epochs[start_index:end_index+1]
        observation_windows.append((min(values), max(values)))

    return observation_windows


# print(get_orbit_based_arc_observation_windows(5, margin=0.05, step_size=0.01, threshold=0.5, pass_interval=1))



#################################################################
###### Generate NavigationOutput objects ########################
#################################################################

def generate_navigation_outputs(observation_windows_settings, seed=0, **kwargs):

    np.random.seed(seed)

    # Run the navigation routine using given settings
    navigation_outputs = {}
    for window_type in observation_windows_settings.keys():

        navigation_output_per_type = []
        for (observation_windows, num_runs) in observation_windows_settings[window_type]:

            navigation_output_per_run = {}
            for run in range(num_runs):

                navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows, **kwargs)
                navigation_output_per_run[run] = navigation_simulator.perform_navigation()

            navigation_output_per_type.append(navigation_output_per_run)

        navigation_outputs[window_type] = navigation_output_per_type

    return navigation_outputs


def generate_navigation_outputs_parameter_sensitivity(observation_windows_settings, arg_dict, **kwargs):

    navigation_outputs_sensitivity = {}
    for arg_name, arg_values in arg_dict.items():
        for arg_index, arg_value in enumerate(arg_values):

            print("Input: \n", {arg_name: arg_value})
            # Run the navigation routine using given settings
            navigation_outputs = generate_navigation_outputs(observation_windows_settings, **{arg_name: arg_value}, **kwargs)

            for type_index, (window_type, navigation_outputs_cases) in enumerate(navigation_outputs.items()):
                for case_index, window_case in enumerate(navigation_outputs_cases):

                    if window_type not in navigation_outputs_sensitivity:
                        navigation_outputs_sensitivity[window_type] = {}

                    if arg_name not in navigation_outputs_sensitivity[window_type]:
                        navigation_outputs_sensitivity[window_type][arg_name] = [window_case]

                    else:
                        navigation_outputs_sensitivity[window_type][arg_name].append(window_case)

    return navigation_outputs_sensitivity


def generate_observation_windows_sensitivity_settings(duration, observation_windows_settings, observation_window_sensitivity_settings):

    observation_windows_sensitivity_settings = {}

    for window_type in observation_windows_settings.keys():

        if window_type not in observation_windows_sensitivity_settings:
            observation_windows_sensitivity_settings[window_type] = {}

        for arg_name, arg_values in observation_window_sensitivity_settings.items():

            for arg_value in arg_values:

                # print({arg_name: arg_value})
                observation_windows = get_constant_arc_observation_windows(duration, **{arg_name: arg_value})

                if arg_name not in observation_windows_sensitivity_settings[window_type]:
                    observation_windows_sensitivity_settings[window_type][arg_name] = {arg_value: observation_windows}

                else:
                    observation_windows_sensitivity_settings[window_type][arg_name].update({arg_value: observation_windows})

    return observation_windows_sensitivity_settings


def generate_objective_value_results(navigation_outputs):

    # Get objective value history
    objective_value_results = {}
    for window_type in navigation_outputs.keys():

        objective_value_results_per_window_case = []
        for window_case, navigation_output_list in enumerate(navigation_outputs[window_type]):

            objective_values = []
            delta_v_per_skm_list = []
            for run, navigation_output in navigation_output_list.items():

                print(f"Results for {window_type} window_case {window_case} run {run}:")

                # Extracting the relevant objects
                navigation_results = navigation_output.navigation_results
                navigation_simulator = navigation_output.navigation_simulator

                # Extracting the relevant results from objects
                delta_v = navigation_results[8][1]
                delta_v_per_skm = np.linalg.norm(delta_v, axis=1)
                objective_value = np.sum(delta_v_per_skm)

                delta_v_per_skm_list.append(delta_v_per_skm.tolist())
                objective_values.append(objective_value)

                print("Objective: ", delta_v_per_skm, objective_value)

            objective_value_results_per_window_case.append((len(objective_values),
                                                        min(objective_values),
                                                        max(objective_values),
                                                        np.mean(objective_values),
                                                        np.std(objective_values),
                                                        objective_values,
                                                        delta_v_per_skm_list))

        objective_value_results[window_type] = objective_value_results_per_window_case

    return objective_value_results


# def generate_objective_value_results(navigation_outputs):

#     # Get objective value history
#     orbit_determination_error_results = {}
#     for window_type in navigation_outputs.keys():

#         orbit_determination_error_results_per_window_case = {}
#         for window_case, navigation_output_list in enumerate(navigation_outputs[window_type]):

#             orbit_determination_errors_per_run = {}
#             objective_values = []
#             delta_v_per_skm_list = []
#             for run, navigation_output in navigation_output_list.items():

#                 print(f"Results for {window_type} window_case {window_case} run {run}:")

#                 # Extracting the relevant objects
#                 navigation_results = navigation_output.navigation_results
#                 navigation_simulator = navigation_output.navigation_simulator

#                 # Extracting the relevant results from objects
#                 delta_v = navigation_results[8][1]
#                 delta_v_per_skm = np.linalg.norm(delta_v, axis=1)
#                 objective_value = np.sum(delta_v_per_skm)

#                 delta_v_per_skm_list.append(delta_v_per_skm.tolist())
#                 objective_values.append(objective_value)

#                 print("Objective: ", delta_v_per_skm, objective_value)

#                 orbit_determination_errors_per_run[run] = delta_v_per_skm.tolist()

#             orbit_determination_error_results_per_window_case[window_case] = {orbit_determination_errors_per_run, (len(objective_values),
#                                                         min(objective_values),
#                                                         max(objective_values),
#                                                         np.mean(objective_values),
#                                                         np.std(objective_values),
#                                                         objective_values)}

#         orbit_determination_error_results[window_type] = orbit_determination_error_results_per_window_case

#     return orbit_determination_error_results










def generate_orbit_determination_error_results(navigation_outputs):

    # Get objective value history
    orbit_determination_error_results = {}
    for window_type in navigation_outputs.keys():

        orbit_determination_error_results_per_window_case = {}
        for window_case, navigation_output_list in enumerate(navigation_outputs[window_type]):

            orbit_determination_errors_per_run = {}
            for run, navigation_output in navigation_output_list.items():

                print(f"Results for {window_type} window_case {window_case} run {run}:")

                # Extracting the relevant objects
                navigation_results = navigation_output.navigation_results
                navigation_simulator = navigation_output.navigation_simulator

                full_estimation_error_dict = navigation_simulator.full_estimation_error_dict
                epochs = list(full_estimation_error_dict.keys())

                orbit_determination_errors_per_window = {}
                for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):

                    # Calculate the absolute difference between each value and the target
                    differences = [abs(epoch - end_epoch) for epoch in epochs]
                    closest_index = differences.index(min(differences))
                    end_epoch = epochs[closest_index]

                    orbit_determination_error = full_estimation_error_dict[end_epoch]
                    orbit_determination_errors_per_window[window_index] = orbit_determination_error.tolist()

                orbit_determination_errors_per_run[run] = orbit_determination_errors_per_window

            orbit_determination_error_results_per_window_case[window_case] = orbit_determination_errors_per_run

        orbit_determination_error_results[window_type] = orbit_determination_error_results_per_window_case

    return orbit_determination_error_results



def generate_reference_orbit_deviation_results(navigation_outputs):

    # Get objective value history
    orbit_determination_error_results = {}
    for window_type in navigation_outputs.keys():

        orbit_determination_error_results_per_window_case = {}
        for window_case, navigation_output_list in enumerate(navigation_outputs[window_type]):

            orbit_determination_errors_per_run = {}
            for run, navigation_output in navigation_output_list.items():

                print(f"Results for {window_type} window_case {window_case} run {run}:")

                # Extracting the relevant objects
                navigation_results = navigation_output.navigation_results
                navigation_simulator = navigation_output.navigation_simulator

                full_reference_state_deviation_dict = navigation_simulator.full_reference_state_deviation_dict
                epochs = list(full_reference_state_deviation_dict.keys())

                orbit_determination_errors_per_window = {}
                for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):

                    # Calculate the absolute difference between each value and the target
                    differences = [abs(epoch - end_epoch) for epoch in epochs]
                    closest_index = differences.index(min(differences))
                    end_epoch = epochs[closest_index]

                    reference_state_deviation = full_reference_state_deviation_dict[end_epoch]
                    orbit_determination_errors_per_window[window_index] = reference_state_deviation.tolist()

                orbit_determination_errors_per_run[run] = orbit_determination_errors_per_window

            orbit_determination_error_results_per_window_case[window_case] = orbit_determination_errors_per_run

        orbit_determination_error_results[window_type] = orbit_determination_error_results_per_window_case

    return orbit_determination_error_results



def generate_global_uncertainty_results(navigation_outputs):

    # Get objective value history
    orbit_determination_error_results = {}
    for window_type in navigation_outputs.keys():

        orbit_determination_error_results_per_window_case = {}
        for window_case, navigation_output_list in enumerate(navigation_outputs[window_type]):

            orbit_determination_errors_per_run = {}
            for run, navigation_output in navigation_output_list.items():

                print(f"Results for {window_type} window_case {window_case} run {run}:")

                # Extracting the relevant objects
                navigation_results = navigation_output.navigation_results
                navigation_simulator = navigation_output.navigation_simulator

                # Extracting the relevant results from objects
                covariance_dict = navigation_simulator.full_propagated_covariance_dict

                covariance_epochs = np.stack(list(covariance_dict.keys()))
                covariance_history = np.stack(list(covariance_dict.values()))

                covariance_history_lpf = covariance_history[:, :6, :6]
                covariance_history_lumio = covariance_history[:, 6:, 6:]

                # beta_lpf = 3*np.max(np.sqrt(np.abs(np.linalg.eigvals(covariance_history_lpf))), axis=1)
                # beta_lumio = 3*np.max(np.sqrt(np.abs(np.linalg.eigvals(covariance_history_lumio))), axis=1)

                beta_aves = []
                for i in range(2):

                    beta_1 = np.max(np.sqrt(np.abs(np.linalg.eigvals(covariance_history[:, 3*i+0:3*i+3, 3*i+0:3*i+3]))), axis=1)
                    beta_2 = np.max(np.sqrt(np.abs(np.linalg.eigvals(covariance_history[:, 3*i+6:3*i+3+6, 3*i+6:3*i+3+6]))), axis=1)

                    beta_bar_1 = np.mean(beta_1)
                    beta_bar_2 = np.mean(beta_2)

                    beta_ave = 1/2*(beta_bar_1+beta_bar_2)

                    beta_aves.append(beta_ave)

                orbit_determination_errors_per_run[run] = beta_aves

            orbit_determination_error_results_per_window_case[window_case] = orbit_determination_errors_per_run

        orbit_determination_error_results[window_type] = orbit_determination_error_results_per_window_case

    return orbit_determination_error_results



#################################################################
###### Plotting bar chart  ######################################
#################################################################


def bar_plot(ax, data, group_stretch=0.8, bar_stretch=0.95,
             legend=True, x_labels=True, label_fontsize=8,
             colors=None, barlabel_offset=1,
             bar_labeler=lambda k, i, s: str(round(s, 3))):

    std_data = {window_type: [case_result[4] for case_result in case_results] for window_type, case_results in data.items()}
    data = {window_type: [case_result[3] for case_result in case_results] for window_type, case_results in data.items()}

    sorted_data = list(data.items())
    sorted_k, sorted_v  = zip(*sorted_data)
    max_n_bars = max(len(v) for v in data.values())
    group_centers = np.cumsum([max_n_bars
                               for _ in sorted_data]) - (max_n_bars / 2)
    bar_offset = (1 - bar_stretch) / 2
    bars = defaultdict(list)

    if colors is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # colors = {g_name: [f"C{i}" for _ in values]
        #           for i, (g_name, values) in enumerate(data.items())}
        colors = {g_name: color_cycle[i]
                for i, (g_name, values) in enumerate(data.items())}

    ax.grid(alpha=0.5)
    ax.set_xticks(group_centers)
    ax.set_xlabel("Tracking window scenario")
    ax.set_ylabel(r'||$\Delta V$|| [m/s]')
    ax.set_title(f'Station keeping costs, simulation of {28} [days]')

    for g_i, ((g_name, vals), g_center) in enumerate(zip(sorted_data,
                                                         group_centers)):

        print(g_name, vals)

        n_bars = len(vals)
        group_beg = g_center - (n_bars / 2) + (bar_stretch / 2)
        for val_i, val in enumerate(vals):

            bar = ax.bar(group_beg + val_i + bar_offset,
                         height=val, width=bar_stretch,
                         color=colors[g_name],
                         yerr=std_data[g_name][val_i],
                         capsize=4)[0]
            bars[g_name].append(bar)
            if bar_labeler is not None:
                x_pos = bar.get_x() + (bar.get_width() / 2.0)
                y_pos = val + barlabel_offset
                barlbl = bar_labeler(g_name, val_i, val)
                ax.text(x_pos, y_pos, barlbl, ha="center", va="bottom",
                        fontsize=label_fontsize)

    # if legend:
    #     ax.legend([bars[k][0] for k in sorted_k if len(bars[k]) !=0], sorted_k, title="Details", bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
        # ax.legend(title="Details", bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

    if x_labels:
        ax.set_xticklabels(sorted_k)
    else:
        ax.set_xticklabels()

    plt.tight_layout()

    return bars, group_centers