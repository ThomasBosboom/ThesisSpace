# Standard
import os
import sys
import re
import numpy as np
import copy
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


def get_random_arc_observation_windows(duration=28, arc_interval_vars=[3.5, 0.1], threshold_vars=[0.5, 0.001], arc_duration_vars=[0.5, 0.1], seed=0, mission_start_epoch=60390):

    np.random.seed(seed)

    arc_interval = np.random.normal(loc=arc_interval_vars[0], scale=arc_interval_vars[1], size=100)
    arc_duration = np.random.normal(loc=arc_duration_vars[0], scale=arc_duration_vars[1], size=100)
    threshold = np.random.normal(loc=threshold_vars[0], scale=threshold_vars[1], size=100)

    # Generate a vector with OD durations
    epoch = mission_start_epoch + threshold[0] + arc_interval[0] + arc_duration[0]
    skm_epochs = []
    i = 1
    while True:
        if epoch <= mission_start_epoch+duration:
            skm_epochs.append(epoch)
            epoch += arc_interval[i]+arc_duration[i]
        else:
            design_vector = arc_duration[i]*np.ones(np.shape(skm_epochs))
            break
        i += 1

    # Extract observation windows
    observation_windows = [(mission_start_epoch, mission_start_epoch+threshold[0])]
    for i, skm_epoch in enumerate(skm_epochs):
        observation_windows.append((skm_epoch-arc_duration[i+1], skm_epoch))

    return observation_windows


def get_constant_arc_observation_windows(duration=28, arc_interval=3, threshold=1, arc_duration=1, mission_start_epoch=60390):

    threshold=arc_duration
    # Generate a vector with OD durations
    epoch = mission_start_epoch + threshold + arc_interval + arc_duration
    skm_epochs = []
    i = 1
    while True:
        if epoch <= mission_start_epoch+duration:
            skm_epochs.append(epoch)
            epoch += arc_interval+arc_duration
        else:
            design_vector = arc_duration*np.ones(np.shape(skm_epochs))
            break
        i += 1

    # Extract observation windows
    observation_windows = [(mission_start_epoch, mission_start_epoch+threshold)]
    for i, skm_epoch in enumerate(skm_epochs):
        observation_windows.append((skm_epoch-arc_duration, skm_epoch))

    return observation_windows


def get_constant_arc_with_subarcs_observation_windows(duration=28, arc_interval=3.5, threshold=[1, 0.1], arc_duration=[0.5, 0.1], mission_start_epoch=60390):

    threshold_subarcs = np.linspace(0, threshold[0], int(threshold[0]/threshold[1]+1))
    arc_subarcs = np.linspace(0, arc_duration[0], int(arc_duration[0]/arc_duration[1]+1))

    # Extract threshold observation windows
    threshold_observation_windows = [(threshold_subarcs[i], threshold_subarcs[i + 1]) for i in range(len(threshold_subarcs) - 1)]
    arc_observation_windows = [(arc_subarcs[i], arc_subarcs[i + 1]) for i in range(len(arc_subarcs) - 1)]

    observation_windows = threshold_observation_windows
    while observation_windows[-1][-1]+arc_interval+arc_subarcs[-1]<duration:
        off_set = observation_windows[-1][-1]+arc_interval
        observation_windows.extend([(tup[0] + off_set, tup[1] + off_set) for tup in arc_observation_windows])

    observation_windows = mission_start_epoch + np.array(observation_windows)
    observation_windows = [tuple(window) for window in observation_windows]

    return observation_windows

# print(get_constant_arc_with_subarcs_observation_windows(duration=28, arc_interval=3.5, threshold=[1, 0.1], arc_duration=[0.5, 0.1], mission_start_epoch=60390))


def get_orbit_based_arc_observation_windows(duration=28, period=0.4597, step_size=0.01, mission_start_epoch=60390, margin=0.05,  apolune=False, pass_interval=7, threshold=0):

    ### Constant arc, around perilune
    epochs = np.arange(0, duration, step_size) + mission_start_epoch
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

# print(get_orbit_based_arc_observation_windows(duration=28, period=0.4597, step_size=0.01, mission_start_epoch=60390, margin=0.05,  apolune=False, pass_interval=7, threshold=0))


#################################################################
###### Generate NavigationOutput objects ########################
#################################################################

def generate_navigation_outputs(observation_windows_settings, **kwargs):

    # Run the navigation routine using given settings
    navigation_outputs = {}
    for window_type in observation_windows_settings.keys():

        navigation_output_per_type = []
        for (observation_windows, num_runs) in observation_windows_settings[window_type]:

            navigation_output_per_run = {}
            for run in range(num_runs):

                navigation_simulator = NavigationSimulator.NavigationSimulator(**kwargs)
                navigation_output_per_run[run] = navigation_simulator.perform_navigation(observation_windows, seed=run)

            navigation_output_per_type.append(navigation_output_per_run)

        navigation_outputs[window_type] = navigation_output_per_type

    return navigation_outputs


def generate_navigation_outputs_sensitivity_analysis(num_runs, sensitivity_settings, default_window_inputs, **kwargs):

    observation_windows_settings = {
        "Constant": [
            (get_constant_arc_observation_windows(**default_window_inputs), num_runs),
        ]
    }

    navigation_outputs_sensitivity = {}

    for arg_name, arg_values in sensitivity_settings.items():
        for arg_index, arg_value in enumerate(arg_values):

            print("Input: \n", {arg_name: arg_value})

            for key, value in kwargs.items():
                if key==arg_name:
                    kwargs.pop(key)

            if arg_name in ["threshold", "arc_interval", "arc_duration", "mission_start_epoch"]:

                window_inputs = copy.deepcopy(default_window_inputs)
                window_inputs[arg_name] = arg_value
                observation_windows_sensitivity_settings = {}
                for window_type in observation_windows_settings.keys():

                    observation_windows_sensitivity_settings[window_type] = [(get_constant_arc_observation_windows(**window_inputs), num_runs)]

                navigation_outputs = generate_navigation_outputs(observation_windows_sensitivity_settings, **{arg_name: arg_value}, **kwargs)

            else:
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


def generate_objective_value_results(navigation_outputs, evaluation_threshold=14):

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
                delta_v_dict = navigation_simulator.delta_v_dict
                delta_v_epochs = np.stack(list(delta_v_dict.keys()))
                delta_v_history = np.stack(list(delta_v_dict.values()))
                delta_v = sum(np.linalg.norm(value) for key, value in delta_v_dict.items() if key > navigation_simulator.mission_start_epoch+evaluation_threshold)

                delta_v_per_skm = np.linalg.norm(delta_v_history, axis=1)
                delta_v_per_skm_list.append(delta_v_per_skm.tolist())
                objective_values.append(delta_v)

                print("Objective: ", delta_v_per_skm, delta_v)

            objective_value_results_per_window_case.append((len(objective_values),
                                                        min(objective_values),
                                                        max(objective_values),
                                                        np.mean(objective_values),
                                                        np.std(objective_values),
                                                        objective_values,
                                                        delta_v_per_skm_list))

        objective_value_results[window_type] = objective_value_results_per_window_case

    return objective_value_results


#################################################################
###### Plotting bar chart  ######################################
#################################################################


def bar_plot(ax, navigation_outputs, evaluation_threshold=14, title="", group_stretch=0.8, bar_stretch=0.95,
             legend=True, x_labels=True, label_fontsize=8,
             colors=None, barlabel_offset=1,
             bar_labeler=lambda k, i, s: str(round(s, 3))):

    for threshold_index, evaluation_threshold in enumerate([0, evaluation_threshold]):
        data = generate_objective_value_results(navigation_outputs, evaluation_threshold=evaluation_threshold)
        std_data = {window_type: [case_result[4] for case_result in case_results] for window_type, case_results in data.items()}
        data = {window_type: [case_result[3] for case_result in case_results] for window_type, case_results in data.items()}

        print(std_data, data)

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
        ax.set_title(title)

        for g_i, ((g_name, vals), g_center) in enumerate(zip(sorted_data,
                                                            group_centers)):

            n_bars = len(vals)
            group_beg = g_center - (n_bars / 2) + (bar_stretch / 2)
            for val_i, val in enumerate(vals):

                if threshold_index == 0:
                    bar = ax.bar(group_beg + val_i + bar_offset,
                                height=val, width=bar_stretch,
                                color=colors[g_name],
                                yerr=std_data[g_name][val_i],
                                capsize=4)[0]

                else:
                    bar = ax.bar(group_beg + val_i + bar_offset,
                                height=val, width=0.8,
                                color="white", hatch='/', edgecolor='black', alpha=0.6,
                                yerr=std_data[g_name][val_i],
                                label=f"Last {evaluation_threshold} days" if g_i == 0 else None,
                                capsize=4)[0]

                bars[g_name].append(bar)
                if bar_labeler is not None:
                    x_pos = bar.get_x() + (bar.get_width() / 2.0)
                    y_pos = val + barlabel_offset
                    barlbl = bar_labeler(g_name, val_i, val)
                    ax.text(x_pos, y_pos, barlbl, ha="center", va="bottom",
                            fontsize=label_fontsize)

    if legend:
        # ax.legend([bars[k][0] for k in sorted_k if len(bars[k]) !=0], sorted_k, title="Details", bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
        ax.legend(title="Details", bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

    if x_labels:
        ax.set_xticklabels(sorted_k)
    else:
        ax.set_xticklabels()

    plt.tight_layout()

    return bars, group_centers


#################################################################
###### Generating LaTeX result table  ###########################
#################################################################

def escape_tex_symbols(string):
    escape_chars = {'%': '\\%', '&': '\\&', '_': '\\_', '#': '\\#', '$': '\\$', '{': '\\{', '}': '\\}'}
    return re.sub(r'[%&_#${}]', lambda match: escape_chars[match.group(0)], string)


def generate_sensitivity_analysis_table(data, caption="Statistical results of Monte Carlo sensitivity analysis", label="tab:SensitivityAnalysis", file_name="sensitivity_analysis.tex", decimals=4):

    # Define the path to the tables folder
    tables_folder = os.path.join(os.path.dirname(__file__), "tables")

    # Create the tables folder if it doesn't exist
    if not os.path.exists(tables_folder):
        os.makedirs(tables_folder)

    # Define the file path for the LaTeX table
    file_path = os.path.join(tables_folder, file_name)

    # Initialize the LaTeX table string
    latex_table = f"""
                \\begin{{table}}[]
                \\centering
                \\begin{{tabular}}{{l l l l}}
                \\rowcolor[HTML]{{EFEFEF}} \\textbf{{Parameter}} & \\textbf{{Value}} & \\textbf{{$\\mu_{{\\Delta V}}$}} & \\textbf{{$\\sigma_{{\\Delta V}}$}} \\\\
                """

    # Iterate through the dictionary to populate the table
    for main_key, sub_dict in data.items():
        main_key_formatted = escape_tex_symbols(main_key)  # Escape special TeX symbols in main key
        for idx, (sub_key, values) in enumerate(sub_dict.items()):
            mean = round(values[0], decimals)
            std_dev = round(values[1], decimals)
            sub_key_formatted = escape_tex_symbols(sub_key)  # Escape special TeX symbols in subkey
            if idx == 0:
                latex_table += f"\\textit{{{main_key_formatted}}} & {sub_key_formatted} & {mean} & {std_dev} \\\\"
                latex_table += "\n"  # No hline here
            else:
                latex_table += f" & {sub_key_formatted} & {mean} & {std_dev} \\\\"
                latex_table += "\n"  # No hline here

    # End the LaTeX table
    latex_table += f"""
                \\end{{tabular}}
                \\caption{{{escape_tex_symbols(caption)}}}
                \\label{{{escape_tex_symbols(label)}}}
                \\end{{table}}
                """

    # Write the LaTeX table to a .tex file
    with open(file_path, 'w') as f:
        f.write(latex_table)

    print(f"LaTeX table code has been written")