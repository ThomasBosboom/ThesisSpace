# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import statistics

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Third party
import pytest
import pytest_html
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Own
from tests import utils
from src.dynamic_models import validation
from src.dynamic_models import Interpolator, FrameConverter, TraditionalLowFidelity
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model



class TestPropagation:

    def test_plot_acceleration_norms(self):

        # Use the full_fidelity dynamic model to generate most sophisticated acceleration terms
        model_type = "full_fidelity"
        model_name = "full_fidelity"
        model_number = 0

        # Use point_mass model to generate the point_mass terms of the Earth and Moon also
        model_type_PM = "high_fidelity"
        model_name_PM = "point_mass"
        model_number_PM = 7

        # Defining dynamic model setup
        simulation_start_epoch_MJD = 60390
        propagation_time = 7
        package_dict = {model_type_PM: [model_name_PM], model_type: [model_name]}
        step_size = 0.01
        epoch_in_MJD = True
        solve_variational_equations = False
        specific_model_list = [model_number_PM, model_number]
        return_dynamic_model_objects = True

        # Initialize dictionaries to store accumulated values
        dynamic_model_objects_results = utils.get_dynamic_model_results(simulation_start_epoch_MJD,
                                                                        propagation_time,
                                                                        package_dict=package_dict,
                                                                        step_size=step_size,
                                                                        epoch_in_MJD=epoch_in_MJD,
                                                                        solve_variational_equations=solve_variational_equations,
                                                                        specific_model_list=specific_model_list,
                                                                        return_dynamic_model_objects=return_dynamic_model_objects)

        epochs, state_history, dependent_variable_history, run_time, dynamic_model_object = dynamic_model_objects_results[model_type][model_name][model_number]
        epochs_PM, state_history_PM, dependent_variable_history_PM, run_time_PM, dynamic_model_object_PM = dynamic_model_objects_results[model_type_PM][model_name_PM][model_number_PM]

        bodies_to_create = dynamic_model_object.bodies_to_create
        bodies_to_create_PM = dynamic_model_object_PM.bodies_to_create
        new_bodies_to_create = dynamic_model_object.new_bodies_to_create

        epochs = epochs - epochs[0]

        satellites = ["LPF", "LUMIO"]
        subtitles = [ "Net acceleration", "Point mass", "Spherical harmonics", "Radiation pressure", "Relativistic corrections"]
        ylabels = [r"$||\mathbf{a}_{net}||$  $[m/s^{2}]$", r"$||\mathbf{a}_{PM}||$  $[m/s^{2}]$", r"$||\mathbf{a}_{SH}||$  $[m/s^{2}]$", r"$||\mathbf{a}_{SRP}||$  $[m/s^{2}]$", r"$||\mathbf{a}_{RC}||$  $[m/s^{2}]$"]
        xlabel = f"Time since MJD {simulation_start_epoch_MJD} [days]"
        plot_labels = [None,
                        bodies_to_create_PM,
                       ["Earth J2,0", "Earth J2,2", "Moon J2,0", "Moon J2,2"],
                       ["Earth", "Moon", "Sun"],
                       bodies_to_create]

        for l in range(2):

            fig, ax = plt.subplots(5, 1, figsize=(8, 9), sharex=True)
            fig.suptitle(f'Absolute accelerations acting on {satellites[l]}', fontsize=14)

            n_net = 1
            n_point_mass = 10
            n_spherical_harmonics = 4
            n_radiation_pressure = 3
            n_relativistic_correction = 10

            data_to_plot = [dependent_variable_history[:,12+n_net*l:12+n_net*(l+1)],
                            dependent_variable_history_PM[:,14+n_point_mass*l:14+n_point_mass*(l+1)],
                            dependent_variable_history[:,30+n_spherical_harmonics*l:30+n_spherical_harmonics*(l+1)],
                            dependent_variable_history[:,38+n_radiation_pressure*l:38+n_radiation_pressure*(l+1)],
                            dependent_variable_history[:,44+n_relativistic_correction*l:44+n_relativistic_correction*(l+1)]]

            # Small plots
            for i in range(5):
                if i == 0:
                    ax[i].plot(epochs, data_to_plot[i])
                else:
                    ax[i].plot(epochs, data_to_plot[i], label=[label for label in plot_labels[i]])
                    ax[i].legend(loc="lower right", ncol=2, fontsize="x-small")
                ax[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax[i].set_title(subtitles[i], fontsize=10)
                if i == 4:
                    ax[i].set_xlabel(xlabel, fontsize=8)
                ax[i].set_ylabel(ylabels[i])
                ax[i].set_yscale("log")
                ax[i].grid(alpha=0.5, linestyle='--')

            plt.tight_layout()

            utils.save_figures_to_folder(figs=[fig], labels=[satellites[l]])

        # plt.show()



    def test_plot_trajectories(self):

        # Defining dynamic model setup
        simulation_start_epoch_MJD = 60390
        propagation_time = 1
        package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}
        get_only_first = True
        step_size = 0.01
        epoch_in_MJD = True
        solve_variational_equations = False
        specific_model_list = None
        return_dynamic_model_objects = True

        # Initialize dictionaries to store accumulated values
        dynamic_model_objects_results = utils.get_dynamic_model_results(simulation_start_epoch_MJD,
                                                                        propagation_time,
                                                                        package_dict=package_dict,
                                                                        get_only_first=get_only_first,
                                                                        step_size=step_size,
                                                                        epoch_in_MJD=epoch_in_MJD,
                                                                        solve_variational_equations=solve_variational_equations,
                                                                        specific_model_list=specific_model_list,
                                                                        return_dynamic_model_objects=return_dynamic_model_objects)

        legend_handles = []
        legend_labels = []
        durations = [7, 14, 21, 28]
        satellites = ["LPF", "LUMIO"]
        plot_labels = [['X [m]', 'Y [m]', 'Z [m]'], ['X [-]', 'Y [-]', 'Z [-]']]
        legend_labels = [["CRTBP"], ["PM", "PMSRP", "SH", "SHSRP"]]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        label_colors = []
        fig, ax = plt.subplots(len(durations), 3, figsize=(8, 2.5*(len(durations))), layout="constrained")
        fig_2, ax_2 = plt.subplots(len(durations), 3, figsize=(8, 2.5*(len(durations))), layout="constrained")
        handles, labels = [], []
        for i, (model_type, model_names) in enumerate(dynamic_model_objects_results.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, model in enumerate(models):

                    # Extract the propagation information
                    epochs, state_history, dependent_variable_history, run_time, dynamic_model_object = dynamic_model_objects_results[model_type][model_name][k]

                    # Plot the trajectories for multiple moments in time in inertial frame
                    for l in range(len(durations)):
                        n = int(durations[l]/(step_size))
                        for m in range(3):
                            ax[l][0].title.set_text(str(durations[l])+ " days")
                            ax[l][m].plot(state_history[0,6+m%3], state_history[0,6+(m+1)%3], marker="o", color=colors[i])
                            line = ax[l][m].plot(state_history[:n,6+m%3], state_history[:n,6+(m+1)%3], color=colors[i])
                            ax[l][m].set_xlabel(plot_labels[0][m%3])
                            ax[l][m].set_ylabel(plot_labels[0][(m+1)%3])
                            ax[l][m].grid(alpha=0.5, linestyle='--')

                    ax[0][-1].legend(line, legend_labels[i][j])

                        # Store handles and labels of the second plot
                        # if l == 0:  # Only store handles and labels once
                        #     legend_handles.append(line[0])
                        #     legend_labels.append(legend_labels[i][j])

                    # Convert back to synodic
                    epochs_synodic, state_history_synodic = \
                        FrameConverter.InertialToSynodicHistoryConverter(dynamic_model_object, step_size=step_size).get_results(state_history)

                    # Plot the trajectories for multiple moments in time in synodic frame

                    for l in range(len(durations)):
                        n = int(durations[l]/(step_size))
                        for m in range(3):
                            ax_2[l][0].title.set_text(str(durations[l])+ " days")
                            ax_2[l][m].plot(state_history_synodic[0,6+m%3], state_history_synodic[0,6+(m+1)%3], marker="o", color=colors[i], label=None)
                            line = ax_2[l][m].plot(state_history_synodic[:n,6+m%3], state_history_synodic[:n,6+(m+1)%3], color=colors[i])
                            ax_2[l][m].set_xlabel(plot_labels[1][m%3])
                            ax_2[l][m].set_ylabel(plot_labels[1][(m+1)%3])
                            ax_2[l][m].grid(alpha=0.5, linestyle='--')

                            ax_2[0][-1].legend(line)

                    # handle, label = ax_2[0][0].get_legend_handles_labels()
                    # handles.extend(handle)
                    # labels.extend(label)

            # Store handles and labels of the second plot
            # if j == 0:
                # print(lines)
                # print(legend_labels[i][j])
                # legend_handles.extend(lines)
                # legend_labels.extend(legend_labels[i][j])

        # fig.legend(legend_handles, legend_labels, loc='upper right')
        # fig_2.legend(legend_handles, legend_labels, loc='upper right')
        plt.tight_layout()

        utils.save_figures_to_folder(figs=[fig], labels=[f"synodic_{satellites[1]}"])
        utils.save_figures_to_folder(figs=[fig_2], labels=[f"inertial_{satellites[1]}"])
        # plt.show()


    def test_dynamic_model_run_times(self):

        package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"], "full_fidelity": ["full_fidelity"]}
        package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass"]}
        propagation_time = 1
        get_only_first = True
        start_epoch = 60390
        end_epoch = 60418
        n = 2
        run_cases = np.linspace(start_epoch, end_epoch, n)

        # Initialize dictionaries to store accumulated values
        accumulator_dict = {
            fidelity_key: {subkey: [[] for value in values]
                for subkey, values in sub_dict.items()
            }
            for fidelity_key, sub_dict in utils.get_dynamic_model_objects(start_epoch, propagation_time, package_dict=package_dict, get_only_first=get_only_first).items()
        }

        for run_case in run_cases:
            params = (run_case, propagation_time, package_dict, get_only_first)
            run_times_dict = utils.get_dynamic_model_results(*params, step_size=0.1, entry_list=[-1])

            # Accumulate values during the loop
            for fidelity_key, sub_dict in run_times_dict.items():
                for i, (subkey, subvalue_list) in enumerate(sub_dict.items()):
                    for j, subvalue in enumerate(subvalue_list):
                        for k, entry in enumerate(subvalue):
                            accumulator_dict[fidelity_key][subkey][j].append(entry)

        # Calculate averages and standard deviations
        result_dict = {
            fidelity_key: {
                subkey: [
                    {
                        "average": statistics.mean(sublist),
                        "std_dev": statistics.stdev(sublist)
                    }
                    for sublist in sublists
                ]
                for subkey, sublists in sub_dict.items()
            }
            for fidelity_key, sub_dict in accumulator_dict.items()
        }

        ### Plot run times for each model
        keys_list = [["CRTBP"], ["PM", "PM SRP", "SH", "SH SRP"], ["FF"]]
        key_count = sum(len(sublist) for sublist in package_dict.values()) #0.75*key_count
        fig, axs = plt.subplots(1, key_count, figsize=(6.4, 5), sharey=True)
        index = 0
        for i, (model_types, model_names) in enumerate(result_dict.items()):
            for j, (key, values) in enumerate(model_names.items()):
                averages = []
                std_devs = []
                for subvalue in values:
                    averages.append(subvalue["average"])
                    std_devs.append(subvalue["std_dev"])
                axs[index].grid(alpha=0.5, linestyle='--')
                axs[index].bar(range(1, len(values)+1), averages, yerr=std_devs, ecolor="black", capsize=4, label=key)
                axs[index].set_xlabel(keys_list[i][j])
                axs[index].set_xticks(range(1, 1+max([len(value) for value in model_names.values()])))
                axs[index].set_yscale("log")
                index += 1

        axs[0].set_ylabel('Run time [s]')
        legend_handles = [plt.Line2D([0], [0], color='black', markersize=5, label=r'1$\sigma$ Std Dev')]
        fig.legend(handles=[legend_handles[0]], loc='upper right')
        fig.suptitle(f"Monte Carlo run time dynamic models, n={len(run_cases)*10}, varying start epoch spread over range [{start_epoch}, {end_epoch}] \nProcessor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz, 2208 Mhz")

        utils.save_figures_to_folder([fig], [run_cases[0], run_cases[-1], n*10])

        # plt.show()


class TestDifferences:

    def test_absolute_difference_to_reference_model(self):

        # Defining dynamic model setup
        simulation_start_epoch_MJD = 60390
        propagation_time = 1
        # {"low_fidelity": ["three_body_problem"]}
        package_dict = {"high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"], "full_fidelity": ["full_fidelity"]}
        package_dict = {"high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]}
        get_only_first = True
        step_size = 0.01
        epoch_in_MJD = True
        solve_variational_equations = False
        specific_model_list = None
        return_dynamic_model_objects = True

        # Initialize dictionaries to store accumulated values
        dynamic_model_objects_results = utils.get_dynamic_model_results(simulation_start_epoch_MJD,
                                                                        propagation_time,
                                                                        package_dict=package_dict,
                                                                        get_only_first=get_only_first,
                                                                        step_size=step_size,
                                                                        epoch_in_MJD=epoch_in_MJD,
                                                                        solve_variational_equations=solve_variational_equations,
                                                                        specific_model_list=specific_model_list,
                                                                        return_dynamic_model_objects=return_dynamic_model_objects)


        satellites = ["LPF", "LUMIO"]
        legend_labels = [["PM", "PMSRP", "SH", "SHSRP"], ["FF"]]
        subplot_ylabels = [r'$x-x_{ref}$ [m]', r'$\Delta$Y [m]', r'$\Delta$Z [m]', \
                    r'$\Delta$VX [m/s]', r'$\Delta$VY [m/s]', r'$\Delta$VZ [m/s]']
        plot_colors = ["red", "blue"]
        plot_ls = [[(0, (1, 5)), '--', '-.', ':'], ['-']]

        fig, ax = plt.subplots(6, 1, figsize=(8, 9), sharex=True)
        fig_2, ax_2 = plt.subplots(6, 1, figsize=(8, 9), sharex=True)
        fig.suptitle(f'Absolute error w.r.t. reference trajectory', fontsize=14)
        for i, (model_type, model_names) in enumerate(dynamic_model_objects_results.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, model in enumerate(models):

                    print(k, model)
                    # Extract the propagation information
                    epochs, state_history, dependent_variable_history, run_time, dynamic_model_object = dynamic_model_objects_results[model_type][model_name][k]

                    # Extract reference trajectory
                    reference_state_history = np.concatenate((validation.get_reference_state_history(simulation_start_epoch_MJD, propagation_time, step_size=step_size, satellite=dynamic_model_object.name_ELO, get_full_history=True),
                                                            validation.get_reference_state_history(simulation_start_epoch_MJD, propagation_time, step_size=step_size, satellite=dynamic_model_object.name_LPO, get_full_history=True)),
                                                            axis=1)


                    data_to_plot = state_history - reference_state_history
                    epochs = epochs - epochs[0]

                    for l in range(6):
                        for n in range(2):
                            if n == 0:
                                ax[l].plot(epochs, data_to_plot[:,6*n+l], label=legend_labels[i][j])
                                ax[-1].set_xlabel(f"Time since MJD {simulation_start_epoch_MJD} [days]")
                                ax[l].set_ylabel(subplot_ylabels[l])
                                ax[l].grid(alpha=0.5, linestyle='--')
                                ax[l].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                                ax[l].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                                ax[-1].legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=6)
                                # ax[l].yscale("log")
                            else:
                                ax_2[l].plot(epochs, data_to_plot[:,6*n+l], label=legend_labels[i][j])
                                ax_2[-1].set_xlabel(f"Time since MJD {simulation_start_epoch_MJD} [days]")
                                ax_2[l].set_ylabel(subplot_ylabels[l])
                                ax_2[l].grid(alpha=0.5, linestyle='--')
                                ax_2[l].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                                ax_2[l].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                                ax_2[-1].legend(bbox_to_anchor=(0.5, 0), loc='upper center', ncol=6)
                                # ax_2[l].yscale("log")

        # fig.legend(bbox_to_anchor=(0.5, 0), loc='upper center', ncol=3)
        fig.suptitle(f"State difference w.r.t. reference for {satellites[0]}", fontsize=14)
        fig_2.suptitle(f"State difference w.r.t. reference for {satellites[1]}", fontsize=14)
        plt.tight_layout()

        plt.show()


a = TestDifferences().test_absolute_difference_to_reference_model()
print(a)