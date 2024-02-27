# Standard
import os
import sys
import copy
import numpy as np
import time
import pytest
import pytest_html
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d

# Tudatpy

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from tests import utils
from src.dynamic_models import Interpolator, NavigationSimulator

# class PlotResults():




class TestNavigation():

    def test_navigation_sequence(self):

        mission_start_epoch = 60390
        propagation_time = 1
        get_only_first = True
        package_dict = {"high_fidelity": ["point_mass"]}
        truth_model_list = ["high_fidelity", "point_mass", 5]
        package_dict = {"high_fidelity": ["spherical_harmonics_srp"]}
        truth_model_list = ["high_fidelity", "spherical_harmonics_srp", 5]
        package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass"]}
        truth_model_list = ["low_fidelity", "three_body_problem", 0]
        sigma_number = 3

        # Save histories of the navigation simulations
        results_dict = dict()


        dynamic_model_objects = utils.get_dynamic_model_objects(mission_start_epoch, propagation_time, get_only_first=True, package_dict=package_dict)
        print(dynamic_model_objects)
        for i, (model_type, model_names) in enumerate(dynamic_model_objects.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, model in enumerate(models):

                    print("MODEL: ", i, j, k, model)


                    # batch_start_times = np.array([60390, 60394.7, 60401.5, 60406.5])
                    # batch_end_times = np.array([60392.5, 60397.2, 60404, 60409])

                    # observation_windows = list(zip(batch_start_times, batch_end_times))

                    # Mission time settings
                    mission_time = 2
                    mission_start_epoch = 60390
                    mission_end_epoch = mission_start_epoch + mission_time
                    mission_epoch = mission_start_epoch

                    # Initial batch timing settings
                    propagation_time = 1
                    batch_start_times = np.arange(mission_start_epoch, mission_end_epoch, propagation_time)
                    batch_end_times = np.arange(propagation_time+mission_start_epoch, propagation_time+mission_end_epoch, propagation_time)
                    observation_windows = list(zip(batch_start_times, batch_end_times))
                    print(observation_windows)

                    dynamic_model_list = [model_type, model_name, k]

                    navigation_simulator = NavigationSimulator.NavigationSimulator(mission_start_epoch, observation_windows, dynamic_model_list, truth_model_list)
                    full_estimation_error_dict, full_reference_state_deviation_dict, full_propagated_covariance_dict, full_propagated_formal_errors_dict = navigation_simulator.perform_navigation(include_station_keeping=True)

                    results_dict[model] = (full_estimation_error_dict, full_reference_state_deviation_dict, full_propagated_covariance_dict, full_propagated_formal_errors_dict)

                    propagated_covariance_epochs, propagated_covariance_history = utils.convert_dictionary_to_array(full_propagated_covariance_dict)
                    full_estimation_error_epochs, full_estimation_error_history = utils.convert_dictionary_to_array(full_estimation_error_dict)
                    full_reference_state_deviation_epochs, full_reference_state_deviation_history = utils.convert_dictionary_to_array(full_reference_state_deviation_dict)
                    full_propagated_formal_errors_epochs, full_propagated_formal_errors_history = utils.convert_dictionary_to_array(full_propagated_formal_errors_dict)

                    full_estimation_error_history = np.array([interp1d(full_estimation_error_epochs, state, kind='linear', fill_value='extrapolate')(propagated_covariance_epochs) for state in full_estimation_error_history.T]).T

                    # Plot how the formal errors grow over time
                    fig1, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
                    reference_epoch_array = mission_start_epoch*np.ones(np.shape(full_propagated_formal_errors_epochs))
                    ax[0].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, 3*full_propagated_formal_errors_history[:,0:3], label=[r"$3\sigma_{x}$", r"$3\sigma_{y}$", r"$3\sigma_{z}$"])
                    ax[1].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, 3*full_propagated_formal_errors_history[:,6:9], label=[r"$3\sigma_{x}$", r"$3\sigma_{y}$", r"$3\sigma_{z}$"])
                    for j in range(2):
                        for i, gap in enumerate(observation_windows):
                            ax[j].axvspan(
                                xmin=gap[0]-mission_start_epoch,
                                xmax=gap[1]-mission_start_epoch,
                                color="gray",
                                alpha=0.1,
                                label="Observation window" if i == 0 else None)
                            ax[j].set_ylabel(r"$\sigma$ [m]")
                            ax[j].grid(alpha=0.5, linestyle='--')
                    ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
                    ax[0].set_title("LPF")
                    ax[1].set_title("LUMIO")
                    fig1.suptitle("Propagated formal errors")
                    plt.legend()
                    # plt.show()

                    # Plot how the uncertainty grows over time
                    fig2, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
                    reference_epoch_array = mission_start_epoch*np.ones(np.shape(full_propagated_formal_errors_epochs))
                    ax[0].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, sigma_number*np.linalg.norm(full_propagated_formal_errors_history[:, 0:3], axis=1))
                    ax[1].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, sigma_number*np.linalg.norm(full_propagated_formal_errors_history[:, 6:9], axis=1))
                    for j in range(2):
                        for i, gap in enumerate(observation_windows):
                            ax[j].axvspan(
                                xmin=gap[0]-mission_start_epoch,
                                xmax=gap[1]-mission_start_epoch,
                                color="gray",
                                alpha=0.1,
                                label="Observation window" if i == 0 else None)
                            ax[j].set_ylabel("3D RSS OD \n position uncertainty [m]")
                            ax[j].grid(alpha=0.5, linestyle='--')
                            ax[j].set_yscale("log")
                    ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
                    ax[0].set_title("LPF")
                    ax[1].set_title("LUMIO")
                    fig2.suptitle("Total position uncertainty")
                    plt.legend()
                    # plt.show()

                    # Plot how the deviation from the reference orbit
                    fig3, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
                    reference_epoch_array = mission_start_epoch*np.ones(np.shape(full_reference_state_deviation_epochs))
                    for j in range(2):
                        labels = ["x", "y", "z"]
                        for i in range(3):
                            ax[j].plot(utils.convert_epochs_to_MJD(full_reference_state_deviation_epochs)-reference_epoch_array, full_reference_state_deviation_history[:,6*j+i], label=labels[i])
                        for i, gap in enumerate(observation_windows):
                            ax[j].axvspan(
                                xmin=gap[0]-mission_start_epoch,
                                xmax=gap[1]-mission_start_epoch,
                                color="gray",
                                alpha=0.1,
                                label="Observation window" if i == 0 else None)
                        ax[j].set_ylabel(r"$\mathbf{r}-\hat{\mathbf{r}}_{ref}$ [m]")
                        ax[j].grid(alpha=0.5, linestyle='--')
                    ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
                    ax[0].set_title("LPF")
                    ax[1].set_title("LUMIO")
                    fig3.suptitle("Deviation from reference orbit")
                    plt.legend()
                    # plt.show()

                    # Plot the estimation error history
                    fig4, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
                    reference_epoch_array = mission_start_epoch*np.ones(np.shape(propagated_covariance_epochs))
                    for k in range(2):
                        for j in range(2):
                            colors = ["red", "green", "blue"]
                            symbols = [[r"x", r"y", r"z"], [r"v_{x}", r"v_{y}", r"v_{z}"]]
                            ylabels = [r"$\mathbf{r}-\hat{\mathbf{r}}$ [m]", r"$\mathbf{v}-\hat{\mathbf{v}}$ [m]"]
                            for i in range(3):
                                sigma = sigma_number*full_propagated_formal_errors_history[:, 3*k+6*j+i]

                                ax[k][j].plot(utils.convert_epochs_to_MJD(propagated_covariance_epochs)-reference_epoch_array, sigma, color=colors[i], ls="--", label=f"$3\sigma_{{{symbols[k][i]}}}$", alpha=0.3)
                                ax[k][j].plot(utils.convert_epochs_to_MJD(propagated_covariance_epochs)-reference_epoch_array, -sigma, color=colors[i], ls="-.", alpha=0.3)
                                ax[k][j].plot(utils.convert_epochs_to_MJD(propagated_covariance_epochs)-reference_epoch_array, full_estimation_error_history[:,3*k+6*j+i], color=colors[i], label=f"${symbols[k][i]}-\hat{{{symbols[k][i]}}}$")
                            ax[k][0].set_ylabel(ylabels[k])
                            ax[k][j].grid(alpha=0.5, linestyle='--')
                            for i, gap in enumerate(observation_windows):
                                ax[k][j].axvspan(
                                    xmin=gap[0]-mission_start_epoch,
                                    xmax=gap[1]-mission_start_epoch,
                                    color="gray",
                                    alpha=0.1,
                                    label="Observation window" if i == 0 else None)

                            ax[0][0].set_ylim(-1000, 1000)
                            ax[1][0].set_ylim(-0.3, 0.3)

                            ax[-1][j].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")

                            # Set y-axis tick label format to scientific notation with one decimal place
                            ax[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                            ax[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                        ax[k][0].set_title("LPF")
                        ax[k][1].set_title("LUMIO")
                        ax[k][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left')

                    fig4.suptitle(r"Estimation error history: $1\sigma_{\rho}$ = 102.44 [$m^2$], $f_{obs}$ = $1/600$ [$s^{-1}$]")
                    plt.tight_layout()

                    # utils.save_figures_to_folder(figs=[fig1, fig2, fig3, fig4], labels=[model_type, model_name, truth_model_list[0], truth_model_list[1]])

        # plt.show()
        shape = (len(results_dict), len(next(iter(results_dict.values()))))
        print(shape)



print(TestNavigation().test_navigation_sequence())


