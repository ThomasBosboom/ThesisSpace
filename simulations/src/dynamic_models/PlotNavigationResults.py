# Standard
import os
import sys
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from tests import utils
from src.dynamic_models import FrameConverter

class PlotNavigationResults():

    def __init__(self, results_dict, sigma_number=3):

        self.results_dict = results_dict
        self.sigma_number = sigma_number

        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, results in enumerate(models):
                    self.mission_start_epoch = results[-1].mission_start_time
                    self.observation_windows = results[-1].observation_windows
                    self.station_keeping_epochs = results[-1].station_keeping_epochs
                    self.step_size = results[-1].step_size


    def plot_full_state_history(self):

        # Plot the trajectory over time
        fig1_3d = plt.figure()
        ax_3d = fig1_3d.add_subplot(111, projection='3d')
        # fig1, ax = plt.subplots(1, 3, figsize=(12, 5), sharex=True)
        # fig2, ax1= plt.subplots(1, 1, figsize=(12, 5), sharex=True)
        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, results in enumerate(models):

                    state_history_reference = results[4][1]
                    state_history_truth = results[5][1]
                    state_history_initial = results[6][1]
                    state_history_final = results[7][1]
                    dynamic_model = results[-2]

                    # Storing some plots
                    ax_3d.plot(state_history_reference[:,0], state_history_reference[:,1], state_history_reference[:,2], label="LPF ref", color="green")
                    ax_3d.plot(state_history_reference[:,6], state_history_reference[:,7], state_history_reference[:,8], label="LUMIO ref", color="green")
                    ax_3d.plot(state_history_initial[:,0], state_history_initial[:,1], state_history_initial[:,2], label="LPF estimated")
                    ax_3d.plot(state_history_initial[:,6], state_history_initial[:,7], state_history_initial[:,8], label="LUMIO estimated")
                    # ax_3d.plot(state_history_final[:,0], state_history_final[:,1], state_history_final[:,2], label="LPF estimated")
                    # ax_3d.plot(state_history_final[:,6], state_history_final[:,7], state_history_final[:,8], label="LUMIO estimated")
                    ax_3d.plot(state_history_truth[:,0], state_history_truth[:,1], state_history_truth[:,2], label="LPF truth", color="black", ls="--")
                    ax_3d.plot(state_history_truth[:,6], state_history_truth[:,7], state_history_truth[:,8], label="LUMIO truth", color="black", ls="--")
                    ax_3d.set_xlabel('X [m]')
                    ax_3d.set_ylabel('Y [m]')
                    ax_3d.set_zlabel('Z [m]')


                    # print("DYBNAMIC MODEL USED", dynamic_model)
                    # print("start epoch", dynamic_model.simulation_start_epoch_MJD)
                    # print("propation_times", dynamic_model.propagation_time)
                    # # Convert state histories to synodic frame
                    # epochs_synodic_reference, state_history_synodic_reference = \
                    #     FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=self.step_size).get_results(state_history_reference)
                    # epochs_synodic_initial, state_history_synodic_initial = \
                    #     FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=self.step_size).get_results(state_history_initial)
                    # epochs_synodic_truth, state_history_synodic_truth = \
                    #     FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=self.step_size).get_results(state_history_truth)

                    # for m in range(3):
                    #     ax[m].plot(state_history_synodic_reference[:,6+m%3], state_history_synodic_reference[:,6+(m+1)%3])
                    #     ax[m].plot(state_history_synodic_initial[:,6+m%3], state_history_synodic_initial[:,6+(m+1)%3])
                    #     ax[m].plot(state_history_synodic_truth[:,6+m%3], state_history_synodic_truth[:,6+(m+1)%3])

                    # ax1.plot(epochs_synodic_reference, state_history_synodic_reference[:,6:9])
                    # ax1.plot(epochs_synodic_initial, state_history_synodic_initial[:,6:9])
                    # ax1.plot(epochs_synodic_truth, state_history_synodic_truth[:,6:9])
        plt.tight_layout()
        plt.legend()


    def plot_formal_error_history(self):

        # Plot how the formal errors grow over time
        fig1, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, results in enumerate(models):

                    full_propagated_formal_errors_epochs = results[3][0]
                    full_propagated_formal_errors_history = results[3][1]
                    relative_epochs = full_propagated_formal_errors_epochs - full_propagated_formal_errors_epochs[0]
                    delta_v = results[8][1]

                    linestyles = ["solid", "dotted", "dashed"]
                    labels = [[r"$x$", r"$y$", r"$z$"], [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"]]
                    ylabels = [r"$\mathbf{r}-\mathbf{r}_{ref}$ [m]", r"$\mathbf{v}-\mathbf{v}_{ref}$ [m/s]"]
                    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    for l in range(2):
                        for m in range(2):
                            for n in range(3):
                                ax[l][m].plot(relative_epochs, self.sigma_number*full_propagated_formal_errors_history[:,3*l+6*m+n], label=model_name if n==0 else None, ls=linestyles[n], color=color_cycle[j])

        for k in range(2):
            for j in range(2):
                for i, gap in enumerate(self.observation_windows):
                    ax[k][j].axvspan(
                        xmin=gap[0]-self.mission_start_epoch,
                        xmax=gap[1]-self.mission_start_epoch,
                        color="gray",
                        alpha=0.1,
                        label="Observation window" if i == 0 else None)
                for i, epoch in enumerate(self.station_keeping_epochs):
                    station_keeping_epoch = epoch - full_propagated_formal_errors_epochs[0]
                    ax[k][j].axvline(x=station_keeping_epoch, color='black', linestyle='--', label="SKM" if i==0 else None)
                ax[k][0].set_ylabel(ylabels[k])
                ax[k][j].grid(alpha=0.5, linestyle='--')
                ax[k][j].set_yscale("log")
                ax[k][0].set_title("LPF")
                ax[k][1].set_title("LUMIO")

                # Set y-axis tick label format to scientific notation with one decimal place
                ax[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax[-1][j].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")

        ax[0][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left')

        fig1.suptitle(r"Formal error history")
        plt.tight_layout()

        # utils.save_figures_to_folder(figs=[fig1], labels=[])
        # plt.show()


    def plot_uncertainty_history(self):

        fig2, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)

        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, results in enumerate(models):

                    full_propagated_formal_errors_epochs = results[3][0]
                    full_propagated_formal_errors_history = results[3][1]
                    propagated_covariance_epochs = results[2][0]

                    # Plot the estimation error history
                    relative_epochs = full_propagated_formal_errors_epochs - full_propagated_formal_errors_epochs[0]
                    for k in range(2):
                        for j in range(2):
                            colors = ["red", "green", "blue"]
                            symbols = [[r"x", r"y", r"z"], [r"v_{x}", r"v_{y}", r"v_{z}"]]
                            ylabels = ["3D RSS OD \n position uncertainty [m]", "3D RSS OD \n velocity uncertainty [m/s]"]
                            ax[k][j].plot(relative_epochs, self.sigma_number*np.linalg.norm(full_propagated_formal_errors_history[:, 3*k+6*j:3*k+6*j+3], axis=1), label=model_name)

        for k in range(2):
            for j in range(2):
                for i, gap in enumerate(self.observation_windows):
                    ax[k][j].axvspan(
                        xmin=gap[0]-self.mission_start_epoch,
                        xmax=gap[1]-self.mission_start_epoch,
                        color="gray",
                        alpha=0.1,
                        label="Observation window" if i == 0 else None)
                for i, epoch in enumerate(self.station_keeping_epochs):
                    station_keeping_epoch = epoch - full_propagated_formal_errors_epochs[0]
                    ax[k][j].axvline(x=station_keeping_epoch, color='black', linestyle='--', label="SKM" if i==0 else None)
                ax[k][0].set_ylabel(ylabels[k])
                ax[k][j].grid(alpha=0.5, linestyle='--')
                ax[k][j].set_yscale("log")
                ax[k][0].set_title("LPF")
                ax[k][1].set_title("LUMIO")

                # Set y-axis tick label format to scientific notation with one decimal place
                ax[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax[-1][j].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")

        ax[0][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left')

        fig2.suptitle(r"Total 3D RSS 3$\sigma$ uncertainty")
        plt.tight_layout()
        # plt.show()


    def plot_reference_deviation_history(self):

        # Plot how the deviation from the reference orbit
        fig3, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, results in enumerate(models):

                    full_reference_state_deviation_epochs = results[1][0]
                    full_reference_state_deviation_history = results[1][1]

                    relative_epochs = full_reference_state_deviation_epochs - full_reference_state_deviation_epochs[0]

                    colors = ["red", "green", "blue"]
                    labels = [[r"$x$", r"$y$", r"$z$"], [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"]]
                    ylabels = [r"$\mathbf{r}-\mathbf{r}_{ref}$ [m]", r"$\mathbf{v}-\mathbf{v}_{ref}$ [m/s]"]
                    for l in range(2):
                        for m in range(2):
                            for i in range(3):
                                ax[l][m].plot(relative_epochs, full_reference_state_deviation_history[:,3*l+6*m+i], label=labels[l][i])

        for k in range(2):
            for j in range(2):
                for i, gap in enumerate(self.observation_windows):
                    ax[k][j].axvspan(
                        xmin=gap[0]-self.mission_start_epoch,
                        xmax=gap[1]-self.mission_start_epoch,
                        color="gray",
                        alpha=0.1,
                        label="Observation window" if i == 0 else None)
                for i, epoch in enumerate(self.station_keeping_epochs):
                    station_keeping_epoch = epoch - full_reference_state_deviation_epochs[0]
                    ax[k][j].axvline(x=station_keeping_epoch, color='black', linestyle='--', label="SKM" if i==0 else None)
                ax[k][0].set_ylabel(ylabels[k])
                ax[k][j].grid(alpha=0.5, linestyle='--')
                ax[k][0].set_title("LPF")
                ax[k][1].set_title("LUMIO")

                # Set y-axis tick label format to scientific notation with one decimal place
                ax[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax[-1][j].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")

            ax[k][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")



        fig3.suptitle("Deviation from reference orbit")
        plt.legend()
        # plt.show()


    def plot_estimation_error_history(self):

        # Plot the estimation error history
        fig4, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                if j == 0:
                    for k, results in enumerate(models):

                        full_estimation_error_epochs = results[0][0]
                        full_estimation_error_history = results[0][1]
                        propagated_covariance_epochs = results[2][0]
                        full_propagated_formal_errors_history = results[3][1]

                        full_estimation_error_history = np.array([interp1d(full_estimation_error_epochs, state, kind='linear', fill_value='extrapolate')(propagated_covariance_epochs) for state in full_estimation_error_history.T]).T

                        relative_epochs = propagated_covariance_epochs - propagated_covariance_epochs[0]
                        for k in range(2):
                            for j in range(2):
                                colors = ["red", "green", "blue"]
                                symbols = [[r"x", r"y", r"z"], [r"v_{x}", r"v_{y}", r"v_{z}"]]
                                ylabels = [r"$\mathbf{r}-\hat{\mathbf{r}}$ [m]", r"$\mathbf{v}-\hat{\mathbf{v}}$ [m/s]"]
                                for i in range(3):
                                    sigma = self.sigma_number*full_propagated_formal_errors_history[:, 3*k+6*j+i]

                                    ax[k][j].plot(relative_epochs, sigma, color=colors[i], ls="--", label=f"$3\sigma_{{{symbols[k][i]}}}$", alpha=0.3)
                                    ax[k][j].plot(relative_epochs, -sigma, color=colors[i], ls="-.", alpha=0.3)
                                    ax[k][j].plot(relative_epochs, full_estimation_error_history[:,3*k+6*j+i], color=colors[i], label=f"${symbols[k][i]}-\hat{{{symbols[k][i]}}}$")

                            ax[k][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

        for k in range(2):
            for j in range(2):
                for i, gap in enumerate(self.observation_windows):
                    ax[k][j].axvspan(
                        xmin=gap[0]-self.mission_start_epoch,
                        xmax=gap[1]-self.mission_start_epoch,
                        color="gray",
                        alpha=0.1,
                        label="Observation window" if i == 0 else None)
                for i, epoch in enumerate(self.station_keeping_epochs):
                    station_keeping_epoch = epoch - propagated_covariance_epochs[0]
                    ax[k][j].axvline(x=station_keeping_epoch, color='black', linestyle='--', label="SKM" if i==0 else None)
                ax[k][0].set_ylabel(ylabels[k])
                ax[k][j].grid(alpha=0.5, linestyle='--')
                ax[0][0].set_ylim(-1000, 1000)
                ax[1][0].set_ylim(-1, 1)
                ax[k][0].set_title("LPF")
                ax[k][1].set_title("LUMIO")

                # Set y-axis tick label format to scientific notation with one decimal place
                ax[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax[-1][j].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")

        fig4.suptitle(r"Estimation error history: range-only, $1\sigma_{\rho}$ = 102.44 [$m$], $f_{obs}$ = $1/600$ [$s^{-1}$]")
        plt.tight_layout()