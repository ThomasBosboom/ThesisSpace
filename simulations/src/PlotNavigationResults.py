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
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Own
from tests import utils
import FrameConverter

class PlotNavigationResults():

    def __init__(self, navigation_output, sigma_number=3):

        self.navigation_output = navigation_output
        self.navigation_simulator = navigation_output.navigation_simulator
        self.navigation_results = navigation_output.navigation_results
        self.sigma_number = sigma_number

        self.mission_start_epoch = self.navigation_simulator.mission_start_epoch
        self.observation_windows = self.navigation_simulator.observation_windows
        self.station_keeping_epochs = self.navigation_simulator.station_keeping_epochs
        self.step_size = self.navigation_simulator.step_size


class PlotNavigationResults():

    def __init__(self, navigation_output, sigma_number=3):

        self.navigation_output = navigation_output
        self.navigation_simulator = navigation_output.navigation_simulator
        self.navigation_results = navigation_output.navigation_results
        self.sigma_number = sigma_number

        self.mission_start_epoch = self.navigation_simulator.mission_start_epoch
        self.observation_windows = self.navigation_simulator.observation_windows
        self.station_keeping_epochs = self.navigation_simulator.station_keeping_epochs
        self.step_size = self.navigation_simulator.step_size


    def plot_full_state_history(self):

        # Plot the trajectory over time
        fig1_3d = plt.figure()
        ax_3d = fig1_3d.add_subplot(111, projection='3d')
        fig1_3d2 = plt.figure()
        ax_3d2 = fig1_3d2.add_subplot(111, projection='3d')
        fig, ax = plt.subplots(2, 3, figsize=(11, 6))

        state_history_reference = self.navigation_results[4][1]
        state_history_truth = self.navigation_results[5][1]
        state_history_initial = self.navigation_results[6][1]
        epochs = self.navigation_results[9][0]
        dependent_variables_history = self.navigation_results[9][1]
        # navigation_simulator = self.navigation_results[-1]

        moon_data_dict = {epoch: state for epoch, state in zip(epochs, dependent_variables_history[:, :6])}
        satellite_data_dict = {epoch: state for epoch, state in zip(epochs, state_history_initial[:, :])}

        G = 6.67430e-11
        m1 = 5.972e24
        m2 = 7.34767309e22
        mu = m2/(m2 + m1)

        # Create the Direct Cosine Matrix based on rotation axis of Moon around Earth
        transformation_matrix_dict = {}
        for epoch, moon_state in moon_data_dict.items():

            moon_position, moon_velocity = moon_state[:3], moon_state[3:]

            # Define the complementary axes of the rotating frame
            rotation_axis = np.cross(moon_position, moon_velocity)
            second_axis = np.cross(moon_position, rotation_axis)

            # Define the rotation matrix (DCM) using the rotating frame axes
            first_axis = moon_position/np.linalg.norm(moon_position)
            second_axis = second_axis/np.linalg.norm(second_axis)
            third_axis = rotation_axis/np.linalg.norm(rotation_axis)
            transformation_matrix = np.array([first_axis, second_axis, third_axis])
            rotation_axis = rotation_axis*m2
            rotation_rate = rotation_axis/(m2*np.linalg.norm(moon_position)**2)

            skew_symmetric_matrix = np.array([[0, -rotation_rate[2], rotation_rate[1]],
                                              [rotation_rate[2], 0, -rotation_rate[0]],
                                              [-rotation_rate[1], rotation_rate[0], 0]])

            transformation_matrix_derivative =  np.dot(transformation_matrix, skew_symmetric_matrix)
            transformation_matrix = np.block([[transformation_matrix, np.zeros((3,3))],
                                              [transformation_matrix_derivative, transformation_matrix]])

            transformation_matrix_dict.update({epoch: transformation_matrix})


        # Generate the synodic states of the satellites
        synodic_satellite_states_dict = {}
        for epoch, state in satellite_data_dict.items():

            transformation_matrix = transformation_matrix_dict[epoch]
            synodic_state = np.concatenate((np.dot(transformation_matrix, state[0:6]), np.dot(transformation_matrix, state[6:12])))

            LU = np.linalg.norm((moon_data_dict[epoch][0:3]))
            TU = np.sqrt(LU**3/(G*(m1+m2)))
            synodic_state[0:3] = synodic_state[0:3]/LU
            synodic_state[6:9] = synodic_state[6:9]/LU
            synodic_state[3:6] = synodic_state[3:6]/(LU/TU)
            synodic_state[9:12] = synodic_state[9:12]/(LU/TU)
            synodic_state = (1-mu)*synodic_state

            synodic_satellite_states_dict.update({epoch: synodic_state})

        synodic_states = np.stack(list(synodic_satellite_states_dict.values()))
        # print(synodic_states[0, :])

        # Generate the synodic states of the moon
        synodic_moon_states_dict = {}
        for epoch, state in moon_data_dict.items():

            transformation_matrix = transformation_matrix_dict[epoch]
            synodic_state = np.dot(transformation_matrix, state)
            LU = np.linalg.norm((moon_data_dict[epoch][0:3]))
            TU = np.sqrt(LU**3/(G*(m1+m2)))
            synodic_state[0:3] = synodic_state[0:3]/LU
            synodic_state[3:6] = synodic_state[3:6]/(LU/TU)
            synodic_state = (1-mu)*synodic_state
            synodic_moon_states_dict.update({epoch: synodic_state})

        synodic_states = np.stack(list(synodic_satellite_states_dict.values()))
        synodic_moon_states = np.stack(list(synodic_moon_states_dict.values()))

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(2):
            if i == 0:
                color="gray"
            else:
                color="black"

            ax[i][0].scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 2], s=50, color="darkgray")
            ax[i][1].scatter(synodic_moon_states[:, 1], synodic_moon_states[:, 2], s=50, color="darkgray")
            ax[i][2].scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 1], s=50, color="darkgray", label="Moon" if i==0 else None)
            ax[i][0].plot(synodic_states[:, 6*i+0], synodic_states[:, 6*i+2], lw=0.5, color=color)
            ax[i][1].plot(synodic_states[:, 6*i+1], synodic_states[:, 6*i+2], lw=0.5, color=color)
            ax[i][2].plot(synodic_states[:, 6*i+0], synodic_states[:, 6*i+1], lw=0.5, color=color, label="LPF" if i==0 else None)
            ax[1][0].plot(synodic_states[:, 6*i+0], synodic_states[:, 6*i+2], lw=0.1, color=color)
            ax[1][1].plot(synodic_states[:, 6*i+1], synodic_states[:, 6*i+2], lw=0.1, color=color)
            ax[1][2].plot(synodic_states[:, 6*i+0], synodic_states[:, 6*i+1], lw=0.1, color=color, label="LUMIO" if i==1 else None)


        ax_3d.scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 1], synodic_moon_states[:, 2], s=50, color="darkgray", label="Moon")
        ax_3d.plot(synodic_states[:, 0], synodic_states[:, 1], synodic_states[:, 2], lw=0.2, color="gray")
        ax_3d.plot(synodic_states[:, 6], synodic_states[:, 7], synodic_states[:, 8], lw=0.7, color="black")
        # ax_3d.scatter(-mu, 0, 0, label="Earth", color="darkblue", s=50)


        # print("INITIAL SYNODIC STATES: ", synodic_states[0, :])
        # print("INITIAL STATES: ", state_history_reference[0, :])
        # print("INITIAL STATES: ", state_history_initial[0, :])



        # ax_3d2.plot(state_history_reference[:,0], state_history_reference[:,1], state_history_reference[:,2], label="LPF ref", color="green")
        # ax_3d2.plot(state_history_reference[:,6], state_history_reference[:,7], state_history_reference[:,8], label="LUMIO ref", color="green")
        ax_3d2.plot(state_history_initial[:,0], state_history_initial[:,1], state_history_initial[:,2], lw=0.5, label="LPF", color="gray")
        ax_3d2.plot(state_history_initial[:,6], state_history_initial[:,7], state_history_initial[:,8], lw=0.5, label="LUMIO", color="black")
        ax_3d2.scatter(0, 0, 0, label="Earth", color="darkblue", s=50)
        # ax_3d2.plot(state_history_truth[:,0], state_history_truth[:,1], state_history_truth[:,2], label="LPF truth", color="black", ls="--")
        # ax_3d2.plot(state_history_truth[:,6], state_history_truth[:,7], state_history_truth[:,8], label="LUMIO truth", color="black", ls="--")


        for num, (start, end) in enumerate(self.navigation_simulator.observation_windows):
            synodic_states_window_dict = {key: value for key, value in synodic_satellite_states_dict.items() if key >= start and key <= end}
            synodic_states_window = np.stack(list(synodic_states_window_dict.values()))

            inertial_states_window_dict = {key: value for key, value in satellite_data_dict.items() if key >= start and key <= end}
            inertial_states_window = np.stack(list(inertial_states_window_dict.values()))

            # print(np.shape(synodic_states_window))
            # print(np.shape(synodic_states_window), color_cycle[num%10])

            for i in range(2):
                linewidth=2

                if num == 0:
                    ax[i][0].scatter(synodic_states_window[0, 6*i+0], synodic_states_window[0, 6*i+2], color=color_cycle[num%10], s=20, marker="X")
                    ax[i][1].scatter(synodic_states_window[0, 6*i+1], synodic_states_window[0, 6*i+2], color=color_cycle[num%10], s=20, marker="X")
                    ax[i][2].scatter(synodic_states_window[0, 6*i+0], synodic_states_window[0, 6*i+1], color=color_cycle[num%10], s=20, marker="X", label="Start" if i == 0 else None)

                ax[i][0].plot(synodic_states_window[:, 6*i+0], synodic_states_window[:, 6*i+2], linewidth=linewidth, color=color_cycle[num%10])
                ax[i][1].plot(synodic_states_window[:, 6*i+1], synodic_states_window[:, 6*i+2], linewidth=linewidth, color=color_cycle[num%10])
                ax[i][2].plot(synodic_states_window[:, 6*i+0], synodic_states_window[:, 6*i+1], linewidth=linewidth, color=color_cycle[num%10], label=f"Arc {num+1}" if i==0 else None)

                ax_3d.plot(synodic_states_window[:, 6*i+0], synodic_states_window[:, 6*i+1], synodic_states_window[:, 6*i+2], linewidth=0.5 if i ==0 else 2, color=color_cycle[num%10], label=f"Arc {num+1}" if i==1 else None)
                ax_3d2.plot(inertial_states_window[:, 6*i+0], inertial_states_window[:, 6*i+1], inertial_states_window[:, 6*i+2], linewidth=2, color=color_cycle[num%10], label=f"Arc {num+1}" if i==0 else None)

            for i in range(len(synodic_states_window[:, 0])):
                ax_3d.plot([synodic_states_window[i, 0], synodic_states_window[i, 6]],
                            [synodic_states_window[i, 1], synodic_states_window[i, 7]],
                            [synodic_states_window[i, 2], synodic_states_window[i, 8]], color=color_cycle[num%10], lw=0.5, alpha=0.2)

                ax_3d2.plot([inertial_states_window[i, 0], inertial_states_window[i, 6]],
                            [inertial_states_window[i, 1], inertial_states_window[i, 7]],
                            [inertial_states_window[i, 2], inertial_states_window[i, 8]], color=color_cycle[num%10], lw=0.5, alpha=0.2)

        axes_labels = ['X [-]', 'Y [-]', 'Z [-]']
        for i in range(2):
            for j in range(3):
                ax[i][j].grid(alpha=0.3)
                ax[i][0].set_xlabel(axes_labels[0])
                ax[i][0].set_ylabel(axes_labels[2])
                ax[i][1].set_xlabel(axes_labels[1])
                ax[i][1].set_ylabel(axes_labels[2])
                ax[i][2].set_xlabel(axes_labels[0])
                ax[i][2].set_ylabel(axes_labels[1])

        ax_3d.set_xlabel('X [-]')
        ax_3d.set_ylabel('Y [-]')
        ax_3d.set_zlabel('Z [-]')
        # ax_3d.set_title("Tracking arcs in synodic frame")
        ax_3d.legend()

        ax_3d2.set_xlabel('X [m]')
        ax_3d2.set_ylabel('Y [m]')
        ax_3d2.set_zlabel('Z [m]')
        # ax_3d2.set_title("Tracking arcs in inertial frame")

        ax_3d2.legend()
        # ax_3d2.set_zlim([-4.5e8, 4.5e8])

        ax[0][2].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

        fig.suptitle(f"Observation windows for {28} days, synodic frame \n Model: on-board: {self.navigation_simulator.model_name}{self.navigation_simulator.model_number}, truth: {self.navigation_simulator.model_name_truth}{self.navigation_simulator.model_number_truth}")
        fig1_3d.suptitle(f"Observation windows for {28} days, synodic frame \n Model: on-board: {self.navigation_simulator.model_name}{self.navigation_simulator.model_number}, truth: {self.navigation_simulator.model_name_truth}{self.navigation_simulator.model_number_truth}")
        fig1_3d2.suptitle(f"Observation windows for {28} days, inertial frame \n Model: on-board: {self.navigation_simulator.model_name}{self.navigation_simulator.model_number}, truth: {self.navigation_simulator.model_name_truth}{self.navigation_simulator.model_number_truth}")
        plt.tight_layout()
        plt.legend()


    def plot_formal_error_history(self):

        # Plot how the formal errors grow over time
        fig, ax = plt.subplots(2, 2, figsize=(11, 4), sharex=True)

        full_propagated_formal_errors_epochs = self.navigation_results[3][0]
        full_propagated_formal_errors_history = self.navigation_results[3][1]
        relative_epochs = full_propagated_formal_errors_epochs - self.mission_start_epoch

        linestyles = ["solid", "dotted", "dashed"]
        labels = [[r"$x$", r"$y$", r"$z$"], [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"]]
        ylabels = [r"$\mathbf{r}-\mathbf{r}_{ref}$ [m]", r"$\mathbf{v}-\mathbf{v}_{ref}$ [m/s]"]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for l in range(2):
            for m in range(2):
                for n in range(3):
                    ax[l][m].plot(relative_epochs, self.sigma_number*full_propagated_formal_errors_history[:,3*l+6*m+n], label=self.navigation_simulator.model_name if n==0 else None, ls=linestyles[n], color="blue")

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
                    station_keeping_epoch = epoch - self.mission_start_epoch
                    ax[k][j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.7, label="SKM" if i==0 else None)

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

        fig.suptitle(f"Formal error history \n Model: on-board: {self.navigation_simulator.model_name}{self.navigation_simulator.model_number}, truth: {self.navigation_simulator.model_name_truth}{self.navigation_simulator.model_number_truth}")
        plt.tight_layout()


    def plot_uncertainty_history(self):

        fig, ax = plt.subplots(2, 2, figsize=(11, 4), sharex=True)
        full_propagated_formal_errors_epochs = self.navigation_results[3][0]
        full_propagated_formal_errors_history = self.navigation_results[3][1]
        propagated_covariance_epochs = self.navigation_results[2][0]
        relative_epochs = full_propagated_formal_errors_epochs - self.mission_start_epoch

        # Plot the estimation error history
        for k in range(2):
            for j in range(2):
                colors = ["red", "green", "blue"]
                symbols = [[r"x", r"y", r"z"], [r"v_{x}", r"v_{y}", r"v_{z}"]]
                ylabels = ["3D RSS OD position \nuncertainty [m]", "3D RSS OD velocity \nuncertainty [m/s]"]
                ax[k][j].plot(relative_epochs, self.sigma_number*np.linalg.norm(full_propagated_formal_errors_history[:, 3*k+6*j:3*k+6*j+3], axis=1), label=self.navigation_simulator.model_name)

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
                    station_keeping_epoch = epoch - self.mission_start_epoch
                    ax[k][j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.7, label="SKM" if i==0 else None)

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

        fig.suptitle(f"Total 3D RSS 3$\sigma$ uncertainty \n Model: on-board: {self.navigation_simulator.model_name}{self.navigation_simulator.model_number}, truth: {self.navigation_simulator.model_name_truth}{self.navigation_simulator.model_number_truth}")
        plt.tight_layout()


    def plot_dispersion_history(self):

        # Plot how the deviation from the reference orbit
        fig, ax = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
        full_reference_state_deviation_epochs = self.navigation_results[1][0]
        full_reference_state_deviation_history = self.navigation_results[1][1]
        relative_epochs = full_reference_state_deviation_epochs - self.mission_start_epoch

        colors = ["red", "green", "blue"]
        labels = [[r"$x$", r"$y$", r"$z$"], [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"]]
        ylabels = [r"$\mathbf{r}-\mathbf{r}_{ref}$ [m]", r"$\mathbf{v}-\mathbf{v}_{ref}$ [m/s]"]

        for j in range(2):

            for i in range(3):
                ax[j].plot(relative_epochs, full_reference_state_deviation_history[:,6+3*j+i], label=labels[j][i])

            for i, gap in enumerate(self.observation_windows):
                ax[j].axvspan(
                    xmin=gap[0]-self.mission_start_epoch,
                    xmax=gap[1]-self.mission_start_epoch,
                    color="gray",
                    alpha=0.1,
                    label="Observation window" if i == 0 else None)

            for i, epoch in enumerate(self.station_keeping_epochs):
                station_keeping_epoch = epoch - self.mission_start_epoch
                ax[j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.7, label="SKM" if i==0 else None)

            ax[j].set_ylabel(ylabels[j])
            ax[j].grid(alpha=0.5, linestyle='--')
            # ax[0].set_title("LUMIO")

            # Set y-axis tick label format to scientific notation with one decimal place
            ax[j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax[-1].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")
            ax[j].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

        plt.tight_layout()
        fig.suptitle(f"Deviation from reference orbit LUMIO \n Model: on-board: {self.navigation_simulator.model_name}{self.navigation_simulator.model_number}, truth: {self.navigation_simulator.model_name_truth}{self.navigation_simulator.model_number_truth}")


    def plot_estimation_error_history(self):

        # Plot the estimation error history
        fig, ax = plt.subplots(2, 2, figsize=(11, 4), sharex=True)

        full_estimation_error_epochs = self.navigation_results[0][0]
        full_estimation_error_history = self.navigation_results[0][1]
        propagated_covariance_epochs = self.navigation_results[2][0]
        full_propagated_formal_errors_history = self.navigation_results[3][1]
        relative_epochs = propagated_covariance_epochs - self.mission_start_epoch

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

                    # ax[k][j].plot(relative_epochs, np.abs(sigma), color=colors[i], ls="--", label=f"$3\sigma_{{{symbols[k][i]}}}$", alpha=0.3)
                    # # ax[k][j].plot(relative_epochs, -sigma, color=colors[i], ls="-.", alpha=0.3)
                    # ax[k][j].plot(relative_epochs, np.abs(full_estimation_error_history[:,3*k+6*j+i]), color=colors[i], label=f"${symbols[k][i]}-\hat{{{symbols[k][i]}}}$")
                    # ax[k][j].set_yscale("log")

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
                    station_keeping_epoch = epoch - self.mission_start_epoch
                    ax[k][j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.2, label="SKM" if i==0 else None)
                ax[k][0].set_ylabel(ylabels[k])
                ax[k][j].grid(alpha=0.5, linestyle='--')
                ax[0][0].set_ylim(-100, 100)
                ax[1][0].set_ylim(-0.03, 0.03)
                # ax[0][0].set_ylim(-1000, 1000)
                # ax[0][0].set_ylim(-1000, 1000)
                ax[k][0].set_title("LPF")
                ax[k][1].set_title("LUMIO")

                # Set y-axis tick label format to scientific notation with one decimal place
                ax[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax[-1][j].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")

        fig.suptitle(f"Estimaton error history | range-only \n Model: on-board: {self.navigation_simulator.model_name}{self.navigation_simulator.model_number}, truth: {self.navigation_simulator.model_name_truth}{self.navigation_simulator.model_number_truth}")
        # fig4.suptitle("Estimation error history: range-only, $1\sigma_{\rho}$ = 102.44 [$m$], $f_{obs}$ = $1/600$ [$s^{-1}$]")
        plt.tight_layout()


    def plot_dispersion_to_estimation_error_history(self):

        # Plot how the deviation from the reference orbit
        fig, ax = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
        full_reference_state_deviation_epochs = self.navigation_results[1][0]
        full_reference_state_deviation_history = self.navigation_results[1][1]
        full_estimation_error_epochs = self.navigation_results[0][0]
        full_estimation_error_history = self.navigation_results[0][1]
        relative_epochs = full_estimation_error_epochs - self.mission_start_epoch

        estimation_error_to_dispersion_history = full_reference_state_deviation_history/full_estimation_error_history

        colors = ["red", "green", "blue"]
        labels = [[r"$x$", r"$y$", r"$z$"], [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"]]
        ylabels = [r"$\mathbf{r}-\mathbf{r}_{ref}$ [m]", r"$\mathbf{v}-\mathbf{v}_{ref}$ [m/s]"]

        for j in range(2):

            for i in range(3):
                ax[j].plot(relative_epochs, estimation_error_to_dispersion_history[:,6+3*j+i], label=labels[j][i])

            for i, gap in enumerate(self.observation_windows):
                ax[j].axvspan(
                    xmin=gap[0]-self.mission_start_epoch,
                    xmax=gap[1]-self.mission_start_epoch,
                    color="gray",
                    alpha=0.1,
                    label="Observation window" if i == 0 else None)

            for i, epoch in enumerate(self.station_keeping_epochs):
                station_keeping_epoch = epoch - self.mission_start_epoch
                ax[j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.7, label="SKM" if i==0 else None)

            ax[j].set_ylabel(ylabels[j])
            ax[j].grid(alpha=0.5, linestyle='--')
            # ax[0].set_title("LUMIO")

            # Set y-axis tick label format to scientific notation with one decimal place
            ax[j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax[-1].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")
            ax[j].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

        plt.tight_layout()
        fig.suptitle(f"Deviation from reference orbit LUMIO \n Model: on-board: {self.navigation_simulator.model_name}{self.navigation_simulator.model_number}, truth: {self.navigation_simulator.model_name_truth}{self.navigation_simulator.model_number_truth}")



    def plot_observations(self):

        fig, ax = plt.subplots(3, 1, figsize=(11, 4), sharex=True)
        arc_nums = len(self.navigation_results[-1].keys())

        # For each arc, plot the observations and its residuals
        for arc_num in range(arc_nums):

            estimation_model = self.navigation_results[-1][arc_num]
            estimation_output = estimation_model.estimation_output

            for i, (observable_type, information_sets) in enumerate(estimation_model.sorted_observation_sets.items()):
                for j, observation_set in enumerate(information_sets.values()):
                    for k, single_observation_set in enumerate(observation_set):

                        color = "blue"
                        s = 0.5

                        observation_times = utils.convert_epochs_to_MJD(single_observation_set.observation_times)
                        observation_times = observation_times - self.mission_start_epoch
                        ax[0].scatter(observation_times, single_observation_set.concatenated_observations, color=color, s=s)

                        residual_history = estimation_output.residual_history
                        best_iteration = estimation_output.best_iteration

                        index = int(len(observation_times))
                        ax[1].scatter(observation_times, residual_history[i*index:(i+1)*index, best_iteration], color=color, s=s)


        # Plot the history of observation angle with respect to the large covariance axis
        state_history = self.navigation_results[6][1]
        epochs = self.navigation_results[9][0]
        dependent_variables_history = self.navigation_results[9][1]
        relative_state_history = dependent_variables_history[:,6:12]
        full_propagated_covariance_epochs = self.navigation_results[2][0]
        full_propagated_covariance_history = self.navigation_results[2][1]

        # Generate history of eigenvectors
        eigenvectors_dict = dict()
        for key, matrix in enumerate(full_propagated_covariance_history):
            eigenvalues, eigenvectors = np.linalg.eigh(matrix[6:9, 6:9])
            max_eigenvalue_index = np.argmax(eigenvalues)
            eigenvector_largest = eigenvectors[:, max_eigenvalue_index]
            eigenvectors_dict.update({full_propagated_covariance_epochs[key]: eigenvector_largest})

        # Store the angles
        angles_dict = dict()
        for i, (key, value) in enumerate(eigenvectors_dict.items()):
            vec1 = relative_state_history[i,:3]
            vec2 = value
            cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle_radians = np.arccos(cosine_angle)
            angle_degrees = np.degrees(angle_radians)
            angles_dict.update({key: np.abs(angle_degrees) if np.abs(angle_degrees)<90 else (180-angle_degrees)})

        # Generate boolans for when treshold condition holds to generate estimation window
        angle_to_range_dict = dict()
        for i, state in enumerate(state_history):

            # Define the 3D vector (replace these values with your actual vector)
            vector = relative_state_history[i, 0:3]

            # Calculate the angles with respect to the x, y, and z axes
            angle_x = np.arctan2(vector[1], vector[0])  # Angle with respect to the x-axis
            angle_y = np.arctan2(vector[2], np.sqrt(vector[0]**2 + vector[1]**2))  # Angle with respect to the y-axis
            angle_z = np.arctan2(np.sqrt(vector[0]**2 + vector[1]**2), vector[2])  # Angle with respect to the z-axis

            # Convert angles from radians to degrees
            angle_x_degrees = np.degrees(angle_x)
            angle_y_degrees = np.degrees(angle_y)
            angle_z_degrees = np.degrees(angle_z)

            angle_to_range_dict.update({epochs[i]: np.array([angle_x_degrees, angle_y_degrees, angle_z_degrees])})

        # fig = plt.figure()
        # ax[2].plot(np.stack(list(angles_dict.keys()))-self.mission_start_epoch, np.stack(list(angles_dict.values())), label="angles in degrees", color=color)
        # plt.show()

        ax[2].plot(np.stack(list(angle_to_range_dict.keys()))-self.mission_start_epoch, np.stack(list(angle_to_range_dict.values())), label=[r"$\alpha$", r"$\beta$", r"$\gamma$"])
        ax[2].legend(loc='upper left', fontsize="small")

        states_history_LPF_moon = state_history[:, 0:3]-dependent_variables_history[:, 0:3]
        states_history_LUMIO_moon = state_history[:, 6:9]-dependent_variables_history[:, 0:3]

        angle_deg = []
        for i in range(len(epochs)):
            cosine_angle = np.dot(relative_state_history[i,:3], states_history_LUMIO_moon[i])/(np.linalg.norm(relative_state_history[i,:3])*np.linalg.norm(states_history_LUMIO_moon[i]))
            angle = np.arccos(cosine_angle)
            angle_deg.append(np.degrees(angle))

        # angle_deg = []
        # for i in range(len(epochs)):
        #     cosine_angle = np.dot(states_history_LPF_moon[i], states_history_LUMIO_moon[i])/(np.linalg.norm(states_history_LPF_moon[i])*np.linalg.norm(states_history_LUMIO_moon[i]))
        #     angle = np.arccos(cosine_angle)
        #     angle_deg.append(np.degrees(angle))


        # ax[2].plot(np.stack(list(angle_to_range_dict.keys()))-self.mission_start_epoch, state_history[:, 0:3]-dependent_variables_history[:, 0:3], label=[r"$\alpha$", r"$\beta$", r"$\gamma$"])
        # ax[2].plot(np.stack(list(angle_to_range_dict.keys()))-self.mission_start_epoch, angle_deg, label=[r"$\alpha$"], color="blue")

        # ax[2].plot(np.stack(list(angle_to_range_dict.keys()))-self.mission_start_epoch, np.linalg.norm(state_history[:, 0:3]-dependent_variables_history[:, 0:3], axis=1), label="Rel pos")
        # ax[2].plot(np.stack(list(angle_to_range_dict.keys()))-self.mission_start_epoch, np.linalg.norm(state_history[:, 3:6]-dependent_variables_history[:, 3:6], axis=1), label="Rel vel")



        for j in range(len(ax)):
            for i, gap in enumerate(self.observation_windows):
                ax[j].axvspan(
                    xmin=gap[0]-self.mission_start_epoch,
                    xmax=gap[1]-self.mission_start_epoch,
                    color="gray",
                    alpha=0.1,
                    label="Observation window" if i == 0 else None)
            for i, epoch in enumerate(self.station_keeping_epochs):
                station_keeping_epoch = epoch - self.mission_start_epoch
                ax[j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.7, label="SKM" if i==0 else None)

            ax[j].grid(alpha=0.5, linestyle='--')

            # Set y-axis tick label format to scientific notation with one decimal place
            ax[j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax[0].set_ylabel("Range [m]")
        ax[1].set_ylabel("Observation \n residual [m]")
        ax[2].set_ylabel("Angle obs. \n w.r.t J2000 [deg]")
        ax[-1].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")
        ax[0].legend(bbox_to_anchor=(1, 1.04), loc='upper left')


        fig.suptitle(f"Intersatellite range observations \n Model: on-board: {self.navigation_simulator.model_name}{self.navigation_simulator.model_number}, truth: {self.navigation_simulator.model_name_truth}{self.navigation_simulator.model_number_truth}")
        plt.tight_layout()
        # plt.show()


    def plot_observability(self):

        fig, ax = plt.subplots(5, 1, figsize=(8, 11), sharex=True)
        arc_nums = len(self.navigation_results[-1].keys())

        for arc_num in range(arc_nums):

            estimation_model = self.navigation_results[-1][arc_num]
            estimation_output = estimation_model.estimation_output
            estimator = estimation_model.estimator
            state_transition_interface = estimator.state_transition_interface

            # Generate information and covariance histories based on all the combinations of observables and link definitions
            total_information_dict = dict()
            total_covariance_dict = dict()
            total_single_information_dict = dict()
            state_transition_matrix_dict = dict()
            len_obs_list = []
            for i, (observable_type, observation_sets) in enumerate(estimation_model.sorted_observation_sets.items()):
                total_information_dict[observable_type] = dict()
                total_covariance_dict[observable_type] = dict()
                total_single_information_dict[observable_type] = dict()
                for j, observation_set in enumerate(observation_sets.values()):
                    total_information_dict[observable_type][j] = list()
                    total_covariance_dict[observable_type][j] = list()
                    total_single_information_dict[observable_type][j] = list()
                    for k, single_observation_set in enumerate(observation_set):

                        epochs = single_observation_set.observation_times
                        len_obs_list.append(len(epochs))

                        weighted_design_matrix_history = np.stack([estimation_model.estimation_output.weighted_design_matrix[sum(len_obs_list[:-1]):sum(len_obs_list), :]], axis=1)

                        information_dict = dict()
                        single_information_dict = dict()
                        information_vector_dict = dict()
                        total_information = 0
                        total_information_vector = 0
                        for index, weighted_design_matrix in enumerate(weighted_design_matrix_history):

                            epoch = epochs[index]

                            state_transition_matrix = state_transition_interface.full_state_transition_sensitivity_at_epoch(epoch)
                            # weighted_design_matrix = np.dot(weighted_design_matrix, np.linalg.inv(state_transition_matrix))
                            state_transition_matrix_product = np.dot(state_transition_matrix, state_transition_matrix.T)
                            weighted_design_matrix_product = np.dot(weighted_design_matrix.T, weighted_design_matrix)

                            # Calculate the information matrix
                            current_information = total_information + weighted_design_matrix_product
                            single_information_dict[epoch] = weighted_design_matrix_product
                            state_transition_matrix_dict[epoch] = state_transition_matrix_product
                            information_dict[epoch] = current_information
                            total_information = current_information

                        covariance_dict = dict()
                        for key in information_dict:
                            if estimation_model.apriori_covariance is not None:
                                information_dict[key] = information_dict[key] + np.linalg.inv(estimation_model.apriori_covariance)
                            covariance_dict[key] = np.linalg.inv(information_dict[key])

                        total_information_dict[observable_type][j].append(information_dict)
                        total_covariance_dict[observable_type][j].append(covariance_dict)
                        total_single_information_dict[observable_type][j].append(single_information_dict)

            for i, (observable_type, information_sets) in enumerate(total_single_information_dict.items()):
                for j, information_set in enumerate(information_sets.values()):
                    for k, single_information_set in enumerate(information_set):

                        information_dict = total_single_information_dict[observable_type][j][k]
                        epochs = utils.convert_epochs_to_MJD(np.array(list(information_dict.keys())))
                        epochs = epochs - self.mission_start_epoch
                        information_matrix_history = np.array(list(information_dict.values()))
                        state_transition_matrix_product_history = np.array(list(state_transition_matrix_dict.values()))

                        for m in range(2):
                            observability_lpf = np.sqrt(np.stack([np.diagonal(matrix) for matrix in information_matrix_history[:,0+3*m:3+3*m,0+3*m:3+3*m]]))
                            observability_lumio = np.sqrt(np.stack([np.diagonal(matrix) for matrix in information_matrix_history[:,6+3*m:9+3*m,6+3*m:9+3*m]]))
                            observability_lpf_total = np.sqrt(np.max(np.linalg.eigvals(information_matrix_history[:,0+3*m:3+3*m,0+3*m:3+3*m]), axis=1, keepdims=True))
                            observability_lumio_total = np.sqrt(np.max(np.linalg.eigvals(information_matrix_history[:,6+3*m:9+3*m,6+3*m:9+3*m]), axis=1, keepdims=True))

                            ax[m].plot(epochs, observability_lpf_total, label="LPF" if m == 0 and arc_num == 0 else None, color="darkred")
                            ax[m].plot(epochs, observability_lumio_total, label="LUMIO" if m == 0 and arc_num == 0 else None, color="darkblue")

                            min_axis_stm_lpf = np.min(np.linalg.eigvals(state_transition_matrix_product_history[:,0+3*m:3+3*m,0+3*m:3+3*m]), axis=1)
                            max_axis_stm_lpf = np.max(np.linalg.eigvals(state_transition_matrix_product_history[:,0+3*m:3+3*m,0+3*m:3+3*m]), axis=1)
                            min_axis_stm_lumio = np.min(np.linalg.eigvals(state_transition_matrix_product_history[:,6+3*m:9+3*m,6+3*m:9+3*m]), axis=1)
                            max_axis_stm_lumio = np.max(np.linalg.eigvals(state_transition_matrix_product_history[:,6+3*m:9+3*m,6+3*m:9+3*m]), axis=1)

                            aspect_ratio_lpf = max_axis_stm_lpf/min_axis_stm_lpf
                            aspect_ratio_lumio = max_axis_stm_lumio/min_axis_stm_lumio

                            if m == 0:
                                ax[2].plot(epochs, aspect_ratio_lpf, label="LPF" if m == 0 and arc_num == 0 else None, color="darkred")
                                ax[3].plot(epochs, aspect_ratio_lumio, label="LUMIO" if m == 0 and arc_num == 0 else None, color="darkblue")
                            ax[2].set_yscale("log")

        for j in range(len(ax)):
            for i, gap in enumerate(self.observation_windows):
                ax[j].axvspan(
                    xmin=gap[0]-self.mission_start_epoch,
                    xmax=gap[1]-self.mission_start_epoch,
                    color="gray",
                    alpha=0.1,
                    label="Observation window" if i == 0 else None)
            for i, epoch in enumerate(self.station_keeping_epochs):
                station_keeping_epoch = epoch - self.mission_start_epoch
                ax[j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.7, label="SKM" if i==0 else None)

            ax[j].grid(alpha=0.5, linestyle='--')
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")

            # Set y-axis tick label format to scientific notation with one decimal place
            ax[j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            # ax[j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax[0].set_ylabel(r'$\sqrt{\max \operatorname{eig}(\delta \mathbf{\Lambda}_{\mathbf{r}, j})}$')
        ax[1].set_ylabel(r'$\sqrt{\max \operatorname{eig}(\delta \mathbf{\Lambda}_{\mathbf{v}, j})}$')
        ax[2].set_ylabel('Aspect ratio \n $\operatorname{eig} \mathbf{\Phi}_{rr}\mathbf{\Phi}_{rr}^T$')
        ax[3].set_ylabel('Aspect ratio \n $\operatorname{eig} \mathbf{\Phi}_{rr}\mathbf{\Phi}_{rr}^T$')
        ax[4].set_ylabel(r'$||\mathbf{v}_{LPF}||$')
        ax[-1].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")
        ax[0].legend(bbox_to_anchor=(1, 1.04), loc='upper left')

        # Plot the history of observation angle with respect to the large covariance axis
        state_history = self.navigation_results[6][1]
        epochs = self.navigation_results[9][0]
        dependent_variables_history = self.navigation_results[9][1]
        moon_state_history = dependent_variables_history[:,0:6]

        state_history_moon_lpf = state_history[:, 0:6] - moon_state_history
        state_history_moon_lumio = state_history[:, 6:12] - moon_state_history

        ax[4].plot(epochs-self.mission_start_epoch, np.linalg.norm(state_history_moon_lpf[:, 3:6], axis=1), color="green")

        fig.suptitle(f"Intersatellite range observability \n Model: on-board: {self.navigation_simulator.model_name}{self.navigation_simulator.model_number}, truth: {self.navigation_simulator.model_name_truth}{self.navigation_simulator.model_number_truth}")
        # plt.tight_layout()
        # plt.show()


    def plot_od_error_dispersion_relation(self):

        fig, axs = plt.subplots(1, 1, figsize=(11, 4), sharex=True)

        full_estimation_error_dict = self.navigation_simulator.full_estimation_error_dict
        full_reference_state_deviation_dict = self.navigation_simulator.full_reference_state_deviation_dict

        epochs = np.stack(list(full_estimation_error_dict.keys()))
        full_estimation_error_history = np.stack(list(full_estimation_error_dict.values()))
        full_reference_state_deviation_history = np.stack(list(full_reference_state_deviation_dict.values()))

        print(len(full_estimation_error_history), len(full_reference_state_deviation_history))

        relative_epochs = epochs - self.mission_start_epoch
        od_error = np.linalg.norm(full_estimation_error_history[:, 6:9], axis=1)
        dispersion = np.linalg.norm(full_reference_state_deviation_history[:, 6:9], axis=1)
        od_error_dispersion_relation = dispersion/od_error

        axs.plot(relative_epochs, dispersion)
        axs.plot(relative_epochs, od_error)
        axs.plot(relative_epochs, od_error_dispersion_relation)

        # axs[1].set_xlabel(r"||$\Delta V$|| [m/s]")
        # axs[0].set_ylabel(r"||$\hat{\mathbf{r}}-\mathbf{r}$|| [m]")
        # axs[1].set_ylabel(r"||$\mathbf{r}-\mathbf{r}_{ref}$|| [m]")
        # axs[0].set_title("Maneuver cost versus OD error")
        # axs[1].set_title("Maneuver cost versus reference orbit deviation")
        # axs[0].legend(title="Arc duration", bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

        # fig.suptitle("Relations between SKM cost for run of 28 days")
        plt.tight_layout()
        # plt.show()


    def plot_correlation_history(self):

        # Plot the estimation error history
        arc_nums = list(self.navigation_results[-1].keys())

        fig, ax = plt.subplots(1, 2, figsize=(9, 4))

        full_propagated_covariance_history = self.navigation_results[2][1]

        correlation_start = np.corrcoef(full_propagated_covariance_history[0])
        correlation_end = np.corrcoef(full_propagated_covariance_history[-1])

        estimation_model = self.navigation_results[-1][arc_nums[0]]
        estimation_output = estimation_model.estimation_output
        correlation_end = estimation_output.correlations

        estimated_param_names = [r"$x_{1}$", r"$y_{1}$", r"$z_{1}$", r"$\dot{x}_{1}$", r"$\dot{y}_{1}$", r"$\dot{z}_{1}$",
                                r"$x_{2}$", r"$y_{2}$", r"$z_{2}$", r"$\dot{x}_{2}$", r"$\dot{y}_{2}$", r"$\dot{z}_{2}$"]

        im_start = ax[0].imshow(correlation_start, cmap="viridis", vmin=-1, vmax=1)
        im_end = ax[1].imshow(correlation_end, cmap="viridis", vmin=-1, vmax=1)

        for i in range(2):
            ax[i].set_xticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)
            ax[i].set_yticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)
            ax[i].set_xlabel("Estimated Parameter")
            ax[i].set_ylabel("Estimated Parameter")

        # ax[0].set_ylabel("Estimated Parameter")
        ax[0].set_title("Before arc")
        ax[1].set_title("After arc")

        plt.colorbar(im_start)
        plt.colorbar(im_end)

        fig.suptitle(f"State correlations for estimation, example arc of {self.navigation_simulator.estimation_arc_durations[0]} days")
        fig.tight_layout()

        # plt.show()