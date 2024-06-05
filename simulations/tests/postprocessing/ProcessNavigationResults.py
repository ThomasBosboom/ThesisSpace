# Standard
import os
import sys
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter
from collections import defaultdict

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(3):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Own
from tests import utils
import FrameConverter


class PlotSingleNavigationResults():

    def __init__(self, navigation_output, sigma_number=3, figure_settings={"save_figure": True, "current_time": float, "file_name": str}):

        self.navigation_output = navigation_output
        self.navigation_simulator = navigation_output.navigation_simulator
        self.navigation_results = navigation_output.navigation_results
        self.sigma_number = sigma_number
        self.mission_start_epoch = self.navigation_simulator.mission_start_epoch
        self.observation_windows = self.navigation_simulator.observation_windows
        self.station_keeping_epochs = self.navigation_simulator.station_keeping_epochs

        for key, value in figure_settings.items():
            setattr(self, key, value)


    def plot_full_state_history(self):

        fig1_3d = plt.figure()
        ax_3d = fig1_3d.add_subplot(111, projection='3d')
        fig1_3d2 = plt.figure()
        ax_3d2 = fig1_3d2.add_subplot(111, projection='3d')
        fig, ax = plt.subplots(2, 3, figsize=(11, 6))

        state_history_reference = self.navigation_results[4][1]
        state_history_truth = self.navigation_results[5][1]
        state_history_estimated = self.navigation_results[6][1]
        epochs = self.navigation_results[9][0]
        dependent_variables_history = self.navigation_results[9][1]
        delta_v_dict = self.navigation_simulator.delta_v_dict
        full_state_history_truth_dict = self.navigation_simulator.full_state_history_truth_dict
        full_state_history_reference_dict = self.navigation_simulator.full_state_history_reference_dict

        moon_data_dict = {epoch: state for epoch, state in zip(epochs, dependent_variables_history[:, :6])}
        full_state_history_estimated_dict = {epoch: state for epoch, state in zip(epochs, state_history_estimated[:, :])}

        G = 6.67430e-11
        m1 = 5.972e24
        m2 = 7.34767309e22
        mu = m2/(m2 + m1)

        # Create the transformation based on rotation axis of Moon around Earth
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
        synodic_full_state_history_estimated_dict = {}
        synodic_full_state_history_truth_dict = {}
        synodic_full_state_history_reference_dict = {}
        synodic_dictionaries = [synodic_full_state_history_estimated_dict, synodic_full_state_history_truth_dict, synodic_full_state_history_reference_dict]
        inertial_dictionaries = [full_state_history_estimated_dict, full_state_history_truth_dict, full_state_history_reference_dict]
        for index, dictionary in enumerate(inertial_dictionaries):
            for epoch, state in dictionary.items():

                transformation_matrix = transformation_matrix_dict[epoch]
                synodic_state = np.concatenate((np.dot(transformation_matrix, state[0:6]), np.dot(transformation_matrix, state[6:12])))

                LU = np.linalg.norm((moon_data_dict[epoch][0:3]))
                TU = np.sqrt(LU**3/(G*(m1+m2)))
                synodic_state[0:3] = synodic_state[0:3]/LU
                synodic_state[6:9] = synodic_state[6:9]/LU
                synodic_state[3:6] = synodic_state[3:6]/(LU/TU)
                synodic_state[9:12] = synodic_state[9:12]/(LU/TU)
                synodic_state = (1-mu)*synodic_state

                synodic_dictionaries[index].update({epoch: synodic_state})


        inertial_states = np.stack(list(full_state_history_estimated_dict.values()))
        inertial_states_truth = np.stack(list(full_state_history_truth_dict.values()))
        inertial_states_reference = np.stack(list(full_state_history_reference_dict.values()))

        synodic_states_estimated = np.stack(list(synodic_full_state_history_estimated_dict.values()))
        synodic_states_truth = np.stack(list(synodic_full_state_history_truth_dict.values()))
        synodic_states_reference = np.stack(list(synodic_full_state_history_reference_dict.values()))

        # print("Initial state estimated inertial: \n", inertial_states[0, :])
        # print("Initial state truth inertial: \n", inertial_states_truth[0, :])
        # print("Initial state estimated synodic: \n", synodic_states_estimated[0, :])
        # print("Initial state truth synodic: \n", synodic_states_truth[0, :])
        # print("Initial state reference inertial: \n", inertial_states_reference[0, :])
        # print("Initial state reference synodic: \n", synodic_states_reference[0, :])

        # Generate the synodic states of station keeping maneuvre vectors
        def closest_key(dictionary, value):
            closest_key = None
            min_difference = float('inf')

            for key in dictionary:
                difference = abs(key - value)
                if difference < min_difference:
                    min_difference = difference
                    closest_key = key

            return closest_key

        synodic_delta_v_dict = {}
        for epoch, delta_v in delta_v_dict.items():

            epoch = closest_key(transformation_matrix_dict, epoch)
            transformation_matrix = transformation_matrix_dict[epoch]
            synodic_state = np.dot(transformation_matrix, np.concatenate((np.zeros((3)), delta_v)))

            LU = np.linalg.norm((moon_data_dict[epoch][0:3]))
            TU = np.sqrt(LU**3/(G*(m1+m2)))
            synodic_delta_v = synodic_state/(LU/TU)
            synodic_delta_v = (1-mu)*synodic_state

            synodic_delta_v_dict.update({epoch: synodic_state[3:6]})

        synodic_delta_v_history = np.stack(list(synodic_delta_v_dict.values()))

        # for start, length, angle in zip(start_positions, arrow_lengths, arrow_angles):
        arrow_plot_dict = {}
        for index, (epoch, delta_v) in enumerate(synodic_delta_v_dict.items()):
            arrow_plot_dict[epoch] = np.concatenate((synodic_full_state_history_estimated_dict[epoch][6:9], delta_v))

        arrow_plot_data = np.stack(list(arrow_plot_dict.values()))
        scale=None
        alpha=0.6
        zorder=10
        for index in range(1):
            ax[1][0].quiver(arrow_plot_data[:, 0], arrow_plot_data[:, 2], arrow_plot_data[:, 3], arrow_plot_data[:, 5],
                        angles='xy', scale_units='xy', scale=scale, zorder=zorder, alpha=alpha)
            ax[1][1].quiver(arrow_plot_data[:, 1], arrow_plot_data[:, 2], arrow_plot_data[:,4], arrow_plot_data[:,5],
                        angles='xy', scale_units='xy', scale=scale, zorder=zorder, alpha=alpha)
            ax[1][2].quiver(arrow_plot_data[:, 0], arrow_plot_data[:, 1], arrow_plot_data[:,3], arrow_plot_data[:,4],
                        angles='xy', scale_units='xy', scale=scale, zorder=zorder, alpha=alpha, label="SKM" if index==0 else None)
            ax_3d.quiver(arrow_plot_data[:, 0], arrow_plot_data[:, 1],  arrow_plot_data[:, 2], arrow_plot_data[:, 3], arrow_plot_data[:, 4], arrow_plot_data[:, 5],
                        alpha=alpha, color="gray", length=2, normalize=False, label="SKM" if index==0 else None)

        # Generate the synodic states of the moon
        synodic_full_state_history_moon_dict = {}
        for epoch, state in moon_data_dict.items():

            transformation_matrix = transformation_matrix_dict[epoch]
            synodic_state = np.dot(transformation_matrix, state)
            LU = np.linalg.norm((moon_data_dict[epoch][0:3]))
            TU = np.sqrt(LU**3/(G*(m1+m2)))
            synodic_state[0:3] = synodic_state[0:3]/LU
            synodic_state[3:6] = synodic_state[3:6]/(LU/TU)
            synodic_state = (1-mu)*synodic_state
            synodic_full_state_history_moon_dict.update({epoch: synodic_state})

        synodic_states_estimated = np.stack(list(synodic_full_state_history_estimated_dict.values()))
        synodic_moon_states = np.stack(list(synodic_full_state_history_moon_dict.values()))

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(2):
            if i == 0:
                color="gray"
            else:
                color="black"

            ax[i][0].scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 2], s=50, color="darkgray")
            ax[i][1].scatter(synodic_moon_states[:, 1], synodic_moon_states[:, 2], s=50, color="darkgray")
            ax[i][2].scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 1], s=50, color="darkgray", label="Moon" if i==0 else None)
            ax[i][0].plot(synodic_states_estimated[:, 6*i+0], synodic_states_estimated[:, 6*i+2], lw=0.5, color=color)
            ax[i][1].plot(synodic_states_estimated[:, 6*i+1], synodic_states_estimated[:, 6*i+2], lw=0.5, color=color)
            ax[i][2].plot(synodic_states_estimated[:, 6*i+0], synodic_states_estimated[:, 6*i+1], lw=0.5, color=color, label="LPF" if i==0 else None)
            ax[1][0].plot(synodic_states_estimated[:, 6*i+0], synodic_states_estimated[:, 6*i+2], lw=0.1, color=color)
            ax[1][1].plot(synodic_states_estimated[:, 6*i+1], synodic_states_estimated[:, 6*i+2], lw=0.1, color=color)
            ax[1][2].plot(synodic_states_estimated[:, 6*i+0], synodic_states_estimated[:, 6*i+1], lw=0.1, color=color, label="LUMIO" if i==1 else None)

        ax_3d.scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 1], synodic_moon_states[:, 2], s=50, color="darkgray", label="Moon")
        ax_3d.plot(synodic_states_estimated[:, 0], synodic_states_estimated[:, 1], synodic_states_estimated[:, 2], lw=0.2, color="gray")
        ax_3d.plot(synodic_states_estimated[:, 6], synodic_states_estimated[:, 7], synodic_states_estimated[:, 8], lw=0.7, color="black")
        # ax_3d.scatter(-mu, 0, 0, label="Earth", color="darkblue", s=50)

        # ax_3d2.plot(state_history_reference[:,0], state_history_reference[:,1], state_history_reference[:,2], label="LPF ref", color="green")
        # ax_3d2.plot(state_history_reference[:,6], state_history_reference[:,7], state_history_reference[:,8], label="LUMIO ref", color="green")
        # ax_3d2.plot(state_history_truth[:,0], state_history_truth[:,1], state_history_truth[:,2], label="LPF truth", color="black", ls="--")
        # ax_3d2.plot(state_history_truth[:,6], state_history_truth[:,7], state_history_truth[:,8], label="LUMIO truth", color="black", ls="--")
        ax_3d2.plot(state_history_estimated[:,0], state_history_estimated[:,1], state_history_estimated[:,2], lw=0.5, label="LPF", color="gray")
        ax_3d2.plot(state_history_estimated[:,6], state_history_estimated[:,7], state_history_estimated[:,8], lw=0.5, label="LUMIO", color="black")
        ax_3d2.scatter(0, 0, 0, label="Earth", color="darkblue", s=50)


        for num, (start, end) in enumerate(self.navigation_simulator.observation_windows):
            synodic_states_window_dict = {key: value for key, value in synodic_full_state_history_estimated_dict.items() if key >= start and key <= end}
            synodic_states_window = np.stack(list(synodic_states_window_dict.values()))

            inertial_states_window_dict = {key: value for key, value in full_state_history_estimated_dict.items() if key >= start and key <= end}
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
        ax_3d.legend()

        ax_3d2.set_xlabel('X [m]')
        ax_3d2.set_ylabel('Y [m]')
        ax_3d2.set_zlabel('Z [m]')

        ax_3d2.legend()

        ax[0][2].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
        ax[1][2].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

        # fig.suptitle(f"Tracking arcs, synodic frame ")
        # fig1_3d.suptitle(f"Tracking arcs, synodic frame ")
        # fig1_3d2.suptitle(f"Tracking arcs, inertial frame ")
        plt.tight_layout()
        plt.legend()


    def plot_formal_error_history(self):

        # Plot how the formal errors grow over time
        fig, ax = plt.subplots(2, 2, figsize=(11, 4), sharex=True)

        full_propagated_formal_errors_epochs = self.navigation_results[3][0]
        full_propagated_formal_errors_history = self.navigation_results[3][1]
        relative_epochs = full_propagated_formal_errors_epochs - self.mission_start_epoch

        colors = ["red", "green", "blue"]
        labels = [[r"$x$", r"$y$", r"$z$"], [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"]]
        ylabels = [r"$\mathbf{r}-\mathbf{r}_{ref}$ [m]", r"$\mathbf{v}-\mathbf{v}_{ref}$ [m/s]"]
        for l in range(2):
            for m in range(2):
                for n in range(3):
                    ax[l][m].plot(relative_epochs, self.sigma_number*full_propagated_formal_errors_history[:,3*l+6*m+n], label=labels[l][n], color=colors[n])

        for k in range(2):
            for j in range(2):

                for i, gap in enumerate(self.observation_windows):
                    ax[k][j].axvspan(
                        xmin=gap[0]-self.mission_start_epoch,
                        xmax=gap[1]-self.mission_start_epoch,
                        color="gray",
                        alpha=0.1,
                        label="Tracking arc" if i == 0 else None)

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
                ax[-1][j].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]", fontsize="small")

        ax[0][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

        # fig.suptitle(f"Formal error history")
        plt.tight_layout()


    def plot_uncertainty_history(self):

        fig, ax = plt.subplots(2, 2, figsize=(11, 4), sharex=True)
        full_propagated_formal_errors_epochs = self.navigation_results[3][0]
        full_propagated_formal_errors_history = self.navigation_results[3][1]
        propagated_covariance_epochs = self.navigation_results[2][0]
        relative_epochs = full_propagated_formal_errors_epochs - self.mission_start_epoch

        # Plot the estimation error history
        colors = ["red", "green", "blue"]
        labels = [[r"$x$", r"$y$", r"$z$"], [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"]]
        ylabels = [r"$\mathbf{r}-\mathbf{r}_{ref}$ [m]", r"$\mathbf{v}-\mathbf{v}_{ref}$ [m/s]"]
        for k in range(2):
            for j in range(2):
                ax[k][j].plot(relative_epochs, self.sigma_number*np.linalg.norm(full_propagated_formal_errors_history[:, 3*k+6*j:3*k+6*j+3], axis=1))

        for k in range(2):
            for j in range(2):
                for i, gap in enumerate(self.observation_windows):
                    ax[k][j].axvspan(
                        xmin=gap[0]-self.mission_start_epoch,
                        xmax=gap[1]-self.mission_start_epoch,
                        color="gray",
                        alpha=0.1,
                        label="Tracking arc" if i == 0 else None)

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

        ax[0][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

        # fig.suptitle(f"Total 3D RSS 3$\sigma$ uncertainty ")
        plt.tight_layout()


    def plot_dispersion_history(self):

        # Plot how the deviation from the reference orbit
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        full_reference_state_deviation_epochs = self.navigation_results[1][0]
        full_reference_state_deviation_history = self.navigation_results[1][1]
        relative_epochs = full_reference_state_deviation_epochs - self.mission_start_epoch

        colors = ["red", "green", "blue"]
        labels = [[r"$x$", r"$y$", r"$z$"], [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"]]
        ylabels = [r"$\mathbf{r}-\mathbf{r}_{ref}$ [m]", r"$\mathbf{v}-\mathbf{v}_{ref}$ [m/s]"]

        for j in range(2):

            for i in range(3):
                ax[j].plot(relative_epochs, full_reference_state_deviation_history[:,6+3*j+i], label=labels[j][i], color=colors[i])

            for i, gap in enumerate(self.observation_windows):
                ax[j].axvspan(
                    xmin=gap[0]-self.mission_start_epoch,
                    xmax=gap[1]-self.mission_start_epoch,
                    color="gray",
                    alpha=0.1,
                    label="Tracking arc" if i == 0 else None)

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
        fig.suptitle(f"Deviation from reference orbit LUMIO ")


    def plot_estimation_error_history(self):

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
                        label="Tracking arc" if i == 0 else None)
                for i, epoch in enumerate(self.station_keeping_epochs):
                    station_keeping_epoch = epoch - self.mission_start_epoch
                    ax[k][j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.2, label="SKM" if i==0 else None)
                ax[k][0].set_ylabel(ylabels[k])
                ax[k][j].grid(alpha=0.5, linestyle='--')
                ax[0][0].set_ylim(-100, 100)
                ax[1][0].set_ylim(-0.03, 0.03)
                ax[k][0].set_title("LPF")
                ax[k][1].set_title("LUMIO")

                # Set y-axis tick label format to scientific notation with one decimal place
                ax[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax[-1][j].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")

        fig.suptitle(f"Estimaton error history")
        plt.tight_layout()


    def plot_observations(self):

        fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
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
            angle_x = np.arctan2(vector[1], vector[0])  # Azimuth
            angle_y = np.arctan2(vector[2], np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2))  # Elevation
            # angle_z = np.arctan2(np.sqrt(vector[0]**2 + vector[1]**2), vector[2])  # Angle with respect to the z-axis

            # Convert angles from radians to degrees
            angle_x_degrees = np.degrees(angle_x)
            angle_y_degrees = np.degrees(angle_y)
            # angle_z_degrees = np.degrees(angle_z)

            angle_to_range_dict.update({epochs[i]: np.array([angle_x_degrees, angle_y_degrees])})

        # fig = plt.figure()
        # ax[2].plot(np.stack(list(angles_dict.keys()))-self.mission_start_epoch, np.stack(list(angles_dict.values())), label="angles in degrees", color=color)
        # plt.show()

        # ax[2].plot(np.stack(list(angle_to_range_dict.keys()))-self.mission_start_epoch, np.stack(list(angle_to_range_dict.values())), label=["Azimuth", "Elevation"])
        # ax[2].legend(loc='upper left', fontsize="small")

        states_history_LPF_moon = state_history[:, 0:3]-dependent_variables_history[:, 0:3]
        states_history_LUMIO_moon = state_history[:, 6:9]-dependent_variables_history[:, 0:3]

        angle_deg = []
        for i in range(len(epochs)):
            cosine_angle = np.dot(relative_state_history[i,:3], states_history_LUMIO_moon[i])/(np.linalg.norm(relative_state_history[i,:3])*np.linalg.norm(states_history_LUMIO_moon[i]))
            angle = np.arccos(cosine_angle)
            angle_deg.append(np.degrees(angle))

        total_accelerations = dependent_variables_history[:, -6:]

        observation_angles = []
        for epoch, dependent_variables in enumerate(dependent_variables_history):
            observation_angle_satellites = []
            for i in reversed(range(2)):

                length = np.shape(dependent_variables_history)[1]
                if i == 0:
                    # total_acceleration = dependent_variables[length-1-3-i*3:length-1-i*3]
                    total_acceleration = dependent_variables[-6:-3]
                else:
                    total_acceleration = dependent_variables[-3:]
                relative_position = dependent_variables[6:9]*(-1+2*i)
                dot_product = np.dot(total_acceleration, relative_position)
                abs_total_acceleration = np.linalg.norm(total_acceleration)
                abs_relative_position = np.linalg.norm(relative_position)
                observation_angle = np.arccos(dot_product/(abs_total_acceleration*abs_relative_position))
                observation_angle_satellites.append(np.degrees(observation_angle))
            observation_angles.append(observation_angle_satellites)

        observation_angles = np.array(observation_angles)

        ax[2].plot(epochs-self.mission_start_epoch, observation_angles[:, 0])
        ax[2].plot(epochs-self.mission_start_epoch, observation_angles[:, 1])




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
                    label="Tracking arc" if i == 0 else None)
            for i, epoch in enumerate(self.station_keeping_epochs):
                station_keeping_epoch = epoch - self.mission_start_epoch
                ax[j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.7, label="SKM" if i==0 else None)

            ax[j].grid(alpha=0.5, linestyle='--')

            # Set y-axis tick label format to scientific notation with one decimal place
            ax[j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax[0].set_ylabel("Range [m]")
        ax[1].set_ylabel("Observation \n residual [m]")
        # ax[2].set_ylabel("Angles obs. \n in ECIJ2000 [deg]")
        ax[2].set_ylabel(r"$\angle \boldsymbol{a}_{j}, \boldsymbol{\rho}$  [deg]")
        ax[-1].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")
        ax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize="small")

        sigma_rho = r"$\sigma_{\rho}$"
        f_obs = r"$\Delta t_{obs}$"
        fig.suptitle(f"{sigma_rho} = {estimation_model.noise_range} [m]    {f_obs} = {estimation_model.observation_step_size_range} [s]")
        plt.tight_layout()
        # plt.show()


    def plot_observability_metrics(self):

        fig, ax = plt.subplots(5, 1, figsize=(8, 6.5), sharex=True)
        arc_nums = len(self.navigation_results[-1].keys())

        for arc_num in range(arc_nums):

            estimation_model = self.navigation_results[-1][arc_num]
            estimation_output = estimation_model.estimation_output
            estimator = estimation_model.estimator
            state_transition_interface = estimator.state_transition_interface
            observation_times_range = estimation_model.observation_times_range
            weighted_design_matrix = estimation_output.weighted_design_matrix
            normalized_design_matrix = estimation_output.normalized_design_matrix
            residual_history = estimation_output.residual_history
            best_iteration = estimation_output.best_iteration
            apriori_covariance = estimation_model.apriori_covariance

            state_transition_matrix_history = {}
            information_matrix_history = {}
            information_vector_history = {}
            for index, epoch in enumerate(observation_times_range):
                state_transition_matrix = state_transition_interface.full_state_transition_sensitivity_at_epoch(epoch)
                state_transition_matrix_history[epoch] = np.outer(state_transition_matrix, state_transition_matrix)
                information_matrix_history[epoch] = np.outer(weighted_design_matrix[index], weighted_design_matrix[index])
                information_vector_history[epoch] = np.dot(weighted_design_matrix[index].T, residual_history[index, best_iteration])

                # print(np.linalg.cond(information_vector_history[epoch][0:3]))
                # print(np.dot(np.linalg.inv(information_matrix_history[epoch])))
            # information_matrix_history = {}
            # state_transition_matrix_history = {}
            # for index, epoch in enumerate(observation_times_range):
            #     state_transition_matrix = state_transition_interface.full_state_transition_sensitivity_at_epoch(epoch)
            #     state_transition_matrix_history[epoch] = np.dot(state_transition_matrix, state_transition_matrix.T)
            #     information_matrix_history[epoch] = np.outer(weighted_design_matrix[index], weighted_design_matrix[index])

            total_information_matrix_history = {}
            total_information_matrix = np.linalg.inv(apriori_covariance)*0
            for epoch, information_matrix in information_matrix_history.items():
                total_information_matrix += information_matrix
                total_information_matrix_history[epoch] = total_information_matrix.copy()

            dilution_of_precision_history = {}
            for epoch, information_matrix in total_information_matrix_history.items():
                dilution_of_precisions = []
                for i in range(2):
                    for j in range(2):
                        dilution_of_precisions.append(np.sqrt(np.trace(np.linalg.inv(information_matrix[0+3*i+j:3+3*i+j,0+3*i+j:3+3*i+j]))))
                dilution_of_precision_history[epoch] = dilution_of_precisions

            epochs = utils.convert_epochs_to_MJD(np.stack(list(total_information_matrix_history.keys()))) - self.mission_start_epoch
            information_matrix_history = np.stack(list(information_matrix_history.values()))
            information_vector_history = np.stack(list(information_vector_history.values()))
            total_information_matrix_history = np.stack(list(total_information_matrix_history.values()))

            state_transition_matrix_history = np.stack(list(state_transition_matrix_history.values()))
            dilution_of_precision_history = np.stack(list(dilution_of_precision_history.values()))

            # print(information_vector_history[0, :])
            # print(information_vector_history[-1, :])



            for m in range(2):
                observability_lpf = np.sqrt(np.stack([np.diagonal(matrix) for matrix in information_matrix_history[:,0+3*m:3+3*m,0+3*m:3+3*m]]))
                observability_lumio = np.sqrt(np.stack([np.diagonal(matrix) for matrix in information_matrix_history[:,6+3*m:9+3*m,6+3*m:9+3*m]]))
                observability_lpf_total = np.sqrt(np.max(np.linalg.eigvals(information_matrix_history[:,0+3*m:3+3*m,0+3*m:3+3*m]), axis=1, keepdims=True))
                observability_lumio_total = np.sqrt(np.max(np.linalg.eigvals(information_matrix_history[:,6+3*m:9+3*m,6+3*m:9+3*m]), axis=1, keepdims=True))

                ax[m].plot(epochs, observability_lpf_total, label="LPF" if m == 0 and arc_num == 0 else None, color="red")
                ax[m].plot(epochs, observability_lumio_total, label="LUMIO" if m == 0 and arc_num == 0 else None, color="blue")

            ax[2].plot(epochs, np.linalg.cond(total_information_matrix_history[:, 0:3, 0:3]), color="red", label="LPF" if arc_num == 0 else None)
            ax[2].plot(epochs, np.linalg.cond(total_information_matrix_history[:, 6:9, 6:9]), color="blue", label="LUMIO" if arc_num == 0 else None)

            ax[3].plot(epochs, np.linalg.cond(total_information_matrix_history[:, 3:6, 3:6]), color="red", label="LPF" if arc_num == 0 else None)
            ax[3].plot(epochs, np.linalg.cond(total_information_matrix_history[:, 9:12, 9:12]), color="blue", label="LUMIO" if arc_num == 0 else None)
            # ax[2].plot(epochs, np.linalg.cond(total_information_matrix_history[:, :, :]), color="blue", label="LUMIO" if arc_num == 0 else None)

            # ax[3].plot(epochs, np.linalg.cond(state_transition_matrix_history[:, 0:3, 0:3]), color="red", label="LPF" if arc_num == 0 else None)
            # ax[3].plot(epochs, np.linalg.cond(state_transition_matrix_history[:, 6:9, 6:9]), color="blue", label="LUMIO" if arc_num == 0 else None)
            # # ax[3].plot(epochs, np.linalg.cond(state_transition_matrix_history[:, :, :]), color="blue", label="LUMIO" if arc_num == 0 else None)
            # ax[3].set_yscale("log")
            # ax[3].grid(alpha=0.5, linestyle='--')
            # ax[3].set_ylabel(r"$\lambda_{max}/\lambda_{min}$ [-]")
            # ax[3].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")
            # ax[3].legend(loc="upper right")

            ax[4].plot(epochs, dilution_of_precision_history[:, 0], color="red", label="LPF" if arc_num == 0 else None)
            ax[4].plot(epochs, dilution_of_precision_history[:, 2], color="blue", label="LUMIO" if arc_num == 0 else None)
            # ax[3].plot(epochs, np.linalg.cond(state_transition_matrix_history[:, :, :]), color="blue", label="LUMIO" if arc_num == 0 else None)

        ax[0].set_ylabel(r'$\sqrt{\max \operatorname{eig}(\delta \mathbf{\Lambda}_{r, j})}$')
        ax[1].set_ylabel(r'$\sqrt{\max \operatorname{eig}(\delta \mathbf{\Lambda}_{v, j})}$')
        ax[2].set_ylabel(r"$cond\left(\mathbf{\Lambda}_{rr, j}\right)$")
        ax[3].set_ylabel(r"$cond\left(\mathbf{\Lambda}_{vv, j}\right)$")
        ax[4].set_ylabel(r"$GDOP_{j}$ [-]")

        for j in range(len(ax)):
            for i, gap in enumerate(self.observation_windows):
                ax[j].axvspan(
                    xmin=gap[0]-self.mission_start_epoch,
                    xmax=gap[1]-self.mission_start_epoch,
                    color="gray",
                    alpha=0.1,
                    label="Tracking arc" if i == 0 else None)
            for i, epoch in enumerate(self.station_keeping_epochs):
                station_keeping_epoch = epoch - self.mission_start_epoch
                ax[j].axvline(x=station_keeping_epoch, color='black', linestyle='--', alpha=0.7, label="SKM" if i==0 else None)

            ax[j].grid(alpha=0.5, linestyle='--')
            ax[j].set_yscale("log")

        # Plot the history of observation angle with respect to the large covariance axis
        # state_history = self.navigation_results[6][1]
        # epochs = self.navigation_results[9][0]
        # dependent_variables_history = self.navigation_results[9][1]
        # moon_state_history = dependent_variables_history[:,0:6]

        # state_history_moon_lpf = state_history[:, 0:6] - moon_state_history
        # state_history_moon_lumio = state_history[:, 6:12] - moon_state_history

        # ax[4].plot(epochs-self.mission_start_epoch, np.linalg.norm(state_history_moon_lpf[:, 3:6], axis=1), color="green")
        # # ax[4].set_ylabel(r'$||\mathbf{v}_{LPF}||$')

        ax[-1].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]", fontsize="small")
        # ax[0].legend(bbox_to_anchor=(1, 1.04), loc='upper left')
        ax[0].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

        plt.tight_layout()
        # plt.suptitle("Observability metrics")


    def plot_dispersion_to_estimation_error_history(self):

        fig, axs = plt.subplots(1, 1, figsize=(11, 4), sharex=True)

        full_estimation_error_dict = self.navigation_simulator.full_estimation_error_dict
        full_reference_state_deviation_dict = self.navigation_simulator.full_reference_state_deviation_dict

        epochs = np.stack(list(full_estimation_error_dict.keys()))
        full_estimation_error_history = np.stack(list(full_estimation_error_dict.values()))
        full_reference_state_deviation_history = np.stack(list(full_reference_state_deviation_dict.values()))

        # print(len(full_estimation_error_history), len(full_reference_state_deviation_history))

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




class PlotMultipleNavigationResults():

    def __init__(self, navigation_outputs, figure_settings={"save_figure": True, "current_time": float, "file_name": str}):

        self.navigation_outputs = navigation_outputs

        for key, value in figure_settings.items():
            setattr(self, key, value)


    def plot_uncertainty_comparison(self, save_figure=True):

        self.save_figure = save_figure

        fig, axs = plt.subplots(2, 2, figsize=(12.5, 5), sharex=True)
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        line_style_cycle = ["solid", "dashed", "dashdot"]
        ylabels = ["3D RSS OD position \nuncertainty [m]", "3D RSS OD velocity \nuncertainty [m/s]"]
        for type_index, (window_type, navigation_outputs_cases) in enumerate(self.navigation_outputs.items()):

            color = color_cycle[int(type_index%len(color_cycle))]
            for case_index, window_case in enumerate(navigation_outputs_cases):

                line_style = line_style_cycle[int(case_index%len(line_style_cycle))]
                full_propagated_formal_errors_histories = []
                for run_index, (run, navigation_output) in enumerate(window_case.items()):

                    # print(f"Results for {window_type} window_case {case_index} run {run}:")

                    # Extracting the relevant objects
                    navigation_results = navigation_output.navigation_results
                    navigation_simulator = navigation_output.navigation_simulator

                    # Extract the relevant information from the objects
                    full_propagated_formal_errors_epochs = navigation_results[3][0]
                    full_propagated_formal_errors_history = navigation_results[3][1]
                    relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch

                    full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

                    # Plot observation windows
                    if run_index==0:

                        for k in range(2):
                            for j in range(2):

                                for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):

                                    axs[k][j].axvline(x=epoch - navigation_simulator.mission_start_epoch,
                                                        color='black',
                                                        linestyle='--',
                                                        alpha=0.3,
                                                        label="SKM" if k==0 and j==1 and i==0 and type_index==0 else None
                                                        )

                                for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):

                                    axs[k][j].axvspan(
                                        xmin=start_epoch-navigation_simulator.mission_start_epoch,
                                        xmax=end_epoch-navigation_simulator.mission_start_epoch,
                                        color=color,
                                        alpha=0.2,
                                        label=f"Arc" if k==0 and j==1 and window_index==0 and case_index==0 else None
                                        )




                                # Plot the results of the first run
                                # axs[k][j].plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 3*k+6*j:3*k+6*j+3], axis=1),
                                #                 # label=window_type if case_index==0 and run_index==0 else None,
                                #                 color=color,
                                #                 ls=line_style,
                                #                 alpha=0.1
                                #                 )

                mean_full_propagated_formal_errors_histories = np.mean(np.array(full_propagated_formal_errors_histories), axis=0)
                for k in range(2):
                    for j in range(2):
                        axs[k][j].plot(relative_epochs, 3*np.linalg.norm(mean_full_propagated_formal_errors_histories[:, 3*k+6*j:3*k+6*j+3], axis=1),
                            # label=f"{window_type}, case {case_index+1}",
                            label=f"{window_type}",
                            color=color,
                            ls=line_style,
                            alpha=1)

        for k in range(2):
            for j in range(2):
                axs[k][0].set_ylabel(ylabels[k])
                axs[k][j].grid(alpha=0.5, linestyle='--', zorder=0)
                axs[k][j].set_yscale("log")
                axs[k][0].set_title("LPF")
                axs[k][1].set_title("LUMIO")
                axs[-1][j].set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")

        axs[0][1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
        # fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(self.navigation_outputs.keys()), fontsize='small')
        fig.suptitle(f"Total 3D RSS 3$\sigma$ uncertainty")
        plt.tight_layout()

        if self.save_figure:
            utils.save_figure_to_folder(figs=[fig], labels=[f"{self.current_time}_uncertianty_comparison"], custom_sub_folder_name=self.file_name)


    def plot_maneuvre_costs(self, save_figure=True):

        self.save_figure = save_figure

        fig, axs = plt.subplots(figsize=(12, 4), sharex=True)
        axs_twin = axs.twinx()
        ylabels = ["3D RSS OD \n position uncertainty [m]", "3D RSS OD \n velocity uncertainty [m/s]"]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        line_style_cycle = ["solid", "dashed", "dashdot"]
        for type_index, (window_type, navigation_outputs_cases) in enumerate(self.navigation_outputs.items()):

            color = color_cycle[int(type_index%len(color_cycle))]

            for case_index, window_case in enumerate(navigation_outputs_cases):

                line_style = line_style_cycle[int(case_index%len(line_style_cycle))]

                full_propagated_formal_errors_histories = []
                delta_v_runs_dict = {}
                for run_index, (run, navigation_output) in enumerate(window_case.items()):

                    # print(f"Results for {window_type} window_case {case_index} run {run}:")

                    # Extracting the relevant objects
                    navigation_results = navigation_output.navigation_results
                    navigation_simulator = navigation_output.navigation_simulator

                    # Extracting the relevant results from objects
                    for window_index, (start_epoch, end_epoch) in enumerate(navigation_simulator.observation_windows):
                        if end_epoch in navigation_simulator.delta_v_dict.keys():

                            delta_v = np.linalg.norm(navigation_simulator.delta_v_dict[end_epoch])

                            if end_epoch in delta_v_runs_dict:
                                delta_v_runs_dict[end_epoch].append(delta_v)
                            else:
                                delta_v_runs_dict[end_epoch] = [delta_v]

                        if run_index==0:

                            axs.axvspan(
                                xmin=start_epoch-navigation_simulator.mission_start_epoch,
                                xmax=end_epoch-navigation_simulator.mission_start_epoch,
                                color=color,
                                alpha=0.2,
                                # label=f"Observation window" if window_index==0 and case_index==0 else None
                                )

                    full_propagated_formal_errors_epochs = navigation_results[3][0]
                    full_propagated_formal_errors_history = navigation_results[3][1]
                    relative_epochs = full_propagated_formal_errors_epochs - navigation_simulator.mission_start_epoch
                    full_propagated_formal_errors_histories.append(full_propagated_formal_errors_history)

                    full_estimation_error_epochs = navigation_results[0][0]
                    full_estimation_error_history = navigation_results[0][1]

                    if run_index == 0:

                        for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):
                            station_keeping_epoch = epoch - navigation_simulator.mission_start_epoch

                            axs.axvline(x=station_keeping_epoch,
                                                color='black',
                                                linestyle='--',
                                                alpha=0.3,
                                                label="SKM" if i==0 and type_index==0 else None)

                        axs_twin.plot(relative_epochs, 3*np.linalg.norm(full_propagated_formal_errors_history[:, 6:9], axis=1),
                                        color=color,
                                        ls=line_style,
                                        alpha=0.7)

                    # axs_twin.plot(relative_epochs, np.linalg.norm(full_estimation_error_history[:, 6:9], axis=1),
                    #                 color=color,
                    #                 ls='--',
                    #                 alpha=0.2)

                # Plot the station keeping costs standard deviations
                for delta_v_runs_dict_index, (end_epoch, delta_v_runs) in enumerate(delta_v_runs_dict.items()):
                    mean_delta_v = np.mean(delta_v_runs)
                    std_delta_v = np.std(delta_v_runs)
                    axs.bar(end_epoch-navigation_simulator.mission_start_epoch, mean_delta_v,
                            color=color,
                            width=0.2,
                            yerr=std_delta_v,
                            capsize=4,
                            label=f"{window_type}" if case_index==0 and delta_v_runs_dict_index==0 else None)

        axs.set_xlabel(f"Time since MJD {navigation_simulator.mission_start_epoch} [days]")
        axs.set_ylabel(r"$||\Delta V||$ [m/s]")
        axs.grid(alpha=0.5, linestyle='--', zorder=0)
        axs.set_title("Station keeping costs")
        axs_twin.set_ylabel(ylabels[0])
        axs.set_yscale("log")
        axs_twin.set_yscale("log")
        axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(self.navigation_outputs.keys())+1, fontsize="small")
        plt.tight_layout()

        if self.save_figure:
            utils.save_figure_to_folder(figs=[fig], labels=[f"{self.current_time}_maneuvre_costs"], custom_sub_folder_name=self.file_name)


    def plot_monte_carlo_estimation_error_history(self, save_figure=True):

        self.save_figure = save_figure

        rows = len(self.navigation_outputs.keys())
        fig, axs = plt.subplots(rows, 4, figsize=(8, 2.5*rows), sharex=True)
        if len(self.navigation_outputs.keys())==1:
            axs = np.array([axs])
        label_index = 0
        detailed_results = [["Perilune", "Apolune", "Random"], [0], [0]]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        line_style_cycle = ["solid", "dashed", "dashdot"]
        colors = ["red", "green", "blue"]
        symbols = [[r"x", r"y", r"z"], [r"x", r"y", r"z"]]
        units = ["[m]", "[m]", "[m/s]", "[m/s]"]
        titles = [r"$\mathbf{r}-\hat{\mathbf{r}}$ LPF", r"$\mathbf{r}-\hat{\mathbf{r}}$ LUMIO", r"$\mathbf{v}-\hat{\mathbf{v}}$ LPF", r"$\mathbf{v}-\hat{\mathbf{v}}$ LUMIO"]
        for type_index, (window_type, navigation_outputs_cases) in enumerate(self.navigation_outputs.items()):

            color = color_cycle[int(type_index%len(color_cycle))]
            for case_index, window_case in enumerate(navigation_outputs_cases):

                line_style = line_style_cycle[int(case_index%len(line_style_cycle))]
                full_estimation_error_histories = []
                for run_index, (run, navigation_output) in enumerate(window_case.items()):

                    # Extracting the relevant objects
                    navigation_results = navigation_output.navigation_results
                    navigation_simulator = navigation_output.navigation_simulator

                    # Extract relevant data from the objects
                    full_estimation_error_epochs = navigation_results[0][0]
                    full_estimation_error_history = navigation_results[0][1]
                    full_propagated_formal_errors_history = navigation_results[3][1]
                    relative_epochs = full_estimation_error_epochs - navigation_simulator.mission_start_epoch
                    full_estimation_error_histories.append(full_estimation_error_history)

                    for n in range(axs.shape[1]):

                        if run_index==0:
                            for i, gap in enumerate(navigation_simulator.observation_windows):
                                axs[type_index][n].axvspan(
                                    xmin=gap[0]-navigation_simulator.mission_start_epoch,
                                    xmax=gap[1]-navigation_simulator.mission_start_epoch,
                                    color="gray",
                                    alpha=0.1,
                                    label="Tracking arc" if i == 0 and type_index==0 else None)

                            for i, epoch in enumerate(navigation_simulator.station_keeping_epochs):
                                station_keeping_epoch = epoch - navigation_simulator.mission_start_epoch
                                axs[type_index][n].axvline(x=station_keeping_epoch,
                                                color='black',
                                                linestyle='--',
                                                alpha=0.2,
                                                label="SKM" if i == 0 and type_index==0 else None)

                            # axs[type_index][n].set_yscale("log")
                            axs[type_index][0].set_ylim(-100, 100)
                            # axs[type_index][1].set_ylim(-100, 100)
                            axs[type_index][2].set_ylim(-0.03, 0.03)
                            # axs[type_index][3].set_ylim(-0.03, 0.03)

                            axs[-1][n].set_xlabel(f"Time since\nMJD {navigation_simulator.mission_start_epoch} [days]", fontsize="small")

                            axs[type_index][0].set_ylabel(window_type, fontsize="small")
                            axs[type_index][n].grid(alpha=0.5, linestyle='--')
                            axs[type_index][n].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                            axs[type_index][n].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                    n = 0
                    for k in range(2):
                        for j in range(2):
                            for i in range(3):

                                sigma = 3*full_propagated_formal_errors_history[:, 3*k+6*j+i]
                                axs[type_index][n].plot(relative_epochs, full_estimation_error_history[:,3*k+6*j+i],
                                                        color=colors[i],
                                                        alpha=0.1,
                                                        # label=f"${symbols[k][i]}-\hat{{{symbols[k][i]}}}$" \
                                                        #     # if label_index in range(6) else None
                                                        #     if n==3 and run_index==0 else None
                                                            )

                                if run_index==0:
                                    axs[type_index][n].plot(relative_epochs, -sigma,
                                                            color=colors[i],
                                                            ls="--",
                                                            alpha=0.3,
                                                            label=f"$3\sigma_{{{symbols[k][i]}}}$" \
                                                                # if label_index in range(6) else None
                                                                if n==3 and run_index==0 else None
                                                                )

                                    axs[type_index][n].plot(relative_epochs, sigma,
                                                            color=colors[i],
                                                            ls="--",
                                                            alpha=0.3)

                            n += 1

                mean_full_estimation_error_histories = np.mean(np.array(full_estimation_error_histories), axis=0)
                # print("Mean: \n", mean_full_estimation_error_histories[-1, :])

                n=0
                for k in range(2):
                    for j in range(2):
                        for i in range(3):
                            axs[type_index][n].plot(relative_epochs, mean_full_estimation_error_histories[:, 3*k+6*j+i],
                                label=f"${symbols[k][i]}-\hat{{{symbols[k][i]}}}$" \
                                    # if label_index in range(6) else None
                                    if n==3 else None,
                                color=colors[i],
                                alpha=1)

                        rss_values = np.sqrt(np.sum(np.square(mean_full_estimation_error_histories[-1, 3*k+6*j:3*k+6*j+3])))
                        if type_index == 0:
                            axs[type_index][n].set_title(titles[n]+f"\nMean RSS: {np.round(rss_values, 3)} "+units[n], fontsize="small")
                        else:
                            axs[type_index][n].set_title(f"Mean RSS: {np.round(rss_values, 3)} "+units[n], fontsize="small")
                        n += 1


        axs[0][-1].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

        # fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        # fig.suptitle(f"Estimaton error history")
        # fig.suptitle(r"Estimation error history \nRange-only, $1\sigma_{\rho}$ = ", navigation_simulator.noise_range, "[$m$], $t_{obs}$ = ", navigation_simulator.observation_step_size_range, "[$s$]")
        plt.tight_layout()

        if self.save_figure:
            utils.save_figure_to_folder(figs=[fig], labels=[f"{self.current_time}_estimation_error_history"], custom_sub_folder_name=self.file_name)


    def plot_maneuvre_costs_bar_chart(self, save_figure=True, evaluation_threshold=14, title="", group_stretch=0.8, bar_stretch=0.95,
             legend=True, x_labels=True, label_fontsize=8,
             colors=None, barlabel_offset=1,
             bar_labeler=lambda k, i, s: str(round(s, 3))):

        self.save_figure = save_figure

        fig, ax = plt.subplots(figsize=(10, 4))

        for threshold_index, evaluation_threshold in enumerate([0, evaluation_threshold]):

            data = {}
            for window_type in self.navigation_outputs.keys():

                objective_value_results_per_window_case = []
                for window_case, navigation_output_list in enumerate(self.navigation_outputs[window_type]):

                    objective_values = []
                    delta_v_per_skm_list = []
                    for run, navigation_output in navigation_output_list.items():

                        # print(f"Results for {window_type} window_case {window_case} run {run}:")

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

                        # print("Objective: ", delta_v_per_skm, delta_v)

                    objective_value_results_per_window_case.append((len(objective_values),
                                                                min(objective_values),
                                                                max(objective_values),
                                                                np.mean(objective_values),
                                                                np.std(objective_values),
                                                                objective_values,
                                                                delta_v_per_skm_list))

                data[window_type] = objective_value_results_per_window_case

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
            ax.legend(loc='upper right', fontsize="small")

        if x_labels:
            ax.set_xticklabels(sorted_k)
        else:
            ax.set_xticklabels()

        plt.tight_layout()

        if self.save_figure:
            utils.save_figure_to_folder(figs=[fig], labels=[f"{self.current_time}_maneuvre_costs_bar_chart"], custom_sub_folder_name=self.file_name)

