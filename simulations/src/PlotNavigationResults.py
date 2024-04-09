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
        fig1_3d2 = plt.figure()
        ax_3d2 = fig1_3d2.add_subplot(111, projection='3d')
        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, results in enumerate(models):

                    state_history_reference = results[4][1]
                    state_history_truth = results[5][1]
                    state_history_initial = results[6][1]
                    epochs = results[9][0]
                    dependent_variables_history = results[9][1]
                    navigation_simulator = results[-1]

                    # Storing some plots
                    ax_3d2.plot(state_history_reference[:,0], state_history_reference[:,1], state_history_reference[:,2], label="LPF ref", color="green")
                    ax_3d2.plot(state_history_reference[:,6], state_history_reference[:,7], state_history_reference[:,8], label="LUMIO ref", color="green")
                    ax_3d2.plot(state_history_initial[:,0], state_history_initial[:,1], state_history_initial[:,2], label="LPF estimated")
                    ax_3d2.plot(state_history_initial[:,6], state_history_initial[:,7], state_history_initial[:,8], label="LUMIO estimated")
                    ax_3d2.plot(state_history_truth[:,0], state_history_truth[:,1], state_history_truth[:,2], label="LPF truth", color="black", ls="--")
                    ax_3d2.plot(state_history_truth[:,6], state_history_truth[:,7], state_history_truth[:,8], label="LUMIO truth", color="black", ls="--")
                    ax_3d2.set_xlabel('X [m]')
                    ax_3d2.set_ylabel('Y [m]')
                    ax_3d2.set_zlabel('Z [m]')


                    moon_data_dict = {epoch: state for epoch, state in zip(epochs, dependent_variables_history[:, :6])}
                    satellite_data_dict = {epoch: state for epoch, state in zip(epochs, state_history_initial[:, :])}

                    mu = 7.34767309e22/(7.34767309e22 + 5.972e24)

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

                        transformation_matrix_dict.update({epoch: transformation_matrix})


                    synodic_satellite_states_dict = {}
                    for epoch, state in satellite_data_dict.items():

                        transformation_matrix = transformation_matrix_dict[epoch]
                        synodic_state = np.dot(transformation_matrix, state[:3])
                        synodic_state = synodic_state/np.linalg.norm((moon_data_dict[epoch][:3]))
                        synodic_state1 = (1-mu)*synodic_state
                        synodic_state = np.dot(transformation_matrix, state[6:9])
                        synodic_state = synodic_state/np.linalg.norm((moon_data_dict[epoch][:3]))
                        synodic_state2 = (1-mu)*synodic_state

                        synodic_satellite_states_dict.update({epoch: np.concatenate((synodic_state1, synodic_state2))})


                    synodic_moon_states_dict = {}
                    for epoch, state in moon_data_dict.items():

                        transformation_matrix = transformation_matrix_dict[epoch]
                        synodic_state = np.dot(transformation_matrix, state[:3])
                        synodic_state = synodic_state/np.linalg.norm(state[:3])
                        synodic_state = (1-mu)*synodic_state
                        synodic_moon_states_dict.update({epoch: synodic_state})

                    synodic_states = np.stack(list(synodic_satellite_states_dict.values()))
                    synodic_moon_states = np.stack(list(synodic_moon_states_dict.values()))

                    fig, ax = plt.subplots(2, 3, figsize=(13, 5))
                    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    for i in range(2):
                        if i == 0:
                            # color = color_cycle[1]
                            color="gray"
                        else:
                            # color = color_cycle[0]
                            color="darkgray"

                        ax[i][0].scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 2], s=50, color="gray")
                        ax[i][1].scatter(synodic_moon_states[:, 1], synodic_moon_states[:, 2], s=50, color="gray")
                        ax[i][2].scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 1], s=50, color="gray", label="Moon" if i==0 else None)
                        ax[i][0].plot(synodic_states[:, 3*i+0], synodic_states[:, 3*i+2], lw=0.5, color=color)
                        ax[i][1].plot(synodic_states[:, 3*i+1], synodic_states[:, 3*i+2], lw=0.5, color=color)
                        ax[i][2].plot(synodic_states[:, 3*i+0], synodic_states[:, 3*i+1], lw=0.5, color=color, label="LPF" if i==0 else None)
                        ax[1][0].plot(synodic_states[:, 3*i+0], synodic_states[:, 3*i+2], lw=0.1, color=color)
                        ax[1][1].plot(synodic_states[:, 3*i+1], synodic_states[:, 3*i+2], lw=0.1, color=color)
                        ax[1][2].plot(synodic_states[:, 3*i+0], synodic_states[:, 3*i+1], lw=0.1, color=color, label="LUMIO" if i==1 else None)

                    ax_3d.scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 1], synodic_moon_states[:, 2], s=50, color="gray", label="Moon")
                    ax_3d.plot(synodic_states[:, 0], synodic_states[:, 1], synodic_states[:, 2], lw=0.2, color="gray")
                    ax_3d.plot(synodic_states[:, 3], synodic_states[:, 4], synodic_states[:, 5], lw=0.7, color="darkgray")

                    for num, (start, end) in enumerate(navigation_simulator.observation_windows):
                        synodic_states_window_dict = {key: value for key, value in synodic_satellite_states_dict.items() if key >= start and key <= end}
                        synodic_states_window = np.stack(list(synodic_states_window_dict.values()))
                        linewidth = 2

                        for i in range(2):
                            ax[i][0].plot(synodic_states_window[:, 3*i+0], synodic_states_window[:, 3*i+2], linewidth=linewidth, color=color_cycle[num])
                            ax[i][1].plot(synodic_states_window[:, 3*i+1], synodic_states_window[:, 3*i+2], linewidth=linewidth, color=color_cycle[num])
                            ax[i][2].plot(synodic_states_window[:, 3*i+0], synodic_states_window[:, 3*i+1], linewidth=linewidth, color=color_cycle[num], label=f"Arc {num+1}" if i==0 else None)
                            ax_3d.plot(synodic_states_window[:, 3*i+0], synodic_states_window[:, 3*i+1], synodic_states_window[:, 3*i+2], linewidth=linewidth, color=color_cycle[num], label=f"Arc {num+1}" if i==0 else None)

                        for i in range(len(synodic_states_window[:, 0])):
                            ax_3d.plot([synodic_states_window[i, 0], synodic_states_window[i, 3]],
                                       [synodic_states_window[i, 1], synodic_states_window[i, 4]],
                                       [synodic_states_window[i, 2], synodic_states_window[i, 5]], color=color_cycle[num], lw=0.5, alpha=0.2)

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
                    # ax_3d.set_xlim([0, 1.4])

                    ax[0][2].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
                    ax_3d.legend()
                    fig.suptitle(f"Observation windows for {epochs[-1]-epochs[0]} days, synodic frame")

        plt.tight_layout()
        plt.legend()


        # # Plot the trajectory over time
        # fig1_3d = plt.figure()
        # ax_3d = fig1_3d.add_subplot(111, projection='3d')
        # for i, (model_type, model_names) in enumerate(self.results_dict.items()):
        #     for j, (model_name, models) in enumerate(model_names.items()):
        #         for k, results in enumerate(models):

        #             state_history_reference = results[4][1]
        #             state_history_truth = results[5][1]
        #             state_history_initial = results[6][1]
        #             epochs = results[9][0]
        #             dependent_variables_history = results[9][1]
        #             navigation_simulator = results[-1]

        #             # # Storing some plots
        #             # ax_3d.plot(state_history_reference[:,0], state_history_reference[:,1], state_history_reference[:,2], label="LPF ref", color="green")
        #             # ax_3d.plot(state_history_reference[:,6], state_history_reference[:,7], state_history_reference[:,8], label="LUMIO ref", color="green")
        #             # ax_3d.plot(state_history_initial[:,0], state_history_initial[:,1], state_history_initial[:,2], label="LPF estimated")
        #             # ax_3d.plot(state_history_initial[:,6], state_history_initial[:,7], state_history_initial[:,8], label="LUMIO estimated")
        #             # # ax_3d.plot(state_history_final[:,0], state_history_final[:,1], state_history_final[:,2], label="LPF estimated")
        #             # # ax_3d.plot(state_history_final[:,6], state_history_final[:,7], state_history_final[:,8], label="LUMIO estimated")
        #             # ax_3d.plot(state_history_truth[:,0], state_history_truth[:,1], state_history_truth[:,2], label="LPF truth", color="black", ls="--")
        #             # ax_3d.plot(state_history_truth[:,6], state_history_truth[:,7], state_history_truth[:,8], label="LUMIO truth", color="black", ls="--")
        #             # ax_3d.set_xlabel('X [m]')
        #             # ax_3d.set_ylabel('Y [m]')
        #             # ax_3d.set_zlabel('Z [m]')


        #             moon_data_dict = {epoch: state for epoch, state in zip(epochs, dependent_variables_history[:, :6])}
        #             satellite_data_dict = {epoch: state for epoch, state in zip(epochs, state_history_initial[:, :])}

        #             mu = 7.34767309e22/(7.34767309e22 + 5.972e24)

        #             # Create the Direct Cosine Matrix based on rotation axis of Moon around Earth
        #             transformation_matrix_dict = {}
        #             for epoch, moon_state in moon_data_dict.items():

        #                 moon_position, moon_velocity = moon_state[:3], moon_state[3:]

        #                 # Define the complementary axes of the rotating frame
        #                 rotation_axis = np.cross(moon_position, moon_velocity)
        #                 second_axis = np.cross(moon_position, rotation_axis)

        #                 # Define the rotation matrix (DCM) using the rotating frame axes
        #                 first_axis = moon_position/np.linalg.norm(moon_position)
        #                 second_axis = second_axis/np.linalg.norm(second_axis)
        #                 third_axis = rotation_axis/np.linalg.norm(rotation_axis)
        #                 transformation_matrix = np.array([first_axis, second_axis, third_axis])

        #                 transformation_matrix_dict.update({epoch: transformation_matrix})


        #             synodic_satellite_states_dict = {}
        #             for satellite in ["LPF", "LUMIO"]:
        #                 for epoch, state in satellite_data_dict.items():

        #                     transformation_matrix = transformation_matrix_dict[epoch]
        #                     if satellite == "LPF":
        #                         state = state[0:3]
        #                     else:
        #                         state = state[6:9]
        #                     synodic_state = np.dot(transformation_matrix, state)
        #                     synodic_state = synodic_state/np.linalg.norm((moon_data_dict[epoch][:3]))
        #                     synodic_state = (1-mu)*synodic_state



        #                     synodic_satellite_states_dict.update({epoch: synodic_state})

        #             synodic_moon_states_dict = {}
        #             for epoch, state in moon_data_dict.items():

        #                 transformation_matrix = transformation_matrix_dict[epoch]
        #                 synodic_state = np.dot(transformation_matrix, state[:3])
        #                 synodic_state = synodic_state/np.linalg.norm(state[:3])
        #                 synodic_state = (1-mu)*synodic_state

        #                 synodic_moon_states_dict.update({epoch: synodic_state})

        #             synodic_states = np.stack(list(synodic_satellite_states_dict.values()))
        #             synodic_moon_states = np.stack(list(synodic_moon_states_dict.values()))
        #             print(np.shape(synodic_states))


        #             fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        #             fig1_3d = plt.figure()
        #             ax_3d = fig1_3d.add_subplot(111, projection='3d')
        #             ax[0].scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 2], s=50, color="gray")
        #             ax[1].scatter(synodic_moon_states[:, 1], synodic_moon_states[:, 2], s=50, color="gray")
        #             ax[2].scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 1], s=50, color="gray", label="Moon")
        #             ax[0].plot(synodic_states[:, 0], synodic_states[:, 2], lw=1)
        #             ax[1].plot(synodic_states[:, 1], synodic_states[:, 2], lw=1)
        #             ax[2].plot(synodic_states[:, 0], synodic_states[:, 1], lw=1)
        #             ax_3d.plot(synodic_states[:, 0], synodic_states[:, 1], synodic_states[:, 2])
        #             ax_3d.scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 1], synodic_moon_states[:, 2], s=50, color="gray", label="Moon")

        #             for num, (start, end) in enumerate(navigation_simulator.observation_windows):
        #                 print(start, end)
        #                 synodic_states_window_dict = {key: value for key, value in synodic_satellite_states_dict.items() if key >= start and key <= end}
        #                 synodic_states_window = np.stack(list(synodic_states_window_dict.values()))
        #                 print(np.shape(synodic_states_window))
        #                 linewidth = 3
        #                 color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        #                 ax[0].plot(synodic_states_window[:, 0], synodic_states_window[:, 2], linewidth=linewidth, color=color_cycle[num])
        #                 ax[1].plot(synodic_states_window[:, 1], synodic_states_window[:, 2], linewidth=linewidth, color=color_cycle[num])
        #                 ax[2].plot(synodic_states_window[:, 0], synodic_states_window[:, 1], linewidth=linewidth, color=color_cycle[num], label=f"Arc {num+1}")
        #                 ax_3d.plot(synodic_states_window[:, 0], synodic_states_window[:, 1], synodic_states_window[:, 2], linewidth=linewidth, color=color_cycle[num], label=f"Arc {num+1}")

        #             axes_labels = ['X [-]', 'Y [-]', 'Z [-]']
        #             for i in range(3):
        #                 ax[i].grid(alpha=0.3)
        #             ax[0].set_xlabel(axes_labels[0])
        #             ax[0].set_ylabel(axes_labels[2])
        #             ax[1].set_xlabel(axes_labels[1])
        #             ax[1].set_ylabel(axes_labels[2])
        #             ax[2].set_xlabel(axes_labels[0])
        #             ax[2].set_ylabel(axes_labels[1])

        #             ax[2].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
        #             ax_3d.legend()
        #             plt.tight_layout()

        #             plt.show()






        # plt.tight_layout()
        # plt.legend()


    def plot_formal_error_history(self):

        # Plot how the formal errors grow over time
        fig1, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, results in enumerate(models):

                    full_propagated_formal_errors_epochs = results[3][0]
                    full_propagated_formal_errors_history = results[3][1]
                    relative_epochs = full_propagated_formal_errors_epochs - full_propagated_formal_errors_epochs[0]
                    navigation_simulator = results[-1]

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

        fig1.suptitle(f"Formal error history \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
        plt.tight_layout()


    def plot_uncertainty_history(self):

        fig2, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)

        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, results in enumerate(models):

                    full_propagated_formal_errors_epochs = results[3][0]
                    full_propagated_formal_errors_history = results[3][1]
                    propagated_covariance_epochs = results[2][0]
                    navigation_simulator = results[-1]

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

        fig2.suptitle(f"Total 3D RSS 3$\sigma$ uncertainty \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
        plt.tight_layout()


    def plot_reference_deviation_history(self):

        # Plot how the deviation from the reference orbit
        fig3, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                for k, results in enumerate(models):

                    full_reference_state_deviation_epochs = results[1][0]
                    full_reference_state_deviation_history = results[1][1]
                    navigation_simulator = results[-1]

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

        fig3.suptitle(f"Deviation from reference orbit \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
        plt.legend()


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
                        navigation_simulator = results[-1]

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
                # ax[0][0].set_ylim(-1000, 1000)
                # ax[1][0].set_ylim(-1, 1)
                ax[k][0].set_title("LPF")
                ax[k][1].set_title("LUMIO")

                # Set y-axis tick label format to scientific notation with one decimal place
                ax[k][j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[k][j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax[-1][j].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")

        fig4.suptitle(f"Estimaton error history | range-only \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
        # fig4.suptitle("Estimation error history: range-only, $1\sigma_{\rho}$ = 102.44 [$m$], $f_{obs}$ = $1/600$ [$s^{-1}$]")
        plt.tight_layout()



    def plot_correlation_history(self):

        # Plot the estimation error history
        # fig4, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                    for k, results in enumerate(models):

                        break
                        arc_nums = len(results[-2].keys())
                        arc_nums = 1

                        fig, ax = plt.subplots(1, arc_nums, figsize=(9, 4), sharey=True)

                        for arc_num in range(arc_nums):

                            estimation_output = results[-2][arc_num]
                            navigation_simulator = results[-1]

                            covariance_output = estimation_output.covariance

                            correlations = estimation_output.correlations
                            estimated_param_names = [r"$x_{1}$", r"$y_{1}$", r"$z_{1}$", r"$\dot{x}_{1}$", r"$\dot{y}_{1}$", r"$\dot{z}_{1}$",
                                                    r"$x_{2}$", r"$y_{2}$", r"$z_{2}$", r"$\dot{x}_{2}$", r"$\dot{y}_{2}$", r"$\dot{z}_{2}$"]

                            im = ax[arc_num].imshow(correlations, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)

                            ax[arc_num].set_xticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)
                            ax[arc_num].set_yticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)

                            # add numbers to each of the boxes
                            for i in range(len(estimated_param_names)):
                                for j in range(len(estimated_param_names)):
                                    text = ax[arc_num].text(
                                        j, i, round(correlations[i, j], 2), ha="center", va="center", color="black"
                                    )

                            ax[arc_num].set_xlabel("Estimated Parameter")
                            ax[0].set_ylabel("Estimated Parameter")

                        cb = plt.colorbar(im)

                        fig.suptitle(f"Correlations for estimated parameters for LPF and LUMIO")
                        fig.tight_layout()
                        plt.show()



    def plot_observations(self):

        fig, ax = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                    for k, results in enumerate(models):

                        arc_nums = len(results[-2].keys())

                        # For each arc, plot the observations and its residuals
                        for arc_num in range(arc_nums):

                            estimation_model = results[-2][arc_num]
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
                        navigation_simulator = results[-1]
                        state_history = results[6][1]
                        epochs = results[9][0]
                        dependent_variables_history = results[9][1]
                        relative_state_history = dependent_variables_history[:,6:12]
                        full_propagated_covariance_epochs = results[2][0]
                        full_propagated_covariance_history = results[2][1]

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

                        # print(angles_dict)

                        # Generate boolans for when treshold condition holds to generate estimation window
                        angle_to_range_dict = dict()
                        for i, state in enumerate(state_history):

                            # # range_direction = np.dotrelative_state_history[0:3]
                            # a = relative_state_history[i, 0:3]
                            # x = np.array([state[6]-state[0], 0, 0])
                            # y = np.array([0, state[7]-state[1], 0])
                            # z = np.array([0, 0, state[8]-state[2]])
                            # axes = [x, y, z]

                            # angle_to_range = []
                            # for axis in axes:
                            #     cosine_angle = np.dot(a, axis)/(np.linalg.norm(a)*np.linalg.norm(axis))
                            #     angle = np.arccos(cosine_angle)
                            #     angle_to_range.append(np.degrees(angle))

                            # angle_to_range_dict.update({epochs[i]: np.array(angle_to_range)})

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

                        ax[2].plot(np.stack(list(angle_to_range_dict.keys()))-self.mission_start_epoch, np.stack(list(angle_to_range_dict.values())), label="angles in degrees")

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
                                ax[j].axvline(x=station_keeping_epoch, color='black', linestyle='--', label="SKM" if i==0 else None)

                            ax[j].grid(alpha=0.5, linestyle='--')

                            # Set y-axis tick label format to scientific notation with one decimal place
                            ax[j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                            ax[j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                        ax[0].set_ylabel("Range [m]")
                        ax[1].set_ylabel("Observation Residual [m]")
                        ax[-1].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")
                        ax[0].legend(bbox_to_anchor=(1, 1.04), loc='upper left')

                        fig.suptitle(f"Intersatellite range observations \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
                        plt.tight_layout()
                        # plt.show()


                        # fig, ax = plt.subplots(1, arc_nums, figsize=(9, 4), sharey=True)

                        # for arc_num in range(arc_nums):

                        #     estimation_output = results[-2][arc_num]
                        #     # total_single_information_dict = results[-2][1]
                        #     # total_covariance_dict = results[-2][2]
                        #     # total_information_dict = results[-2][3]
                        #     sorted_observation_sets = results[-2][arc_num][4]


                        #     print(estimation_output)

                        #     for i, (observable_type, information_sets) in enumerate(sorted_observation_sets.items()):
                        #         for j, observation_set in enumerate(information_sets.values()):
                        #             for k, single_observation_set in enumerate(observation_set):

                        #                 residual_history = estimation_output.residual_history

                        #                 fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
                        #                 subplots_list = [ax1, ax2, ax3, ax4]

                        #                 index = int(len(single_observation_set.observation_times))
                        #                 for l in range(4):
                        #                     subplots_list[l].scatter(single_observation_set.observation_times, residual_history[i*index:(i+1)*index, l])
                        #                     subplots_list[l].set_ylabel("Observation Residual")
                        #                     subplots_list[l].set_title("Iteration "+str(l+1))

                        #                 ax3.set_xlabel("Time since J2000 [s]")
                        #                 ax4.set_xlabel("Time since J2000 [s]")

                        #                 plt.figure(figsize=(9,5))
                        #                 plt.hist(residual_history[i*index:(i+1)*index, -1], 25)
                        #                 plt.xlabel('Final iteration range residual')
                        #                 plt.ylabel('Occurences [-]')
                        #                 plt.title('Histogram of residuals on final iteration')

                        #                 plt.tight_layout()
                        #                 plt.show()




    def plot_observability(self):

        fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        for i, (model_type, model_names) in enumerate(self.results_dict.items()):
            for j, (model_name, models) in enumerate(model_names.items()):
                    for k, results in enumerate(models):

                        arc_nums = len(results[-2].keys())

                        for arc_num in range(arc_nums):

                            estimation_model = results[-2][arc_num]
                            estimation_output = estimation_model.estimation_output
                            navigation_simulator = results[-1]

                            # Generate information and covariance histories based on all the combinations of observables and link definitions
                            total_information_dict = dict()
                            total_covariance_dict = dict()
                            total_single_information_dict = dict()
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
                                            weighted_design_matrix_product = np.dot(weighted_design_matrix.T, weighted_design_matrix)

                                            # Calculate the information matrix
                                            current_information = total_information + weighted_design_matrix_product
                                            single_information_dict[epoch] = weighted_design_matrix_product
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

                                        for m in range(2):
                                            observability_lpf = np.sqrt(np.stack([np.diagonal(matrix) for matrix in information_matrix_history[:,0+3*m:3+3*m,0+3*m:3+3*m]]))
                                            observability_lumio = np.sqrt(np.stack([np.diagonal(matrix) for matrix in information_matrix_history[:,6+3*m:9+3*m,6+3*m:9+3*m]]))
                                            observability_lpf_total = np.sqrt(np.max(np.linalg.eigvals(information_matrix_history[:,0+3*m:3+3*m,0+3*m:3+3*m]), axis=1, keepdims=True))
                                            observability_lumio_total = np.sqrt(np.max(np.linalg.eigvals(information_matrix_history[:,6+3*m:9+3*m,6+3*m:9+3*m]), axis=1, keepdims=True))

                                            ax[m].plot(epochs, observability_lpf_total, label="LPF" if m == 0 and arc_num == 0 else None, color="darkred")
                                            ax[m].plot(epochs, observability_lumio_total, label="LUMIO" if m == 0 and arc_num == 0 else None, color="darkblue")

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
                                ax[j].axvline(x=station_keeping_epoch, color='black', linestyle='--', label="SKM" if i==0 else None)

                            ax[j].grid(alpha=0.5, linestyle='--')
                            ax[j].set_yscale("log")

                            # Set y-axis tick label format to scientific notation with one decimal place
                            ax[j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                            ax[j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                        ax[0].set_ylabel(r'$\sqrt{\max \operatorname{eig}(\delta \mathbf{\Lambda}_{\mathbf{r}, j})}$ [m]')
                        ax[1].set_ylabel(r'$\sqrt{\max \operatorname{eig}(\delta \mathbf{\Lambda}_{\mathbf{v}, j})}$ [m]')
                        ax[-1].set_xlabel(f"Time since MJD {self.mission_start_epoch} [days]")
                        ax[0].legend(bbox_to_anchor=(1, 1.04), loc='upper left')

                        fig.suptitle(f"Intersatellite range observability \n Model: on-board: {navigation_simulator.model_name}{navigation_simulator.model_number}, truth: {navigation_simulator.model_name_truth}{navigation_simulator.model_number_truth}")
                        plt.tight_layout()
                        plt.show()
