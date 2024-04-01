# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Third party
import pytest
import pytest_html
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Own
from tests import utils
import reference_data, Interpolator, FrameConverter
from src.dynamic_models.LF.CRTBP import *
from src.dynamic_models.HF.PM import *
from src.dynamic_models.HF.PMSRP import *
from src.dynamic_models.HF.SH import *
from src.dynamic_models.HF.SHSRP import *
from src.estimation_models import estimation_model



def comparison_high_and_LF(simulation_start_epoch_MJD, propagation_time, durations, step_size=0.001):

    custom_model_dict={"LF": ["CRTBP"], "HF": ["PM", "PMSRP", "SH", "SHSRP"]}
    dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time, custom_model_dict=custom_model_dict)

    # Pick only the first of teach model name
    dynamic_models = utils.get_first_of_model_types(dynamic_model_objects)

    # Extract simulation histories of classic CRTBP halo continuation model
    synodic_state_history_erdem = reference_data.get_synodic_state_history_erdem()[:int(propagation_time/0.001),1:]

    epochs_classic_erdem, state_history_classic_erdem, dependent_variables_history_classic_erdem = \
        FrameConverter.SynodicToInertialHistoryConverter(dynamic_models["LF"]["CRTBP"][0], step_size=step_size).get_results(synodic_state_history_erdem)

    # Generate figures
    y_scale = 3
    fig1, axs1 = plt.subplots(len(durations), 3, figsize=(14, y_scale*(len(durations))), layout="constrained")
    fig2, axs2 = plt.subplots(len(durations), 3, figsize=(14, y_scale*(len(durations))), layout="constrained")
    fig3, axs3 = plt.subplots(len(durations), 3, figsize=(14, y_scale*(len(durations))), layout="constrained")
    fig4, axs4 = plt.subplots(len(durations), 3, figsize=(14, y_scale*(len(durations))), layout="constrained")

    figs_list = [[fig1, fig2],[fig3, fig4]]
    axs_list = [[axs1, axs2],[axs3, axs4]]

    fontsize = 16
    figure_titles = ["position", "velocity"]
    frames = ["inertial", "synodic"]
    satellite_labels = ["LPF", "LUMIO"]
    subplot_labels = [[r'X [m]', r'Y [m]', r'Z [m]', r'VX [m/s]', r'VY [m/s]', r'VZ [m/s]'],
                        [r'X [-]', r'Y [-]', r'Z [-]', r'VX [-]', r'VY [-]', r'VZ [-]']]
    labels = ["LPF ideal", "LUMIO ideal", "Moon", "LPF tudat high", "LUMIO tudat high", "LPF tudat low", "LUMIO tudat low"]
    figure_colors = ["lightgray", "lightgray", "gray", "red", "blue", "red", "blue"]

    fig1_3d = plt.figure()
    ax_3d = fig1_3d.add_subplot(111, projection='3d')
    ax_3d.set_title(f"Tudat high versus low models, {propagation_time} days")


    for m, dynamic_model in enumerate([dynamic_models["LF"]["CRTBP"][0], dynamic_models["HF"]["PM"][0]]):

        # Extra simulation histories of tudat models
        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(step_size=step_size).get_propagation_results(dynamic_model)

        epochs_synodic, state_history_synodic = \
            FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=step_size).get_results(state_history)


        ax_3d.plot(state_history_classic_erdem[:,0], state_history_classic_erdem[:,1], state_history_classic_erdem[:,2], label="LPF LF", color="red", ls="--")
        ax_3d.plot(state_history_classic_erdem[:,6], state_history_classic_erdem[:,7], state_history_classic_erdem[:,8], label="LUMIO LF", color="blue", ls="--")
        ax_3d.plot(state_history[:,0], state_history[:,1], state_history[:,2], label="LPF HF", color="red")
        ax_3d.plot(state_history[:,6], state_history[:,7], state_history[:,8], label="LUMIO HF", color="blue")

        # Plot the plane time evolutions in synodic and inertial frame
        for index, figs in enumerate(figs_list):
            axs = axs_list[index]
            for i, fig in enumerate(figs):
                for k, ax in enumerate(axs[i]):
                    j = int(durations[k]/step_size)
                    for l in range(len(ax)):
                        for n in range(len(satellite_labels)-1):

                            if index == 0:

                                # ax[l].plot(state_history_classic_erdem[:j,6*n+3*i+l%3], state_history_classic_erdem[:j,6*n+3*i+(l+1)%3], color=figure_colors[0])
                                # ax[l].plot(state_history_classic_erdem[:j,6*(n+1)+3*i+l%3], state_history_classic_erdem[:j,6*(n+1)+3*i+(l+1)%3], color=figure_colors[1])

                                if m != 0:
                                    ax[l].plot(dependent_variables_history[:j,3*i+l%3], dependent_variables_history[:j,3*i+(l+1)%3], color=figure_colors[2], lw=0.5, label=labels[2])
                                    ax[l].plot(state_history[:j,6*n+3*i+l%3], state_history[:j,6*n+3*i+(l+1)%3], color=figure_colors[3], label=labels[3])
                                    ax[l].plot(state_history[:j,6*(n+1)+3*i+l%3], state_history[:j,6*(n+1)+3*i+(l+1)%3], color=figure_colors[4], label=labels[4])
                                else:
                                    ax[l].plot(dependent_variables_history[:j,3*i+l%3], dependent_variables_history[:j,3*i+(l+1)%3], color=figure_colors[2], lw=0.5, ls="--", label=labels[2])
                                    ax[l].plot([None, None], color=figure_colors[3], label=labels[3])
                                    ax[l].plot([None, None], color=figure_colors[4], label=labels[4])
                                    ax[l].plot(state_history[:j,6*n+3*i+l%3], state_history[:j,6*n+3*i+(l+1)%3], color=figure_colors[5], ls="--", label=labels[5])
                                    ax[l].plot(state_history[:j,6*(n+1)+3*i+l%3], state_history[:j,6*(n+1)+3*i+(l+1)%3], color=figure_colors[6], ls="--", label=labels[6])

                            else:

                                # ax[l].plot(synodic_state_history_erdem[:j,6*n+3*i+l%3], synodic_state_history_erdem[:j,6*n+3*i+(l+1)%3], color=figure_colors[0])
                                # ax[l].plot(synodic_state_history_erdem[:j,6*(n+1)+3*i+l%3], synodic_state_history_erdem[:j,6*(n+1)+3*i+(l+1)%3], color=figure_colors[1])
                                # ax[l].plot([None,None])
                                if m != 0:
                                    ax[l].plot(state_history_synodic[:j,6*n+3*i+l%3], state_history_synodic[:j,6*n+3*i+(l+1)%3], color=figure_colors[3])
                                    ax[l].plot(state_history_synodic[:j,6*(n+1)+3*i+l%3], state_history_synodic[:j,6*(n+1)+3*i+(l+1)%3], color=figure_colors[4])
                                else:
                                    ax[l].plot([None, None], color=figure_colors[3])
                                    ax[l].plot([None, None], color=figure_colors[4])
                                    ax[l].plot(state_history_synodic[:j,6*n+3*i+l%3], state_history_synodic[:j,6*n+3*i+(l+1)%3], color=figure_colors[3], ls="--")
                                    ax[l].plot(state_history_synodic[:j,6*(n+1)+3*i+l%3], state_history_synodic[:j,6*(n+1)+3*i+(l+1)%3], color=figure_colors[4], ls="--")

                            ax[l].set_xlabel(subplot_labels[index][3*i+l%3])
                            ax[l].set_ylabel(subplot_labels[index][3*i+(l+1)%3])
                            ax[l].grid(alpha=0.5, linestyle='--')
                            ax[l].ticklabel_format(axis="y", scilimits=(0,0))

                        if index == 0:
                            fig.legend(["Moon", "LPF tudat high", "LUMIO tudat high", "LPF tudat low", "LUMIO tudat low"])
                        else:
                            fig.legend(["LPF tudat high", "LUMIO tudat high", "LPF tudat low", "LUMIO tudat low"])

                    if l == 3:
                        ax[l].invert_xaxis()
                        ax[l].invert_yaxis()

                    ax[0].title.set_text(str(durations[k])+ " days")
                fig.suptitle(f"Comparison {figure_titles[i]} low and high fidelity models, {frames[index]} frame", fontsize=fontsize)

    plt.tight_layout()
    # plt.show()

    ax_3d.set_xlabel('X [m]')
    ax_3d.set_ylabel('Y [m]')
    ax_3d.set_zlabel('Z [m]')
    fig1_3d.legend(loc="upper right")

    utils.save_figure_to_folder([fig1, fig2, fig3, fig4], [])


comparison_high_and_LF(60390, 28, [7, 14, 21, 28])