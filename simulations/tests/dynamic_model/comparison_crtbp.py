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
from src.dynamic_models import TraditionalLowFidelity
from src.dynamic_models.LF.CRTBP import *
from src.dynamic_models.HF.PM import *
from src.dynamic_models.HF.PMSRP import *
from src.dynamic_models.HF.SH import *
from src.dynamic_models.HF.SHSRP import *
from src.estimation_models import EstimationModel




def comparison_crtbp(simulation_start_epoch_MJD, propagation_time, durations, step_size=0.001):

    # Define initial state for both tudatpy and classic simulations
    custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
                                    1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])

    # Generate LowFidelityDynamicModel object only
    dynamic_model = LF_CRTBP.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state, use_synodic_state=True)

    # Extract simulation histories tudatpy solution
    epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
        Interpolator.Interpolator(step_size=step_size).get_propagation_results(dynamic_model, custom_initial_state=custom_initial_state)

    # Convert back to synodic
    epochs_synodic, state_history_synodic = \
        FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=step_size).get_results(state_history)

    # Extract synodic simulation histories classical solution
    epochs_synodic, synodic_state_history = reference_data.get_synodic_state_history(constants.GRAVITATIONAL_CONSTANT,
                                                                                        dynamic_model.bodies.get("Earth").mass,
                                                                                        dynamic_model.bodies.get("Moon").mass,
                                                                                        dynamic_model.distance_between_primaries,
                                                                                        propagation_time,
                                                                                        step_size,
                                                                                        custom_initial_state=custom_initial_state)

    # Extract synodic simulation histories classical solution Erdem's continuation model (breaks of other than 0.001)
    synodic_state_history_erdem = reference_data.get_synodic_state_history_erdem()[:int(propagation_time/0.001),1:]

    # Extract converted inertial states (works only with low-fidelity dynamic model)
    epochs_classic, state_history_classic, dependent_variables_history_classic = \
        FrameConverter.SynodicToInertialHistoryConverter(dynamic_model, step_size=step_size).get_results(synodic_state_history)

    # Extract converted inertial states Erdem's continuation model
    epochs_classic_erdem, state_history_classic_erdem, dependent_variables_history_classic_erdem = \
        FrameConverter.SynodicToInertialHistoryConverter(dynamic_model, step_size=step_size).get_results(synodic_state_history_erdem)


    fig1_3d = plt.figure()
    ax = fig1_3d.add_subplot(111, projection='3d')
    plt.title("Tudat versus true halo versus classical CRTBP")
    plt.plot(state_history_classic[:,0], state_history_classic[:,1], state_history_classic[:,2], label="LPF classic", color="gray")
    plt.plot(state_history_classic[:,6], state_history_classic[:,7], state_history_classic[:,8], label="LUMIO classic", color="gray")
    plt.plot(state_history_classic_erdem[:,0], state_history_classic_erdem[:,1], state_history_classic_erdem[:,2], label="LPF ideal", color="black")
    plt.plot(state_history_classic_erdem[:,6], state_history_classic_erdem[:,7], state_history_classic_erdem[:,8], label="LUMIO ideal", color="black")
    plt.plot(state_history[:,0], state_history[:,1], state_history[:,2], label="LPF tudat", color="red")
    plt.plot(state_history[:,6], state_history[:,7], state_history[:,8], label="LUMIO tudat", color="blue")
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.legend(loc="upper right")
    # plt.show()


    y_scale = 3

    ### Printing histories in inertial frame ###
    fontsize = 16
    fig1, axs1 = plt.subplots(len(durations), 3, figsize=(14, y_scale*(len(durations))), layout="constrained")
    fig2, axs2 = plt.subplots(len(durations), 3, figsize=(14, y_scale*(len(durations))), layout="constrained")

    figs = [fig1, fig2]
    axs = [axs1, axs2]

    for k, fig in enumerate(figs):
        for i, ax in enumerate(axs[k]):

            ax[0].title.set_text(str(durations[i])+ " days")
            [axis.grid(alpha=0.5, linestyle='--') for axis in ax]

            j = int(durations[i]/(step_size))

            if k==0:

                ax[0].plot(synodic_state_history_erdem[:j,0], synodic_state_history_erdem[:j,1], label="LPF ideal", color="black")
                ax[0].plot(synodic_state_history[:j,0], synodic_state_history[:j,1], label="LPF classic", color="gray")
                ax[0].plot(state_history_synodic[:j,0], state_history_synodic[:j,1], label="LPF tudat", color="red")
                ax[0].plot(synodic_state_history[:j,6], synodic_state_history[:j,7], label="LUMIO classic", color="gray")
                ax[0].plot(state_history_synodic[:j,6], state_history_synodic[:j,7], label="LUMIO tudat", color="blue")
                ax[0].plot(synodic_state_history_erdem[:j,6], synodic_state_history_erdem[:j,7], label="LUMIO ideal", color="black")
                ax[0].set_xlabel('X [-]')
                ax[0].set_ylabel('Y [-]')

                ax[1].plot(synodic_state_history_erdem[:j,1], synodic_state_history_erdem[:j,2], label="LPF ideal", color="black")
                ax[1].plot(synodic_state_history[:j,1], synodic_state_history[:j,2], label="LPF classic", color="gray")
                ax[1].plot(state_history_synodic[:j,1], state_history_synodic[:j,2], label="LPF tudat", color="red")
                ax[1].plot(synodic_state_history[:j,7], synodic_state_history[:j,8], label="LUMIO classic", color="gray")
                ax[1].plot(state_history_synodic[:j,7], state_history_synodic[:j,8], label="LUMIO tudat", color="blue")
                ax[1].plot(synodic_state_history_erdem[:j,7], synodic_state_history_erdem[:j,8], label="LUMIO ideal", color="black")
                ax[1].set_xlabel('Y [-]')
                ax[1].set_ylabel('Z [-]')

                ax[2].plot(synodic_state_history_erdem[:j,0], synodic_state_history_erdem[:j,2], label="LPF ideal", color="black")
                ax[2].plot(synodic_state_history[:j,0], synodic_state_history[:j,2], label="LPF classic", color="gray")
                ax[2].plot(state_history_synodic[:j,0], state_history_synodic[:j,2], label="LPF tudat", color="red")
                ax[2].plot(synodic_state_history[:j,6], synodic_state_history[:j,8], label="LUMIO classic", color="gray")
                ax[2].plot(state_history_synodic[:j,6], state_history_synodic[:j,8], label="LUMIO tudat", color="blue")
                ax[2].plot(synodic_state_history_erdem[:j,6], synodic_state_history_erdem[:j,8], label="LUMIO ideal", color="black")
                ax[2].set_xlabel('X [-]')
                ax[2].set_ylabel('Z [-]')

                fig.legend(["LPF ideal", "LPF classic", "LPF tudat", "LUMIO classic", "LUMIO tudat", "LUMIO ideal"])

            if k==1:

                ax[0].plot(synodic_state_history_erdem[:j,0], synodic_state_history_erdem[:j,1], label="LPF ideal", color="black")
                ax[0].plot(synodic_state_history[:j,0], synodic_state_history[:j,1], label="LPF classic", color="gray")
                ax[0].plot(state_history_synodic[:j,0], state_history_synodic[:j,1], label="LPF tudat", color="red")
                ax[0].set_xlabel('X [-]')
                ax[0].set_ylabel('Y [-]')

                ax[1].plot(synodic_state_history_erdem[:j,1], synodic_state_history_erdem[:j,2], label="LPF ideal", color="black")
                ax[1].plot(synodic_state_history[:j,1], synodic_state_history[:j,2], label="LPF classic", color="gray")
                ax[1].plot(state_history_synodic[:j,1], state_history_synodic[:j,2], label="LPF tudat", color="red")
                ax[1].set_xlabel('Y [-]')
                ax[1].set_ylabel('Z [-]')

                ax[2].plot(synodic_state_history_erdem[:j,0], synodic_state_history_erdem[:j,2], label="LPF ideal", color="black")
                ax[2].plot(synodic_state_history[:j,0], synodic_state_history[:j,2], label="LPF classic", color="gray")
                ax[2].plot(state_history_synodic[:j,0], state_history_synodic[:j,2], label="LPF tudat", color="red")
                ax[2].set_xlabel('X [-]')
                ax[2].set_ylabel('Z [-]')

                fig.legend(["LPF ideal", "LPF classic", "LPF tudat"])

            fig.suptitle("Comparison position CRTBP models, synodic frame", fontsize=fontsize)

    plt.tight_layout()
    # plt.show()


    fig3, axs3 = plt.subplots(len(durations), 3, figsize=(14, y_scale*(len(durations))), layout="constrained")
    fig4, axs4 = plt.subplots(len(durations), 3, figsize=(14, y_scale*(len(durations))), layout="constrained")

    figs = [fig3, fig4]
    axs = [axs3, axs4]

    for k, fig in enumerate(figs):
        for i, ax in enumerate(axs[k]):

            ax[0].title.set_text(str(durations[i])+ " days")
            [axis.grid(alpha=0.5, linestyle='--') for axis in ax]

            j = int(durations[i]/(step_size))

            if k==0:

                ax[0].plot(synodic_state_history_erdem[:j,3], synodic_state_history_erdem[:j,4], label="LPF ideal", color="black")
                ax[0].plot(synodic_state_history[:j,3], synodic_state_history[:j,4], label="LPF classic", color="gray")
                ax[0].plot(state_history_synodic[:j,3], state_history_synodic[:j,4], label="LPF tudat", color="red")
                ax[0].plot(synodic_state_history[:j,9], synodic_state_history[:j,10], label="LUMIO classic", color="gray")
                ax[0].plot(state_history_synodic[:j,9], state_history_synodic[:j,10], label="LUMIO tudat", color="blue")
                ax[0].plot(synodic_state_history_erdem[:j,9], synodic_state_history_erdem[:j,10], label="LUMIO ideal", color="black")
                ax[0].set_xlabel('VX [-]')
                ax[0].set_ylabel('VY [-]')

                ax[1].plot(synodic_state_history_erdem[:j,4], synodic_state_history_erdem[:j,5], label="LPF ideal", color="black")
                ax[1].plot(synodic_state_history[:j,4], synodic_state_history[:j,5], label="LPF classic", color="gray")
                ax[1].plot(state_history_synodic[:j,4], state_history_synodic[:j,5], label="LPF tudat", color="red")
                ax[1].plot(synodic_state_history[:j,10], synodic_state_history[:j,11], label="LUMIO classic", color="gray")
                ax[1].plot(state_history_synodic[:j,10], state_history_synodic[:j,11], label="LUMIO tudat", color="blue")
                ax[1].plot(synodic_state_history_erdem[:j,10], synodic_state_history_erdem[:j,11], label="LUMIO ideal", color="black")
                ax[1].set_xlabel('VY [-]')
                ax[1].set_ylabel('VZ [-]')

                ax[2].plot(synodic_state_history_erdem[:j,3], synodic_state_history_erdem[:j,5], label="LPF ideal", color="black")
                ax[2].plot(synodic_state_history[:j,3], synodic_state_history[:j,5], label="LPF classic", color="gray")
                ax[2].plot(state_history_synodic[:j,3], state_history_synodic[:j,5], label="LPF tudat", color="red")
                ax[2].plot(synodic_state_history[:j,9], synodic_state_history[:j,11], label="LUMIO classic", color="gray")
                ax[2].plot(state_history_synodic[:j,9], state_history_synodic[:j,11], label="LUMIO tudat", color="blue")
                ax[2].plot(synodic_state_history_erdem[:j,9], synodic_state_history_erdem[:j,11], label="LUMIO ideal", color="black")
                ax[2].set_xlabel('VX [-]')
                ax[2].set_ylabel('VZ [-]')

                fig.legend(["LPF ideal", "LPF classic", "LPF tudat", "LUMIO classic", "LUMIO tudat", "LUMIO ideal"])

            if k==1:

                ax[0].plot(synodic_state_history[:j,9], synodic_state_history[:j,10], label="LUMIO classic", color="gray")
                ax[0].plot(state_history_synodic[:j,9], state_history_synodic[:j,10], label="LUMIO tudat", color="blue")
                ax[0].plot(synodic_state_history_erdem[:j,9], synodic_state_history_erdem[:j,10], label="LUMIO ideal", color="black")
                ax[0].set_xlabel('VX [-]')
                ax[0].set_ylabel('VY [-]')

                ax[1].plot(synodic_state_history[:j,10], synodic_state_history[:j,11], label="LUMIO classic", color="gray")
                ax[1].plot(state_history_synodic[:j,10], state_history_synodic[:j,11], label="LUMIO tudat", color="blue")
                ax[1].plot(synodic_state_history_erdem[:j,10], synodic_state_history_erdem[:j,11], label="LUMIO ideal", color="black")
                ax[1].set_xlabel('VY [-]')
                ax[1].set_ylabel('VZ [-]')

                ax[2].plot(synodic_state_history[:j,9], synodic_state_history[:j,11], label="LUMIO classic", color="gray")
                ax[2].plot(state_history_synodic[:j,9], state_history_synodic[:j,11], label="LUMIO tudat", color="blue")
                ax[2].plot(synodic_state_history_erdem[:j,9], synodic_state_history_erdem[:j,11], label="LUMIO ideal", color="black")
                ax[2].set_xlabel('VX [-]')
                ax[2].set_ylabel('VZ [-]')

                fig.legend(["LUMIO classic", "LUMIO tudat", "LUMIO ideal"])

            fig.suptitle("Comparison velocities CRTBP models, synodic frame", fontsize=fontsize)

    plt.tight_layout()
    # plt.show()



    ### Printing histories in inertial frame ###

    fig5, axs5 = plt.subplots(len(durations), 3, figsize=(14, y_scale*(len(durations))), layout="constrained", sharex=True, sharey=True)

    figs = [fig5]
    axs = [axs5]

    for k, fig in enumerate(figs):
        for i, ax in enumerate(axs[k]):

            ax[0].title.set_text(str(durations[i])+ " days")
            [axis.grid(alpha=0.5, linestyle='--') for axis in ax]

            j = int(durations[i]/(step_size))

            if k==0:

                ax[0].plot(state_history_classic_erdem[:j,0], state_history_classic_erdem[:j,1], label="LPF ideal", color="black")
                ax[0].plot(state_history_classic[:j,0], state_history_classic[:j,1], label="LPF classic", color="gray")
                ax[0].plot(state_history[:j,0], state_history[:j,1], label="LPF tudat", color="red")
                ax[0].plot(state_history_classic_erdem[:j,6], state_history_classic_erdem[:j,7], label="LUMIO ideal", color="black")
                ax[0].plot(state_history_classic[:j,6], state_history_classic[:j,7], label="LUMIO classic", color="gray")
                ax[0].plot(state_history[:j,6], state_history[:j,7], label="LUMIO tudat", color="blue")
                ax[0].set_xlabel('X [m]')
                ax[0].set_ylabel('Y [m]')

                ax[1].plot(state_history_classic_erdem[:j,1], state_history_classic_erdem[:j,2], label="LPF ideal", color="black")
                ax[1].plot(state_history_classic[:j,1], state_history_classic[:j,2], label="LPF classic", color="gray")
                ax[1].plot(state_history[:j,1], state_history[:j,2], label="LPF tudat", color="red")
                ax[1].plot(state_history_classic_erdem[:j,7], state_history_classic_erdem[:j,8], label="LUMIO ideal", color="black")
                ax[1].plot(state_history_classic[:j,7], state_history_classic[:j,8], label="LUMIO classic", color="gray")
                ax[1].plot(state_history[:j,7], state_history[:j,8], label="LUMIO tudat", color="blue")
                ax[1].set_xlabel('Y [m]')
                ax[1].set_ylabel('Z [m]')

                ax[2].plot(state_history_classic_erdem[:j,0], state_history_classic_erdem[:j,2], label="LPF ideal", color="black")
                ax[2].plot(state_history_classic[:j,0], state_history_classic[:j,2], label="LPF classic", color="gray")
                ax[2].plot(state_history[:j,0], state_history[:j,2], label="LPF tudat", color="red")
                ax[2].plot(state_history_classic_erdem[:j,6], state_history_classic_erdem[:j,8], label="LUMIO ideal", color="black")
                ax[2].plot(state_history_classic[:j,6], state_history_classic[:j,8], label="LUMIO classic", color="gray")
                ax[2].plot(state_history[:j,6], state_history[:j,8], label="LUMIO tudat", color="blue")
                ax[2].set_xlabel('X [m]')
                ax[2].set_ylabel('Z [m]')

                fig.legend(["LPF ideal", "LPF classic", "LPF tudat", "LUMIO ideal", "LUMIO classic", "LUMIO tudat"])

            fig.suptitle("Comparison position CRTBP models, inertial frame", fontsize=fontsize)

    plt.tight_layout()
    # plt.show()


    fig6, axs6 = plt.subplots(len(durations), 3, figsize=(14, y_scale*(len(durations))), layout="constrained")

    figs = [fig6]
    axs = [axs6]

    for k, fig in enumerate(figs):
        for i, ax in enumerate(axs[k]):

            ax[0].title.set_text(str(durations[i])+ " days")
            [axis.grid(alpha=0.5, linestyle='--') for axis in ax]

            j = int(durations[i]/(step_size))

            ax[0].plot(state_history_classic_erdem[:j,3], state_history_classic_erdem[:j,4], label="LPF ideal", color="black")
            ax[0].plot(state_history_classic[:j,3], state_history_classic[:j,4], label="LPF classic", color="gray")
            ax[0].plot(state_history[:j,3], state_history[:j,4], label="LPF tudat", color="red")
            ax[0].plot(state_history_classic[:j,9], state_history_classic[:j,10], label="LUMIO classic", color="gray")
            ax[0].plot(state_history[:j,9], state_history[:j,10], label="LUMIO tudat", color="blue")
            ax[0].plot(state_history_classic_erdem[:j,9], state_history_classic_erdem[:j,10], label="LUMIO ideal", color="black")
            ax[0].set_xlabel('VX [m/s]')
            ax[0].set_ylabel('VY [m/s]')

            ax[1].plot(state_history_classic_erdem[:j,4], state_history_classic_erdem[:j,5], label="LPF ideal", color="black")
            ax[1].plot(state_history_classic[:j,4], state_history_classic[:j,5], label="LPF classic", color="gray")
            ax[1].plot(state_history[:j,4], state_history[:j,5], label="LPF tudat", color="red")
            ax[1].plot(state_history_classic[:j,10], state_history_classic[:j,11], label="LUMIO classic", color="gray")
            ax[1].plot(state_history[:j,10], state_history[:j,11], label="LUMIO tudat", color="blue")
            ax[1].plot(state_history_classic_erdem[:j,10], state_history_classic_erdem[:j,11], label="LUMIO ideal", color="black")
            ax[1].set_xlabel('VY [m/s]')
            ax[1].set_ylabel('VZ [m/s]')

            ax[2].plot(state_history_classic_erdem[:j,3], state_history_classic_erdem[:j,5], label="LPF ideal", color="black")
            ax[2].plot(state_history_classic[:j,3], state_history_classic[:j,5], label="LPF classic", color="gray")
            ax[2].plot(state_history[:j,3], state_history[:j,5], label="LPF tudat", color="red")
            ax[2].plot(state_history_classic[:j,9], state_history_classic[:j,11], label="LUMIO classic", color="gray")
            ax[2].plot(state_history[:j,9], state_history[:j,11], label="LUMIO tudat", color="blue")
            ax[2].plot(state_history_classic_erdem[:j,9], state_history_classic_erdem[:j,11], label="LUMIO ideal", color="black")
            ax[2].set_xlabel('VX [m/s]')
            ax[2].set_ylabel('VZ [m/s]')

            fig.legend(["LPF ideal", "LPF classic", "LPF tudat", "LUMIO classic", "LUMIO tudat", "LUMIO ideal"])
        fig.suptitle("Comparison velocities CRTBP models, inertial frame", fontsize=fontsize)
    plt.tight_layout()
    # plt.show()


    fig7, axs = plt.subplots(6, 1, figsize=(6.4,8.3), constrained_layout=True, sharex=True)


    n = int(14/step_size)
    epochs = epochs[:n] - epochs[0]

    # axs[0].set_title("Position states tudat")
    axs[0].set_ylabel(r"$||\mathbf{r}_{tudat}||$ [m]")
    axs[0].plot(epochs, np.linalg.norm(state_history[:n,0:3], axis=1), label="LPF w.r.t. Earth", color="red")
    axs[0].plot(epochs, np.linalg.norm(state_history[:n,6:9], axis=1), label="LUMIO w.r.t. Earth", color="blue")
    axs[0].plot(epochs, np.linalg.norm(dependent_variables_history[:n,0:3], axis=1), label="Moon w.r.t. Earth", color="gray")
    axs[0].plot(epochs, np.linalg.norm(state_history[:n,0:3]-dependent_variables_history[:n,0:3], axis=1), label="LPF w.r.t. Moon", color="red", ls="--")
    axs[0].plot(epochs, np.linalg.norm(state_history[:n,6:9]-dependent_variables_history[:n,0:3], axis=1), label="LUMIO w.r.t. Moon", color="blue", ls="--")
    axs[0].legend(loc="upper right", ncol=2, fontsize="x-small")

    # axs[1].set_title("Position states classic")
    axs[1].set_ylabel(r"$||\mathbf{r}_{classic}||$ [m]")
    axs[1].plot(epochs, np.linalg.norm(state_history_classic[:n,0:3], axis=1), label="LPF w.r.t. Earth", color="red")
    axs[1].plot(epochs, np.linalg.norm(state_history_classic[:n,6:9], axis=1), label="LUMIO w.r.t. Earth", color="blue")
    axs[1].plot(epochs, np.linalg.norm(dependent_variables_history_classic[:n,6:9], axis=1), label="Moon w.r.t. Earth", color="gray")
    axs[1].plot(epochs, np.linalg.norm(state_history_classic[:n,0:3]-dependent_variables_history_classic[:n,6:9], axis=1), label="LPF w.r.t. Moon", color="red", ls="--")
    axs[1].plot(epochs, np.linalg.norm(state_history_classic[:n,6:9]-dependent_variables_history_classic[:n,6:9], axis=1), label="LUMIO w.r.t. Moon", color="blue", ls="--")

    # axs[2].set_title("Velocity states tudat")
    axs[2].set_ylabel(r"$||\mathbf{v}_{tudat}||$ [m/s]")
    axs[2].plot(epochs, np.linalg.norm(state_history[:n,3:6], axis=1), label="LPF w.r.t. Earth", color="red")
    axs[2].plot(epochs, np.linalg.norm(state_history[:n,9:12], axis=1), label="LUMIO w.r.t. Earth", color="blue")
    axs[2].plot(epochs, np.linalg.norm(dependent_variables_history[:n,3:6], axis=1), label="Moon w.r.t. Earth", color="gray")
    axs[2].plot(epochs, np.linalg.norm(state_history[:n,3:6]-dependent_variables_history[:n,3:6], axis=1), label="LPF w.r.t. Moon", color="red", ls="--")
    axs[2].plot(epochs, np.linalg.norm(state_history[:n,9:12]-dependent_variables_history[:n,3:6], axis=1), label="LUMIO w.r.t. Moon", color="blue", ls="--")

    # axs[3].set_title("Velocity states classic")
    axs[3].set_ylabel(r"$||\mathbf{v}_{classic}||$ [m/s]")
    axs[3].plot(epochs, np.linalg.norm(state_history_classic[:n,3:6], axis=1), label="LPF w.r.t. Earth", color="red")
    axs[3].plot(epochs, np.linalg.norm(state_history_classic[:n,9:12], axis=1), label="LUMIO w.r.t. Earth", color="blue")
    axs[3].plot(epochs, np.linalg.norm(dependent_variables_history_classic[:n,9:12], axis=1), label="Moon w.r.t. Earth", color="gray")
    axs[3].plot(epochs, np.linalg.norm(state_history_classic[:n,3:6]-dependent_variables_history_classic[:n,9:12], axis=1), label="LPF w.r.t. Moon", color="red", ls="--")
    axs[3].plot(epochs, np.linalg.norm(state_history_classic[:n,9:12]-dependent_variables_history_classic[:n,9:12], axis=1), label="LUMIO w.r.t. Moon", color="blue", ls="--")

    # axs[4].set_title("Difference position states")
    axs[4].set_ylabel(r"$||\mathbf{r}_{tudat}-\mathbf{r}_{classic}||$")
    axs[4].plot(epochs, np.linalg.norm(state_history[:n,0:3]-state_history_classic[:n,0:3], axis=1), label="LPF w.r.t. Earth", color="red")
    axs[4].plot(epochs, np.linalg.norm(state_history[:n,6:9]-state_history_classic[:n,6:9], axis=1), label="LUMIO w.r.t. Earth", color="blue")
    axs[4].plot(epochs, np.linalg.norm(dependent_variables_history[:n,0:3]-dependent_variables_history_classic[:n,6:9], axis=1), label="Moon w.r.t. Earth", color="gray")
    axs[4].plot(epochs, np.linalg.norm((state_history[:n,0:3]-dependent_variables_history[:n,0:3])-(state_history_classic[:n,0:3]-dependent_variables_history_classic[:n,6:9]), axis=1), label="LPF w.r.t. Moon", color="red")
    axs[4].plot(epochs, np.linalg.norm((state_history[:n,6:9]-dependent_variables_history[:n,0:3])-(state_history_classic[:n,6:9]-dependent_variables_history_classic[:n,6:9]), axis=1), label="LUMIO w.r.t. Moon", color="blue")

    # axs[5].set_title("Difference velocity states")
    axs[5].set_ylabel(r"$||\mathbf{v}_{tudat}-\mathbf{v}_{classic}||$")
    axs[5].plot(epochs, np.linalg.norm(state_history[:n,3:6]-state_history_classic[:n,3:6], axis=1), label="LPF w.r.t. Earth", color="red")
    axs[5].plot(epochs, np.linalg.norm(state_history[:n,9:12]-state_history_classic[:n,9:12], axis=1), label="LUMIO w.r.t. Earth", color="blue")
    axs[5].plot(epochs, np.linalg.norm(dependent_variables_history[:n,3:6]-dependent_variables_history_classic[:n,9:12], axis=1), label="Moon w.r.t. Earth", color="gray")
    axs[5].plot(epochs, np.linalg.norm((state_history[:n,3:6]-dependent_variables_history[:n,3:6])-(state_history_classic[:n,3:6]-dependent_variables_history_classic[:n,9:12]), axis=1), label="LPF w.r.t. Moon", color="red")
    axs[5].plot(epochs, np.linalg.norm((state_history[:n,9:12]-dependent_variables_history[:n,3:6])-(state_history_classic[:n,9:12]-dependent_variables_history_classic[:n,9:12]), axis=1), label="LUMIO w.r.t. Moon", color="blue")
    axs[5].set_xlabel(f"Time since MJD {simulation_start_epoch_MJD} [days]")

    plt.suptitle("State comparison of classic and tudat CRTBP models")
    [ax.grid(alpha=0.5, linestyle='--') for ax in axs]
    [ax.set_yscale("log") for ax in axs]
    plt.tight_layout()

    utils.save_figure_to_folder([fig1, fig2, fig3, fig4, fig5, fig6, fig7], [propagation_time])


comparison_crtbp(60390, 50, [7, 14, 28, 42, 49])