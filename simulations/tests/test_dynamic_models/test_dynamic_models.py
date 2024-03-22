# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

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


class TestFrameConversions:

    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time, durations",
    [
        (60390, 50, [7, 14, 28, 42, 49]),
    ])


    def test_validation_crtbp(self, simulation_start_epoch_MJD, propagation_time, durations, extras, step_size = 0.001):

        # Define initial state for both tudatpy and classic simulations
        custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
                                        1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])

        # Generate LowFidelityDynamicModel object only
        dynamic_model = low_fidelity.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state, use_synodic_state=True)

        # Extract simulation histories tudatpy solution
        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(step_size=step_size).get_propagation_results(dynamic_model, custom_initial_state=custom_initial_state)

        # Convert back to synodic
        epochs_synodic, state_history_synodic = \
            FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=step_size).get_results(state_history)

        # Extract synodic simulation histories classical solution
        epochs_synodic, synodic_state_history = validation.get_synodic_state_history(constants.GRAVITATIONAL_CONSTANT,
                                                                                           dynamic_model.bodies.get("Earth").mass,
                                                                                           dynamic_model.bodies.get("Moon").mass,
                                                                                           dynamic_model.distance_between_primaries,
                                                                                           propagation_time,
                                                                                           step_size,
                                                                                           custom_initial_state=custom_initial_state)

        # Extract synodic simulation histories classical solution Erdem's continuation model (breaks of other than 0.001)
        synodic_state_history_erdem = validation.get_synodic_state_history_erdem()[:int(propagation_time/0.001),1:]

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

        utils.save_figures_to_folder([fig1, fig2, fig3, fig4, fig5, fig6, fig7], [simulation_start_epoch_MJD, propagation_time])
        utils.save_figures_to_folder([fig1_3d], [simulation_start_epoch_MJD, propagation_time], save_to_report=False)



class TestOutputsDynamicModels:

    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time, durations",
    [
        (60390, 50, [7, 14, 28, 42, 49]),
    ])

    def test_difference_high_and_low_fidelity(self, simulation_start_epoch_MJD, propagation_time, durations, extras, step_size=0.001):

        custom_model_dict={"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]}
        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time, custom_model_dict=custom_model_dict)

        # Pick only the first of teach model name
        dynamic_models = utils.get_first_of_model_types(dynamic_model_objects)

        # Adjust to match initial state
        # dynamic_models[0] = low_fidelity.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=utils.synodic_initial_state)

        # Extract simulation histories of classic CRTBP halo continuation model
        synodic_state_history_erdem = validation.get_synodic_state_history_erdem()[:int(propagation_time/0.001),1:]

        epochs_classic_erdem, state_history_classic_erdem, dependent_variables_history_classic_erdem = \
            FrameConverter.SynodicToInertialHistoryConverter(dynamic_models[0], step_size=step_size).get_results(synodic_state_history_erdem)

        # Generate figures
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
        ax_3d.suptitle(f"Tudat high versus low models, {propagation_time} days")


        for m, dynamic_model in enumerate(dynamic_models[:2]):

            # Extra simulation histories of tudat models
            epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
                Interpolator.Interpolator(step_size=step_size).get_propagation_results(dynamic_model)

            epochs_synodic, state_history_synodic = \
                FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=step_size).get_results(state_history)


            ax_3d.plot(state_history_classic_erdem[:,0], state_history_classic_erdem[:,1], state_history_classic_erdem[:,2], label="LPF low_fidelity", color="red", ls="--")
            ax_3d.plot(state_history_classic_erdem[:,6], state_history_classic_erdem[:,7], state_history_classic_erdem[:,8], label="LUMIO low_fidelity", color="blue", ls="--")
            ax_3d.plot(state_history[:,0], state_history[:,1], state_history[:,2], label="LPF high_fidelity", color="red")
            ax_3d.plot(state_history[:,6], state_history[:,7], state_history[:,8], label="LUMIO high_fidelity", color="blue")

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

        utils.save_figures_to_folder([fig1_3d], [simulation_start_epoch_MJD, propagation_time], save_to_report=False)
        utils.save_figures_to_folder(list(itertools.chain(*figs_list)), [simulation_start_epoch_MJD, propagation_time])
