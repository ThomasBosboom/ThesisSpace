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
from src.dynamic_models import validation_LUMIO
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
        dynamic_model = low_fidelity.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state)

        # Extract simulation histories tudatpy solution
        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(step_size=step_size).get_propagator_results(dynamic_model)

        # Convert back to synodic
        epochs_synodic, state_history_synodic = \
            FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=step_size).get_results(state_history)

        # Extract synodic simulation histories classical solution
        epochs_synodic, synodic_state_history = validation_LUMIO.get_synodic_state_history(constants.GRAVITATIONAL_CONSTANT,
                                                                                           dynamic_model.bodies.get("Earth").mass,
                                                                                           dynamic_model.bodies.get("Moon").mass,
                                                                                           dynamic_model.distance_between_primaries,
                                                                                           propagation_time,
                                                                                           step_size,
                                                                                           custom_initial_state=custom_initial_state)

        # Extract synodic simulation histories classical solution Erdem's continuation model (breaks of other than 0.001)
        synodic_state_history_erdem = validation_LUMIO.get_synodic_state_history_erdem()[:int(propagation_time/0.001),1:]

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
        plt.plot(state_history_classic_erdem[:,0], state_history_classic_erdem[:,1], state_history_classic_erdem[:,2], label="LPF halo", color="black")
        plt.plot(state_history_classic_erdem[:,6], state_history_classic_erdem[:,7], state_history_classic_erdem[:,8], label="LUMIO halo", color="black")
        plt.plot(state_history[:,0], state_history[:,1], state_history[:,2], label="LPF tudat", color="red")
        plt.plot(state_history[:,6], state_history[:,7], state_history[:,8], label="LUMIO tudat", color="blue")
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.legend(loc="upper right")
        # plt.show()



        ### Printing histories in inertial frame ###
        fontsize = 16
        fig1, axs1 = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained")
        fig2, axs2 = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained")

        figs = [fig1, fig2]
        axs = [axs1, axs2]

        for k, fig in enumerate(figs):
            for i, ax in enumerate(axs[k]):

                ax[0].title.set_text(str(durations[i])+ " days")
                [axis.grid(alpha=0.5, linestyle='--') for axis in ax]

                j = int(durations[i]/(step_size))

                if k==0:

                    ax[0].plot(synodic_state_history_erdem[:j,0], synodic_state_history_erdem[:j,1], label="LPF halo", color="black")
                    ax[0].plot(synodic_state_history[:j,0], synodic_state_history[:j,1], label="LPF classic", color="gray")
                    ax[0].plot(state_history_synodic[:j,0], state_history_synodic[:j,1], label="LPF tudat", color="red")
                    ax[0].plot(synodic_state_history[:j,6], synodic_state_history[:j,7], label="LUMIO classic", color="gray")
                    ax[0].plot(state_history_synodic[:j,6], state_history_synodic[:j,7], label="LUMIO tudat", color="blue")
                    ax[0].plot(synodic_state_history_erdem[:j,6], synodic_state_history_erdem[:j,7], label="LUMIO halo", color="black")
                    ax[0].set_xlabel('X [-]')
                    ax[0].set_ylabel('Y [-]')

                    ax[1].plot(synodic_state_history_erdem[:j,1], synodic_state_history_erdem[:j,2], label="LPF halo", color="black")
                    ax[1].plot(synodic_state_history[:j,1], synodic_state_history[:j,2], label="LPF classic", color="gray")
                    ax[1].plot(state_history_synodic[:j,1], state_history_synodic[:j,2], label="LPF tudat", color="red")
                    ax[1].plot(synodic_state_history[:j,7], synodic_state_history[:j,8], label="LUMIO classic", color="gray")
                    ax[1].plot(state_history_synodic[:j,7], state_history_synodic[:j,8], label="LUMIO tudat", color="blue")
                    ax[1].plot(synodic_state_history_erdem[:j,7], synodic_state_history_erdem[:j,8], label="LUMIO halo", color="black")
                    ax[1].set_xlabel('Y [-]')
                    ax[1].set_ylabel('Z [-]')

                    ax[2].plot(synodic_state_history_erdem[:j,0], synodic_state_history_erdem[:j,2], label="LPF halo", color="black")
                    ax[2].plot(synodic_state_history[:j,0], synodic_state_history[:j,2], label="LPF classic", color="gray")
                    ax[2].plot(state_history_synodic[:j,0], state_history_synodic[:j,2], label="LPF tudat", color="red")
                    ax[2].plot(synodic_state_history[:j,6], synodic_state_history[:j,8], label="LUMIO classic", color="gray")
                    ax[2].plot(state_history_synodic[:j,6], state_history_synodic[:j,8], label="LUMIO tudat", color="blue")
                    ax[2].plot(synodic_state_history_erdem[:j,6], synodic_state_history_erdem[:j,8], label="LUMIO halo", color="black")
                    ax[2].set_xlabel('X [-]')
                    ax[2].set_ylabel('Z [-]')

                    fig.legend(["LPF halo", "LPF classic", "LPF tudat", "LUMIO classic", "LUMIO tudat", "LUMIO halo"])

                if k==1:

                    ax[0].plot(synodic_state_history_erdem[:j,0], synodic_state_history_erdem[:j,1], label="LPF halo", color="black")
                    ax[0].plot(synodic_state_history[:j,0], synodic_state_history[:j,1], label="LPF classic", color="gray")
                    ax[0].plot(state_history_synodic[:j,0], state_history_synodic[:j,1], label="LPF tudat", color="red")
                    ax[0].set_xlabel('X [-]')
                    ax[0].set_ylabel('Y [-]')

                    ax[1].plot(synodic_state_history_erdem[:j,1], synodic_state_history_erdem[:j,2], label="LPF halo", color="black")
                    ax[1].plot(synodic_state_history[:j,1], synodic_state_history[:j,2], label="LPF classic", color="gray")
                    ax[1].plot(state_history_synodic[:j,1], state_history_synodic[:j,2], label="LPF tudat", color="red")
                    ax[1].set_xlabel('Y [-]')
                    ax[1].set_ylabel('Z [-]')

                    ax[2].plot(synodic_state_history_erdem[:j,0], synodic_state_history_erdem[:j,2], label="LPF halo", color="black")
                    ax[2].plot(synodic_state_history[:j,0], synodic_state_history[:j,2], label="LPF classic", color="gray")
                    ax[2].plot(state_history_synodic[:j,0], state_history_synodic[:j,2], label="LPF tudat", color="red")
                    ax[2].set_xlabel('X [-]')
                    ax[2].set_ylabel('Z [-]')

                    fig.legend(["LPF halo", "LPF classic", "LPF tudat"])

                fig.suptitle("Comparison position CRTBP models, synodic frame", fontsize=fontsize)

        plt.tight_layout()
        # plt.show()


        fig3, axs3 = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained")
        fig4, axs4 = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained")

        figs = [fig3, fig4]
        axs = [axs3, axs4]

        for k, fig in enumerate(figs):
            for i, ax in enumerate(axs[k]):

                ax[0].title.set_text(str(durations[i])+ " days")
                [axis.grid(alpha=0.5, linestyle='--') for axis in ax]

                j = int(durations[i]/(step_size))

                if k==0:

                    ax[0].plot(synodic_state_history_erdem[:j,3], synodic_state_history_erdem[:j,4], label="LPF halo", color="black")
                    ax[0].plot(synodic_state_history[:j,3], synodic_state_history[:j,4], label="LPF classic", color="gray")
                    ax[0].plot(state_history_synodic[:j,3], state_history_synodic[:j,4], label="LPF tudat", color="red")
                    ax[0].plot(synodic_state_history[:j,9], synodic_state_history[:j,10], label="LUMIO classic", color="gray")
                    ax[0].plot(state_history_synodic[:j,9], state_history_synodic[:j,10], label="LUMIO tudat", color="blue")
                    ax[0].plot(synodic_state_history_erdem[:j,9], synodic_state_history_erdem[:j,10], label="LUMIO halo", color="black")
                    ax[0].set_xlabel('VX [-]')
                    ax[0].set_ylabel('VY [-]')

                    ax[1].plot(synodic_state_history_erdem[:j,4], synodic_state_history_erdem[:j,5], label="LPF halo", color="black")
                    ax[1].plot(synodic_state_history[:j,4], synodic_state_history[:j,5], label="LPF classic", color="gray")
                    ax[1].plot(state_history_synodic[:j,4], state_history_synodic[:j,5], label="LPF tudat", color="red")
                    ax[1].plot(synodic_state_history[:j,10], synodic_state_history[:j,11], label="LUMIO classic", color="gray")
                    ax[1].plot(state_history_synodic[:j,10], state_history_synodic[:j,11], label="LUMIO tudat", color="blue")
                    ax[1].plot(synodic_state_history_erdem[:j,10], synodic_state_history_erdem[:j,11], label="LUMIO halo", color="black")
                    ax[1].set_xlabel('VY [-]')
                    ax[1].set_ylabel('VZ [-]')

                    ax[2].plot(synodic_state_history_erdem[:j,3], synodic_state_history_erdem[:j,5], label="LPF halo", color="black")
                    ax[2].plot(synodic_state_history[:j,3], synodic_state_history[:j,5], label="LPF classic", color="gray")
                    ax[2].plot(state_history_synodic[:j,3], state_history_synodic[:j,5], label="LPF tudat", color="red")
                    ax[2].plot(synodic_state_history[:j,9], synodic_state_history[:j,11], label="LUMIO classic", color="gray")
                    ax[2].plot(state_history_synodic[:j,9], state_history_synodic[:j,11], label="LUMIO tudat", color="blue")
                    ax[2].plot(synodic_state_history_erdem[:j,9], synodic_state_history_erdem[:j,11], label="LUMIO halo", color="black")
                    ax[2].set_xlabel('VX [-]')
                    ax[2].set_ylabel('VZ [-]')

                    fig.legend(["LPF halo", "LPF classic", "LPF tudat", "LUMIO classic", "LUMIO tudat", "LUMIO halo"])

                if k==1:

                    ax[0].plot(synodic_state_history[:j,9], synodic_state_history[:j,10], label="LUMIO classic", color="gray")
                    ax[0].plot(state_history_synodic[:j,9], state_history_synodic[:j,10], label="LUMIO tudat", color="blue")
                    ax[0].plot(synodic_state_history_erdem[:j,9], synodic_state_history_erdem[:j,10], label="LUMIO halo", color="black")
                    ax[0].set_xlabel('VX [-]')
                    ax[0].set_ylabel('VY [-]')

                    ax[1].plot(synodic_state_history[:j,10], synodic_state_history[:j,11], label="LUMIO classic", color="gray")
                    ax[1].plot(state_history_synodic[:j,10], state_history_synodic[:j,11], label="LUMIO tudat", color="blue")
                    ax[1].plot(synodic_state_history_erdem[:j,10], synodic_state_history_erdem[:j,11], label="LUMIO halo", color="black")
                    ax[1].set_xlabel('VY [-]')
                    ax[1].set_ylabel('VZ [-]')

                    ax[2].plot(synodic_state_history[:j,9], synodic_state_history[:j,11], label="LUMIO classic", color="gray")
                    ax[2].plot(state_history_synodic[:j,9], state_history_synodic[:j,11], label="LUMIO tudat", color="blue")
                    ax[2].plot(synodic_state_history_erdem[:j,9], synodic_state_history_erdem[:j,11], label="LUMIO halo", color="black")
                    ax[2].set_xlabel('VX [-]')
                    ax[2].set_ylabel('VZ [-]')

                    fig.legend(["LUMIO classic", "LUMIO tudat", "LUMIO halo"])

                fig.suptitle("Comparison velocities CRTBP models, synodic frame", fontsize=fontsize)

        plt.tight_layout()
        # plt.show()



        ### Printing histories in inertial frame ###

        fig5, axs5 = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained", sharex=True, sharey=True)

        figs = [fig5]
        axs = [axs5]

        for k, fig in enumerate(figs):
            for i, ax in enumerate(axs[k]):

                ax[0].title.set_text(str(durations[i])+ " days")
                [axis.grid(alpha=0.5, linestyle='--') for axis in ax]

                j = int(durations[i]/(step_size))

                if k==0:

                    ax[0].plot(state_history_classic_erdem[:j,0], state_history_classic_erdem[:j,1], label="LPF halo", color="black")
                    ax[0].plot(state_history_classic[:j,0], state_history_classic[:j,1], label="LPF classic", color="gray")
                    ax[0].plot(state_history[:j,0], state_history[:j,1], label="LPF tudat", color="red")
                    ax[0].plot(state_history_classic_erdem[:j,6], state_history_classic_erdem[:j,7], label="LUMIO halo", color="black")
                    ax[0].plot(state_history_classic[:j,6], state_history_classic[:j,7], label="LUMIO classic", color="gray")
                    ax[0].plot(state_history[:j,6], state_history[:j,7], label="LUMIO tudat", color="blue")
                    ax[0].set_xlabel('X [m]')
                    ax[0].set_ylabel('Y [m]')

                    ax[1].plot(state_history_classic_erdem[:j,1], state_history_classic_erdem[:j,2], label="LPF halo", color="black")
                    ax[1].plot(state_history_classic[:j,1], state_history_classic[:j,2], label="LPF classic", color="gray")
                    ax[1].plot(state_history[:j,1], state_history[:j,2], label="LPF tudat", color="red")
                    ax[1].plot(state_history_classic_erdem[:j,7], state_history_classic_erdem[:j,8], label="LUMIO halo", color="black")
                    ax[1].plot(state_history_classic[:j,7], state_history_classic[:j,8], label="LUMIO classic", color="gray")
                    ax[1].plot(state_history[:j,7], state_history[:j,8], label="LUMIO tudat", color="blue")
                    ax[1].set_xlabel('Y [m]')
                    ax[1].set_ylabel('Z [m]')

                    ax[2].plot(state_history_classic_erdem[:j,0], state_history_classic_erdem[:j,2], label="LPF halo", color="black")
                    ax[2].plot(state_history_classic[:j,0], state_history_classic[:j,2], label="LPF classic", color="gray")
                    ax[2].plot(state_history[:j,0], state_history[:j,2], label="LPF tudat", color="red")
                    ax[2].plot(state_history_classic_erdem[:j,6], state_history_classic_erdem[:j,8], label="LUMIO halo", color="black")
                    ax[2].plot(state_history_classic[:j,6], state_history_classic[:j,8], label="LUMIO classic", color="gray")
                    ax[2].plot(state_history[:j,6], state_history[:j,8], label="LUMIO tudat", color="blue")
                    ax[2].set_xlabel('X [m]')
                    ax[2].set_ylabel('Z [m]')

                    fig.legend(["LPF halo", "LPF classic", "LPF tudat", "LUMIO halo", "LUMIO classic", "LUMIO tudat"])

                fig.suptitle("Comparison position CRTBP models, inertial frame", fontsize=fontsize)

        plt.tight_layout()
        # plt.show()


        fig6, axs6 = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained")

        figs = [fig6]
        axs = [axs6]

        for k, fig in enumerate(figs):
            for i, ax in enumerate(axs[k]):

                ax[0].title.set_text(str(durations[i])+ " days")
                [axis.grid(alpha=0.5, linestyle='--') for axis in ax]

                j = int(durations[i]/(step_size))

                ax[0].plot(state_history_classic_erdem[:j,3], state_history_classic_erdem[:j,4], label="LPF halo", color="black")
                ax[0].plot(state_history_classic[:j,3], state_history_classic[:j,4], label="LPF classic", color="gray")
                ax[0].plot(state_history[:j,3], state_history[:j,4], label="LPF tudat", color="red")
                ax[0].plot(state_history_classic[:j,9], state_history_classic[:j,10], label="LUMIO classic", color="gray")
                ax[0].plot(state_history[:j,9], state_history[:j,10], label="LUMIO tudat", color="blue")
                ax[0].plot(state_history_classic_erdem[:j,9], state_history_classic_erdem[:j,10], label="LUMIO halo", color="black")
                ax[0].set_xlabel('VX [m/s]')
                ax[0].set_ylabel('VY [m/s]')

                ax[1].plot(state_history_classic_erdem[:j,4], state_history_classic_erdem[:j,5], label="LPF halo", color="black")
                ax[1].plot(state_history_classic[:j,4], state_history_classic[:j,5], label="LPF classic", color="gray")
                ax[1].plot(state_history[:j,4], state_history[:j,5], label="LPF tudat", color="red")
                ax[1].plot(state_history_classic[:j,10], state_history_classic[:j,11], label="LUMIO classic", color="gray")
                ax[1].plot(state_history[:j,10], state_history[:j,11], label="LUMIO tudat", color="blue")
                ax[1].plot(state_history_classic_erdem[:j,10], state_history_classic_erdem[:j,11], label="LUMIO halo", color="black")
                ax[1].set_xlabel('VY [m/s]')
                ax[1].set_ylabel('VZ [m/s]')

                ax[2].plot(state_history_classic_erdem[:j,3], state_history_classic_erdem[:j,5], label="LPF halo", color="black")
                ax[2].plot(state_history_classic[:j,3], state_history_classic[:j,5], label="LPF classic", color="gray")
                ax[2].plot(state_history[:j,3], state_history[:j,5], label="LPF tudat", color="red")
                ax[2].plot(state_history_classic[:j,9], state_history_classic[:j,11], label="LUMIO classic", color="gray")
                ax[2].plot(state_history[:j,9], state_history[:j,11], label="LUMIO tudat", color="blue")
                ax[2].plot(state_history_classic_erdem[:j,9], state_history_classic_erdem[:j,11], label="LUMIO halo", color="black")
                ax[2].set_xlabel('VX [m/s]')
                ax[2].set_ylabel('VZ [m/s]')

                fig.legend(["LPF halo", "LPF classic", "LPF tudat", "LUMIO classic", "LUMIO tudat", "LUMIO halo"])
            fig.suptitle("Comparison velocities CRTBP models, inertial frame", fontsize=fontsize)
        plt.tight_layout()
        # plt.show()


        fig7, axs = plt.subplots(6, 1, figsize=(9,10), constrained_layout=True)

        axs[0].set_title("Position states tudat")
        axs[0].set_ylabel(r"$||\mathbf{r}||$ [m]")
        axs[0].plot(epochs, np.linalg.norm(state_history[:,0:3], axis=1), label="LPF w.r.t. Earth", color="red")
        axs[0].plot(epochs, np.linalg.norm(state_history[:,6:9], axis=1), label="LUMIO w.r.t. Earth", color="blue")
        axs[0].plot(epochs, np.linalg.norm(dependent_variables_history[:,0:3], axis=1), label="Moon w.r.t. Earth", color="gray")
        axs[0].plot(epochs, np.linalg.norm(state_history[:,0:3]-dependent_variables_history[:,0:3], axis=1), label="LPF w.r.t. Moon", color="red", ls="--")
        axs[0].plot(epochs, np.linalg.norm(state_history[:,6:9]-dependent_variables_history[:,0:3], axis=1), label="LUMIO w.r.t. Moon", color="blue", ls="--")

        axs[1].set_title("Position states classic")
        axs[1].set_ylabel(r"$||\mathbf{r}||$ [m]")
        axs[1].plot(epochs, np.linalg.norm(state_history_classic[:,0:3], axis=1), label="LPF w.r.t. Earth", color="red")
        axs[1].plot(epochs, np.linalg.norm(state_history_classic[:,6:9], axis=1), label="LUMIO w.r.t. Earth", color="blue")
        axs[1].plot(epochs, np.linalg.norm(dependent_variables_history_classic[:,6:9], axis=1), label="Moon w.r.t. Earth", color="gray")
        axs[1].plot(epochs, np.linalg.norm(state_history_classic[:,0:3]-dependent_variables_history_classic[:,6:9], axis=1), label="LPF w.r.t. Moon", color="red", ls="--")
        axs[1].plot(epochs, np.linalg.norm(state_history_classic[:,6:9]-dependent_variables_history_classic[:,6:9], axis=1), label="LUMIO w.r.t. Moon", color="blue", ls="--")

        axs[2].set_title("Velocity states tudat")
        axs[2].set_ylabel(r"$||\mathbf{v}||$ [m/s]")
        axs[2].plot(epochs, np.linalg.norm(state_history[:,3:6], axis=1), label="LPF w.r.t. Earth", color="red")
        axs[2].plot(epochs, np.linalg.norm(state_history[:,9:12], axis=1), label="LUMIO w.r.t. Earth", color="blue")
        axs[2].plot(epochs, np.linalg.norm(dependent_variables_history[:,3:6], axis=1), label="Moon w.r.t. Earth", color="gray")
        axs[2].plot(epochs, np.linalg.norm(state_history[:,3:6]-dependent_variables_history[:,3:6], axis=1), label="LPF w.r.t. Moon", color="red", ls="--")
        axs[2].plot(epochs, np.linalg.norm(state_history[:,9:12]-dependent_variables_history[:,3:6], axis=1), label="LUMIO w.r.t. Moon", color="blue", ls="--")

        axs[3].set_title("Velocity states classic")
        axs[3].set_ylabel(r"$||\mathbf{v}||$ [m/s]")
        axs[3].plot(epochs, np.linalg.norm(state_history_classic[:,3:6], axis=1), label="LPF w.r.t. Earth", color="red")
        axs[3].plot(epochs, np.linalg.norm(state_history_classic[:,9:12], axis=1), label="LUMIO w.r.t. Earth", color="blue")
        axs[3].plot(epochs, np.linalg.norm(dependent_variables_history_classic[:,9:12], axis=1), label="Moon w.r.t. Earth", color="gray")
        axs[3].plot(epochs, np.linalg.norm(state_history_classic[:,3:6]-dependent_variables_history_classic[:,9:12], axis=1), label="LPF w.r.t. Moon", color="red", ls="--")
        axs[3].plot(epochs, np.linalg.norm(state_history_classic[:,9:12]-dependent_variables_history_classic[:,9:12], axis=1), label="LUMIO w.r.t. Moon", color="blue", ls="--")

        axs[4].set_title("Difference position states")
        axs[4].set_yscale("log")
        axs[4].set_ylabel(r"$||\mathbf{r}_{tudat}-\mathbf{r}_{classic}||$ [m]")
        axs[4].plot(epochs, np.linalg.norm(state_history[:,0:3]-state_history_classic[:,0:3], axis=1), label="LPF w.r.t. Earth", color="red")
        axs[4].plot(epochs, np.linalg.norm(state_history[:,6:9]-state_history_classic[:,6:9], axis=1), label="LUMIO w.r.t. Earth", color="blue")
        axs[4].plot(epochs, np.linalg.norm(dependent_variables_history[:,0:3]-dependent_variables_history_classic[:,6:9], axis=1), label="Moon w.r.t. Earth", color="gray")
        axs[4].plot(epochs, np.linalg.norm((state_history[:,0:3]-dependent_variables_history[:,0:3])-(state_history_classic[:,0:3]-dependent_variables_history_classic[:,6:9]), axis=1), label="LPF w.r.t. Moon", color="red")
        axs[4].plot(epochs, np.linalg.norm((state_history[:,6:9]-dependent_variables_history[:,0:3])-(state_history_classic[:,6:9]-dependent_variables_history_classic[:,6:9]), axis=1), label="LUMIO w.r.t. Moon", color="blue")

        axs[5].set_title("Difference velocity states")
        axs[5].set_yscale("log")
        axs[5].set_ylabel(r"$||\mathbf{v}_{tudat}-\mathbf{v}_{classic}||$ [m/s]")
        axs[5].plot(epochs, np.linalg.norm(state_history[:,3:6]-state_history_classic[:,3:6], axis=1), label="LPF w.r.t. Earth", color="red")
        axs[5].plot(epochs, np.linalg.norm(state_history[:,9:12]-state_history_classic[:,9:12], axis=1), label="LUMIO w.r.t. Earth", color="blue")
        axs[5].plot(epochs, np.linalg.norm(dependent_variables_history[:,3:6]-dependent_variables_history_classic[:,9:12], axis=1), label="Moon w.r.t. Earth", color="gray")
        axs[5].plot(epochs, np.linalg.norm((state_history[:,3:6]-dependent_variables_history[:,3:6])-(state_history_classic[:,3:6]-dependent_variables_history_classic[:,9:12]), axis=1), label="LPF w.r.t. Moon", color="red")
        axs[5].plot(epochs, np.linalg.norm((state_history[:,9:12]-dependent_variables_history[:,3:6])-(state_history_classic[:,9:12]-dependent_variables_history_classic[:,9:12]), axis=1), label="LUMIO w.r.t. Moon", color="blue")

        plt.suptitle("State comparison of classic and tudat CRTBP models. Time step: "+str(propagation_time)+" days")
        [ax.grid(alpha=0.5, linestyle='--') for ax in axs]
        [ax.set_yscale("log") for ax in axs]
        lines, labels = axs[3].get_legend_handles_labels()
        plt.legend()
        # plt.show()

        utils.save_figures_to_folder("test_validation_crtbp", extras, [fig1, fig2, fig3, fig4, fig5, fig6, fig7], [simulation_start_epoch_MJD, propagation_time])
        utils.save_figures_to_folder("test_validation_crtbp", extras, [fig1_3d], [simulation_start_epoch_MJD, propagation_time], save_to_report=False)



class TestOutputsDynamicModels:

    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time",
    [
        # (60390, 14),
    ])

    def test_loading_time_models(self, simulation_start_epoch_MJD, propagation_time, extras):

        time_dict = {}
        package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}
        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time)
        print(dynamic_model_objects)
        for model_type, model_names in dynamic_model_objects.items():
            for model_name, dynamic_models in model_names.items():
                time_list = []
                for dynamic_model in dynamic_models:
                    start_time = time.time()
                    _, _ = dynamic_model.get_propagated_orbit()
                    time_list.append(time.time()-start_time)
                time_dict[model_name] = time_list

        fig, axs = plt.subplots(1, len(list(time_dict.keys())), figsize=(10, 4), sharey=True)
        for i, (key, values) in enumerate(time_dict.items()):
            axs[i].bar(range(1, len(values)+1), values, label=key)
            axs[i].set_xlabel(key)
            axs[i].set_xticks(range(1, 1+max([len(value) for value in time_dict.values()])))
            axs[i].set_yscale("log")

        axs[0].set_ylabel('Run time [s]')
        fig.suptitle(f"Run time dynamic models, {simulation_start_epoch_MJD} MJD, {propagation_time} days")

        utils.save_figures_to_folder("test_loading_time_models", extras, [fig], [simulation_start_epoch_MJD, propagation_time])


    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time, durations",
    [
        # (60390, 50, [7, 14, 28, 42, 49]),
    ])

    def test_difference_high_and_low_fidelity(self, simulation_start_epoch_MJD, propagation_time, durations, extras, step_size=0.001):

        package_dict={"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]}
        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time, package_dict=package_dict)

        # Pick only the first of teach model name
        dynamic_models = utils.get_first_of_model_types(dynamic_model_objects)

        # Adjust to match initial state
        # dynamic_models[0] = low_fidelity.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=utils.synodic_initial_state)

        # Extract simulation histories of classic CRTBP halo continuation model
        synodic_state_history_erdem = validation_LUMIO.get_synodic_state_history_erdem()[:int(propagation_time/0.001),1:]

        epochs_classic_erdem, state_history_classic_erdem, dependent_variables_history_classic_erdem = \
            FrameConverter.SynodicToInertialHistoryConverter(dynamic_models[0], step_size=step_size).get_results(synodic_state_history_erdem)

        # Generate figures
        fig1, axs1 = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained")
        fig2, axs2 = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained")
        fig3, axs3 = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained")
        fig4, axs4 = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained")

        figs_list = [[fig1, fig2],[fig3, fig4]]
        axs_list = [[axs1, axs2],[axs3, axs4]]

        fontsize = 16
        figure_titles = ["position", "velocity"]
        frames = ["inertial", "synodic"]
        satellite_labels = ["LPF", "LUMIO"]
        subplot_labels = [[r'X [m]', r'Y [m]', r'Z [m]', r'VX [m/s]', r'VY [m/s]', r'VZ [m/s]'],
                            [r'X [-]', r'Y [-]', r'Z [-]', r'VX [-]', r'VY [-]', r'VZ [-]']]
        labels = ["LPF halo", "LUMIO halo", "Moon", "LPF tudat high", "LUMIO tudat high", "LPF tudat low", "LUMIO tudat low"]
        figure_colors = ["lightgray", "lightgray", "gray", "red", "blue", "red", "blue"]

        fig1_3d = plt.figure()
        ax_3d = fig1_3d.add_subplot(111, projection='3d')
        ax_3d.suptitle(f"Tudat high versus low models, {propagation_time} days")


        for m, dynamic_model in enumerate(dynamic_models[:2]):

            # Extra simulation histories of tudat models
            epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
                Interpolator.Interpolator(step_size=step_size).get_propagator_results(dynamic_model)

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

        utils.save_figures_to_folder("test_difference_high_and_low_fidelity", extras, [fig1_3d], [simulation_start_epoch_MJD, propagation_time], save_to_report=False)
        utils.save_figures_to_folder("test_difference_high_and_low_fidelity", extras, list(itertools.chain(*figs_list)), [simulation_start_epoch_MJD, propagation_time])



    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time",
    [
        (60390, 1)
        # (60390, 10),
    ])

    def test_difference_to_reference_model(self, simulation_start_epoch_MJD, propagation_time, extras, step_size=0.01):

        # Generate plot settings
        fontsize = 12
        figs = []
        axs = []
        for i in range(2):
            fig, ax = plt.subplots(2, 1, figsize=(14, 5), layout="constrained", sharex=True)
            figs.append(fig)
            axs.append(ax)
        ylabels = [r"||$\Delta \mathbf{r}$||",r"||$\Delta \mathbf{v}$||"]
        colors = [[["black"], ["black"]],[["darkred", "red", "darksalmon", "peachpuff"],["steelblue", "blue", "deepskyblue", "lightblue"]]]
        satellite_labels = ["LPF", "LUMIO"]
        model_name_abbreviations = [["CRTBP"],["PM", "PM SRP", "SH", "SH SRP"]]
        markers = ["o", "v", "^", "<", ">", "s", "h", "D", "*", "p","8"]

        # Generate dynamic model objects
        run_only_first = False
        package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}
        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time)
        for f, fig in enumerate(figs):
            ax = axs[f]
            legend_labels = []
            for m, (model_type, model_names) in enumerate(dynamic_model_objects.items()):

                labels = []
                for i, (model_name, dynamic_models) in enumerate(model_names.items()):

                    if run_only_first:
                        dynamic_models = list(dynamic_models[:1])
                    for d, dynamic_model in enumerate(dynamic_models):

                        # Extract simulation histories numerical solution
                        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
                            Interpolator.Interpolator(epoch_in_MJD=True, step_size=step_size).get_propagator_results(dynamic_model)

                        reference_state_history = np.concatenate((validation_LUMIO.get_reference_state_history(simulation_start_epoch_MJD, propagation_time, step_size=step_size, satellite=dynamic_model.name_ELO, get_full_history=True),
                                                                validation_LUMIO.get_reference_state_history(simulation_start_epoch_MJD, propagation_time, step_size=step_size, satellite=dynamic_model.name_LPO, get_full_history=True)),
                                                                axis=1)

                        data_to_plot = state_history-reference_state_history

                        for k in range(2):
                            ax[0+k].plot(epochs, np.linalg.norm(data_to_plot[:,6*f+3*k:6*f+3*k+3], axis=1), color=colors[m][f][i], marker=markers[d], label=model_name, markersize=3)

                        labels.append(model_name_abbreviations[m][i]+" "+str(d+1))
                        print(labels)

                legend_labels.append(labels)

                for i in range(2):
                    ax[i].grid(alpha=0.5, linestyle='--')
                    ax[i].set_ylabel(ylabels[i%2])
                    ax[i].set_yscale("log")

            fig.legend(list(np.concatenate(legend_labels)), ncol=3, title="Models", loc="upper left")
            fig.suptitle(f"Absolute difference dynamic models and reference, {satellite_labels[f]}", fontsize=fontsize)
            plt.xlabel(f"Epoch since first measurement", fontsize=fontsize)

        plt.show()

        # utils.save_figures_to_folder("test_difference_to_reference_model", extras, figs, [simulation_start_epoch_MJD, propagation_time])




        package_dict={"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]}
        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time, package_dict=package_dict)

        figs = []
        axs = []
        for i, (model_type, model_names) in enumerate(dynamic_model_objects.items()):

            model_names = model_names.keys()

            fig_model_type = []
            axs_model_type = []
            for j in range(len(model_names)):
                fig, ax = plt.subplots(6, 1, figsize=(13, 7), sharex=True)
                fig_model_type.append(fig)
                axs_model_type.append(ax)

            figs.append(fig_model_type)
            axs.append(axs_model_type)

            for k, model_name in enumerate(model_names):
                for dynamic_model in dynamic_model_objects[model_type][model_name]:

                    # Extract simulation histories numerical solution
                    epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
                        Interpolator.Interpolator(epoch_in_MJD=True, step_size=step_size).get_propagator_results(dynamic_model)

                    reference_state_history = np.concatenate((validation_LUMIO.get_reference_state_history(simulation_start_epoch_MJD, propagation_time, step_size=step_size, satellite=dynamic_model.name_ELO, get_full_history=True),
                                                            validation_LUMIO.get_reference_state_history(simulation_start_epoch_MJD, propagation_time, step_size=step_size, satellite=dynamic_model.name_LPO, get_full_history=True)),
                                                            axis=1)

                    fig1_3d = plt.figure()
                    ax = fig1_3d.add_subplot(111, projection='3d')
                    plt.title("Comparison states tudat and reference")
                    plt.plot(state_history[:,0], state_history[:,1], state_history[:,2], label="LPF tudat", color="red")
                    plt.plot(state_history[:,6], state_history[:,7], state_history[:,8], label="LUMIO tudat", color="blue")
                    plt.plot(reference_state_history[:,0], reference_state_history[:,1], reference_state_history[:,2], label="LPF reference", color="salmon")
                    plt.plot(reference_state_history[:,6], reference_state_history[:,7], reference_state_history[:,8], label="LUMIO reference", color="lightskyblue")
                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Y [m]')
                    ax.set_zlabel('Z [m]')
                    plt.legend()
                    # plt.show()

                    # Define the titles and data for the subplots
                    fontsize = 16
                    subplot_ylabels = [r'$\Delta$X [m]', r'$\Delta$Y [m]', r'$\Delta$Z [m]', \
                                       r'$\Delta$VX [m/s]', r'$\Delta$VY [m/s]', r'$\Delta$VZ [m/s]']
                    subplot_labels = ["LPF", "LUMIO"]
                    data_to_plot = [state_history-reference_state_history]

                    # Plot the state histories for each entry across all models
                    for state_index in range(6):
                        axs[i][k][state_index].plot(epochs, data_to_plot[0][:,state_index], color="red")
                        axs[i][k][state_index].plot(epochs, data_to_plot[0][:,6+state_index], color="blue")

                for state_index in range(6):
                    axs[i][k][state_index].set_ylabel(subplot_ylabels[state_index])
                    axs[i][k][state_index].grid(alpha=0.5, linestyle='--')
                    axs[i][k][state_index].ticklabel_format(axis='y', scilimits=(0,0))
                    figs[i][k].suptitle(f"State difference w.r.t. reference, {model_type}, {model_name}", fontsize=fontsize)
                    axs[i][k][-1].set_xlabel(f"Epoch (MJD)")

                figs[i][k].legend(subplot_labels)
                plt.tight_layout()
        # plt.show()

        utils.save_figures_to_folder("test_difference_to_reference_model", extras, [fig1_3d], [simulation_start_epoch_MJD, propagation_time], save_to_report=False)
        utils.save_figures_to_folder("test_difference_to_reference_model", extras, list(np.concatenate(figs)), [simulation_start_epoch_MJD, propagation_time])







    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time",
    [
        (60390, 14),
    ])

    def test_observability_effectiveness(self, simulation_start_epoch_MJD, propagation_time, extras):

        # Plot the observability history and compare them to the different models
        fig, axs = plt.subplots(2, 1, figsize=(10, 5))

        package_dict={"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass"]}
        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time, package_dict=package_dict)
        custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
                                        1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])
        dynamic_model_objects["low_fidelity"]["three_body_problem"][0] = low_fidelity.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state)
        estimation_model_objects = utils.get_estimation_model_objects(estimation_model, dynamic_model_objects)

        for estimation_model in utils.convert_model_objects_to_list(estimation_model_objects):

            estimation_results = estimation_model.get_estimation_results()
            observations_range = estimation_results[-2][observation.one_way_range_type]
            observations_doppler = estimation_results[-2][observation.one_way_instantaneous_doppler_type]

            epoch_range, information_matrix_history_range = utils.convert_dictionary_to_array(observations_range)
            epoch_doppler, information_matrix_history_doppler = utils.convert_dictionary_to_array(observations_doppler)
            epoch_range, epoch_doppler = utils.convert_epochs_to_MJD(epoch_range), utils.convert_epochs_to_MJD(epoch_doppler)

            # Define the titles for the subplots
            fontsize = 16
            subplot_titles = ["Range measurements", "Range-rate measurements"]
            subplot_labels = ["LPF", "LUMIO"]
            data_to_plot = [epoch_range, information_matrix_history_range, epoch_doppler, information_matrix_history_doppler]

            # Iterate through subplots and data
            for i, ax in enumerate(axs):
                for data in data_to_plot:
                    epoch = data_to_plot[i*2]
                    ax.plot(epoch, np.max(np.linalg.eigvals(data_to_plot[1+i*2][:,6:9,6:9]), axis=1, keepdims=True), color="blue")
                    ax.plot(epoch, np.max(np.linalg.eigvals(data_to_plot[1+i*2][:,0:3,0:3]), axis=1, keepdims=True), color="red")
                    ax.set_xlabel(r"Time since start propagation")
                    ax.set_ylabel(r"$\sqrt{\max(\lambda_r(t))}$ [-]")
                    # ax.set_xlim(min(epoch), max(epoch))
                    ax.set_title(subplot_titles[i])

                ax.set_yscale("log")
                ax.legend(subplot_labels)
                ax.grid(alpha=0.5, linestyle='--')

        plt.suptitle("Observability effectiveness comparison of high and low fidelity models. Time step: "+str(propagation_time)+" days", fontsize=fontsize)
        plt.tight_layout()
        # plt.show()

        utils.save_figures_to_folder("test_observability_effectiveness", extras, [fig], [simulation_start_epoch_MJD, propagation_time])
