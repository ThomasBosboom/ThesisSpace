# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
from src.dynamic_models.low_fidelity.integration_settings import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import EstimationModel



class TestFrameConversions:

    simulation_start_epoch_MJD = [60390]
    propagation_time = [28]
    tolerance = [1e-3]

    all_combinations = itertools.product(simulation_start_epoch_MJD, propagation_time, tolerance)
    @pytest.mark.parametrize("simulation_start_epoch_MJD, propagation_time, tolerance", all_combinations)
    def test_validation_crtbp(self, simulation_start_epoch_MJD, propagation_time, tolerance, extras, step_size = 0.001):

        # Define initial state for both tudatpy and classic simulations
        custom_initial_state = np.array([0.98512,0.0014765,	0.0049255,	-0.8733,	-1.6119,	0,	\
            1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])

        # Generate LowFidelityDynamicModel object only
        dynamic_model = low_fidelity.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state)

        # Extract simulation histories tudatpy solution
        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(dynamic_model, step_size=step_size).get_results()

        # Extract synodic simulation histories classical solution
        epochs_synodic, synodic_state_history = validation_LUMIO.get_synodic_state_history(constants.GRAVITATIONAL_CONSTANT,
                                                                                           dynamic_model.bodies.get("Earth").mass,
                                                                                           dynamic_model.bodies.get("Moon").mass,
                                                                                           dynamic_model.distance_between_primaries,
                                                                                           propagation_time,
                                                                                           step_size,
                                                                                           custom_initial_state=custom_initial_state)

        # Extract converted inertial states (works only with low-fidelity dynamic model)
        epochs_classic, state_history_classic, dependent_variables_history_classic = \
            FrameConverter.SynodicToInertialHistoryConverter(dynamic_model, step_size=step_size).get_results(synodic_state_history)

        # Convert back to synodic
        epochs_synodic, state_history_synodic = \
            FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=step_size).get_results(state_history)

        print("actual initial state: ", custom_initial_state)
        print("converted back: ", state_history_synodic)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title("test figure")
        plt.plot(state_history_synodic[:,0], state_history_synodic[:,1], state_history_synodic[:,2])
        plt.plot(state_history_synodic[:,6], state_history_synodic[:,7], state_history_synodic[:,8])
        # plt.plot(state_history_classic[:,0], state_history_classic[:,1], state_history_classic[:,2])
        # plt.plot(state_history_classic[:,6], state_history_classic[:,7], state_history_classic[:,8])
        # plt.show()



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title("Comparison tudat versus classical CRTBP")
        plt.plot(state_history[:,0], state_history[:,1], state_history[:,2])
        plt.plot(state_history[:,6], state_history[:,7], state_history[:,8])
        plt.plot(state_history_classic[:,0], state_history_classic[:,1], state_history_classic[:,2])
        plt.plot(state_history_classic[:,6], state_history_classic[:,7], state_history_classic[:,8])
        # plt.show()

        synodic_state_history_erdem = validation_LUMIO.get_synodic_state_history_erdem()[:,1:]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title("Comparison Erdem versus own-built CRTBP")
        plt.plot(synodic_state_history[:,0], synodic_state_history[:,1], synodic_state_history[:,2])
        plt.plot(synodic_state_history[:,6], synodic_state_history[:,7], synodic_state_history[:,8])
        plt.plot(synodic_state_history_erdem[:,0], synodic_state_history_erdem[:,1], synodic_state_history_erdem[:,2])
        plt.plot(synodic_state_history_erdem[:,6], synodic_state_history_erdem[:,7], synodic_state_history_erdem[:,8])
        # plt.show()


        # Create a figure and three subplots side by side
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # axs[0].plot(state_history_synodic[:,0], state_history_synodic[:,1], label="LPF", color="red", ls="--")
        # axs[0].plot(synodic_state_history[:,0], synodic_state_history[:,1], label="LPF classic", color="black")
        axs[0].plot(state_history_synodic[:,6], state_history_synodic[:,7], label="LUMIO tudat", color="blue", ls="--")
        axs[0].plot(synodic_state_history[:,6], synodic_state_history[:,7], label="LUMIO classic", color="grey")
        axs[0].plot(synodic_state_history_erdem[:,6], synodic_state_history_erdem[:,7], label="LUMIO halo", color="black")
        axs[0].grid(alpha=0.5, linestyle='--')
        axs[0].set_title('Plot 1')

        axs[1].plot(state_history_synodic[:,7], state_history_synodic[:,8], label="LUMIO tudat", color="blue", ls="--")
        axs[1].plot(synodic_state_history[:,7], synodic_state_history[:,8], label="LUMIO classic", color="grey")
        axs[1].plot(synodic_state_history_erdem[:,7], synodic_state_history_erdem[:,8], label="LUMIO halo", color="black")
        axs[1].grid(alpha=0.5, linestyle='--')
        axs[1].set_title('Plot 2')

        axs[2].plot(state_history_synodic[:,8], state_history_synodic[:,6], label="LUMIO tudat", color="blue", ls="--")
        axs[2].plot(synodic_state_history[:,8], synodic_state_history[:,6], label="LUMIO classic", color="grey")
        axs[2].plot(synodic_state_history_erdem[:,8], synodic_state_history_erdem[:,6], label="LUMIO halo", color="black")
        axs[2].grid(alpha=0.5, linestyle='--')
        axs[2].set_title('Plot 3')

        plt.suptitle("Comparison tudat and classical CRTBP synodic non-dim rotating frame")
        plt.tight_layout()
        plt.legend()
        plt.show()





        # Calculate the percentage difference for each entry
        percentage_difference = np.abs((dependent_variables_history[0,0:6] - dependent_variables_history_classic[0,6:12])/dependent_variables_history_classic[0,6:12]) * 100
        print(percentage_difference)
        percentage_difference = np.abs((state_history[0,0:6] - state_history_classic[0,0:6])/state_history_classic[0,0:6]) * 100
        print(percentage_difference)
        percentage_difference = np.abs((state_history[0,6:12] - state_history_classic[0,6:12])/state_history_classic[0,6:12]) * 100
        print(percentage_difference)
        percentage_difference = np.abs((state_history[0,0:6]-dependent_variables_history[0,6:12] - (state_history_classic[0,0:6]-dependent_variables_history[0,6:12]))/(state_history_classic[0,0:6]-dependent_variables_history[0,6:12])) * 100
        print(percentage_difference)



        Keplerian_state_Moon_Earth_tudat = element_conversion.cartesian_to_keplerian(dependent_variables_history[0,0:6], dynamic_model.gravitational_parameter_primary+dynamic_model.gravitational_parameter_secondary)
        Keplerian_state_Moon_Earth_classic = element_conversion.cartesian_to_keplerian(dependent_variables_history_classic[0,6:12], dynamic_model.gravitational_parameter_primary+dynamic_model.gravitational_parameter_secondary)

        Keplerian_state_LPF_Moon_tudat = element_conversion.cartesian_to_keplerian(state_history[0,0:6]-dependent_variables_history[0,0:6], dynamic_model.gravitational_parameter_secondary)
        Keplerian_state_LPF_Moon_classic = element_conversion.cartesian_to_keplerian(state_history_classic[0,0:6]-dependent_variables_history_classic[0,6:12], dynamic_model.gravitational_parameter_secondary)

        percentage_difference = np.abs((Keplerian_state_Moon_Earth_tudat-Keplerian_state_Moon_Earth_classic)/Keplerian_state_Moon_Earth_classic) * 100
        print(percentage_difference)
        percentage_difference = np.abs((Keplerian_state_LPF_Moon_tudat-Keplerian_state_LPF_Moon_classic)/Keplerian_state_LPF_Moon_classic) * 100
        print(percentage_difference)

        Keplerian_state_LPF_Moon_frontiers = np.array([5737.4e3, 6.1e-1, 57.83*np.pi/180, 90*np.pi/180, 61.55*np.pi/180, 0])
        print(Keplerian_state_LPF_Moon_frontiers)

        percentage_difference = np.abs((Keplerian_state_LPF_Moon_frontiers-Keplerian_state_LPF_Moon_tudat)/Keplerian_state_LPF_Moon_tudat) * 100
        print(percentage_difference)
        percentage_difference = np.abs((Keplerian_state_LPF_Moon_frontiers-Keplerian_state_LPF_Moon_classic)/Keplerian_state_LPF_Moon_classic) * 100
        print(percentage_difference)


        print("==== Cartesian states tudat ====")
        print("Moon w.r.t. Earth: ", dependent_variables_history[0,0:6])
        print("LPF w.r.t. Earth: ", state_history[0,0:6])
        print("LUMIO w.r.t. Earth: ", state_history[0,6:12])
        # print("Moon w.r.t. Earth: ", dependent_variables_history[0,0:6])
        print("LPF w.r.t. Moon: ", state_history[0,0:6]-dependent_variables_history[0,6:12])
        print("LUMIO w.r.t. Moon: ", state_history[0,6:12]-dependent_variables_history[0,6:12])

        print("==== Cartesian states classic ====")
        print("Moon w.r.t. Earth: ", dependent_variables_history_classic[0,6:12])
        print("LPF w.r.t. Earth: ", state_history_classic[0,0:6])
        print("LUMIO w.r.t. Earth: ", state_history_classic[0,6:12])
        # print("Moon w.r.t. Earth, tudat: ", dependent_variables_history_classic[0,6:12])
        print("LPF w.r.t. Moon: ", state_history_classic[0,0:6]-dependent_variables_history[0,6:12])
        print("LUMIO w.r.t. Moon: ", state_history_classic[0,6:12]-dependent_variables_history[0,6:12])

        print("==== Keplerian states tudat ====")
        print("Moon w.r.t. Earth: ", element_conversion.cartesian_to_keplerian(dependent_variables_history[0,0:6], dynamic_model.gravitational_parameter_primary+dynamic_model.gravitational_parameter_secondary))
        print("LPF w.r.t. Moon: ", element_conversion.cartesian_to_keplerian(state_history[0,0:6]-dependent_variables_history[0,0:6], dynamic_model.gravitational_parameter_secondary))

        print("==== Keplerian states classic ====")
        print("Moon w.r.t. Earth: ", element_conversion.cartesian_to_keplerian(dependent_variables_history_classic[0,6:12], dynamic_model.gravitational_parameter_primary+dynamic_model.gravitational_parameter_secondary))
        print("LPF w.r.t. Moon: ", element_conversion.cartesian_to_keplerian(state_history_classic[0,0:6]-dependent_variables_history_classic[0,6:12], dynamic_model.gravitational_parameter_secondary))



        fig1, axs = plt.subplots(6, 1, figsize=(9,10), constrained_layout=True)

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
        fig1.legend(lines, labels, loc='center right', bbox_to_anchor=(1.1, 0.5))
        # plt.show()




        keplerian_state_Moon_Earth = np.empty(np.shape(state_history_classic[:,:6]))
        keplerian_state_LPF_Moon = keplerian_state_Moon_Earth.copy()
        keplerian_state_Moon_Earth_classic = keplerian_state_Moon_Earth.copy()
        keplerian_state_LPF_Moon_classic = keplerian_state_Moon_Earth.copy()
        for epoch, state in enumerate(keplerian_state_Moon_Earth):
            keplerian_state_Moon_Earth[epoch] = element_conversion.cartesian_to_keplerian(dependent_variables_history[epoch,0:6], dynamic_model.gravitational_parameter_primary+dynamic_model.gravitational_parameter_secondary)
            keplerian_state_LPF_Moon[epoch] = element_conversion.cartesian_to_keplerian(state_history[epoch,0:6]-dependent_variables_history[epoch,0:6], dynamic_model.gravitational_parameter_secondary)
            keplerian_state_Moon_Earth_classic[epoch] = element_conversion.cartesian_to_keplerian(dependent_variables_history_classic[epoch,6:12], dynamic_model.gravitational_parameter_primary+dynamic_model.gravitational_parameter_secondary)
            keplerian_state_LPF_Moon_classic[epoch] = element_conversion.cartesian_to_keplerian(state_history_classic[epoch,0:6]-dependent_variables_history_classic[epoch,6:12], dynamic_model.gravitational_parameter_secondary)


        fig2, axs = plt.subplots(2, 1, figsize=(11.69,8.27), constrained_layout=True)
        axs[0].set_title("Moon w.r.t. Earth")
        axs[0].plot(epochs, dependent_variables_history[:,18:24], label=[r"$a$", r"$e$", r"$i$", r"$\omega$", r"$\Omega$", r"$\theta$"])
        axs[1].set_title("LPF w.r.t. Moon")
        axs[1].plot(epochs, dependent_variables_history[:,24:30], label=[r"$a$", r"$e$", r"$i$", r"$\omega$", r"$\Omega$", r"$\theta$"])

        plt.suptitle("State comparison of classic and tudat CRTBP model parameters. Time step: "+str(propagation_time)+" days")
        [ax.grid(alpha=0.5, linestyle='--') for ax in axs]
        plt.suptitle("Keplerian states")
        fig2.legend(loc='lower center', bbox_to_anchor=(0.0, -1))
        plt.legend()
        # plt.show()


        # Create a figure and three subplots side by side
        fig3, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].plot(state_history[:,0], state_history[:,1], label="LPF", color="red", ls="--")
        axs[0].plot(state_history_classic[:,0], state_history_classic[:,1], label="LPF classic", color="black")
        axs[0].plot(state_history[:,6], state_history[:,7], label="LUMIO", color="blue", ls="--")
        axs[0].plot(state_history_classic[:,6], state_history_classic[:,7], label="LUMIO classic", color="grey")
        axs[0].grid(alpha=0.5, linestyle='--')
        axs[0].set_title('Plot 1')

        axs[1].plot(state_history[:,1], state_history[:,2])
        axs[1].plot(state_history_classic[:,1], state_history_classic[:,2])
        axs[1].grid(alpha=0.5, linestyle='--')
        axs[1].set_title('Plot 2')

        axs[2].plot(state_history[:,2], state_history[:,0])
        axs[2].plot(state_history_classic[:,2], state_history_classic[:,0])
        axs[2].grid(alpha=0.5, linestyle='--')
        axs[2].set_title('Plot 3')

        plt.tight_layout()
        plt.legend()
        plt.show()

        utils.save_figures_to_folder("test_initial_state_conversion", extras, [fig1, fig2, fig3], [simulation_start_epoch_MJD, propagation_time, tolerance])




class TestOutputsDynamicalModels:

    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time",
    [
        # (60390, 10),
    ])

    def test_estimation_models(self, simulation_start_epoch_MJD, propagation_time):

        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time)
        estimation_model_objects = utils.get_estimation_model_objects(EstimationModel, dynamic_model_objects)

        for estimation_model in utils.convert_model_objects_to_list(estimation_model_objects):

            print(estimation_model.get_estimation_results()[0].correlations)


    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time",
    [
        # (60390, 10),
        # (60395, 10),
        # (60400, 10),
    ])

    def test_compare_dynamic_models(self, simulation_start_epoch_MJD, propagation_time, extras):

        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time)

        # Create a figure and three subplots side by side
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for dynamic_model in utils.convert_model_objects_to_list(dynamic_model_objects):

            setattr(dynamic_model, "current_coefficient_set", propagation_setup.integrator.CoefficientSets.rkf_89)

            # Extract simulation histories numerical solution
            epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
                Interpolator.Interpolator(dynamic_model, step_size=0.005, epoch_in_MJD=False).get_results()

            # Define the titles and data for the subplots
            subplot_titles = ['Plot 1', 'Plot 2', 'Plot 3']
            data_to_plot = [dependent_variables_history[:, 6:9]]

            # Iterate through subplots and data
            for i, ax in enumerate(axs):
                for data in data_to_plot:
                    ax.plot(data[:, i % 3], data[:, (i + 1) % 3], label=f"{dynamic_model}")

                ax.grid(alpha=0.5, linestyle='--')
                ax.set_title(subplot_titles[i])

            assert dynamic_model.name_ELO == "LPF"

        plt.legend()

        # Create a folder named after the function
        folder_name = "test_compare_dynamic_models"
        os.makedirs(folder_name, exist_ok=True)
        figure_path = os.path.join(folder_name, f"{simulation_start_epoch_MJD}_{propagation_time}.png")
        fig.savefig(figure_path)
        extras.append(pytest_html.extras.png(figure_path))








    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time",
    [
        # (60390, 20),
        # (60395, 10),
        # (60400, 10),
    ])

    def test_observability_effectiveness(self, simulation_start_epoch_MJD, propagation_time):

        # Plot the observability history and compare them to the different models
        fig, axs = plt.subplots(2, 1, figsize=(15, 5))

        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time)
        estimation_model_objects = utils.get_estimation_model_objects(EstimationModel, dynamic_model_objects)

        for estimation_model in utils.convert_model_objects_to_list(estimation_model_objects):

            estimation_results = estimation_model.get_estimation_results()
            observations_range = estimation_results[-1][observation.one_way_range_type]
            observations_doppler = estimation_results[-1][observation.one_way_instantaneous_doppler_type]

            epoch_range, information_matrix_history_range = utils.convert_dictionary_to_array(observations_range)
            epoch_doppler, information_matrix_history_doppler = utils.convert_dictionary_to_array(observations_doppler)

            # Define the titles for the subplots
            subplot_titles = ["Range measurements", "Range-rate measurements"]
            subplot_labels = ["LPF", "LUMIO"]

            # Data to plot
            data_to_plot = [epoch_range, information_matrix_history_range, epoch_doppler, information_matrix_history_doppler]

            # Iterate through subplots and data
            for i, ax in enumerate(axs):
                for data in data_to_plot:
                    epoch = data_to_plot[i*2]
                    ax.plot(epoch, np.max(np.linalg.eigvals(data_to_plot[1+i*2][:,6:9,6:9]), axis=1, keepdims=True), color="blue")
                    ax.plot(epoch, np.max(np.linalg.eigvals(data_to_plot[1+i*2][:,0:3,0:3]), axis=1, keepdims=True), color="red")
                    ax.set_xlabel(r"Time since $t_0$"+ " [days] ("+str(simulation_start_epoch_MJD)+" MJD)")
                    ax.set_ylabel(r"$\sqrt{\max(\lambda_r(t))}$ [-]")
                    ax.set_xlim(min(epoch), max(epoch))
                    ax.set_title(subplot_titles[i])

                ax.set_yscale("log")
                ax.legend(subplot_labels)
                ax.grid(alpha=0.5, linestyle='--')

        plt.suptitle("Observability effectiveness comparison of high and low fidelity models. Time step: "+str(propagation_time)+" days")
        plt.tight_layout()
        plt.show()



import numpy as np
import pandas as pd

# Example data (replace these with your actual arrays)
array1 = np.array([1, 2, 3, 4])
array2 = np.array([1.2, 1.8, 3.1, 4.5])

# Calculate the percentage difference for each entry
percentage_difference = np.abs((array2 - array1) / array1) * 100

# Create a DataFrame
data = {'Array 1': array1, 'Array 2': array2, 'Percentage Difference': percentage_difference}
df = pd.DataFrame(data)

# Transpose the DataFrame
df_transposed = df.transpose()

# Export the transposed DataFrame to a LaTeX table
latex_table_transposed = df_transposed.to_latex()

# Display the LaTeX table
print(latex_table_transposed)

