# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)


# Third party
import pytest
import pytest_html
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
# import mpld3

# Own
from tests import utils
from src.dynamic_models import validation_LUMIO
from src.dynamic_models import Interpolator, FrameConverter
from src.dynamic_models.low_fidelity.integration_settings import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import EstimationModel


class TestOutputsDynamicalModels:

    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time",
    [
        (60390, 10),
    ])

    def test_estimation_models(self, simulation_start_epoch_MJD, propagation_time):

        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time)
        estimation_model_objects = utils.get_estimation_model_objects(EstimationModel, dynamic_model_objects)

        for package_type, package_names in estimation_model_objects.items():
            for package_name, estimation_model_objects in package_names.items():
                for estimation_model_object in estimation_model_objects:

                    print(estimation_model_object.get_estimation_results()[0])


    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time",
    [
        (60390, 10),
        # (60395, 10),
        # (60400, 10),
    ])

    def test_compare_dynamic_models(self, simulation_start_epoch_MJD, propagation_time, extras):

        dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time)

        # Create a figure and three subplots side by side
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for package_type, package_names in dynamic_model_objects.items():
            for package_name, dynamic_models in package_names.items():
                for dynamic_model in dynamic_models:

                    # Create a folder named after the function
                    folder_name = "test_compare_dynamic_models"
                    os.makedirs(folder_name, exist_ok=True)

                    setattr(dynamic_model, "current_coefficient_set", propagation_setup.integrator.CoefficientSets.rkf_89)

                    # Extract simulation histories numerical solution
                    epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
                        Interpolator.Interpolator(dynamic_model, step_size=0.005, epoch_in_MJD=False).get_results()

                    # Define the titles for the subplots
                    subplot_titles = ['Plot 1', 'Plot 2', 'Plot 3']

                    # Data to plot
                    data_to_plot = [dependent_variables_history[:, 6:9]]

                    # Iterate through subplots and data
                    for i, ax in enumerate(axs):
                        for data in data_to_plot:
                            ax.plot(data[:, i % 3], data[:, (i + 1) % 3], label=f"{dynamic_model}")

                        ax.grid(alpha=0.5, linestyle='--')
                        ax.set_title(subplot_titles[i])

                    assert dynamic_model.name_ELO == "LPF"

        plt.legend()
        figure_path = os.path.join(folder_name, f"test_low_fidelity_dynamic_models_{simulation_start_epoch_MJD}_{propagation_time}.png")
        fig.savefig(figure_path)
        extras.append(pytest_html.extras.png(figure_path))


    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time",
    [
        (60390, 10),
        # (60395, 10),
        # (60400, 10),
    ])

    def test_low_fidelity_dynamic_models(self, simulation_start_epoch_MJD, propagation_time, extras):

        custom_initial_state = np.array([0.985141349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0, \
                                         1.1473302, 0, -0.15142308, 0, -0.21994554, 0])

        # Generate LowFidelityDynamicModel object only
        dynamic_model = low_fidelity.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state)

        # Define step size that both tudatpy and classic model work run (variable step size epochs are adjusted in Interpolator)
        step_size = 0.005

        # Extract simulation histories tudatpy solution
        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(dynamic_model, step_size=step_size).get_results()

        # Extract simulation histories classical solution
        epochs_classic, state_history_classic, dependent_variables_history_classic = \
            FrameConverter.SynodicToInertialHistoryConverter(dynamic_model, step_size=step_size).get_results()

        fig1, ax = plt.subplots()
        plt.plot(epochs, np.abs(state_history[:,0]-state_history_classic[:,0]), label="LPF x")
        plt.plot(epochs, np.abs(state_history[:,1]-state_history_classic[:,1]), label="LPF y")
        plt.plot(epochs, np.abs(state_history[:,2]-state_history_classic[:,2]),  label="LPF z")
        plt.xlabel("Epoch")
        plt.ylabel("Difference [m]")
        plt.title("Absolute position difference tudatpy versus classical CRTBP")
        ax.legend()
        plt.grid(alpha=0.5)
        plt.yscale("log")
        # plt.show()

        fig2, ax = plt.subplots()
        plt.plot(epochs, np.linalg.norm(state_history[:,0:3]-state_history_classic[:,0:3], axis=1), label="LPF w.r.t. Earth")
        plt.plot(epochs, np.linalg.norm(state_history[:,6:9]-state_history_classic[:,6:9], axis=1), label="LUMIO w.r.t. Earth")
        plt.plot(epochs, np.linalg.norm((state_history[:,0:3]-dependent_variables_history[:,0:3])-(state_history_classic[:,0:3]-dependent_variables_history_classic[:,6:9]), axis=1), label="LPF w.r.t. Moon")
        plt.plot(epochs, np.linalg.norm((state_history[:,6:9]-dependent_variables_history[:,0:3])-(state_history_classic[:,6:9]-dependent_variables_history_classic[:,6:9]), axis=1), label="LUMIO w.r.t. Moon")
        plt.plot(epochs, np.linalg.norm(np.zeros(np.shape(dependent_variables_history[:,0:3]))-dependent_variables_history_classic[:,0:3], axis=1), label="Earth w.r.t. Earth")
        plt.plot(epochs, np.linalg.norm(dependent_variables_history[:,0:3]-dependent_variables_history_classic[:,6:9], axis=1), label="Moon w.r.t. Earth")
        plt.xlabel("Epoch")
        plt.ylabel("Difference [m]")
        plt.title("Absolute position difference tudatpy versus classical CRTBP")
        ax.legend()
        plt.grid(alpha=0.5)
        plt.yscale("log")
        # plt.show()

        fig3, ax = plt.subplots()
        plt.plot(epochs, np.linalg.norm(state_history[:,3:6]-state_history_classic[:,3:6], axis=1), label="LPF")
        plt.plot(epochs, np.linalg.norm(state_history[:,9:12]-state_history_classic[:,9:12], axis=1), label="LUMIO")
        plt.plot(epochs, np.linalg.norm((state_history[:,3:6]-dependent_variables_history[:,3:6])-(state_history_classic[:,3:6]-dependent_variables_history_classic[:,9:12]), axis=1), label="LPF w.r.t. Moon")
        plt.plot(epochs, np.linalg.norm((state_history[:,9:12]-dependent_variables_history[:,3:6])-(state_history_classic[:,9:12]-dependent_variables_history_classic[:,9:12]), axis=1), label="LUMIO w.r.t. Moon")
        plt.plot(epochs, np.linalg.norm(dependent_variables_history[:,3:6]-dependent_variables_history_classic[:,9:12], axis=1), label="Moon w.r.t. Earth")
        plt.xlabel("Epoch")
        plt.ylabel("Difference [m/s]")
        plt.title("Absolute velocity difference tudatpy versus classical CRTBP")
        ax.legend()
        plt.grid(alpha=0.5)
        plt.yscale("log")
        # plt.show()

        ax = plt.figure().add_subplot(projection='3d')
        plt.plot(state_history_classic[:,0], state_history_classic[:,1], state_history_classic[:,2], label="lpf classic")
        plt.plot(state_history_classic[:,6], state_history_classic[:,7], state_history_classic[:,8], label="lumio classic")
        plt.plot(state_history[:,0], state_history[:,1], state_history[:,2], label="lpf tudatpy")
        plt.plot(state_history[:,6], state_history[:,7], state_history[:,8], label="lumio tudatpy")
        plt.plot(dependent_variables_history_classic[:,0], dependent_variables_history_classic[:,1], dependent_variables_history_classic[:,2], label="earth classic")
        plt.plot(dependent_variables_history_classic[:,6], dependent_variables_history_classic[:,7], dependent_variables_history_classic[:,8], label="moon classic")
        plt.plot(dependent_variables_history[:,0], dependent_variables_history[:,1], dependent_variables_history[:,2], label="moon tudatpy")
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_zlabel("Z [km]")
        plt.title("Trajectories in inertial Earth-centered J2000 frame")
        ax.legend()
        plt.axis('equal')
        # plt.show()


        # Create a figure and three subplots side by side
        fig4, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot on the first subplot
        axs[0].plot(state_history[:,0], state_history[:,1])
        axs[0].plot(state_history_classic[:,0], state_history_classic[:,1])
        axs[0].grid(alpha=0.5, linestyle='--')
        axs[0].set_title('Plot 1')

        # Plot on the second subplot
        axs[1].plot(state_history[:,1], state_history[:,2])
        axs[1].plot(state_history_classic[:,1], state_history_classic[:,2])
        axs[1].grid(alpha=0.5, linestyle='--')
        axs[1].set_title('Plot 2')

        # Plot on the third subplot
        axs[2].plot(state_history[:,2], state_history[:,0])
        axs[2].plot(state_history_classic[:,2], state_history_classic[:,0])
        axs[2].grid(alpha=0.5, linestyle='--')
        axs[2].set_title('Plot 3')

        plt.tight_layout()
        # plt.show()

        # Save the figure
        for i, fig in enumerate([fig1, fig2, fig3, fig4]):
            figure_path = os.path.join(script_directory, f"test_low_fidelity_dynamic_models_{simulation_start_epoch_MJD}_{propagation_time}_{i}.png")
            fig.savefig(figure_path)
            extras.append(pytest_html.extras.png(figure_path))








    # def test_example(self, extras):

    #     # Assume some test logic that generates a plot
    #     for i in range(5):

    #         fig = plt.figure()
    #         ax = fig.add_subplot(1,1,1)
    #         ax.plot([1,2,3], color="red")
    #         ax.plot([2,3,2], color="green")
    #         plt.title("figure")

    #         # Save the figure
    #         figure_path = f"C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/tests/test_dynamic_models/figure{i}.png"
    #         fig.savefig(figure_path)

    #         extras.append(pytest_html.extras.png(figure_path))

    #     assert True




print(os.getcwd())
print(os.path.dirname(os.path.realpath(__file__)))


DynamicModel = high_fidelity_point_mass_01.HighFidelityDynamicModel(60390, 10)

print(DynamicModel.current_coefficient_set)
setattr(DynamicModel, "current_coefficient_set", "CoefficientSets.rkf_89")
print(DynamicModel.current_coefficient_set)