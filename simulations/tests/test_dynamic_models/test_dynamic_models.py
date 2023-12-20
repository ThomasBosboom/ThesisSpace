# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Define path to import src files
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Third party
import pytest
import pytest_html
from tudatpy.kernel import constants
import mpld3

# Own
from src.dynamic_models import validation_LUMIO
from src.dynamic_models import Interpolator, FrameConverter
from src.dynamic_models.low_fidelity.integration_settings import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
# from src.estimation_models import EstimationModel



class TestOutputsDynamicalModels:

    @pytest.mark.parametrize(
    "first_value, second_value",
    [
        (10, 8),
        (8, 6)
    ])

    def test_parametrize(self, first_value, second_value):
        assert first_value - 2 == second_value

# package_dict={"low_fidelity": ["integration_settings"], "high_fidelity": ["point_mass", "point_mass_srp"]}
    def get_dynamic_model_objects(self, simulation_start_epoch_MJD, propagation_time, package_dict={"low_fidelity": ["integration_settings"], "high_fidelity": ["point_mass", "point_mass_srp"]}):

        # Get a list of all relevant packages within 'high_fidelity'
        dynamic_model_objects = {}
        for package_type, package_name_list in package_dict.items():

            sub_dict = {package_name_list[i]: [] for i in range(len(package_name_list))}
            packages_dir = os.path.join(parent_dir, 'src', 'dynamic_models', package_type)

            package_name_counter = 0
            for package_name in package_dict[package_type]:

                package_module_path = f'dynamic_models.{package_type}.{package_name}'
                package_module = __import__(package_module_path, fromlist=[package_name])
                package_files = os.listdir(os.path.join(packages_dir, package_name))

                for file_name in package_files:

                    if file_name.endswith('.py') and not file_name.startswith('__init__'):
                        module_path = f'{package_module_path}.{os.path.splitext(file_name)[0]}'
                        module = __import__(module_path, fromlist=[file_name])

                        if package_type == "low_fidelity":
                            DynamicModel = module.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time)
                        else:
                            DynamicModel = module.HighFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time)

                        sub_dict[package_name_list[package_name_counter]].extend([DynamicModel])
                        dynamic_model_objects[package_type] = sub_dict

                package_name_counter += 1

        return dynamic_model_objects


    @pytest.mark.parametrize(
    "simulation_start_epoch_MJD, propagation_time",
    [
        (60390, 30),
        # (60395, 1),
    ])

    # def test_compare_dynamic_models(self, simulation_start_epoch_MJD, propagation_time, extras):

    #     dynamic_model_objects = self.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time)

    #     # Loop through each key in the outer dictionary
    #     i = 0
    #     for package_type, package_names in dynamic_model_objects.items():
    #         for package_name, dynamic_models in package_names.items():
    #             for dynamic_model in dynamic_models:

    #                 print(f"dynamic_model: {dynamic_model}")

    #                 # Extract simulation histories numerical solution
    #                 epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
    #                     Interpolator.Interpolator(dynamic_model, step_size=0.005*constants.JULIAN_DAY, epoch_in_MJD=False).get_results()

    #                 # Extra simulation histories analytical solution
    #                 state_history_barycentric_primary = dependent_variables_history[:,:6]*(-dynamic_model.mu)
    #                 state_history_barycentric_secondary = dependent_variables_history[:,:6]*(1-dynamic_model.mu)
    #                 state_history_barycentric = np.add(state_history[:,:6], state_history_barycentric_primary)

    #                 print(epochs, np.shape(epochs))
    #                 # print("state_history", state_history)
    #                 # print("state_history_barycentric", state_history_barycentric)
    #                 # print("difference: ", np.linalg.norm(state_history_barycentric_primary-state_history_barycentric_secondary, axis=1))

    #                 # fig = plt.figure()
    #                 # ax = fig.add_subplot(1,1,1)
    #                 # ax.plot([1,2,3], color="red")
    #                 # ax.plot([2,3,2], color="green")
    #                 # plt.title("figure")

    #                 # # Save the figure
    #                 # figure_path = f"C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/tests/test_dynamic_models/figure{i}.png"
    #                 # fig.savefig(figure_path)

    #                 # extras.append(pytest_html.extras.png(figure_path))


    #                 i += 1

    #         assert dynamic_model.name_ELO == "LPF"
    #         assert dynamic_model.simulation_start_epoch_MJD == simulation_start_epoch_MJD


    def test_low_fidelity_dynamic_models(self, simulation_start_epoch_MJD, propagation_time, extras):

        custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0, \
                                         1.1473302, 0, -0.15142308, 0, -0.21994554, 0])

        # Generate LowFidelityDynamicModel object only
        dynamic_model = low_fidelity.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time, custom_initial_state=custom_initial_state)

        # Define step size that both tudatpy and classic model work run (variable step size epochs are adjusted in Interpolator)
        step_size = 0.001

        # Extract simulation histories tudatpy solution
        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(dynamic_model, step_size=step_size).get_results()

        # Extract simulation histories classical solution

        custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861,  0, \
                                         1.1473302, 0, -0.15142308, 0, -0.21994554, 0])

        epochs_classic, state_history_classic, state_history_primaries = \
            FrameConverter.SynodicToInertialHistoryConverter(dynamic_model, step_size=step_size).get_results(custom_initial_state=custom_initial_state)


        print(epochs, np.shape(epochs))
        print(epochs_classic, np.shape(epochs_classic))
        print("tudatpy: ", state_history[0], "classic: ", state_history_classic[0])

        ax = plt.figure()
        plt.plot(epochs, np.abs(state_history[:,0]-state_history_classic[:,0]), label="LPF x")
        plt.plot(epochs, np.abs(state_history[:,1]-state_history_classic[:,1]), label="LPF y")
        plt.plot(epochs, np.abs(state_history[:,2]-state_history_classic[:,2]),  label="LPF z")

        plt.plot(epochs, np.linalg.norm(state_history[:,:3]-state_history_classic[:,:3], axis=1), label="LPF")
        plt.plot(epochs, np.linalg.norm(state_history[:,6:9]-state_history_classic[:,6:9], axis=1), label="LUMIO")
        # plt.plot(epochs, np.linalg.norm(dependent_variables_history[:,:3]-state_history_primaries[:,:3], axis=1), label="Earth")
        plt.plot(epochs, np.linalg.norm(dependent_variables_history[:,:3]-state_history_primaries[:,6:9], axis=1), label="Moon")
        plt.xlabel("Epoch")
        plt.ylabel("Difference [m]")
        plt.title("Absolute position difference tudatpy versus classical CRTBP")
        ax.legend()
        plt.grid(alpha=0.5)
        plt.yscale("log")
        plt.show()

        ax = plt.figure()
        plt.plot(epochs, np.linalg.norm(state_history[:,3:6]-state_history_classic[:,3:6], axis=1), label="LPF")
        plt.plot(epochs, np.linalg.norm(state_history[:,9:12]-state_history_classic[:,9:12], axis=1), label="LUMIO")
        # plt.plot(epochs, np.linalg.norm(dependent_variables_history[:,3:6]-state_history_primaries[:,3:6], axis=1), label="Earth")
        plt.plot(epochs, np.linalg.norm(dependent_variables_history[:,3:6]-state_history_primaries[:,9:12], axis=1), label="Moon")
        plt.xlabel("Epoch")
        plt.ylabel("Difference [m/s]")
        plt.title("Absolute velocity difference tudatpy versus classical CRTBP")
        ax.legend()
        plt.grid(alpha=0.5)
        plt.yscale("log")
        plt.show()

        ax = plt.figure().add_subplot(projection='3d')
        # plt.plot(dependent_variables_history[:,0], dependent_variables_history[:,1], dependent_variables_history[:,2], label="moon w.r.t earth")
        plt.plot(state_history_classic[:,0], state_history_classic[:,1], state_history_classic[:,2], label="lpf classic")
        plt.plot(state_history[:,0], state_history[:,1], state_history[:,2], label="lpf tudatpy")
        plt.plot(state_history_classic[:,6], state_history_classic[:,7], state_history_classic[:,8], label="lumio classic")
        plt.plot(state_history[:,6], state_history[:,7], state_history[:,8], label="lumio tudatpy")
        plt.plot(state_history_primaries[:,0], state_history_primaries[:,1], state_history_primaries[:,2], label="earth classic")
        plt.plot(state_history_primaries[:,6], state_history_primaries[:,7], state_history_primaries[:,8], label="moon classic")
        plt.plot(dependent_variables_history[:,0], dependent_variables_history[:,1], dependent_variables_history[:,2], label="moon tudatpy")
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_zlabel("Z [km]")
        plt.title("Trajectories in inertial Earth-centered J2000 frame")
        ax.legend()
        plt.axis('equal')
        plt.show()



        # Create a figure and three subplots side by side
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

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
        plt.show()

        # Save the figure
        figure_path = f"C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/tests/test_dynamic_models/comparison.png"
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




# test = TestOutputsDynamicalModels()
# dynamic_model_objects = test.get_dynamic_model_objects(60390, 10)
# model = dynamic_model_objects["high_fidelity"]["point_mass"][0]

# dynamics_simulator, variational_equations_solver = model.get_propagated_orbit()
# print(model)

# print(Interpolator.Interpolator(model).get_results()[0], np.shape(Interpolator.Interpolator(model).get_results()[0]))
# # print(Interpolator.Interpolator(model).get_results()[1], np.shape(Interpolator.Interpolator(model).get_results()[1]))
# # print(Interpolator.Interpolator(model).get_results()[2], np.shape(Interpolator.Interpolator(model).get_results()[2]))
# # print(Interpolator.Interpolator(model).get_results()[3], np.shape(Interpolator.Interpolator(model).get_results()[3]))

# state_history = Interpolator.Interpolator(model).get_results()[1]
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(state_history[:,0], state_history[:,1], state_history[:,2], color="red", label="LPF")
# ax.plot(state_history[:,6], state_history[:,7], state_history[:,8], color="blue", label="LUMIO")
# ax.set_xlabel("X [m]")
# ax.set_ylabel("Y [m]")
# ax.set_zlabel("Z [m]")
# plt.axis('equal')
# plt.legend()
# plt.grid(alpha=0.2)
# plt.tight_layout()
# plt.show()