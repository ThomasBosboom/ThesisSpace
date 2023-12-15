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
# import mpld3

# Own
from src.dynamic_models import validation_LUMIO
from src.dynamic_models.low_fidelity.integration_settings import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
# from src.estimation_models import EstimationModel

from src.dynamic_models import Interpolator


class TestOutputsDynamicalModels:

    @pytest.mark.parametrize(
    "first_value, second_value",
    [
        (10, 8),
        (8, 6)
    ])

    def test_parametrize(self, first_value, second_value):
        assert first_value - 2 == second_value

    def get_dynamic_model_objects(self, simulation_start_epoch_MJD, propagation_time, package_dict={"low_fidelity": ["integration_settings"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics"]}):

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
        (60390, 1),
        # (60395, 1),
    ])

    def test_low_fidelity(self, simulation_start_epoch_MJD, propagation_time, extras):

        dynamic_model_objects = self.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time, package_dict={"high_fidelity": ["point_mass", "point_mass_srp"]})

        # Loop through each key in the outer dictionary
        for package_type, package_names in dynamic_model_objects.items():
            for package_name, dynamic_models in package_names.items():
                for dynamic_model in dynamic_models:

                    print(f"dynamic_model: {dynamic_model}")

                    # Extract simulation histories
                    epochs, state_history, dependent_variables_history, state_transition_matrix_history = Interpolator.Interpolator(dynamic_model).get_results()
                    # print(epochs)
                    # print(state_history[0,:])
                    # print(dependent_variables_history[0,:])
                    # print(state_transition_matrix_history[0,:])

                    mass_primary = dependent_variables_history[0,-2]
                    mass_secondary = dependent_variables_history[0,-1]
                    state_history_barycentric = dependent_variables_history[:,:6]*(1-1/(1+mass_primary/mass_secondary))

                    print(state_history)
                    print(state_history_barycentric)

            # Test initial state


            assert dynamic_model.name_ELO == "LPF"
            assert dynamic_model.simulation_start_epoch_MJD == simulation_start_epoch_MJD


    def test_example(self, extras):

        # Assume some test logic that generates a plot
        for i in range(5):

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot([1,2,3], color="red")
            ax.plot([2,3,2], color="green")
            plt.title("figure")

            # Save the figure
            figure_path = f"C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/tests/test_dynamic_models/figure{i}.png"
            fig.savefig(figure_path)

            extras.append(pytest_html.extras.png(figure_path))

        assert True




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