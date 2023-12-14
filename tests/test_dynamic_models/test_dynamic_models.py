# Standard
import os
import sys
import matplotlib.pyplot as plt

# Define path to import src files
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Third party
import pytest
import mpld3

# Own
from src.dynamic_models import validation_LUMIO
from src.dynamic_models.low_fidelity.integration_settings import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
# from src.estimation_models import EstimationModel



# Get a list of all relevant packages within 'high_fidelity'
# packages_dir = os.path.join(parent_dir, 'src', 'dynamic_models', 'high_fidelity')
# package_list = [d for d in os.listdir(packages_dir) if d != "__pycache__" if os.path.isdir(os.path.join(packages_dir, d))]
# package_list.remove("spherical_harmonics")
# package_list.remove("spherical_harmonics_srp")


def custom_figure_test(func):
    func.custom_figure_data = "your_custom_figure_data"
    return func


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
        (60390, 10),
        (60395, 10),
    ])

    def test_initial_state(self, simulation_start_epoch_MJD, propagation_time, all_specified_models=True):

        dynamic_model_objects = self.get_dynamic_model_objects(simulation_start_epoch_MJD, propagation_time)
        # Run the low_fidelity models
        for dynamic_model in dynamic_model_objects["high_fidelity"]["point_mass"]:

            print(dynamic_model)

            assert dynamic_model.name_ELO == "LPF"
            assert dynamic_model.simulation_start_epoch_MJD == simulation_start_epoch_MJD


    # @custom_figure_test
    def test_example(self):
        # Your test code here

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

            # html_code = mpld3.fig_to_html(fig)

        # print(html_code)


        # Your test assertions go here
        assert True  # Replace with your actual test assertions




test = TestOutputsDynamicalModels()
# # dynamic_model_objects = test.get_dynamic_model_objects(60390, 10)
# # test_initial_state = test.test_initial_state(60390, 10)
test.test_example()

# low_fidelity_dynamic_model_objects = dynamic_model_objects["low_fidelity"]
# high_fidelity_dynamic_model_objects = dynamic_model_objects["high_fidelity"]["point_mass_srp"]
# print(high_fidelity_dynamic_model_objects)


