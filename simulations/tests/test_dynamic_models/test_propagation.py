# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import statistics

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



class TestPropagation:

    def test_plot_acceleration_norms(self):

        # package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]}
        # Test case dynamic model
        model_type = "full_fidelity"
        model_name = "full_fidelity"
        model_number = 0

        model_type = "low_fidelity"
        model_name = "three_body_problem"
        model_number = 0

        model_type = "high_fidelity"
        model_name = "point_mass_srp"
        model_number = 7

        simulation_start_epoch_MJD = 60390
        propagation_time = 14
        package_dict = {model_type: [model_name]}
        get_only_first = False
        custom_initial_state = None
        step_size = 0.01
        epoch_in_MJD = True
        entry_list = None
        solve_variational_equations = False
        custom_propagation_time = None
        specific_model_list = [model_number]
        return_dynamic_model_objects = True

        # Initialize dictionaries to store accumulated values
        dynamic_model_objects_results = utils.get_dynamic_model_results(simulation_start_epoch_MJD,
                                                                        propagation_time,
                                                                        package_dict=package_dict,
                                                                        get_only_first=get_only_first,
                                                                        custom_initial_state=custom_initial_state,
                                                                        step_size=step_size,
                                                                        epoch_in_MJD=epoch_in_MJD,
                                                                        entry_list=entry_list,
                                                                        solve_variational_equations=solve_variational_equations,
                                                                        custom_propagation_time=custom_propagation_time,
                                                                        specific_model_list=specific_model_list,
                                                                        return_dynamic_model_objects=return_dynamic_model_objects)

        print(dynamic_model_objects_results)


        epochs, state_history, dependent_variable_history, run_time, dynamic_model_object = dynamic_model_objects_results[model_type][model_name][model_number]


        print(dynamic_model_object)

        fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        len_total_accs = 2
        len_rel_states = 12

        satellites = ["LPF", "LUMIO"]
        for i, ax in enumerate(axs):
            # for dep_var_i, dep_vars in enumerate(dependent_variable_history.T):

            # for j in range(2):
            # if dep_var_i in range(12+i, 12+(i+1)):
            axs[i].plot(epochs, dependent_variable_history[:,12+i], label="Total acceleration")

            bodies_to_create = dynamic_model_object.bodies_to_create
            for j in range(len(bodies_to_create)):
                axs[i].plot(epochs, dependent_variable_history[:,14+8*i+j], label=bodies_to_create[j])

            # for j in range(2):
            #     axs[i].plot(epochs, dependent_variable_history[:,14+8*i+j])

            # for j in range(3):
            #     axs[i].plot(epochs, dependent_variable_history[:,30+8*i+j])
                # axs[i].set_ylabel(r"$||\mathbf{a}_{PM}||$  $[m/s^{2}]$")

            # if dep_var_i in range(16, 16):

            axs[i].set_ylabel(r"$||\mathbf{a}_{total}||$  $[m/s^{2}]$")
            axs[i].set_yscale("log")
            axs[i].grid(alpha=0.5, linestyle='--')
            axs[i].set_title(satellites[i])

        axs[-1].set_xlabel(f"Time since MJD {simulation_start_epoch_MJD} [days]")
        fig.suptitle("Norms of acceleration terms")
        plt.legend()

        utils.save_figures_to_folder(figs=[fig], labels=[], save_to_report=True)

        plt.show()

        #         self.dependent_variables_to_save.extend([propagation_setup.dependent_variable.total_acceleration_norm(self.name_ELO),
        #                                          propagation_setup.dependent_variable.total_acceleration_norm(self.name_LPO)])

        # self.dependent_variables_to_save.extend([
        #     propagation_setup.dependent_variable.single_acceleration_norm(
        #             propagation_setup.acceleration.point_mass_gravity_type, body_to_propagate, body_to_create) \
        #                 for body_to_propagate in self.bodies_to_propagate for body_to_create in self.bodies_to_create])

        # self.dependent_variables_to_save.extend([
        #     propagation_setup.dependent_variable.single_acceleration_norm(
        #             propagation_setup.acceleration.radiation_pressure_type, body_to_propagate, "Sun") \
        #                 for body_to_propagate in self.bodies_to_propagate])
