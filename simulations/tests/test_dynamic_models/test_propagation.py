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
        simulation_start_epoch_MJD = 60390
        propagation_time = 10
        package_dict = {"low_fidelity": ["three_body_problem"]}
        get_only_first = False
        custom_initial_state = None
        step_size = 0.01
        epoch_in_MJD = True
        entry_list = None
        solve_variational_equations = False
        custom_propagation_time = None

        # Test case dynamic model
        model_type = "low_fidelity"
        model_name = "three_body_problem"
        model_number = 0

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
                                                                        custom_propagation_time=custom_propagation_time)

        epochs, state_history, dependent_variable_history, run_time = dynamic_model_objects_results[model_type][model_name][model_number]

        fig, axs = plt.subplots(2, 1, figsize=(14, 6))

        for i, dep_vars in enumerate(dependent_variable_history.T):
            if i == (13 or 14):
                axs[0].plot(epochs, dependent_variable_history[:,i])


        print(dynamic_model_objects_results)

        utils.save_figures_to_folder(figs=[fig], labels=[], save_to_report=True)

        plt.show()
