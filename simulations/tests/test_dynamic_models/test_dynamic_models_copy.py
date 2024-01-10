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







# # Fixture function to set up multiple sets of sample data
# @pytest.fixture(params=[("simulation_start_epoch_MJD", "propagation_time")], scope="module")
# def generated_test_data(request):
#     package_dict = {"high_fidelity": ["point_mass", "point_mass_srp"]}
#     return utils.get_interpolated_dynamic_model_objects_results(*request.param, package_dict=package_dict, step_size=0.1)
#     # return utils.get_interpolated_estimation_model_objects_results(estimation_model, dynamic_model_objects)["high_fidelity"]["point_mass"][-1]


# # Using the parametrize decorator with the indirect parameter
# @pytest.mark.parametrize("generated_test_data", [(60390,1), (60395, 1), (60450,1)], indirect=True)
# def test_sample_data_with_indirect(generated_test_data):
#     print(generated_test_data)
#     assert isinstance(generated_test_data, dict)


# @pytest.mark.parametrize("generated_test_data", [(60390,1), (60390,1), (60450,1)], indirect=True)
# def test_sample_data_with_indirect_2(generated_test_data):
#     print(generated_test_data)
#     assert isinstance(generated_test_data, dict)


package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]}

@pytest.fixture(scope="module")
def generated_test_data():
    return utils.get_interpolated_dynamic_model_objects_results(60390,50, package_dict=package_dict, step_size=0.1)

@pytest.fixture(scope="module")
def generated_test_data_estimation():

    return utils.get_interpolated_estimation_model_objects_results(60390,50, estimation_model, package_dict=package_dict)["high_fidelity"]["point_mass"][-1]

# Using the parametrize decorator with the indirect parameter
# @pytest.mark.parametrize("generated_test_data", [(60390,1), (60395, 1), (60450,1)], indirect=True)
def test_sample_data_with_indirect(generated_test_data):
    print(generated_test_data)
    assert isinstance(generated_test_data, dict)

def test_sample_data_with_indirect_1(generated_test_data):
    print(generated_test_data)
    assert isinstance(generated_test_data, dict)

def test_sample_data_with_indirect_2(generated_test_data):
    print(generated_test_data)
    assert isinstance(generated_test_data, dict)


# @pytest.mark.parametrize("generated_test_data", [(60390,1), (60390,1), (60450,1)], indirect=True)
def test_sample_data_with_indirect_3(generated_test_data_estimation):
    print(generated_test_data_estimation)
    assert isinstance(generated_test_data_estimation, dict)

def test_sample_data_with_indirect_3(generated_test_data_estimation):
    print(generated_test_data_estimation)
    assert isinstance(generated_test_data_estimation, dict)
