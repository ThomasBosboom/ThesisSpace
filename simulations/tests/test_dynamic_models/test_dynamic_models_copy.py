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
from src.estimation_models import EstimationModel





package_dict = {"high_fidelity": ["point_mass", "point_mass_srp"]}
dynamic_model_objects = utils.get_dynamic_model_objects(60390, 0.2, package_dict=package_dict)
# print(utils.get_interpolated_dynamic_model_objects_results(60390, .2, package_dict=package_dict)["high_fidelity"]["point_mass"])
# print(utils.get_interpolated_estimation_model_objects_results(EstimationModel, dynamic_model_objects)["high_fidelity"]["point_mass"][-1])

def test(arg1, arg2):
    return arg1+arg2

# Define a fixture to calculate the expensive result
@pytest.fixture
def expensive_result(request):
    # return utils.get_interpolated_dynamic_model_objects_results(request.param, request.param, package_dict=package_dict)
    return test(request.param ,request.param)

# Use the fixture with parametrize in your test function
@pytest.mark.parametrize("expensive_result", [(60390, 1)], indirect=["expensive_result"])
def test_your_function(expensive_result, input_args):

    initial_state = expensive_result["high_fidelity"]["point_mass"][0][0,0]
    assert initial_state == 1
