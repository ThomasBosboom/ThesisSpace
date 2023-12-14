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

# Own
from src.estimation_models import somefile
from src.dynamic_models import validation_LUMIO

class TestOutputsDynamicalModels:

    def test_initial_state(self):

        # initial_state = HighFidelity_2.HighFidelity_2(60390, 10).get_propagated_orbit()[0][0][6:12]

        test = somefile.sum(1,3)

        initial_state_validation = validation_LUMIO.get_reference_state_history(60390, 10)[0][0]

        # assert initial_state[0] == initial_state_validation[0]

        assert test == 4


    def test_something_else(self):

        assert 1 == 1