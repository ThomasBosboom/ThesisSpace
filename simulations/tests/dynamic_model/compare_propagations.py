# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Third party
import pytest
import pytest_html
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Own
from tests import utils
import reference_data, Interpolator, FrameConverter
from src.dynamic_models import TraditionalLowFidelity
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model


# Old parameter
old_parameter = [-3.62949097e+08, -1.59106985e+08, -8.00358902e+07, 8.45228035e+02,
                 -5.91485039e+02, -8.49468017e+02, -4.14859239e+08, -1.64597204e+08,
                 -1.43094890e+08, 4.30054740e+02, -7.12404884e+02, -4.46824114e+02]

# Updated parameter
updated_parameter = [-3.62949068e+08, -1.59106979e+08, -8.00358852e+07, 8.45228069e+02,
                     -5.91483104e+02, -8.49467516e+02, -4.14858803e+08, -1.64597442e+08,
                     -1.43094882e+08, 4.30055711e+02, -7.12404090e+02, -4.46823221e+02]

# Create a 2D NumPy array
custom_initial_states = np.array([old_parameter, updated_parameter])


fig, ax = plt.subplots(1, 1)
state_histories = []
for custom_initial_state in custom_initial_states:

    dynamic_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(60393, 3.95, custom_initial_state=custom_initial_state)

    # Extract simulation histories tudatpy solution
    epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
        Interpolator.Interpolator(step_size=0.01).get_propagation_results(dynamic_model, custom_initial_state=custom_initial_state)

    # ax.plot(epochs, state_history[:, :3] ,label="LPF")
    # ax.plot(epochs, state_history[:, 6:9] ,label="LUMIO")

    state_histories.append(state_history)

ax.plot(state_histories[1][:,6:9]-state_histories[0][:,6:9])
# fig1_3d = plt.figure()
# ax = fig1_3d.add_subplot(111, projection='3d')
# plt.title("Tudat versus true halo versus classical CRTBP")
# plt.plot(state_history_classic[:,0], state_history_classic[:,1], state_history_classic[:,2], label="LPF classic", color="gray")
# plt.plot(state_history_classic[:,6], state_history_classic[:,7], state_history_classic[:,8], label="LUMIO classic", color="gray")
# plt.plot(state_history_classic_erdem[:,0], state_history_classic_erdem[:,1], state_history_classic_erdem[:,2], label="LPF ideal", color="black")
# plt.plot(state_history_classic_erdem[:,6], state_history_classic_erdem[:,7], state_history_classic_erdem[:,8], label="LUMIO ideal", color="black")
# plt.plot(state_history[:,0], state_history[:,1], state_history[:,2], label="LPF tudat", color="red")
# plt.plot(state_history[:,6], state_history[:,7], state_history[:,8], label="LUMIO tudat", color="blue")
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')
plt.legend(loc="upper right")
plt.show()
