# Standard
import os
import sys
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Tudatpy
from tudatpy.kernel.numerical_simulation import estimation
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from tests import utils
from src.dynamic_models import validation
from src.dynamic_models import Interpolator, StationKeeping
from src.dynamic_models.full_fidelity.full_fidelity import *
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model

model_type = "high_fidelity"
model_name = "spherical_harmonics_srp"
model_number = 4

# model_type = "low_fidelity"
# model_name = "three_body_problem"
# model_number = 0

step_size = 0.01

# Define dynamic models and select one to test the estimation on
dynamic_model_objects = utils.get_dynamic_model_objects(60390, 14)
dynamic_model = dynamic_model_objects[model_type][model_name][model_number]

custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
                                1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])
dynamic_model = low_fidelity.LowFidelityDynamicModel(60390, 14, custom_initial_state=custom_initial_state, use_synodic_state=True)

epochs, state_history, dependent_variables_history, state_transition_history = \
    Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(dynamic_model)

# Define the relative state of LPF with respect to LUMIO
relative_states = dependent_variables_history[:,6:12]



apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2

propagated_covariance_dict = dict()
for i in range(len(epochs)):
    propagated_covariance = state_transition_history[i] @ apriori_covariance @ state_transition_history[i].T
    propagated_covariance_dict.update({epochs[i]: propagated_covariance})

# propagated_covariance_array = np.stack(list(propagated_covariance_dict.values()))[:, 6:9, 6:9]

# print(propagated_covariance_array)

eigenvectors_dict = dict()
for key, matrix in propagated_covariance_dict.items():
    eigenvalues, eigenvectors = np.linalg.eigh(matrix[6:9, 6:9])
    # eigenvalues, eigenvectors = np.linalg.eigh(matrix[0:3, 0:3])
    max_eigenvalue_index = np.argmax(eigenvalues)
    eigenvector_largest = eigenvectors[:, max_eigenvalue_index]
    eigenvectors_dict.update({key: eigenvector_largest})

# eigenvectors_array = np.array(eigenvectors_list)

print(eigenvectors_dict)



# Initialize an empty list to store the angles
angles_dict = dict()
for i, (key, value) in enumerate(eigenvectors_dict.items()):
    vec1 = relative_states[i,:3]
    vec2 = value
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    cosine_angle = dot_product / (magnitude_vec1 * magnitude_vec2)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    angles_dict.update({key: angle_degrees if angle_degrees<90 else 180-angle_degrees})

# # Convert the list of angles to a NumPy array
# angles_array = np.array([180-angle for angle in angles])

print(angles_dict)


threshold = 20
filtered_dict = dict()
observation_windows = dict()
for key, value in angles_dict.items():

    # Check if any element of the value vector is below the threshold
    above_threshold = value < threshold
    filtered_dict[key] = above_threshold

print(filtered_dict)







def generate_boundary_tuples(input_dict):
    boundaries = []
    start = None
    for i, (key, value) in enumerate(input_dict.items()):
        if i == 0 and value:  # Handle the first element separately
            start = key
        elif value and not input_dict[list(input_dict.keys())[i - 1]]:  # Transition from False to True
            start = key
        elif not value and input_dict[list(input_dict.keys())[i - 1]]:  # Transition from True to False
            boundaries.append((start, list(input_dict.keys())[i - 1]))
            start = None
    # If the last region continues until the end, add it
    if start is not None:
        boundaries.append((start, list(input_dict.keys())[-1]))
    return boundaries

boundaries = generate_boundary_tuples(filtered_dict)
print(boundaries)



fig = plt.figure()
# plt.plot(epochs, relative_states[:,:3])
# plt.plot(epochs, np.linalg.norm(relative_states[:,:3], axis=1), label="abs position")
plt.plot(epochs, np.stack(list(eigenvectors_dict.values())))
plt.legend()

from tudatpy.kernel import constants

fig = plt.figure()
plt.plot((epochs-epochs[0])/constants.JULIAN_DAY, np.stack(list(angles_dict.values())), label="angles in degrees")
for i, gap in enumerate(boundaries):
    plt.axvspan(
        xmin=(gap[0]-epochs[0])/constants.JULIAN_DAY,
        xmax=(gap[1]-epochs[0])/constants.JULIAN_DAY,
        color="gray",
        alpha=0.1,
        label="Observation window" if i == 0 else None)
plt.xlabel("Time since MJD 60390 [days]")
plt.legend()

plt.show()
