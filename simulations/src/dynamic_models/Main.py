# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Third party
import pytest
import pytest_html
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from tests import utils
from src.dynamic_models import validation
from src.dynamic_models import Interpolator, FrameConverter, TraditionalLowFidelity
from src.dynamic_models.full_fidelity import *
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model


mission_time = 25
mission_start_epoch = 60390
mission_end_epoch = 60390 + mission_time

# Argument settings for dynamic models to be used in estimation
simulation_start_epoch = mission_start_epoch
propagation_time = 1
package_dict = {"high_fidelity": ["point_mass"]}
# package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}
get_only_first = True
custom_initial_state = None
custom_initial_state_truth = None

apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2

fig1_3d = plt.figure()
ax = fig1_3d.add_subplot(111, projection='3d')

estimation_errors_list = list()
formal_errors_list = list()
epochs_list = list()
full_covariance_dict = dict()
full_estimation_error_dict = dict()

while simulation_start_epoch < mission_end_epoch:

    print("Simulation epoch of batch: ", simulation_start_epoch)

    # Define dynamic models and select one to test the estimation on
    dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch,
                                                            propagation_time,
                                                            package_dict=package_dict,
                                                            get_only_first=get_only_first,
                                                            custom_initial_state=custom_initial_state)

    dynamic_model_object = dynamic_model_objects["high_fidelity"]["point_mass"][0]

    # Define the truth model to simulate the observations
    # high_fidelity_spherical_harmonics_srp_01_2_2_2_2
    truth_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(simulation_start_epoch,
                                                                       propagation_time,
                                                                       custom_initial_state=custom_initial_state_truth)

    # Obtain estimation results for given batch
    estimation_model_objects_results = utils.get_estimation_model_results(dynamic_model_objects,
                                                                          estimation_model,
                                                                          custom_truth_model=truth_model,
                                                                          apriori_covariance=apriori_covariance)

    # Extract all the results
    estimation_model_objects_result = estimation_model_objects_results["high_fidelity"]["point_mass"][0]
    estimation_output = estimation_model_objects_result[0]
    total_covariance_dict = estimation_model_objects_result[1]

    for i, (observable_type, information_sets) in enumerate(total_covariance_dict.items()):
        if i == 0:
            for j, information_set in enumerate(information_sets.values()):
                for k, single_information_set in enumerate(information_set):
                    covariance_dict = total_covariance_dict[observable_type][j][k]
                    # covariance_epochs = np.stack(list(covariance_dict.keys()))
                    # covariance_history = np.stack(list(covariance_dict.values()))

    parameter_history = estimation_output.parameter_history
    final_covariance = np.linalg.inv(estimation_output.inverse_covariance)
    formal_errors = estimation_output.formal_errors

    # Update the reference orbit that the estimated orbit should follow
    reference_state_history = list()
    for body in dynamic_model_object.bodies_to_propagate:
        reference_state_history.append(validation.get_reference_state_history(simulation_start_epoch,
                                                                              propagation_time,
                                                                                satellite=body,
                                                                                step_size=0.01,
                                                                                get_full_history=True))

    reference_state_history = np.concatenate(reference_state_history, axis=1)


    # Update the estimator with the final parameter values to generate dynamics simulation with final dynamics for next batch
    epochs, state_history_final, dependent_variables_history, state_transition_matrix_history = \
        Interpolator.Interpolator(epoch_in_MJD=False, step_size=0.01).get_propagation_results(dynamic_model_object,
                                                                                              estimated_parameter_vector=estimation_output.parameter_history[:,-1])

    epochs, state_history_truth, dependent_variables_history, state_transition_matrix_history = \
        Interpolator.Interpolator(epoch_in_MJD=False, step_size=0.01).get_propagation_results(truth_model,
                                                                                              estimated_parameter_vector=custom_initial_state_truth)

    ax.plot(reference_state_history[:,0], reference_state_history[:,1], reference_state_history[:,2], label="LPF ref", color="green")
    ax.plot(reference_state_history[:,6], reference_state_history[:,7], reference_state_history[:,8], label="LUMIO ref", color="green")
    ax.plot(state_history_final[:,0], state_history_final[:,1], state_history_final[:,2], label="LPF final", )
    ax.plot(state_history_final[:,6], state_history_final[:,7], state_history_final[:,8], label="LUMIO final")
    ax.plot(state_history_truth[:,0], state_history_truth[:,1], state_history_truth[:,2], label="LPF truth", color="black", ls="--")
    ax.plot(state_history_truth[:,6], state_history_truth[:,7], state_history_truth[:,8], label="LUMIO truth", color="black", ls="--")

    # Add current relevant information to the total
    estimation_errors = state_history_truth[0,:] - parameter_history[:,-1]
    estimation_errors_list.append(estimation_errors)
    formal_errors_list.append(formal_errors)
    epochs_list.append(simulation_start_epoch)
    full_covariance_dict.update(covariance_dict)
    full_estimation_error_dict.update(dict(zip(epochs, state_history_truth-state_history_final)))

    # Update settings for next batch
    simulation_start_epoch += propagation_time
    custom_initial_state = state_history_final[-1,:]
    custom_initial_state_truth = state_history_truth[-1,:]
    apriori_covariance = final_covariance

print("Done with the estimation process")

# covariance_epochs = np.stack(list(full_covariance_dict.keys()))
# covariance_history = np.stack(list(full_covariance_dict.values()))
# estimation_error_epochs = np.stack(list(full_estimation_error_dict.keys()))
# estimation_error_history = np.stack(list(full_estimation_error_dict.values()))

# print(full_estimation_error_dict)
# print(estimation_error_epochs, estimation_error_history)
# fig = plt.figure()
# plt.scatter(covariance_epochs, np.array([np.sqrt(np.diagonal(covariance)) for covariance in covariance_history[:,0:1,0:1]]))
# plt.scatter(estimation_error_epochs, estimation_error_history[:,0:1])
# plt.show()


dynamic_model_objects = utils.get_dynamic_model_objects(mission_start_epoch,
                                                        mission_time,
                                                        package_dict=None,
                                                        get_only_first=False,
                                                        custom_initial_state=None)

epochs, state_history_full, _ = \
    Interpolator.Interpolator(epoch_in_MJD=False, step_size=0.01).get_propagation_results(dynamic_model_objects["high_fidelity"]["point_mass"][0],
                                                                                          estimated_parameter_vector=None,
                                                                                          solve_variational_equations=False)

epochs, state_history_full_truth, _ = \
    Interpolator.Interpolator(epoch_in_MJD=False, step_size=0.01).get_propagation_results(dynamic_model_objects["high_fidelity"]["point_mass"][0],
                                                                                          estimated_parameter_vector=None,
                                                                                          solve_variational_equations=False)



# ax.plot(state_history_full_truth[:,0], state_history_full_truth[:,1], state_history_full_truth[:,2], label="LPF truth", color="red", ls="--")
# ax.plot(state_history_full_truth[:,6], state_history_full_truth[:,7], state_history_full_truth[:,8], label="LUMIO truth", color="blue", ls="--")
# ax.plot(state_history_full[:,0], state_history_full[:,1], state_history_full[:,2], label="LPF", color="gray", ls="--")
# ax.plot(state_history_full[:,6], state_history_full[:,7], state_history_full[:,8], label="LUMIO", color="gray", ls="--")

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
# plt.legend(loc="upper right")
plt.show()

estimation_errors_history = np.array(estimation_errors_list)
formal_errors_history = np.array(formal_errors_list)
epochs = np.array(epochs_list)

fig = plt.figure()
plt.plot(epochs, estimation_errors_history[:,:3], color="blue")
plt.plot(epochs, estimation_errors_history[:,:3]+formal_errors_history[:,:3], color="red", ls="--")
plt.plot(epochs, estimation_errors_history[:,:3]-formal_errors_history[:,:3], color="red", ls="-.")
# plt.plot(np.linalg.norm(estimation_errors_history[:,:3]+formal_errors_history[:,:3], axis=1), color="red", ls="--")
# plt.plot(np.linalg.norm(estimation_errors_history[:,:3]-formal_errors_history[:,:3], axis=1), color="red", ls="-.")
plt.plot(epochs, 100*np.ones(len(estimation_errors_history)), color="gray", ls="--")
plt.show()