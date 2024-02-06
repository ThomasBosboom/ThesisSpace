# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Tudatpy
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
from src.dynamic_models import Interpolator, StationKeeping
from src.dynamic_models.full_fidelity import *
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model

# Mission time settings
mission_time = 5
mission_start_epoch = 60390
mission_end_epoch = 60390 + mission_time

# Argument settings for dynamic models to be used in estimation
simulation_start_epoch = mission_start_epoch
propagation_time = 0.5
package_dict = {"high_fidelity": ["point_mass"]}
get_only_first = True
custom_initial_state = None
custom_initial_state_truth = None
model_type = "high_fidelity"
model_name = "point_mass"
model_number = 0


initial_state_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])**2
initial_state_error = None
apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
# apriori_covariance = np.diag([1e5, 1e5, 1e5, 1e-0, 1e-0, 1e-0, 1e5, 1e5, 1e5, 1e-0, 1e-0, 1e-0])**2


fig1_3d = plt.figure()
ax = fig1_3d.add_subplot(111, projection='3d')

estimation_error_dict = dict()
formal_error_dict = dict()
full_covariance_dict = dict()
full_estimation_error_dict = dict()
reference_state_deviation_dict = dict()

while simulation_start_epoch < mission_end_epoch:

    print("Estimation of batch ", simulation_start_epoch, " till ", simulation_start_epoch+propagation_time)

    # Define dynamic models and select one to test the estimation on
    dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch,
                                                            propagation_time,
                                                            package_dict=package_dict,
                                                            get_only_first=get_only_first,
                                                            custom_initial_state=custom_initial_state)

    dynamic_model_object = dynamic_model_objects[model_type][model_name][model_number]

    # Define the truth model to simulate the observations
    # high_fidelity_spherical_harmonics_srp_01_2_2_2_2
    truth_model = high_fidelity_point_mass_srp_06.HighFidelityDynamicModel(simulation_start_epoch,
                                                                       propagation_time,
                                                                       custom_initial_state=custom_initial_state_truth)

    # Obtain estimation results for given batch
    estimation_model_objects_results = utils.get_estimation_model_results(dynamic_model_objects,
                                                                          estimation_model,
                                                                          custom_truth_model=truth_model,
                                                                          apriori_covariance=apriori_covariance,
                                                                          initial_state_error=initial_state_error)

    # Extract all the results
    estimation_model_objects_result = estimation_model_objects_results[model_type][model_name][model_number]
    estimation_output = estimation_model_objects_result[0]
    total_covariance_dict = estimation_model_objects_result[2]

    for i, (observable_type, information_sets) in enumerate(total_covariance_dict.items()):
        if i == 0:
            for j, information_set in enumerate(information_sets.values()):
                for k, single_information_set in enumerate(information_set):
                    covariance_dict = total_covariance_dict[observable_type][j][k]

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
    epochs, state_history_final, dependent_variables_history = \
        Interpolator.Interpolator(epoch_in_MJD=False, step_size=0.01).get_propagation_results(dynamic_model_object,
                                                                                              estimated_parameter_vector=estimation_output.parameter_history[:,-1],
                                                                                              solve_variational_equations=False)

    epochs, state_history_truth, dependent_variables_history = \
        Interpolator.Interpolator(epoch_in_MJD=False, step_size=0.01).get_propagation_results(truth_model,
                                                                                              estimated_parameter_vector=custom_initial_state_truth,
                                                                                              solve_variational_equations=False)


    # Set condition on the maximum amount of error allowed before the station-keeping should take place
    # print("Difference final and reference: ", np.abs(reference_state_final[0,6:9]-reference_state_history[0,6:9]))
    # print("param vec: ", estimation_output.parameter_history[:,-1])
    station_keeping_object = StationKeeping.StationKeeping(dynamic_model_object,
                                                           estimated_parameter_vector=estimation_output.parameter_history[:,-1],
                                                           custom_propagation_time=8)

    delta_v = station_keeping_object.get_corrected_state_vector(correction_epoch=0+propagation_time,
                                                                target_point_epoch=7+propagation_time,
                                                                cut_off_epoch=0)
    print("done station keeping: ", delta_v)

    # Add current relevant information to the total
    estimation_errors = state_history_truth[0,:] - parameter_history[:,-1]
    formal_error_dict.update({simulation_start_epoch: formal_errors})
    estimation_error_dict.update({simulation_start_epoch: estimation_errors})
    full_covariance_dict.update(covariance_dict)
    full_estimation_error_dict.update(dict(zip(epochs, state_history_truth-state_history_final)))
    reference_state_deviation_dict.update(dict(zip(epochs, reference_state_history-state_history_final)))

    # Update settings for next batch
    # print(custom_initial_state, simulation_start_epoch)
    custom_initial_state = state_history_final[-1,:]
    simulation_start_epoch += propagation_time
    state_history_final[-1,9:12] = state_history_final[-1,9:12] + delta_v
    custom_initial_state = state_history_final[-1,:]
    custom_initial_state_truth = state_history_truth[-1,:]
    apriori_covariance = final_covariance
    # initial_state_error = estimation_errors

    # Storing some plots
    ax.plot(reference_state_history[:,0], reference_state_history[:,1], reference_state_history[:,2], label="LPF ref", color="green")
    ax.plot(reference_state_history[:,6], reference_state_history[:,7], reference_state_history[:,8], label="LUMIO ref", color="green")
    ax.plot(state_history_final[:,0], state_history_final[:,1], state_history_final[:,2], label="LPF estimated", )
    ax.plot(state_history_final[:,6], state_history_final[:,7], state_history_final[:,8], label="LUMIO estimated")
    ax.plot(state_history_truth[:,0], state_history_truth[:,1], state_history_truth[:,2], label="LPF truth", color="black", ls="--")
    ax.plot(state_history_truth[:,6], state_history_truth[:,7], state_history_truth[:,8], label="LUMIO truth", color="black", ls="--")



print("Done with the estimation process")

plt.show()

print(formal_error_dict, estimation_error_dict)

covariance_epochs = np.stack(list(full_covariance_dict.keys()))
covariance_history = np.stack(list(full_covariance_dict.values()))
estimation_error_epochs = np.stack(list(full_estimation_error_dict.keys()))
estimation_error_history = np.stack(list(full_estimation_error_dict.values()))

formal_error_history = np.array([np.sqrt(np.diagonal(covariance)) for covariance in covariance_history])
estimation_error_history = np.array([interp1d(estimation_error_epochs, state, kind='linear', fill_value='extrapolate')(covariance_epochs) for state in estimation_error_history.T]).T

fig = plt.figure()
plt.plot(covariance_epochs, formal_error_history[:,6:9], color="red", ls="--", label=r"$1\sigma$ std. dev.")
plt.plot(covariance_epochs, estimation_error_history[:,6:9], color="blue", label=r"$\theta$-$\hat{\theta}$")
plt.grid()
plt.legend()
plt.show()


# dynamic_model_objects = utils.get_dynamic_model_objects(mission_start_epoch,
#                                                         mission_time,
#                                                         package_dict=None,
#                                                         get_only_first=False,
#                                                         custom_initial_state=None)

# epochs, state_history_full, _ = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=0.01).get_propagation_results(dynamic_model_objects[model_type][model_name][model_number],
#                                                                                           estimated_parameter_vector=None,
#                                                                                           solve_variational_equations=False)

# epochs, state_history_full_truth, _ = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=0.01).get_propagation_results(dynamic_model_objects["high_fidelity"]["point_mass_srp"][0],
#                                                                                           estimated_parameter_vector=None,
#                                                                                           solve_variational_equations=False)



# ax.plot(state_history_full_truth[:,0], state_history_full_truth[:,1], state_history_full_truth[:,2], label="LPF truth", color="red", ls="--")
# ax.plot(state_history_full_truth[:,6], state_history_full_truth[:,7], state_history_full_truth[:,8], label="LUMIO truth", color="blue", ls="--")
# ax.plot(state_history_full[:,0], state_history_full[:,1], state_history_full[:,2], label="LPF", color="gray", ls="--")
# ax.plot(state_history_full[:,6], state_history_full[:,7], state_history_full[:,8], label="LUMIO", color="gray", ls="--")

# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')
# plt.legend(loc="upper right")
# plt.show()

# estimation_errors_history = np.array(estimation_error_dict)
# formal_errors_history = np.array(formal_error_dict)
# epochs = np.array(epochs_list)

# fig = plt.figure()
# plt.plot(epochs, estimation_errors_history[:,:3], color="blue")
# plt.plot(epochs, estimation_errors_history[:,:3]+formal_errors_history[:,:3], color="red", ls="--")
# plt.plot(epochs, estimation_errors_history[:,:3]-formal_errors_history[:,:3], color="red", ls="-.")
# # plt.plot(np.linalg.norm(estimation_errors_history[:,:3]+formal_errors_history[:,:3], axis=1), color="red", ls="--")
# # plt.plot(np.linalg.norm(estimation_errors_history[:,:3]-formal_errors_history[:,:3], axis=1), color="red", ls="-.")
# plt.plot(epochs, 100*np.ones(len(estimation_errors_history)), color="gray", ls="--")
# plt.show()