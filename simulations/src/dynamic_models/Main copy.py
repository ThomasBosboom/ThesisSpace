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
mission_time = 3
mission_start_epoch = 60390
mission_end_epoch = 60390 + mission_time

# Initial estimation setup
simulation_start_epoch = mission_start_epoch
propagation_time = 1
package_dict = {"high_fidelity": ["point_mass"]}
get_only_first = True
custom_initial_state = None
custom_initial_state_truth = None
initial_state_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])**2
apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
step_size = 0.001

# Specify a test case
model_type = "high_fidelity"
model_name = "point_mass"
model_number = 0




fig1_3d = plt.figure()
ax = fig1_3d.add_subplot(111, projection='3d')

estimation_error_dict = dict()
formal_error_dict = dict()
full_covariance_dict = dict()
full_estimation_error_dict = dict()
reference_state_deviation_dict = dict()
propagated_covariance_dict = dict()
propagated_formal_errors_dict = dict()

while simulation_start_epoch < mission_end_epoch:

    print("Estimation of batch ", simulation_start_epoch, " until ", simulation_start_epoch+propagation_time)

    # Define dynamic models and select one to test the estimation on
    dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch,
                                                            propagation_time,
                                                            package_dict=package_dict,
                                                            get_only_first=get_only_first,
                                                            custom_initial_state=custom_initial_state)

    dynamic_model_object = dynamic_model_objects[model_type][model_name][model_number]

    # Define the truth model to simulate the observations
    truth_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(simulation_start_epoch,
                                                                       propagation_time,
                                                                       custom_initial_state=custom_initial_state_truth)

    estimation_model_objects = utils.get_estimation_model_objects(dynamic_model_objects,
                                                                    custom_truth_model=truth_model,
                                                                    apriori_covariance=apriori_covariance,
                                                                    initial_state_error=initial_state_error)

    # print(estimation_model_objects, estimation_model_objects[model_type][model_name][model_number].observation_step_size_range)
    # estimation_model_objects[model_type][model_name][model_number].observation_step_size_range = 200
    # estimation_model_objects[model_type][model_name][model_number].observation_step_size_doppler = 200
    # estimation_model_objects[model_type][model_name][model_number].noise_range = 200
    # print(estimation_model_objects, estimation_model_objects[model_type][model_name][model_number].observation_step_size_range)

    # Obtain estimation results for given batch
    estimation_model_objects_results = utils.get_estimation_model_results(dynamic_model_objects,
                                                                          custom_estimation_model_objects=estimation_model_objects,
                                                                          get_only_first=True,
                                                                          custom_truth_model=truth_model,
                                                                          apriori_covariance=apriori_covariance,
                                                                          initial_state_error=initial_state_error)

    # Extract all the results
    estimation_model_objects_result = estimation_model_objects_results[model_type][model_name][model_number]
    estimation_output = estimation_model_objects_result[0]
    parameter_history = estimation_output.parameter_history
    final_covariance = estimation_output.covariance
    formal_errors = estimation_output.formal_errors
    total_covariance_dict = estimation_model_objects_result[2]
    propagated_covariance = estimation_model_objects_result[4]
    propagated_formal_errors = estimation_model_objects_result[5]

    # Update the reference orbit that the estimated orbit should follow
    reference_state_history = list()
    for body in dynamic_model_object.bodies_to_propagate:
        reference_state_history.append(validation.get_reference_state_history(simulation_start_epoch,
                                                                              propagation_time,
                                                                                satellite=body,
                                                                                step_size=step_size,
                                                                                get_full_history=True))

    reference_state_history = np.concatenate(reference_state_history, axis=1)


    # Update the estimator with the final parameter values to generate dynamics simulation with final dynamics for next batch
    epochs, state_history_final, dependent_variables_history = \
        Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(dynamic_model_object,
                                                                                              estimated_parameter_vector=estimation_output.parameter_history[:,-1],
                                                                                              solve_variational_equations=False)

    epochs, state_history_truth, dependent_variables_history = \
        Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(truth_model,
                                                                                              estimated_parameter_vector=custom_initial_state_truth,
                                                                                              solve_variational_equations=False)



    # Add current relevant information to the total

    estimation_errors = state_history_truth[0,:] - state_history_final[0,:]
    formal_error_dict.update({simulation_start_epoch: formal_errors})
    estimation_error_dict.update({simulation_start_epoch: estimation_errors})
    full_covariance_dict.update(total_covariance_dict)
    full_estimation_error_dict.update(dict(zip(epochs, state_history_truth-state_history_final)))
    reference_state_deviation_dict.update(dict(zip(epochs, reference_state_history-state_history_final)))
    propagated_covariance_dict.update(propagated_covariance)
    propagated_formal_errors_dict.update(propagated_formal_errors)

    # print("estimation errors")
    # estimation_errors_magnitude = np.sqrt(np.square(estimation_errors[6:9]).sum())
    # print(estimation_errors[6:9], estimation_errors_magnitude)
    # print("formal errors")
    # propagated_formal_errors_magnitude = np.sqrt(np.square(np.stack(list(propagated_formal_errors.values()))[0, 6:9]).sum())
    # print(np.stack(list(propagated_formal_errors.values()))[0, 6:9], propagated_formal_errors_magnitude)
    # if propagated_formal_errors_magnitude < 1000:
    #     print("CONVERGENCE AT ", simulation_start_epoch)

    # Update settings for next batch
    custom_initial_state = state_history_final[-1,:]
    custom_initial_state_truth = state_history_truth[-1,:]
    apriori_covariance = np.array(list(propagated_covariance.values()))[-1]

    # Set condition on the maximum amount of error allowed before the station-keeping should take place
    # if int((simulation_start_epoch-mission_start_epoch)%4) == 0 and simulation_start_epoch != mission_start_epoch:

    #     station_keeping_object = StationKeeping.StationKeeping(dynamic_model_object,
    #                                                         estimated_parameter_vector=estimation_output.parameter_history[:,-1],
    #                                                         custom_propagation_time=6)

    #     custom_initial_state = station_keeping_object.get_corrected_state_vector(correction_epoch=0+propagation_time,
    #                                                                 target_point_epoch=4+propagation_time,
    #                                                                 cut_off_epoch=0+propagation_time)

    simulation_start_epoch += propagation_time


    # Storing some plots
    ax.plot(reference_state_history[:,0], reference_state_history[:,1], reference_state_history[:,2], label="LPF ref", color="green")
    ax.plot(reference_state_history[:,6], reference_state_history[:,7], reference_state_history[:,8], label="LUMIO ref", color="green")
    ax.plot(state_history_final[:,0], state_history_final[:,1], state_history_final[:,2], label="LPF estimated", )
    ax.plot(state_history_final[:,6], state_history_final[:,7], state_history_final[:,8], label="LUMIO estimated")
    ax.plot(state_history_truth[:,0], state_history_truth[:,1], state_history_truth[:,2], label="LPF truth", color="black", ls="--")
    ax.plot(state_history_truth[:,6], state_history_truth[:,7], state_history_truth[:,8], label="LUMIO truth", color="black", ls="--")



print("Done with the estimation process")

plt.show()

propagated_covariance_epochs = np.stack(list(propagated_covariance_dict.keys()))
propagated_covariance_history = np.stack(list(propagated_covariance_dict.values()))
estimation_error_epochs = np.stack(list(full_estimation_error_dict.keys()))
estimation_error_history = np.stack(list(full_estimation_error_dict.values()))
reference_state_deviation_epochs = np.stack(list(reference_state_deviation_dict.keys()))
reference_state_deviation_history = np.stack(list(reference_state_deviation_dict.values()))
propagated_formal_errors_epochs = np.stack(list(propagated_formal_errors_dict.keys()))
propagated_formal_errors_history = np.stack(list(propagated_formal_errors_dict.values()))

estimation_error_history = np.array([interp1d(estimation_error_epochs, state, kind='linear', fill_value='extrapolate')(propagated_covariance_epochs) for state in estimation_error_history.T]).T


plt.plot(propagated_formal_errors_history[:,6:9])
plt.show()

plt.plot(reference_state_deviation_history[:,6:9])
plt.show()

fig = plt.figure()
plt.plot(propagated_covariance_epochs, estimation_error_history[:,6:9]+propagated_formal_errors_history[:,6:9], color="red", ls="--", label=r"$1\sigma$ std. dev.")
plt.plot(propagated_covariance_epochs, estimation_error_history[:,6:9]-propagated_formal_errors_history[:,6:9], color="orange", ls="-.", label=r"$1\sigma$ std. dev.")
plt.plot(propagated_covariance_epochs, estimation_error_history[:,6:9], color="blue", label=r"$\theta$-$\hat{\theta}$")
plt.grid()
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(propagated_covariance_epochs, estimation_error_history[:,0:3]+propagated_formal_errors_history[:,0:3], color="red", ls="--", label=r"$1\sigma$ std. dev.")
plt.plot(propagated_covariance_epochs, estimation_error_history[:,0:3]-propagated_formal_errors_history[:,0:3], color="red", ls="-.", label=r"$1\sigma$ std. dev.")
plt.plot(propagated_covariance_epochs, estimation_error_history[:,0:3], color="blue", label=r"$\theta$-$\hat{\theta}$")
plt.grid()
plt.legend()
plt.show()