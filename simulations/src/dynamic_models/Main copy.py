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
from tudatpy.kernel.numerical_simulation import estimation
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.astro import time_conversion

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

# Mission time settings
mission_time = 5
mission_start_epoch = 60390
mission_end_epoch = mission_start_epoch + mission_time

# Initial estimation setup
simulation_start_epoch = mission_start_epoch
propagation_time = 1
batch_start_times = np.arange(mission_start_epoch, mission_end_epoch, propagation_time)
batch_end_times = np.arange(propagation_time+mission_start_epoch, propagation_time+mission_end_epoch, propagation_time)
# batch_start_times = np.array([60390, 60391, 60392, 60393, 60394])
# batch_end_times = np.array([60390.5, 60391.5, 60392.5, 60394, 60395])
# batch_start_times = np.array([60390, 60390.7, 60391.5, 60393, 60394])
# batch_end_times = np.array([60390.5, 60391, 60392.5, 60394, 60395])
batch_times = np.round(batch_end_times - batch_start_times, 2)
propagation_times = np.round(np.concatenate((np.diff(batch_start_times), [batch_times[-1]])), 2)
observation_windows = list(zip(batch_start_times, batch_end_times))
print(observation_windows)
package_dict = {"high_fidelity": ["point_mass"]}
# package_dict = {"low_fidelity": ["three_body_problem"]}
custom_initial_state = None
custom_initial_state_truth = None
initial_state_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
step_size = 0.01
sigma_number = 3
batch_count = 0

# Specify a test case
model_type = "high_fidelity"
model_name = "point_mass"
model_number = 0


fig1_3d = plt.figure()
ax = fig1_3d.add_subplot(111, projection='3d')

batch_estimation_error_dict = dict()
batch_formal_error_dict = dict()
intermediate_covariance_dict = dict()
full_estimation_error_dict = dict()
full_reference_state_deviation_dict = dict()
full_state_errors_dict = dict()
full_propagated_covariance_dict = dict()
full_propagated_formal_errors_dict = dict()

while simulation_start_epoch < mission_end_epoch:

    print(f"Estimation of batch {batch_count}, duration {batch_times[batch_count]} days: {batch_start_times[batch_count]} until {batch_end_times[batch_count]}")

    print(batch_start_times[batch_count])
    print(batch_times[batch_count])
    print(custom_initial_state)
    print(custom_initial_state_truth)

    # Define dynamic models and select one to test the estimation on
    dynamic_model_objects = utils.get_dynamic_model_objects(batch_start_times[batch_count],
                                                            batch_times[batch_count],
                                                            package_dict=package_dict,
                                                            get_only_first=True,
                                                            custom_initial_state=custom_initial_state)

    dynamic_model_object = dynamic_model_objects[model_type][model_name][model_number]

    print(dynamic_model_object)

    # Define the truth model to simulate the observations
    # high_fidelity_spherical_harmonics_srp_04_2_2_20_20
    truth_model = low_fidelity.LowFidelityDynamicModel(batch_start_times[batch_count],
                                                                       batch_times[batch_count],
                                                                       custom_initial_state=custom_initial_state_truth)

    truth_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(batch_start_times[batch_count],
                                                                       batch_times[batch_count],
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
                                                                          custom_estimation_model_objects=None,
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
    estimator = estimation_model_objects_result[-2]

    output_times = np.arange(utils.convert_MJD_to_epoch(batch_start_times[batch_count], full_array=False),
                             utils.convert_MJD_to_epoch(batch_start_times[batch_count]+propagation_times[batch_count], full_array=False),
                             100)

    propagated_formal_errors = dict()
    propagated_formal_errors.update(zip(*estimation.propagate_formal_errors_split_output(
        initial_covariance=estimation_output.covariance,
        state_transition_interface=estimator.state_transition_interface,
        output_times=output_times)))

    propagated_covariance = dict()
    propagated_covariance.update(zip(*estimation.propagate_covariance_split_output(
        initial_covariance=estimation_output.covariance,
        state_transition_interface=estimator.state_transition_interface,
        output_times=output_times)))

    # Update the reference orbit that the estimated orbit should follow
    reference_state_history = list()
    for body in dynamic_model_object.bodies_to_propagate:
        reference_state_history.append(validation.get_reference_state_history(batch_start_times[batch_count],
                                                                              propagation_times[batch_count],
                                                                                satellite=body,
                                                                                step_size=step_size,
                                                                                get_full_history=True))

    reference_state_history = np.concatenate(reference_state_history, axis=1)


    # Update the estimator with the final parameter values to generate dynamics simulation with final dynamics for next batch

    epochs, state_history_initial, dependent_variables_history_initial = \
        Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(dynamic_model_object,
                                                                                              custom_initial_state=parameter_history[:,0],
                                                                                              custom_propagation_time=propagation_times[batch_count],
                                                                                              solve_variational_equations=False)

    epochs, state_history_final, dependent_variables_history_final = \
        Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(dynamic_model_object,
                                                                                              custom_initial_state=parameter_history[:,-1],
                                                                                              custom_propagation_time=propagation_times[batch_count],
                                                                                              solve_variational_equations=False)

    epochs, state_history_truth, dependent_variables_history_truth = \
        Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(truth_model,
                                                                                              custom_initial_state=custom_initial_state_truth,
                                                                                              custom_propagation_time=propagation_times[batch_count],
                                                                                              solve_variational_equations=False)

    print("shape of state_histories: ", np.shape(state_history_initial), np.shape(state_history_final), np.shape(state_history_truth))

    # Add current relevant information to the total
    estimation_errors = state_history_truth[0,:] - state_history_final[0,:]
    batch_formal_error_dict.update({simulation_start_epoch: formal_errors})
    batch_estimation_error_dict.update({simulation_start_epoch: estimation_errors})
    intermediate_covariance_dict.update(total_covariance_dict)
    full_estimation_error_dict.update(dict(zip(epochs, state_history_truth-state_history_final)))
    full_reference_state_deviation_dict.update(dict(zip(epochs, reference_state_history-state_history_final)))
    full_state_errors_dict.update(dict(zip(epochs, state_history_initial-state_history_final)))
    full_propagated_covariance_dict.update(propagated_covariance)
    full_propagated_formal_errors_dict.update(propagated_formal_errors)


    # pos_uncertainties_rss_LPF = np.sqrt(np.sum(sigma_number*np.diag(final_covariance[:3,:3])))
    # pos_uncertainties_rss_LUMIO = np.sqrt(np.sum(sigma_number*np.diag(final_covariance[6:9,6:9])))
    # print("RSS of Position Uncertainties LPF and LUMIO:", pos_uncertainties_rss_LPF, pos_uncertainties_rss_LUMIO)

    # pos_uncertainties_LPF = sigma_number*np.sqrt(np.diag(final_covariance[:3,:3]))
    # pos_uncertainties_LUMIO = sigma_number*np.sqrt(np.diag(final_covariance[6:9,6:9]))
    # print("Sigma of Position Uncertainties LPF and LUMIO:", pos_uncertainties_LPF, pos_uncertainties_LUMIO)



    initial_state_error = state_history_initial[-1,:]-state_history_final[-1,:]
    custom_initial_state = state_history_final[-1,:]
    custom_initial_state_truth = state_history_truth[-1,:]
    apriori_covariance = np.array(list(propagated_covariance.values()))[-1]

    # Set condition on the maximum amount of error allowed before the station-keeping should take place
    # if int((simulation_start_epoch-mission_start_epoch)%4) == 0 and simulation_start_epoch != mission_start_epoch:

    #     station_keeping_object = StationKeeping.StationKeeping(dynamic_model_object,
    #                                                         custom_initial_state=estimation_output.parameter_history[:,-1],
    #                                                         custom_propagation_time=6)

    #     custom_initial_state = station_keeping_object.get_corrected_state_vector(correction_epoch=0+propagation_time,
    #                                                                 target_point_epoch=4+propagation_time,
    #                                                                 cut_off_epoch=0+propagation_time)

    simulation_start_epoch += propagation_times[batch_count]
    batch_count += 1


    # Storing some plots
    ax.plot(reference_state_history[:,0], reference_state_history[:,1], reference_state_history[:,2], label="LPF ref", color="green")
    ax.plot(reference_state_history[:,6], reference_state_history[:,7], reference_state_history[:,8], label="LUMIO ref", color="green")
    ax.plot(state_history_final[:,0], state_history_final[:,1], state_history_final[:,2], label="LPF estimated", )
    ax.plot(state_history_final[:,6], state_history_final[:,7], state_history_final[:,8], label="LUMIO estimated")
    ax.plot(state_history_truth[:,0], state_history_truth[:,1], state_history_truth[:,2], label="LPF truth", color="black", ls="--")
    ax.plot(state_history_truth[:,6], state_history_truth[:,7], state_history_truth[:,8], label="LUMIO truth", color="black", ls="--")




print("Done with the estimation process")

# dynamic_model_objects = utils.get_dynamic_model_objects(mission_start_epoch,
#                                                         mission_time,
#                                                         package_dict=package_dict,
#                                                         get_only_first=True,
#                                                         custom_initial_state=custom_initial_state)

# dynamic_model_object = dynamic_model_objects[model_type][model_name][model_number]

# truth_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(mission_start_epoch,
#                                                                                         mission_time,
#                                                                                         custom_initial_state=None)

# epochs, state_history_final, dependent_variables_history = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(dynamic_model_object,
#                                                                                             custom_initial_state=None,
#                                                                                             solve_variational_equations=False)

# epochs, state_history_truth, dependent_variables_history = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(truth_model,
#                                                                                             custom_initial_state=None,
#                                                                                             solve_variational_equations=False)

# ax.plot(state_history_final[:,0], state_history_final[:,1], state_history_final[:,2], label="LPF estimated", color="red", ls="-.")
# ax.plot(state_history_final[:,6], state_history_final[:,7], state_history_final[:,8], label="LUMIO estimated", color="red", ls="-.")
# ax.plot(state_history_truth[:,0], state_history_truth[:,1], state_history_truth[:,2], label="LPF truth", color="yellow", ls="--")
# ax.plot(state_history_truth[:,6], state_history_truth[:,7], state_history_truth[:,8], label="LUMIO truth", color="yellow", ls="--")

# plt.legend()
plt.title(f"propagation_time: {propagation_time}, {mission_time} [days], obs. period {estimation_model_objects[model_type][model_name][model_number].observation_step_size_range} [s]")
# plt.show()

propagated_covariance_epochs = np.stack(list(full_propagated_covariance_dict.keys()))
propagated_covariance_history = np.stack(list(full_propagated_covariance_dict.values()))
full_estimation_error_epochs = np.stack(list(full_estimation_error_dict.keys()))
full_estimation_error_history = np.stack(list(full_estimation_error_dict.values()))
full_reference_state_deviation_epochs = np.stack(list(full_reference_state_deviation_dict.keys()))
full_reference_state_deviation_history = np.stack(list(full_reference_state_deviation_dict.values()))
full_state_errors_epochs = np.stack(list(full_state_errors_dict.keys()))
full_state_errors_history = np.stack(list(full_state_errors_dict.values()))
full_propagated_formal_errors_epochs = np.stack(list(full_propagated_formal_errors_dict.keys()))
full_propagated_formal_errors_history = np.stack(list(full_propagated_formal_errors_dict.values()))

full_estimation_error_history = np.array([interp1d(full_estimation_error_epochs, state, kind='linear', fill_value='extrapolate')(propagated_covariance_epochs) for state in full_estimation_error_history.T]).T


# show areas where there are no observations:
fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
reference_epoch_array = mission_start_epoch*np.ones(np.shape(full_propagated_formal_errors_epochs))
ax[0].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, 3*full_propagated_formal_errors_history[:,0:3], label=[r"$3\sigma_{x}$", r"$3\sigma_{y}$", r"$3\sigma_{z}$"])
ax[1].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, 3*full_propagated_formal_errors_history[:,6:9], label=[r"$3\sigma_{x}$", r"$3\sigma_{y}$", r"$3\sigma_{z}$"])
for j in range(2):
    for i, gap in enumerate(observation_windows):
        ax[j].axvspan(
            xmin=gap[0]-mission_start_epoch,
            xmax=gap[1]-mission_start_epoch,
            color="gray",
            alpha=0.1,
            label="Observation window" if i == 0 else None)
        ax[j].set_ylabel(r"$\sigma$ [m]")
        ax[j].grid(alpha=0.5, linestyle='--')
ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
ax[0].set_title("LPF")
ax[1].set_title("LUMIO")
fig.suptitle("Propagated formal errors")
plt.legend()
# plt.show()

fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
reference_epoch_array = mission_start_epoch*np.ones(np.shape(full_state_errors_epochs))
for j in range(2):
    labels = ["x", "y", "z"]
    for i in range(3):
        ax[j].plot(utils.convert_epochs_to_MJD(full_state_errors_epochs)-reference_epoch_array, sigma_number*full_state_errors_history[:,6*j+i], label=labels[i])
    for i, gap in enumerate(observation_windows):
        ax[j].axvspan(
            xmin=gap[0]-mission_start_epoch,
            xmax=gap[1]-mission_start_epoch,
            color="gray",
            alpha=0.1,
            label="Observation window" if i == 0 else None)
        ax[j].set_ylabel(r"$||\mathbf{x_{initial}}-\mathbf{x_{final}}||$ [m]")
        ax[j].grid(alpha=0.5, linestyle='--')
ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
ax[0].set_title("LPF")
ax[1].set_title("LUMIO")
fig.suptitle("State errors")
plt.legend()
# plt.show()

fig, ax = plt.subplots(2, 1, figsize=(9, 5))
reference_epoch_array = mission_start_epoch*np.ones(np.shape(full_propagated_formal_errors_epochs))
ax[0].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, np.linalg.norm(3*full_propagated_formal_errors_history[:, 0:3], axis=1))
ax[1].plot(utils.convert_epochs_to_MJD(full_propagated_formal_errors_epochs)-reference_epoch_array, np.linalg.norm(3*full_propagated_formal_errors_history[:, 6:9], axis=1))
for j in range(2):
    for i, gap in enumerate(observation_windows):
        ax[j].axvspan(
            xmin=gap[0]-mission_start_epoch,
            xmax=gap[1]-mission_start_epoch,
            color="gray",
            alpha=0.1,
            label="Observation window" if i == 0 else None)
        ax[j].set_ylabel(r"$\sqrt{(3\sigma_{x})^2 + (3\sigma_{y})^2 + (3\sigma_{z})^2}$ [m]")
        ax[j].grid(alpha=0.5, linestyle='--')
ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
ax[0].set_title("LPF")
ax[1].set_title("LUMIO")
fig.suptitle("Total position uncertainty")
plt.legend()
# plt.show()


fig, ax = plt.subplots(2, 1, figsize=(9, 5))
reference_epoch_array = mission_start_epoch*np.ones(np.shape(full_reference_state_deviation_epochs))
for j in range(2):
    labels = ["x", "y", "z"]
    for i in range(3):
        ax[j].plot(utils.convert_epochs_to_MJD(full_reference_state_deviation_epochs)-reference_epoch_array, full_reference_state_deviation_history[:,6*j+i], label=labels[i])
    for i, gap in enumerate(observation_windows):
        ax[j].axvspan(
            xmin=gap[0]-mission_start_epoch,
            xmax=gap[1]-mission_start_epoch,
            color="gray",
            alpha=0.1,
            label="Observation window" if i == 0 else None)
    ax[j].set_ylabel(r"Deviation [m]")
    ax[j].grid(alpha=0.5, linestyle='--')
ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
ax[0].set_title("LPF")
ax[1].set_title("LUMIO")
fig.suptitle("Deviation from reference orbit")
plt.legend()
# plt.show()

fig, ax = plt.subplots(2, 1, figsize=(9, 5))
reference_epoch_array = mission_start_epoch*np.ones(np.shape(propagated_covariance_epochs))
for j in range(2):
    colors = ["red", "green", "blue"]
    symbols = ["x", "y", "z"]
    for i in range(3):
        sigma = sigma_number*full_propagated_formal_errors_history[:, 6*j+i]
        ax[j].plot(utils.convert_epochs_to_MJD(propagated_covariance_epochs)-reference_epoch_array, sigma, color=colors[i], ls="--", label=f"$3\sigma_{symbols[i]}$", alpha=0.3)
        ax[j].plot(utils.convert_epochs_to_MJD(propagated_covariance_epochs)-reference_epoch_array, -sigma, color=colors[i], ls="-.", alpha=0.3)
        ax[j].plot(utils.convert_epochs_to_MJD(propagated_covariance_epochs)-reference_epoch_array, full_estimation_error_history[:,6*j+i], color=colors[i], label=f"${symbols[i]}-\hat{{{symbols[i]}}}$")
    ax[j].set_ylabel(r"Deviation [m]")
    ax[j].grid(alpha=0.5, linestyle='--')
    for i, gap in enumerate(observation_windows):
        ax[j].axvspan(
            xmin=gap[0]-mission_start_epoch,
            xmax=gap[1]-mission_start_epoch,
            color="gray",
            alpha=0.1,
            label="Observation window" if i == 0 else None)
ax[-1].set_xlabel(f"Time since MJD {mission_start_epoch} [days]")
ax[0].set_title("LPF")
ax[1].set_title("LUMIO")
fig.suptitle("Estimation error")
ax[0].legend(bbox_to_anchor=(1.03, 1), loc='upper left')
plt.tight_layout()
plt.show()
