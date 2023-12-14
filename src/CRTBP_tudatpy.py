# General imports
import copy
import numpy as np
import json
import os
from matplotlib import pyplot as plt

# Tudatpy imports
import tudatpy
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.kernel.interface import spice
from tudatpy.kernel.astro import time_conversion, element_conversion, frame_conversion
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup, environment_setup, estimation_setup
from tudatpy.kernel.astro import polyhedron_utilities
from tudatpy.kernel.math import interpolators, root_finders



## Auxiliary Functions
import dynamic_models.validation_LUMIO

# Since the CR3BP is being used, functions to compute the units of length and time used to make the CR3BP dimensionless are first defined.

########################################################################################################################
# Compute unit of length of the CR3BP
def cr3bp_unit_of_length (distance_between_primaries: float) -> float:
    return distance_between_primaries

########################################################################################################################
# Compute unit of time of the CR3BP
def cr3bp_unit_of_time (gravitational_parameter_primary: float,
                        gravitational_parameter_secondary: float,
                        distance_between_primaries: float) -> float:

    mean_motion = np.sqrt((gravitational_parameter_primary + gravitational_parameter_secondary) / \
                          distance_between_primaries ** 3)
    unit = 1/mean_motion

    return unit


# Load spice kernels.
spice.load_standard_kernels()

time_from_start = 0
simulation_start_epoch = 60390+time_from_start
propagation_time = 28

G  = 6.67408E-11
m1 = 5.97219E+24
m2 = 7.34767E+22
a  = 3.84747963e8

name_primary = "Earth"
name_secondary = "Moon"
name_spacecraft = "Spacecraft"

gravitational_parameter_primary = spice.get_body_gravitational_parameter(name_primary)
gravitational_parameter_secondary = spice.get_body_gravitational_parameter(name_secondary)
distance_between_primaries = a

mu = gravitational_parameter_secondary/(gravitational_parameter_primary+gravitational_parameter_secondary)

# Get CR3BP units
tu_cr3bp = cr3bp_unit_of_time(gravitational_parameter_primary, gravitational_parameter_secondary,
                              distance_between_primaries)
lu_cr3bp = cr3bp_unit_of_length(distance_between_primaries)

simulation_start_epoch = time_conversion.julian_day_to_seconds_since_epoch(time_conversion.modified_julian_day_to_julian_day(simulation_start_epoch))

# Frame origin and orientation
global_frame_origin = name_primary
global_frame_orientation = "J2000"

bodies_to_create = [name_primary, name_secondary, "Sun"]

# Define body settings
body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation)

# semimajor_axis = 3.8474796e8
moon_initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name = name_secondary,
        observer_body_name = name_primary,
        reference_frame_name = global_frame_orientation,
        aberration_corrections = 'NONE',
        ephemeris_time = simulation_start_epoch
    )

# Define the computation of the Kepler orbit ephemeris
central_body_gravitational_parameter = spice.get_body_gravitational_parameter(name_primary) + spice.get_body_gravitational_parameter(name_secondary) 
initial_keplerian_moon_state = element_conversion.cartesian_to_keplerian(moon_initial_state, central_body_gravitational_parameter)
initial_keplerian_moon_state[0], initial_keplerian_moon_state[1] = distance_between_primaries, 0      # Make orbit circular at the right radius

initial_cartesian_moon_state = element_conversion.keplerian_to_cartesian(initial_keplerian_moon_state, central_body_gravitational_parameter)


### Is the earth-moon distance the same everywhere?: YES


# Create ephemeris settings and add to body settings of "Moon"
body_settings.get(name_secondary).ephemeris_settings = environment_setup.ephemeris.keplerian(
    initial_keplerian_moon_state,
    simulation_start_epoch,
    central_body_gravitational_parameter,
    global_frame_origin, global_frame_orientation)

initial_state_lumio_barycenter_fixed = np.array([1.1473302, 0, -0.15142308, 0, -0.21994554, 0])
initial_state_lumio_barycenter_fixed[0] = initial_state_lumio_barycenter_fixed[0] - (1-mu)
initial_state_lumio_moon_fixed = initial_state_lumio_barycenter_fixed


rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(initial_cartesian_moon_state)

rotation_rate = 1 / cr3bp_unit_of_time(
    gravitational_parameter_primary, gravitational_parameter_secondary, distance_between_primaries)

omega_w = np.dot(rsw_to_inertial_rotation_matrix[:, 2], rotation_rate)
omega_w_norm = np.linalg.norm(omega_w)
Omega = np.array([[0, omega_w_norm, 0],[-omega_w_norm, 0, 0],[0, 0, 0]])

time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, -Omega)

total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
                  [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

initial_state_lumio_moon_fixed = np.concatenate((initial_state_lumio_moon_fixed[:3]*lu_cr3bp, initial_state_lumio_moon_fixed[3:]*lu_cr3bp/tu_cr3bp))
initial_state_history_lumio = initial_cartesian_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_lumio_moon_fixed)

print("initial_state_history_lumio", initial_state_history_lumio)
# initial_state_history_lumio = np.array([[-3.09965465e+08,  3.07254364e+08,  1.09190626e+08, -6.94514593e+02, -5.92981679e+02, -3.02950273e+02])

### Is the satellite in the proper position w.r.t the moon and the J2000?

### Is the history of the tudatpy integrated result the same as the true crtbp?

# Spacecraft
body_settings.add_empty_settings(name_spacecraft)
body_settings.get(name_spacecraft).constant_mass = 0

# Create system of selected celestial bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create acceleration models

# Define bodies that are propagated.
bodies_to_propagate = [name_spacecraft]
central_bodies = [name_primary]

# Define accelerations acting on spacecraft
acceleration_settings_on_spacecraft = {
    name_primary: [propagation_setup.acceleration.point_mass_gravity()],
    name_secondary: [propagation_setup.acceleration.point_mass_gravity()]
}

# Create global accelerations settings dictionary.
acceleration_settings = {name_spacecraft: acceleration_settings_on_spacecraft}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies)

# Create integrator settings

# current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkdp_87
# # Define absolute and relative tolerance
# current_tolerance = 1e-12
# initial_time_step = 1e-6
# # Maximum step size: inf; minimum step size: eps
# integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(initial_time_step,
#                                                                                   current_coefficient_set,
#                                                                                   np.finfo(float).eps, 
#                                                                                   np.inf,
#                                                                                   current_tolerance, 
#                                                                                   current_tolerance)

integrator_settings = propagation_setup.integrator.runge_kutta_4(simulation_start_epoch, 0.005*86400)

# Select dependent variables
dependent_variables_to_save = [propagation_setup.dependent_variable.relative_position(name_secondary, name_primary),
                               propagation_setup.dependent_variable.relative_velocity(name_secondary, name_primary),
                               propagation_setup.dependent_variable.relative_position(name_spacecraft, name_primary),
                               propagation_setup.dependent_variable.relative_velocity(name_spacecraft, name_primary),
                               propagation_setup.dependent_variable.relative_distance(name_secondary, name_primary),
                               propagation_setup.dependent_variable.relative_distance(name_spacecraft, name_secondary),
                               propagation_setup.dependent_variable.relative_speed(name_spacecraft, name_secondary),
                               propagation_setup.dependent_variable.keplerian_state(name_secondary, name_primary), 
                               propagation_setup.dependent_variable.total_acceleration(name_spacecraft)]

# Create propagator settings
termination_settings = propagation_setup.propagator.time_termination(simulation_start_epoch+propagation_time*86400)
propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_state_history_lumio,
            simulation_start_epoch,
            integrator_settings,
            termination_settings,
            output_variables= dependent_variables_to_save
        )

print("PROPAGATOR SETTINGS: ", propagator_settings)
print(central_bodies)
print(acceleration_models)
print(bodies_to_propagate)
print(initial_state_history_lumio)
print(simulation_start_epoch)
print(integrator_settings)
print(termination_settings)

# Propagate variational equations, propagating just the STM
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
variational_equations_solver = numerical_simulation.create_variational_equations_solver(
        bodies,
        propagator_settings,
        parameters_to_estimate,
        simulate_dynamics_on_creation=True)

# Retrieve state and STM history and convert them to body-fixed frame
state_history_lumio = variational_equations_solver.state_history
stm_history_lumio_inertial = variational_equations_solver.state_transition_matrix_history

# Create simulation object and propagate dynamics.
dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
dependent_variable_history_lumio_inertial = dynamics_simulator.dependent_variable_history

state_history_lumio = np.vstack(list(state_history_lumio.values()))
dependent_variables_to_save = np.vstack(list(dependent_variable_history_lumio_inertial.values()))


print("Moon states: ", dependent_variables_to_save[0,:6], initial_cartesian_moon_state)
print("Initial state lumio", initial_state_history_lumio)


# [-2.66109637e+08  2.41012575e+08  1.38309778e+08 -7.39210793e+02
#  -6.31143624e+02 -3.22446949e+02]
# 2.6616994633772318e-06
# [-2.65747649e+08  2.39100536e+08  1.39418264e+08  1.07394573e+03
#  -1.73251153e+02 -1.23853301e+02 -3.09965465e+08  3.07254364e+08
#   1.09190626e+08 -6.94514593e+02 -5.92981679e+02 -3.02950273e+02]














# ######## TESTS ##############################################################




import CRTBP_traditional

G  = 6.67408E-11
m1 = 5.97219E+24
m2 = 7.34767E+22
a  = 3.8474796e8
print(bodies.get("Earth").mass)

state_rotating_bary_lumio_0 = [1.1473302, 0, -0.15142308, 0, -0.21994554, 0]
# state_rotating_bary_LPF_0   = [0.98512134, 0.00147649, 0.00492546, -0.87329730, -1.61190048, 0]
start = 0
stop = propagation_time
step = 0.005

system   = CRTBP_traditional.CRTBP(G, m1, m2, a)
t, state_rotating_bary_lumio = system.get_state_history(state_rotating_bary_lumio_0, start, stop, step)
state_rotating_secondary_lumio = system.convert_state_barycentric_to_body(state_rotating_bary_lumio, "secondary", state_type="rotating")
# state_rotating_bary_lumio[:, 0] = state_rotating_bary_lumio[:,0] - (1-mu)
# state_rotating_secondary_lumio = state_rotating_bary_lumio
# t, state_rotating_bary_LPF   = system.get_state_history(state_rotating_bary_LPF_0, start, stop, step)[1]

# Looping through all epochs to convert each synodic frame element to J2000 Earth-centered
state_history_lumio_CRTBP = np.empty(np.shape(state_rotating_secondary_lumio))
for epoch, state in enumerate(state_rotating_secondary_lumio):

    rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(dependent_variables_to_save[epoch, :6])

    rotation_rate = 1 / cr3bp_unit_of_time(
        gravitational_parameter_primary, gravitational_parameter_secondary, distance_between_primaries)

    omega_w = np.dot(rsw_to_inertial_rotation_matrix[:, 2], rotation_rate)
    omega_w_norm = np.linalg.norm(omega_w)
    Omega = np.array([[0, omega_w_norm, 0],[-omega_w_norm, 0, 0],[0, 0, 0]])

    time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, -Omega)

    total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
                    [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

    initial_state_lumio_moon_fixed = np.concatenate((state_rotating_secondary_lumio[epoch, :3]*lu_cr3bp, state_rotating_secondary_lumio[epoch, 3:]*lu_cr3bp/tu_cr3bp))
    state_history_lumio_CRTBP[epoch] = dependent_variables_to_save[epoch, :6] + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_lumio_moon_fixed)

    
array_dict = {float(f'{simulation_start_epoch+i*0.005*86400}'): row.tolist() for i, row in enumerate(state_history_lumio_CRTBP)}
file_name = 'lumio_crtbp_j2000.txt'
with open(file_name, 'w') as file:
    json.dump(array_dict, file, indent=4)




ax = plt.figure().add_subplot(projection='3d')
plt.plot(dependent_variables_to_save[:,0], dependent_variables_to_save[:,1], dependent_variables_to_save[:,2], label="moon w.r.t earth")
plt.plot(state_history_lumio_CRTBP[:,0], state_history_lumio_CRTBP[:,1], state_history_lumio_CRTBP[:,2], label="lumio w.r.t earth method 1")
plt.plot(state_history_lumio[:,0], state_history_lumio[:,1], state_history_lumio[:,2], label="lumio w.r.t earth method 2")
ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")
plt.title("Trajectories in inertial Earth-centered J2000 frame")
ax.legend()
plt.axis('equal')
plt.show()

# reference_states_lumio = validation_LUMIO.get_reference_state_history(60390, 28, fixed_step_size=0.005)

# ax = plt.figure().add_subplot(projection='3d')
# plt.plot(reference_states_lumio[0][:,0], reference_states_lumio[0][:,1], reference_states_lumio[0][:,2], label="sat w.r.t earth ref")
# plt.plot(state_history_lumio_CRTBP[:,0], state_history_lumio_CRTBP[:,1], state_history_lumio_CRTBP[:,2], label="sat w.r.t earth")
# ax.set_xlabel("X [km]")
# ax.set_ylabel("Y [km]")
# ax.set_zlabel("Z [km]")
# plt.title("Trajectories in inertial Earth-centered J2000 frame (unshifted)")
# ax.legend()
# plt.axis('equal')

# # 1380 to match value of time dependent crtbp result to actual J2000 epoch.
# sat_moon_distance_reference = np.linalg.norm(reference_states_lumio[0][:,:3]-reference_states_lumio[1][:,:3], axis=1)
# sat_moon_distance_CRTBP = np.linalg.norm(state_rotating_secondary_lumio[1380:,:3]*lu_cr3bp, axis=1)
# sat_moon_distance = np.linalg.norm(state_history_lumio_CRTBP[1380:,:3]-dependent_variables_to_save[1380:,:3], axis=1)
# sat_moon_distance_tudatpy = np.linalg.norm(dependent_variables_to_save[1380:,6:9]-dependent_variables_to_save[1380:,0:3], axis=1)
# # sat_moon_distance_tudatpy = dependent_variables_to_save[1380:,-8]

# ax = plt.figure()
# plt.plot(sat_moon_distance_reference, label="sat-moon refence (not CRTBP, but higher fidelity)")
# plt.plot(sat_moon_distance_CRTBP, label="sat-moon directly from CRTBP")
# plt.plot(sat_moon_distance, label="sat-moon CRTBP converted to J2000 (method 1)")
# plt.plot(sat_moon_distance_tudatpy, label="sat-moon directly from tudatpy (method 2)")
# plt.xlabel("indices")
# plt.ylabel("distance lumio w.r.t moon [m]")
# ax.legend()


# # 1380 to match value of time dependent crtbp result to actual J2000 epoch.
# sat_moon_distance_reference = np.linalg.norm(reference_states_lumio[0][:,3:]-reference_states_lumio[1][:,3:], axis=1)
# sat_moon_distance_CRTBP = np.linalg.norm(state_rotating_secondary_lumio[1380:,3:]*lu_cr3bp/tu_cr3bp, axis=1)
# sat_moon_distance = np.linalg.norm(state_history_lumio_CRTBP[1380:,3:]-dependent_variables_to_save[1380:,3:6], axis=1)
# sat_moon_distance_tudatpy = np.linalg.norm(dependent_variables_to_save[1380:,9:12]-dependent_variables_to_save[1380:,3:6], axis=1)
# # sat_moon_distance_tudatpy = dependent_variables_to_save[1380:,-7]

# ax = plt.figure()
# plt.plot(sat_moon_distance_reference, label="sat-moon refence (not CRTBP, but higher fidelity)")
# plt.plot(sat_moon_distance_CRTBP, label="sat-moon directly from CRTBP")
# plt.plot(sat_moon_distance, label="sat-moon CRTBP converted to J2000 (method 1)")
# plt.plot(sat_moon_distance_tudatpy, label="sat-moon directly from tudatpy (method 2)")
# plt.xlabel("indices")
# plt.ylabel("speed lumio w.r.t moon [m/s]")
# ax.legend()
# # plt.show()


# print("Initial moon state: ", initial_cartesian_moon_state)
# print("Initial moon state in propagator: ", dependent_variables_to_save[0,:6])

# # print("moon state day 1: ", initial_cartesian_moon_state)
# print("Initial moon state in propagator day 1: ", dependent_variables_to_save[int(1/0.005),:6])

# print("Initial lumio state CRTBP: ", state_history_lumio_CRTBP[0])
# print("Initial lumio state tudatpy before: ", initial_state_history_lumio)
# print("Initial lumio state tudatpy: ", state_history_lumio[0])


# ax = plt.figure()
# # plt.plot(state_history_lumio_CRTBP[:,:3], label="position w.r.t. Earth CRTBP", color="blue")
# # plt.plot(dependent_variables_to_save[:,6:9], label="position w.r.t. Earth", color="blue", ls="--")
# plt.plot(state_history_lumio_CRTBP[:,3:], label="velocity w.r.t. Earth CRTBP", color="red")
# plt.plot(dependent_variables_to_save[:,9:12], label="velocity w.r.t. Earth", color="red", ls="--")
# ax.legend()
# plt.show()