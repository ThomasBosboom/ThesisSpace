# General imports
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# tudatpy imports
from tudatpy import util
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import time_conversion, element_conversion, frame_conversion
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup


# Tudatpy imports
import tudatpy
from tudatpy.util import result2array
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup, environment_setup, estimation_setup
from tudatpy.kernel.astro import time_conversion, element_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel.math import interpolators, root_finders
from tudatpy.kernel import constants
import dynamic_models.validation_LUMIO
import json



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
tu_cr3bp = 1/np.sqrt((gravitational_parameter_primary + gravitational_parameter_secondary) / \
                          distance_between_primaries ** 3)
lu_cr3bp = distance_between_primaries

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

rotation_rate = 1 / tu_cr3bp

omega_w = np.dot(rsw_to_inertial_rotation_matrix[:, 2], rotation_rate)
omega_w_norm = np.linalg.norm(omega_w)
Omega = np.array([[0, omega_w_norm, 0],[-omega_w_norm, 0, 0],[0, 0, 0]])

time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, -Omega)

total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
                  [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

initial_state_lumio_moon_fixed = np.concatenate((initial_state_lumio_moon_fixed[:3]*lu_cr3bp, initial_state_lumio_moon_fixed[3:]*lu_cr3bp/tu_cr3bp))
initial_state_history_lumio = initial_cartesian_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_lumio_moon_fixed)

print("State history: ", initial_state_history_lumio)

### Is the satellite in the proper position w.r.t the moon and the J2000?

### Is the history of the tudatpy integrated result the same as the true crtbp?

# Spacecraft
body_settings.add_empty_settings(name_spacecraft)
body_settings.get(name_spacecraft).constant_mass = 0.0


with open("lumio_crtbp_j2000.txt", 'r') as file:
    my_dict = json.load(file)
    new_dict = {float(key): value for key, value in my_dict.items()}

print(new_dict)

body_settings.get(name_spacecraft).ephemeris_settings = environment_setup.ephemeris.tabulated(new_dict,
    global_frame_origin,
    global_frame_orientation)

# Create system of selected celestial bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create acceleration models

# Define bodies that are propagated.
bodies_to_propagate = [name_spacecraft]
# Define central bodies.
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
                               propagation_setup.dependent_variable.keplerian_state(name_secondary, name_primary)]

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


state_history_lumio_CRTBP = np.array(list(new_dict.values()))

print(simulation_start_epoch, list(new_dict.keys())[0])

ax = plt.figure().add_subplot(projection='3d')
plt.plot(state_history_lumio_CRTBP[:,0], state_history_lumio_CRTBP[:,1], state_history_lumio_CRTBP[:,2], label="lumio w.r.t earth method 1")
plt.plot(state_history_lumio[:,0], state_history_lumio[:,1], state_history_lumio[:,2], label="lumio w.r.t earth method 2")
ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")
plt.title("Trajectories in inertial Earth-centered J2000 frame")
ax.legend()
plt.axis('equal')
plt.show()



###########################################################################
# CREATE ORBITAL ESTIMATION ###############################################
###########################################################################

## Orbital Estimation
"""
Having defined all settings required for the simulation of the moons' orbits, the orbital estimation can finally be discussed - we will have to create the required link ends for the Galilean moons, define the observation model and simulation settings, simulate the states of the moons based on their associated ephemerides, define the estimable parameters, and finally perform the estimation itself.
"""


### Create Link Ends for the Moons
"""
Since we will be using the [cartesian_position](https://py.api.tudat.space/en/latest/observation.html#tudatpy.numerical_simulation.estimation_setup.observation.cartesian_position) type of observable to simulate the ephemeris-states of the moons, we will have to define the link-ends for all four moons to be of the `observed_body` type. Finally, we will also have to create the complete set of link definitions for each moon individually.
"""

link_ends_lumio = dict()
link_ends_lumio[estimation_setup.observation.observed_body] = estimation_setup.observation.\
    body_origin_link_end_id(name_spacecraft)
link_definition_lumio = estimation_setup.observation.LinkDefinition(link_ends_lumio)

link_definition_dict = {
    name_spacecraft: link_definition_lumio
}


### Observation Model Settings
"""
As mentioned above, we will 'observe' the state of the moons at every epoch as being perfectly cartesian and handily available to the user. However, note that the `cartesian_position` observable is typically not realized in reality but mainly serves verification or analysis purposes.
"""

position_observation_settings = [estimation_setup.observation.cartesian_position(link_definition_lumio)
                                ]


### Observation Simulation Settings
"""
To simulate the states of the moons at every given epochs, we will have to define the simulation settings for all moons. For the problem at hand, they will be entirely identical - we have to define the correct `observable_type` that is associated with the `cartesian_position` observable, give the above-realised `link_definition`, and finally define the epochs at which we want to take the states from the respective ephemerides.

Finally, realise that the default setting for the `reference_link_end_type` argument of the [`tabulated_simulation_settings`](https://py.api.tudat.space/en/latest/observation.html#tudatpy.numerical_simulation.estimation_setup.observation.tabulated_simulation_settings) function is set to `LinkEndType`.receiver. However, to satisfy the estimators expectation when using the `position_observable_type` the default value has to be overwritten and set to `observed_body`. This might be different on a case-by-case situation and should carefully be evaluated when using different types of observables, since the estimation will crash otherwise.
"""

# Define epochs at which the ephemerides shall be checked
observation_times = np.arange(simulation_start_epoch, simulation_start_epoch+propagation_time*86400, 30)

# Create the observation simulation settings per moon
observation_simulation_settings = list()
for body in link_definition_dict.keys():
    observation_simulation_settings.append(estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_dict[body],
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body))

print("Observation settings: ", observation_simulation_settings)


### Simulate Ephemeris' States of Satellites
"""
In a nutshell, what we want to do is to check the ephemeris every three hours - as defined just above - and take the associated (cartesian) state of all four moons at that moment as our observable. However, in order to automatically satisfy all requirements in terms of inputs to the estimator, we have to manually create an `observation_simulator` object, since we explicitly do not want to use the (propagating) simulators that get created alongside the estimator.

The way custom-implemented observation simulators are implemented is that they do not propagate any bodies themselves but simulate the observations based on the (tabulated) ephemerides of all involved bodies. To this end, while setting up the environment we have already set the NOE-5 ephemeris as tabulated ephemerides for all Galilean moons. Thanks to this, we can directly create the required observation simulator object and finally simulate the observations according to the above-defined settings.
"""

# Create observation simulators
ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
    position_observation_settings, bodies)
# Get ephemeris states as ObservationCollection
print('Checking ephemerides...')
ephemeris_satellite_states = estimation.simulate_observations(
    observation_simulation_settings,
    ephemeris_observation_simulators,
    bodies)



### Define Estimable Parameters
"""
Given the problem at hand - minimising the discrepancy between the NOE-5 ephemeris and the states of the moons when propagated under the influence of the above-defined accelerations - we are mainly interested in an improved initial state of all four Galilean moons. We will thus restrict the set of estimable parameters to the moons' initial states.
"""

parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)
original_parameter_vector = parameters_to_estimate.parameter_vector


### Perform the Estimation
"""
Using the set of artificial cartesian 'observations' of the moons' ephemerides we are finally able to estimate improved initial states for each of the four Galilean satellites. To this end we will make use of the known estimation functionality of tudat - nevertheless, note that in order to easily post-process the results we have changed the associated settings such that the moons' state histories will be saved for every iteration of the estimation. All other settings remain unchanged and thus equal to their default values (for more details see [here](https://py.api.tudat.space/en/latest/estimation.html#tudatpy.numerical_simulation.estimation.EstimationInput.define_estimation_settings)).
"""

print('Running propagation...')
with util.redirect_std():
    estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate,
                                            position_observation_settings, propagator_settings)


# Create input object for the estimation
convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=5)
estimation_input = estimation.EstimationInput(ephemeris_satellite_states, convergence_checker=convergence_checker)
# Set methodological options
estimation_input.define_estimation_settings(save_state_history_per_iteration=True)
# Perform the estimation
print('Performing the estimation...')
print(f'Original initial states: {original_parameter_vector}')


with util.redirect_std(redirect_out=False):
    estimation_output = estimator.perform_estimation(estimation_input)
initial_states_updated = parameters_to_estimate.parameter_vector
print('Done with the estimation...')
print(f'Updated initial states: {initial_states_updated}')





## Set up the inversion
"""
To set up the inversion of the problem, we collect all relevant inputs in the form of a covariance input object and define some basic settings of the inversion. Most crucially, this is the step where we can account for different weights - if any - of the different observations, to give the estimator knowledge about the quality of the individual types of observations.
"""

# Create input object for covariance analysis
covariance_input = estimation.CovarianceAnalysisInput(
    ephemeris_satellite_states)

# Set methodological options
covariance_input.define_covariance_settings(
    reintegrate_variational_equations=False)

# Define weighting of the observations in the inversion
# weights_per_observable = {estimation_setup.observation.one_way_instantaneous_doppler_type: noise_level ** -2}
# covariance_input.set_constant_weight_per_observable(weights_per_observable)


### Propagate the covariance matrix
"""
Using the just defined inputs, we can ultimately run the computation of our covariance matrix. Printing the resulting formal errors will give us the diagonal entries of the matrix - while the first six entries represent the uncertainties in the (cartesian) initial state, the seventh and eighth are the errors associated with the gravitational parameter of Earth and the aerodynamic drag coefficient, respectively.
"""

# Perform the covariance analysis
covariance_output = estimator.compute_covariance(covariance_input)


# Print the covariance matrix
print("Formal_errors: ", covariance_output.formal_errors)


## Results post-processing
"""
Finally, to further process the obtained data, one can - exemplary - plot the correlation between the individual estimated parameters, or the behaviour of the formal error over time.
"""


### Correlation
"""
When dealing with the results of covariance analyses - as a measure of how the estimated variable differs from the 'thought' true value - it is important to underline that the correlation between the parameters is another important aspect to take into consideration. In particular, correlation describes how two parameters are related with each other. Typically, a value of 1.0 indicates entirely correlated elements (thus always present on the diagonal, indicating the correlation of an element with itself), a value of 0.0 indicates perfectly uncorrelated elements.
"""

plt.figure(figsize=(9, 6))

plt.imshow(np.abs(covariance_output.correlations), aspect='auto', interpolation='none')
plt.colorbar()

plt.title("Correlation Matrix")
plt.xlabel("Index - Estimated Parameter")
plt.ylabel("Index - Estimated Parameter")

plt.tight_layout()
# plt.show()


### Propagated Formal Errors
"""
"""

initial_covariance = covariance_output.covariance
state_transition_interface = estimator.state_transition_interface
output_times = observation_times

# Propagate formal errors over the course of the orbit
propagated_formal_errors = estimation.propagate_formal_errors_split_output(
    initial_covariance=initial_covariance,
    state_transition_interface=state_transition_interface,
    output_times=output_times)
# Split tuple into epochs and formal errors
epochs = np.array(propagated_formal_errors[0])
formal_errors = np.array(propagated_formal_errors[1])

plt.figure(figsize=(9, 5))
plt.title("Observations as a function of time")
plt.plot(output_times / (24*3600), formal_errors[:, 0], label=r"$x$")
plt.plot(output_times / (24*3600), formal_errors[:, 1], label=r"$y$")
plt.plot(output_times / (24*3600), formal_errors[:, 2], label=r"$z$")

plt.xlabel("Time [days]")
plt.ylabel("Formal Errors in Position [m]")
plt.legend()
plt.grid()

plt.tight_layout()
# plt.show()



# simulation_start_epoch_MJD = 60390
# propagation_time = 40
# fixed_step_size = 0.005
# propagate_moon = True
# central_body = "Earth"
# # initial_state = np.array([[-3.13025630e+08,  2.44821287e+08,  1.72785418e+08, -9.23161169e+02, -6.87350606e+02, -5.23421993e+02]])
# initial_state = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, "LUMIO", "low", propagate_moon=False, central_body=central_body)
# print(initial_state)

