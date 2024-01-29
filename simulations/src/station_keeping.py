# General imports
import copy
import numpy as np
import os
import time
from matplotlib import pyplot as plt

# Tudatpy imports
import tudatpy
from tudatpy.util import result2array
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup, environment_setup, estimation_setup
from tudatpy.kernel.astro import time_conversion, element_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel.math import interpolators, root_finders
from tudatpy.kernel import constants
import validation

import miscellaneous

def get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, name_spacecraft, fidelity, propagate_moon=True, initial_state="standard", central_body="Earth"):

    # print("Simulation started: "+str(name_spacecraft)+", fidelity: "+str(fidelity))

    # Load spice kernels.
    spice.load_standard_kernels()

    # Convert MJD time to J2000 epochs
    simulation_start_epoch = time_conversion.julian_day_to_seconds_since_epoch(time_conversion.modified_julian_day_to_julian_day(simulation_start_epoch_MJD))
    simulation_end_epoch   = time_conversion.julian_day_to_seconds_since_epoch(time_conversion.modified_julian_day_to_julian_day(simulation_start_epoch_MJD+propagation_time))


    ###########################################################################
    # CREATE ENVIRONMENT ######################################################
    ###########################################################################

    name_primary = "Earth"
    name_secondary = "Moon"

    # Create settings for celestial bodies
    bodies_to_create = [name_primary, name_secondary, "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    bodies_to_propagate = [name_spacecraft]
    if central_body == "Earth":
        central_bodies = [name_primary]
        global_frame_origin = name_primary
    elif central_body == "Moon":
        central_bodies = [name_secondary]
        global_frame_origin = name_secondary
    global_frame_orientation = 'J2000'
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)




    # if fidelity=="low":

    #     # Define the ephemeris frame
    #     body_settings.get(name_secondary).ephemeris_settings = environment_setup.ephemeris.custom_ephemeris(
    #         miscellaneous.get_circular_orbit_states_1,
    #         global_frame_origin,
    #         global_frame_orientation)

    # Create environment
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create vehicle object
    bodies.create_empty_body(name_spacecraft)

    if name_spacecraft == "LUMIO":
        if fidelity == "low":
            bodies.get_body(name_spacecraft).mass = 0
        elif fidelity == "high":
            bodies.get_body(name_spacecraft).mass = 22.3
            radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
                "Sun", 0.410644, 1.08, [name_primary, name_secondary])
            environment_setup.add_radiation_pressure_interface(bodies, name_spacecraft, radiation_pressure_settings)

    elif name_spacecraft == "LPF":
        if fidelity == "low":
            bodies.get_body(name_spacecraft).mass = 0
        elif fidelity == "high":
            bodies.get_body(name_spacecraft).mass = 280
            radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
                "Sun", 3.0, 1.8, [name_primary, name_secondary])
            environment_setup.add_radiation_pressure_interface(bodies, name_spacecraft, radiation_pressure_settings)


    ###########################################################################
    # CREATE ACCELERATIONS ####################################################
    ###########################################################################


    if fidelity == "high":

        # Define accelerations acting on vehicle.
        acceleration_settings_on_spacecraft = dict(
                Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(10, 10)],
                Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(10, 10)],
                Sun=[propagation_setup.acceleration.point_mass_gravity(),
                     propagation_setup.acceleration.cannonball_radiation_pressure()],
                Mercury=[propagation_setup.acceleration.point_mass_gravity()],
                Venus=[propagation_setup.acceleration.point_mass_gravity()],
                Mars=[propagation_setup.acceleration.point_mass_gravity()],
                Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
                Saturn=[propagation_setup.acceleration.point_mass_gravity()],
                Uranus=[propagation_setup.acceleration.point_mass_gravity()],
                Neptune=[propagation_setup.acceleration.point_mass_gravity()],
            )

        acceleration_settings_on_moon = dict(
                Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(10, 10)],
                Sun=[propagation_setup.acceleration.point_mass_gravity()],
                Mercury=[propagation_setup.acceleration.point_mass_gravity()],
                Venus=[propagation_setup.acceleration.point_mass_gravity()],
                Mars=[propagation_setup.acceleration.point_mass_gravity()],
                Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
                Saturn=[propagation_setup.acceleration.point_mass_gravity()],
                Uranus=[propagation_setup.acceleration.point_mass_gravity()],
                Neptune=[propagation_setup.acceleration.point_mass_gravity()],
            )


    elif fidelity == "low":

        # Define accelerations acting on vehicle.
        acceleration_settings_on_spacecraft = dict(
                Earth=[propagation_setup.acceleration.point_mass_gravity()],
                Moon=[propagation_setup.acceleration.point_mass_gravity()],
            )

        acceleration_settings_on_moon = dict(
                Earth=[propagation_setup.acceleration.point_mass_gravity()],
            )

    # Create global accelerations dictionary.
    acceleration_settings = {
        name_spacecraft: acceleration_settings_on_spacecraft
        }

    if propagate_moon==True:

        bodies_to_propagate = [name_spacecraft, name_secondary]
        central_bodies = [name_primary, name_primary]
        acceleration_settings = {
            name_spacecraft: acceleration_settings_on_spacecraft,
            name_secondary: acceleration_settings_on_moon
            }

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies)


    ###########################################################################
    # CREATE INITIAL STATE ####################################################
    ###########################################################################


    moon_initial_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name = name_secondary,
            observer_body_name = central_bodies[0],
            reference_frame_name = global_frame_orientation,
            aberration_corrections = 'NONE',
            ephemeris_time = simulation_start_epoch
        )

    if fidelity == "low":
        moon_initial_state_adjusted = miscellaneous.get_circular_initial_orbit_state(semimajor_axis=3.8474796e8,
                                                                                     position_vector=moon_initial_state[:3],
                                                                                     gravitational_parameter=spice.get_body_gravitational_parameter(name_primary))

        translation_vector = moon_initial_state[:3] - moon_initial_state_adjusted[:3]
        moon_initial_state = moon_initial_state_adjusted + np.hstack((translation_vector, np.zeros((3))))
        moon_initial_state = moon_initial_state_adjusted

        # print(moon_initial_state_adjusted, np.linalg.norm(moon_initial_state_adjusted[:3]))
        # print("kepler: ", element_conversion.cartesian_to_keplerian(np.array([moon_initial_state]).T, spice.get_body_gravitational_parameter(name_primary)))

    if isinstance(initial_state, str):

        if name_spacecraft == "LUMIO":

            system_initial_state = validation.get_reference_state_history(simulation_start_epoch_MJD, propagation_time)[0][0]

        if name_spacecraft == "LPF":

            initial_state_lpf_moon = element_conversion.keplerian_to_cartesian_elementwise(
                gravitational_parameter=spice.get_body_gravitational_parameter(name_secondary),
                semi_major_axis=5737.4E3,
                eccentricity=0.61,
                inclination=np.deg2rad(57.83),
                argument_of_periapsis=np.deg2rad(90),
                longitude_of_ascending_node=np.rad2deg(61.55),
                true_anomaly=np.deg2rad(0)
            )

            system_initial_state = np.add(initial_state_lpf_moon, moon_initial_state)

        if propagate_moon==True:

            system_initial_state = np.concatenate((system_initial_state,moon_initial_state))


    else:

        system_initial_state = initial_state

        if propagate_moon==True:

            system_initial_state = np.concatenate((system_initial_state,moon_initial_state))



    ###########################################################################
    # CREATE PROPAGATION SETTINGS #############################################
    ###########################################################################

    # Define required outputs
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.total_acceleration(name_spacecraft),
        propagation_setup.dependent_variable.relative_distance(name_secondary, name_primary),
        propagation_setup.dependent_variable.relative_velocity(name_secondary, name_primary)
    ]

    # Create termination settings
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

    # Create numerical integrator settings
    fixed_step_size = fixed_step_size*constants.JULIAN_DAY
    integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)

    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        system_initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings,
        output_variables= dependent_variables_to_save
    )


    ###########################################################################
    # PROPAGATE ORBIT #########################################################
    ###########################################################################

    # Create simulation object and propagate dynamics.
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        propagator_settings
    )

    # Setup parameters settings to propagate the state transition matrix
    parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)# Create the parameters that will be estimated
    parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
    variational_equations_solver = numerical_simulation.create_variational_equations_solver(
            bodies,
            propagator_settings,
            parameters_to_estimate,
            simulate_dynamics_on_creation=True
    )

    # Extract the simulation results
    state_history                   = np.vstack(list(variational_equations_solver.state_history.values()))
    dependent_variables_history     = np.vstack(list(dynamics_simulator.dependent_variable_history.values()))
    state_transition_matrix_history = np.vstack(list(variational_equations_solver.state_transition_matrix_history.values())).reshape((np.shape(state_history)[0], np.shape(state_history)[1], np.shape(state_history)[1]))

    # print("Simulation ended: "+str(name_spacecraft)+", fidelity: "+str(fidelity))

    return state_history[:,:6], dependent_variables_history[:,:6], state_transition_matrix_history[:,:6,:6]






def get_corrected_initial_state(state_history, state_transition_matrix_history, reference_state_history, cut_off_epoch, correction_epoch, target_point_epoch):

    R_i = 1e-2*np.eye(3)
    Q = 1e-1*np.eye(3)

    sum_array = np.zeros((3,3))
    sum_array_2 = np.zeros(3)
    for index, t_i in enumerate(target_point_epochs):
        t_i = int(t_i)
        t_c = int(t_c)
        t_v = int(t_v)

        Phi_tvti_rv = np.dot(state_transition_matrix_history[t_v], np.linalg.inv(state_transition_matrix_history[t_i]))[3:,:3]
        Phi_tcti_rv = np.dot(state_transition_matrix_history[t_c], np.linalg.inv(state_transition_matrix_history[t_i]))[3:,:3]
        Phi_tcti_rr = np.dot(state_transition_matrix_history[t_c], np.linalg.inv(state_transition_matrix_history[t_i]))[:3,:3]
        sum_array   = np.add(sum_array, Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tvti_rv)

        alpha_i = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rr
        beta_i  = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rv

        if index == 0: t_v_old = correction_epochs[index]
        else: t_v_old = correction_epochs[index-1]
        # print(index, t_i, t_v, t_c, int(t_v-t_v_old))
        delta_state_cutoff = np.dot(state_transition_matrix_history[t_c], np.linalg.inv(state_transition_matrix_history[0])) @ initial_delta_state # assuming no OD
        sum_array_2 = np.add(sum_array_2, alpha_i @ delta_state_cutoff[:3] + beta_i @ delta_state_cutoff[3:])

    A = -np.linalg.inv(sum_array+ (Q.T+Q))
    delta_v_sk_array = A @ sum_array_2

    # delta_state_correction = np.dot(state_transition_matrix_history[t_v], np.linalg.inv(state_transition_matrix_history[int(t_v-t_v_old)])) @ initial_delta_state

    corrected_initial_state = np.concatenate((np.zeros(3),delta_v_sk_array))

    return corrected_initial_state




def get_corrected_initial_state(state_history, state_transition_matrix_history, reference_state_history, cut_off_epoch, correction_epoch, target_point_epoch):

    state_deviation_history = state_history - reference_state_history

    dr_tc = state_deviation_history[correction_epoch,:3]
    dv_tc = state_deviation_history[correction_epoch,3:]
    dr_ti = state_deviation_history[target_point_epoch,:3]

    Phi = state_transition_matrix_history
    Phi_tcti = np.linalg.inv(Phi[target_point_epoch]) @ Phi[cut_off_epoch]
    Phi_tvti = np.linalg.inv(Phi[target_point_epoch]) @ Phi[correction_epoch]
    Phi_tvti_rr = Phi_tvti[:3,:3]
    Phi_tvti_rv = Phi_tvti[:3,3:]
    Phi_tcti_rr = Phi_tcti[:3,:3]
    Phi_tcti_rv = Phi_tcti[:3,3:]

    delta_v = -np.linalg.inv(Phi_tvti_rv) @ Phi_tvti_rr @ dr_tc - dv_tc

    state_history[correction_epoch, 3:] = state_history[correction_epoch, 3:] + delta_v

    print(delta_v)

    # # return state_history[correction_epoch]



    R_i = 1e-2*np.eye(3)
    Q = 1e-1*np.eye(3)

    A = -np.linalg.inv(np.add((Q.T+Q), Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tvti_rv))
    alpha_i = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rr
    beta_i  = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rv

    delta_v = A @ (alpha_i @ dr_tc + beta_i @ dv_tc)

    print(delta_v)

    state_history[correction_epoch, 3:] = state_history[correction_epoch, 3:] + delta_v


    delta_v = np.linalg.inv(Phi[target_point_epoch,:3,3:]) @ (dr_ti)

    print(delta_v)

    state_history[correction_epoch, 3:] = state_history[correction_epoch, 3:] + delta_v

    return state_history[correction_epoch]



def get_differential_corrector(simulation_start_epoch_MJD, propagation_time, fixed_step_size, name_spacecraft, fidelity, target_point_epoch, iterations=5):

    state_history, _, state_transition_matrix_history = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, name_spacecraft, fidelity)
    # reference_state_history = validation.get_reference_state_history(simulation_start_epoch_MJD, propagation_time, fixed_step_size=fixed_step_size)[0]
    reference_state_history = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, name_spacecraft, "high")[0]
    final_state_deviation = state_history[target_point_epoch] - reference_state_history[target_point_epoch]
    final_state_transition_matrix = state_transition_matrix_history[target_point_epoch]

    delta_v = np.linalg.inv(-final_state_transition_matrix[:3,3:]) @ final_state_deviation[:3]
    initial_state_corrected = np.add(state_history[0], np.concatenate((np.zeros(3),delta_v)))

    state_history_corrected, _, state_transition_matrix_history_corrected = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, name_spacecraft, fidelity, initial_state=initial_state_corrected)
    final_state_deviation_corrected = state_history_corrected[target_point_epoch]  - reference_state_history[target_point_epoch]

    # print(np.linalg.norm(delta_v), delta_v)
    # print(initial_state_corrected)

    for i in range(iterations):

        delta_v = np.linalg.inv(-final_state_transition_matrix[:3,3:]) @ final_state_deviation_corrected[:3]
        initial_state_corrected = initial_state_corrected + np.concatenate((np.zeros(3),delta_v))

        state_history_corrected, _, _ = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, name_spacecraft, fidelity, initial_state=initial_state_corrected)

        final_state_deviation_corrected = state_history_corrected[target_point_epoch]  - reference_state_history[target_point_epoch]

        # print(np.linalg.norm(delta_v), delta_v)
        # print(initial_state_corrected)

    return initial_state_corrected




# def get_simulation_results_corrected(simulation_start_epoch_MJD, propagation_time, fixed_step_size, name_spacecraft, fidelity, initial_delta_state, propagate_moon=True, initial_state="standard", central_body="Earth"):

#     state_history, _, state_transition_matrix_history = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, name_spacecraft, fidelity, propagate_moon=propagate_moon, initial_state="standard", central_body=central_body)

#     arc_epochs = np.arange(0, np.shape(state_history)[0], 7/fixed_step_size)
#     correction_epochs = np.delete(arc_epochs, np.arange(3, len(arc_epochs), 4))
#     # cutoff_epochs = correction_epochs - 0.5/fixed_step_size*np.ones(np.shape(correction_epochs))
#     # target_point_epochs = correction_epochs - 1000*np.ones(np.shape(correction_epochs))

#     print(arc_epochs, correction_epochs)

#     for index, correction_epoch in enumerate(correction_epochs):

#         # Compute initial and final time for arc
#         if index < len(correction_epochs)-1:
#             current_arc_initial_time = correction_epochs[index]*fixed_step_size
#             current_arc_final_time = correction_epochs[index+1]*fixed_step_size
#             current_arc_duration = current_arc_final_time - current_arc_initial_time

#         print(current_arc_initial_time, current_arc_final_time, current_arc_duration)

#         state_history, _, state_transition_matrix_history = get_simulation_results(simulation_start_epoch_MJD+current_arc_initial_time, current_arc_duration, fixed_step_size, name_spacecraft, fidelity, propagate_moon=propagate_moon, central_body=central_body)

#         print(state_history[0], state_history[-1])
#         # corrected_initial_state = get_corrected_initial_state(state_history, state_transition_matrix_history, initial_delta_state, target_point_epochs, cutoff_epochs[index], correction_epochs[index])

#         # print(corrected_initial_state)




simulation_start_epoch_MJD = 60390
propagation_time = 30
fixed_step_size = 0.005
propagate_moon = True
central_body = "Earth"

state_history, _, state_transition_matrix_history = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, "LUMIO", "high", propagate_moon=propagate_moon, central_body=central_body)

reference_state_history = validation.get_reference_state_history(simulation_start_epoch_MJD,propagation_time, fixed_step_size=fixed_step_size)[0]
correction_epoch = int((1)/fixed_step_size)
cut_off_epoch = correction_epoch
target_point_epoch = int((8)/fixed_step_size)
initial_state = get_corrected_initial_state(state_history, state_transition_matrix_history, reference_state_history, cut_off_epoch, correction_epoch, target_point_epoch)

initial_state = get_differential_corrector(simulation_start_epoch_MJD, propagation_time, fixed_step_size, "LUMIO", "low", target_point_epoch, iterations=5)

states_history_LUMIO_high = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, "LUMIO", "high", propagate_moon=propagate_moon, central_body=central_body)[0]
states_history_LUMIO_low  = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, "LUMIO",  "low", initial_state=initial_state, propagate_moon=propagate_moon, central_body=central_body)[0]
states_history_LPF_high   = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, "LPF",   "high", propagate_moon=propagate_moon, central_body=central_body)[0]
states_history_LPF_low    = get_simulation_results(simulation_start_epoch_MJD, propagation_time, fixed_step_size, "LPF",    "low", propagate_moon=propagate_moon, central_body=central_body)[0]


ax = plt.figure().add_subplot(projection='3d')
ax.set_title("States of satellite, propagation time: "+str(propagation_time)+" days since MJD 60390, step size: "+str(fixed_step_size)+" days")
plt.plot(states_history_LUMIO_high[:,0], states_history_LUMIO_high[:,1], states_history_LUMIO_high[:,2], color="blue", label="LUMIO, high")
plt.plot(states_history_LUMIO_low[:,0], states_history_LUMIO_low[:,1], states_history_LUMIO_low[:,2],  color="blue", label="LUMIO, low", linestyle='dashed')
plt.plot(states_history_LPF_high[:,0], states_history_LPF_high[:,1], states_history_LPF_high[:,2], color="red", label="LPF, high")
plt.plot(states_history_LPF_low[:,0], states_history_LPF_low[:,1], states_history_LPF_low[:,2],  color="red", label="LPF, low", linestyle='dashed')
plt.plot(reference_state_history[:,0], reference_state_history[:,1], reference_state_history[:,2], color="green", label="validation data")
ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")
ax.legend()
plt.axis('equal')
plt.show()