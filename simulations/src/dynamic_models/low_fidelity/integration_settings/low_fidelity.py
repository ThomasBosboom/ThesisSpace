# General imports
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

# Tudatpy imports
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup, environment_setup, estimation_setup
from tudatpy.kernel.astro import element_conversion, frame_conversion
from tudatpy.kernel.interface import spice

# Define path to import src files
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(parent_dir))

# Own
from dynamic_models import validation_LUMIO
from DynamicModelBase import DynamicModelBase


class LowFidelityDynamicModel(DynamicModelBase):

    def __init__(self, simulation_start_epoch_MJD, propagation_time):
        super().__init__(simulation_start_epoch_MJD, propagation_time)

        # Get CRTBP characteristics
        self.distance_between_primaries = 3.84747963e8
        self.eccentricity = 0
        self.bodies_mass = [0,0]
        self.lu_cr3bp = self.distance_between_primaries
        self.tu_cr3bp = 1/np.sqrt((self.gravitational_parameter_primary + self.gravitational_parameter_secondary)/self.distance_between_primaries**3)
        self.mu = self.gravitational_parameter_secondary/(self.gravitational_parameter_primary+self.gravitational_parameter_secondary)
        self.rotation_rate = 1/self.tu_cr3bp


    def set_initial_cartesian_moon_state(self):

        # Adjust the orbit of the moon to perfectly circular
        self.moon_initial_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name = self.name_secondary,
            observer_body_name = self.name_primary,
            reference_frame_name = self.global_frame_orientation,
            aberration_corrections = 'NONE',
            ephemeris_time = self.simulation_start_epoch)

        central_body_gravitational_parameter = self.gravitational_parameter_primary + self.gravitational_parameter_secondary
        initial_keplerian_moon_state = element_conversion.cartesian_to_keplerian(self.moon_initial_state,
                                                                                 central_body_gravitational_parameter)
        initial_keplerian_moon_state[0] = self.distance_between_primaries
        initial_keplerian_moon_state[1] = self.eccentricity
        self.initial_cartesian_moon_state = element_conversion.keplerian_to_cartesian(initial_keplerian_moon_state,
                                                                                      central_body_gravitational_parameter)


    def convert_synodic_to_inertial_state(self, initial_state_barycenter_fixed):

        self.set_initial_cartesian_moon_state()

        # Convert barycentric CRTBP non-dimensionalized state into moon fixed frame.
        initial_state_barycenter_fixed[0] = initial_state_barycenter_fixed[0] - (1-self.mu)
        initial_state_barycenter_fixed[6] = initial_state_barycenter_fixed[6] - (1-self.mu)
        initial_state_moon_fixed = initial_state_barycenter_fixed

        # Compute the transformation matrix from CRTBP synodic frame to Earth-centered J2000 frame
        rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(self.initial_cartesian_moon_state)

        # Compute time derivative element for the conversion of the velocity state elements and combine into final matrix
        omega_w_norm = np.linalg.norm(np.dot(rsw_to_inertial_rotation_matrix[:, 2], self.rotation_rate))
        Omega = np.array([[0, omega_w_norm, 0],[-omega_w_norm, 0, 0],[0, 0, 0]])

        # rotational_rates = np.cross(self.initial_cartesian_moon_state[:3], self.initial_cartesian_moon_state[3:])/np.linalg.norm(self.initial_cartesian_moon_state[:3])**2
        # w1, w2, w3 = rotational_rates[:3]
        # Omega = np.array([[0, -w3, w2],
        #                   [w3, 0, -w1],
        #                   [-w2, w1, 0]])
        time_derivative_rsw_to_inertial_rotation_matrix = -np.dot(rsw_to_inertial_rotation_matrix, Omega)
        total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
                        [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

        # Convert initial state defined in CRTBP synodic frame to Earth-centered J2000 frame
        initial_state_moon_fixed_lpf = np.concatenate((initial_state_moon_fixed[:3]*self.lu_cr3bp, initial_state_moon_fixed[3:6]*self.lu_cr3bp/self.tu_cr3bp))
        initial_state_moon_fixed_lumio = np.concatenate((initial_state_moon_fixed[6:9]*self.lu_cr3bp, initial_state_moon_fixed[9:12]*self.lu_cr3bp/self.tu_cr3bp))
        initial_state_lpf = self.initial_cartesian_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_moon_fixed_lpf)
        initial_state_lumio = self.initial_cartesian_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_moon_fixed_lumio)
        self.initial_state = np.concatenate((initial_state_lpf, initial_state_lumio))


    # def convert_cartesian_state_to_synodic(self):

    #     self.set_initial_cartesian_moon_state()

    #     # initial_state_LPF = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_ELO, )
    #     state_history_moon = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_LPO, body="moon", get_epoch_in_array=True, get_full_history=True)
    #     state_history_lumio = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_LPO, body="satellite", get_epoch_in_array=True, get_full_history=True)


    #     # Extract time and state vectors from your data
    #     epochs = state_history_moon[:, 0]  # Assuming the first column represents time
    #     state_moon = state_history_moon[:, 1:]  # Assuming the remaining columns represent the state

    #     state_lumio = state_history_lumio[:, 1:]


    #     initial_states = np.empty((len(epochs),12))
    #     state_history_erdem = validation_LUMIO.get_state_history_erdem()[:,1:]
    #     state_history_erdem_bary = state_history_erdem
    #     # state_history_erdem_bary = np.array([0.97533,	-0.0035687,	-0.020545,	0.12461,	-0.40584,	0.045309,	1.0974,	0.12988	,-0.070446,	0.10002,	-0.05159,	-0.19379])
    #     # state_history_erdem_bary[0] = state_history_erdem_bary[0] - (1-self.mu)
    #     # state_history_erdem_bary[6] = state_history_erdem_bary[6] - (1-self.mu)
    #     for index, epoch in enumerate(epochs):

    #         rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(state_history_moon[index, 1:])

    #         rotational_rates = np.cross(state_moon[index, :3], state_moon[index, 3:])/np.linalg.norm(state_moon[index, :3])**2
    #         w1, w2, w3 = rotational_rates[0], rotational_rates[1], rotational_rates[2]
    #         Omega = np.array([[0, -w3, w2],
    #                         [w3, 0, -w1],
    #                         [-w2, w1, 0]])
    #         # omega_w_norm = np.linalg.norm(np.dot(rsw_to_inertial_rotation_matrix[:, 2], self.rotation_rate))
    #         # Omega = np.array([[0, omega_w_norm, 0],[-omega_w_norm, 0, 0],[0, 0, 0]])
    #         time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, -Omega)
    #         total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
    #                     [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

    #         # Convert barycentric CRTBP non-dimensionalized state into moon fixed frame.
    #         # initial_state_moon_fixed = state_history_erdem_bary
    #         epoch = index*7
    #         state_history_erdem_bary[epoch, 0] = state_history_erdem_bary[epoch, 0] - (1-self.mu)
    #         state_history_erdem_bary[epoch, 6] = state_history_erdem_bary[epoch, 6] - (1-self.mu)
    #         initial_state_moon_fixed = state_history_erdem_bary[epoch]

    #         # Convert initial state defined in CRTBP synodic frame to Earth-centered J2000 frame
    #         self.lu_cr3bp = np.linalg.norm(state_history_moon[index, 1:4])
    #         self.tu_cr3bp = 1/np.sqrt((self.gravitational_parameter_primary + self.gravitational_parameter_secondary)/self.lu_cr3bp**3)
    #         initial_state_moon_fixed_lpf = np.concatenate((initial_state_moon_fixed[:3]*self.lu_cr3bp, initial_state_moon_fixed[3:6]*self.lu_cr3bp/self.tu_cr3bp))
    #         initial_state_moon_fixed_lumio = np.concatenate((initial_state_moon_fixed[6:9]*self.lu_cr3bp, initial_state_moon_fixed[9:12]*self.lu_cr3bp/self.tu_cr3bp))
    #         initial_state_lpf = state_history_moon[index, 1:] + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_moon_fixed_lpf)
    #         initial_state_lumio = state_history_moon[index, 1:] + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_moon_fixed_lumio)

    #         self.initial_state = np.concatenate((initial_state_lpf, initial_state_lumio))

    #         initial_states[index]  = self.initial_state


    #     ax = plt.figure().add_subplot(projection='3d')
    #     plt.plot(initial_states[:,0], initial_states[:,1], initial_states[:,2])
    #     plt.plot(initial_states[:,6], initial_states[:,7], initial_states[:,8])
    #     plt.plot(state_history_moon[:,1], state_history_moon[:,2], state_history_moon[:,3])
    #     plt.axis("equal")
    #     plt.show()


    # def convert_cartesian_state_to_synodic(self):

    #     self.set_initial_cartesian_moon_state()

    #     # initial_state_LPF = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_ELO, )
    #     state_history_moon = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_LPO, body="satellite", get_epoch_in_array=True, get_full_history=True)
    #     state_history_lpf = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_ELO, body="satellite", get_epoch_in_array=True, get_full_history=True)
    #     state_history_lumio = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_LPO, body="moon", get_epoch_in_array=True, get_full_history=True)


    #     # Extract time and state vectors from your data
    #     epochs = state_history_moon[:, 0]  # Assuming the first column represents time
    #     state = state_history_moon[:, 1:]  # Assuming the remaining columns represent the state

    #     state_lpf = state_history_lpf[:, 1:]
    #     state_lumio = state_history_lumio[:, 1:]

    #     print(np.shape(state_lpf), np.shape(state_lumio))


    #     initial_states = np.empty((len(epochs),12))
    #     state_history_erdem = validation_LUMIO.get_state_history_erdem()[0,1:]
    #     state_history_erdem_bary = state_history_erdem
    #     # state_history_erdem_bary = np.array([0.97533,	-0.0035687,	-0.020545,	0.12461,	-0.40584,	0.045309,	1.0974,	0.12988	,-0.070446,	0.10002,	-0.05159,	-0.19379])
    #     state_history_erdem_bary[0] = state_history_erdem_bary[0] - (1-self.mu)
    #     state_history_erdem_bary[6] = state_history_erdem_bary[6] - (1-self.mu)
    #     for index, epoch in enumerate(epochs):

    #         rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(state_history_moon[index, 1:])

    #         rotational_rates = np.cross(state[index, :3], state[index, 3:]) / np.linalg.norm(state[index, :3])**2
    #         w1, w2, w3 = rotational_rates[0], rotational_rates[1], rotational_rates[2]
    #         Omega = np.array([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
    #         time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, -Omega)
    #         total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
    #                     [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

    #         total_inertial_to_rsw_rotation_matrix = np.linalg.inv(total_rsw_to_inertial_rotation_matrix)
    #         if index == 0:
    #             print(total_rsw_to_inertial_rotation_matrix[:3,:3], total_inertial_to_rsw_rotation_matrix[:3,:3])

    #         # Convert J2000 states to Moon-fixed coordinates
    #         state_history_lpf_moon_dim = np.dot(total_inertial_to_rsw_rotation_matrix, state_lpf[index]-state_history_moon[index, 1:])
    #         state_history_lumio_moon_dim = np.dot(total_inertial_to_rsw_rotation_matrix, state_lumio[index]-state_history_moon[index, 1:])

    #         # print(state_history_lpf_moon_dim, state_history_lumio_moon_dim)

    #         # Convert Moon-fixed dimensional coordinates to non-dimensional form
    #         self.lu_cr3bp = np.linalg.norm(state_history_moon[0, 1:4])
    #         self.tu_cr3bp = 1/np.sqrt((self.gravitational_parameter_primary + self.gravitational_parameter_secondary)/self.lu_cr3bp**3)
    #         state_history_lpf_moon_nondim = np.concatenate((state_history_lpf_moon_dim[0:3]/self.lu_cr3bp, state_history_lpf_moon_dim[3:6]/(self.lu_cr3bp/self.tu_cr3bp)))
    #         state_history_lumio_moon_nondim = np.concatenate((state_history_lumio_moon_dim[0:3]/self.lu_cr3bp, state_history_lumio_moon_dim[3:6]/(self.lu_cr3bp/self.tu_cr3bp)))

    #         # Convert Moon-centric non-dimensionalized state to barycentric states.
    #         state_history_lpf_moon_nondim = state_history_lpf_moon_nondim + (1-self.mu)
    #         state_history_lumio_moon_nondim = state_history_lumio_moon_nondim + (1-self.mu)

    #         # print(state_history_lpf_moon_nondim, state_history_lumio_moon_nondim)

    #         # Save state history
    #         initial_states[index]  = np.concatenate((state_history_lpf_moon_nondim, state_history_lumio_moon_nondim))


    #     ax = plt.figure().add_subplot(projection='3d')
    #     plt.plot(initial_states[:,0], initial_states[:,1], initial_states[:,2])
    #     plt.plot(initial_states[:,6], initial_states[:,7], initial_states[:,8])
    #     plt.axis('equal')
    #     # plt.plot(state_history_moon[:,1], state_history_moon[:,2], state_history_moon[:,3])
    #     plt.show()


    def get_closest_initial_state(self):

        self.set_initial_cartesian_moon_state()

        state_history_erdem = validation_LUMIO.get_state_history_erdem()[:,1:]
        state_history_polimi = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_LPO, get_full_history=True, get_dict=False)
        distance_array = np.empty((0, 1))
        initial_state_history = np.empty((0, 12))
        for initial_state_barycenter_fixed in state_history_erdem:

            self.convert_synodic_to_inertial_state(initial_state_barycenter_fixed)

            distance = np.linalg.norm(self.initial_state[6:12]-state_history_polimi[0])

            distance_array = np.vstack((distance_array, distance))
            initial_state_history = np.vstack((initial_state_history, self.initial_state))

        min_distance_index = np.argmin(distance_array)
        closest_initial_state = initial_state_history[min_distance_index]

        # print(state_history_erdem[min_distance_index])

        # ax = plt.figure().add_subplot(projection='3d')
        # plt.plot(initial_state_history[:, 6], initial_state_history[:, 7], initial_state_history[:, 8])
        # plt.plot(state_history_polimi[0, 0], state_history_polimi[0,1], state_history_polimi[0,2], marker='o', color="red")
        # plt.plot(closest_initial_state[6], closest_initial_state[7], closest_initial_state[8], marker='o', color="blue")
        # ax.set_xlabel('X [m]')
        # ax.set_ylabel('Y [m]')
        # ax.set_zlabel('Z [m]')
        # plt.show()

        # plt.plot(distance_array)
        # plt.show()

        # print(closest_initial_state)

        # closest_state[6:12] = np.array([-3.09965465e+08,  3.07254364e+08,  1.09190626e+08, -6.94514593e+02, -5.92981679e+02, -3.02950273e+02])
        return closest_initial_state


    def set_environment_settings(self):

        # Create default body settings
        self.body_settings = environment_setup.get_default_body_settings(
            self.bodies_to_create, self.global_frame_origin, self.global_frame_orientation)

        # Create spacecraft bodies
        for index, body in enumerate(self.bodies_to_propagate):
            self.body_settings.add_empty_settings(body)
            self.body_settings.get(body).constant_mass = self.bodies_mass[index]

        # Adjust the orbit of the moon to perfectly circular
        self.moon_initial_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name = self.name_secondary,
            observer_body_name = self.name_primary,
            reference_frame_name = self.global_frame_orientation,
            aberration_corrections = 'NONE',
            ephemeris_time = self.simulation_start_epoch)

        central_body_gravitational_parameter = self.gravitational_parameter_primary + self.gravitational_parameter_secondary
        initial_keplerian_moon_state = element_conversion.cartesian_to_keplerian(self.moon_initial_state, central_body_gravitational_parameter)
        initial_keplerian_moon_state[0], initial_keplerian_moon_state[1] = self.distance_between_primaries, self.eccentricity
        self.initial_cartesian_moon_state = element_conversion.keplerian_to_cartesian(initial_keplerian_moon_state, central_body_gravitational_parameter)

        # Add ephemeris settings to body settings of the Moon
        self.body_settings.get(self.name_secondary).ephemeris_settings = environment_setup.ephemeris.keplerian(
            initial_keplerian_moon_state,
            self.simulation_start_epoch,
            central_body_gravitational_parameter,
            self.global_frame_origin, self.global_frame_orientation)

        # Update the body settings
        self.bodies = environment_setup.create_system_of_bodies(self.body_settings)

        # print(self.bodies.get(self.name_primary).mass,\
        #     self.bodies.get(self.name_secondary).mass)
        # print(self.bodies.get(self.name_ELO).mass,\
        #     self.bodies.get(self.name_LPO).mass)

        # print(self.mu, constants.GRAVITATIONAL_CONSTANT)


    def set_acceleration_settings(self):

        self.set_environment_settings()

        # Define accelerations acting on vehicle.
        self.acceleration_settings_on_spacecrafts = dict()
        for index, spacecraft in enumerate([self.name_ELO, self.name_LPO]):
            acceleration_settings_on_spacecraft = {
                    self.name_primary: [propagation_setup.acceleration.point_mass_gravity()],
                    self.name_secondary: [propagation_setup.acceleration.point_mass_gravity()]
            }
            self.acceleration_settings_on_spacecrafts[spacecraft] = acceleration_settings_on_spacecraft

        # Create global accelerations dictionary.
        self.acceleration_settings = self.acceleration_settings_on_spacecrafts

        # Create acceleration models.
        self.acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, self.acceleration_settings, self.bodies_to_propagate, self.central_bodies)


    def set_initial_state(self):

        self.set_acceleration_settings()

        self.initial_state = self.get_closest_initial_state()


    def set_integration_settings(self):

        self.set_initial_state()

        current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkdp_87
        current_tolerance = 1e-10*constants.JULIAN_DAY
        initial_time_step = 1e-3*constants.JULIAN_DAY
        self.integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(initial_time_step,
                                                                                        current_coefficient_set,
                                                                                        np.finfo(float).eps,
                                                                                        np.inf,
                                                                                        current_tolerance,
                                                                                        current_tolerance)

        self.integrator_settings = propagation_setup.integrator.runge_kutta_4(self.simulation_start_epoch, 0.005*86400)

    def set_dependent_variables_to_save(self):

        self.set_integration_settings()

        # Define required outputs
        self.dependent_variables_to_save = [
            propagation_setup.dependent_variable.keplerian_state(self.name_secondary, self.name_primary),
            propagation_setup.dependent_variable.relative_position(self.name_primary, self.name_secondary),
            propagation_setup.dependent_variable.relative_velocity(self.name_primary, self.name_secondary),
            propagation_setup.dependent_variable.relative_distance(self.name_primary, self.name_secondary),
            propagation_setup.dependent_variable.total_acceleration(self.name_ELO),
            propagation_setup.dependent_variable.total_acceleration(self.name_LPO)
        ]


    def set_termination_settings(self):

        self.set_dependent_variables_to_save()

        # Create termination settings
        self.termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch)


    def set_propagator_settings(self):

        self.set_termination_settings()

        # Create propagation settings
        self.propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            self.acceleration_models,
            self.bodies_to_propagate,
            self.initial_state,
            self.simulation_start_epoch,
            self.integrator_settings,
            self.termination_settings,
            output_variables= self.dependent_variables_to_save
        )


    def get_propagated_orbit(self):

        self.set_propagator_settings()

        # Create simulation object and propagate dynamics.
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            self.bodies,
            self.propagator_settings)

        # Setup parameters settings to propagate the state transition matrix
        self.parameter_settings = estimation_setup.parameter.initial_states(self.propagator_settings, self.bodies)
        self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.bodies)
        variational_equations_solver = numerical_simulation.create_variational_equations_solver(
                self.bodies,
                self.propagator_settings,
                self.parameters_to_estimate,
                simulate_dynamics_on_creation=True)

        return dynamics_simulator, variational_equations_solver



# test2 = LowFidelityDynamicModel(60390, 28)
# states2 = test2.get_propagated_orbit()[0]
# # print(test2.convert_cartesian_state_to_synodic())

# ax = plt.figure().add_subplot(projection='3d')
# # plt.plot(states[:,0], states[:,1], states[:,2])
# # plt.plot(states[:,6], states[:,7], states[:,8])
# plt.plot(states2[:,0], states2[:,1], states2[:,2])
# plt.plot(states2[:,6], states2[:,7], states2[:,8])
# plt.legend()
# plt.axis('equal')
# plt.show()

# states2 = test2.get_propagated_orbit()[1]

# ax = plt.figure()
# plt.plot(states2[:,:6])
# plt.legend()
# plt.show()