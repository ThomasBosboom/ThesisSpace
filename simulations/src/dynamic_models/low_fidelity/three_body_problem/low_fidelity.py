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
from dynamic_models import validation
from DynamicModelBase import DynamicModelBase


class LowFidelityDynamicModel(DynamicModelBase):

    def __init__(self, simulation_start_epoch_MJD, propagation_time, custom_initial_state=None):
        super().__init__(simulation_start_epoch_MJD, propagation_time)

        # Get CRTBP characteristics
        self.custom_initial_state = custom_initial_state
        self.distance_between_primaries = 3.84747963e8
        self.eccentricity = 0
        self.bodies_mass = [0,0]
        self.lu_cr3bp = self.distance_between_primaries
        self.tu_cr3bp = 1/np.sqrt((self.gravitational_parameter_primary + self.gravitational_parameter_secondary)/self.distance_between_primaries**3)
        self.rotation_rate = 1/self.tu_cr3bp


    def set_initial_cartesian_moon_state(self):

        # Adjust the orbit of the moon to perfectly circular
        moon_initial_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name = self.name_secondary,
            observer_body_name = self.name_primary,
            reference_frame_name = self.global_frame_orientation,
            aberration_corrections = 'NONE',
            ephemeris_time = self.simulation_start_epoch)

        self.central_body_gravitational_parameter = self.gravitational_parameter_primary + self.gravitational_parameter_secondary
        self.initial_keplerian_moon_state = element_conversion.cartesian_to_keplerian(moon_initial_state,
                                                                                      self.central_body_gravitational_parameter)
        self.initial_keplerian_moon_state[0] = self.distance_between_primaries
        self.initial_keplerian_moon_state[1] = self.eccentricity
        # print("initial_keplerian_moon_state: ", initial_keplerian_moon_state)
        self.initial_cartesian_moon_state = element_conversion.keplerian_to_cartesian(self.initial_keplerian_moon_state,
                                                                                      self.central_body_gravitational_parameter)
        # print("initial_cartesian_moon_state: ", self.initial_cartesian_moon_state)



    def convert_synodic_to_inertial_state(self, initial_state_barycenter_fixed):

        self.set_initial_cartesian_moon_state()

        # Convert barycentric CRTBP non-dimensionalized state into moon fixed frame.
        initial_state_moon_fixed = initial_state_barycenter_fixed
        initial_state_moon_fixed[0] = initial_state_moon_fixed[0] - (1-self.mu)
        initial_state_moon_fixed[6] = initial_state_moon_fixed[6] - (1-self.mu)

        # Compute the transformation matrix from CRTBP synodic frame to Earth-centered J2000 frame
        rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(self.initial_cartesian_moon_state)

        # Compute time derivative element for the conversion of the velocity state elements and combine into final matrix
        omega_w_norm = self.rotation_rate
        m = self.gravitational_parameter_secondary/constants.GRAVITATIONAL_CONSTANT
        r_norm = np.linalg.norm(self.initial_cartesian_moon_state[:3])
        v_norm = np.linalg.norm(self.initial_cartesian_moon_state[3:])
        h = m*r_norm*v_norm
        rotation_rate = h/(m*r_norm**2)
        omega_w_norm = rotation_rate
        Omega = np.array([[0, omega_w_norm, 0],[-omega_w_norm, 0, 0],[0, 0, 0]])

        time_derivative_rsw_to_inertial_rotation_matrix = -np.dot(rsw_to_inertial_rotation_matrix, Omega)
        total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
                        [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

        # Convert initial state defined in CRTBP synodic frame to Earth-centered J2000 frame
        initial_state_moon_fixed_lpf = np.concatenate((initial_state_moon_fixed[:3]*self.lu_cr3bp, initial_state_moon_fixed[3:6]*self.lu_cr3bp/self.tu_cr3bp))
        initial_state_moon_fixed_lumio = np.concatenate((initial_state_moon_fixed[6:9]*self.lu_cr3bp, initial_state_moon_fixed[9:12]*self.lu_cr3bp/self.tu_cr3bp))
        initial_state_lpf = self.initial_cartesian_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_moon_fixed_lpf)
        initial_state_lumio = self.initial_cartesian_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_moon_fixed_lumio)
        initial_state = np.concatenate((initial_state_lpf, initial_state_lumio))

        return initial_state


    def get_closest_initial_state(self, step_size=0.05):

        self.set_initial_cartesian_moon_state()

        # state_history_synodic = validation.get_synodic_state_history_erdem()[:,1:]
        # print(constants.GRAVITATIONAL_CONSTANT, self.bodies.get("Earth").mass, self.bodies.get("Moon").mass, self.distance_between_primaries)
        _, state_history_synodic = validation.get_synodic_state_history(constants.GRAVITATIONAL_CONSTANT,
                                                                            self.bodies.get("Earth").mass,
                                                                            self.bodies.get("Moon").mass,
                                                                            self.distance_between_primaries,
                                                                            14, # max 14 days, to save run time, full period halo orbit approximate
                                                                            step_size)
        reference_state_LUMIO = validation.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, step_size=step_size, satellite=self.name_LPO, get_full_history=True, get_dict=False)
        # reference_state_LPF = validation.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_ELO, get_full_history=False, get_dict=False)
        distance_array = np.empty((0, 1))
        initial_state_history = np.empty((0, 12))
        for initial_state_barycenter_fixed in state_history_synodic:

            self.initial_state = self.convert_synodic_to_inertial_state(initial_state_barycenter_fixed)

            distance = np.linalg.norm(self.initial_state[6:12]-reference_state_LUMIO[0])

            distance_array = np.vstack((distance_array, distance))
            initial_state_history = np.vstack((initial_state_history, self.initial_state))

        min_distance_index = np.argmin(distance_array)
        closest_initial_state = initial_state_history[min_distance_index]

        # print("reference lpf", reference_state_LPF)
        # print("initial state early 1: ", closest_initial_state)
        # closest_initial_state[:6] = initial_state_history[0,:6]

        # print("initial state early 2: ", closest_initial_state)

        # ax = plt.figure().add_subplot(projection='3d')
        # plt.plot(initial_state_history[:, 6], initial_state_history[:, 7], initial_state_history[:, 8])
        # plt.plot(reference_state_LUMIO[0, 0], reference_state_LUMIO[0,1], reference_state_LUMIO[0,2], marker='o', color="red")
        # plt.plot(closest_initial_state[6], closest_initial_state[7], closest_initial_state[8], marker='o', color="blue")
        # ax.set_xlabel('X [m]')
        # ax.set_ylabel('Y [m]')
        # ax.set_zlabel('Z [m]')
        # plt.show()

        # plt.plot(distance_array)
        # plt.show()

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
        self.set_initial_cartesian_moon_state()

        # Add ephemeris settings to body settings of the Moon
        self.body_settings.get(self.name_secondary).ephemeris_settings = environment_setup.ephemeris.keplerian(
            self.initial_keplerian_moon_state,
            self.simulation_start_epoch,
            self.central_body_gravitational_parameter,
            self.global_frame_origin,
            self.global_frame_orientation)

        # define parameters describing the synchronous rotation model for Moon fixed frame w.r.t. central body Earth
        self.body_settings.get(self.name_secondary).rotation_model_settings = environment_setup.rotation_model.synchronous(
            self.name_primary, self.global_frame_orientation, "Moon_fixed")

        # Update the body settings
        self.bodies = environment_setup.create_system_of_bodies(self.body_settings)


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

        if self.custom_initial_state is None:
            self.initial_state = self.get_closest_initial_state()
        else:
            self.initial_state = self.convert_synodic_to_inertial_state(self.custom_initial_state)
            self.custom_initial_state[0] = self.custom_initial_state[0] + (1-self.mu)
            self.custom_initial_state[6] = self.custom_initial_state[6] + (1-self.mu)


    def set_integration_settings(self):

        self.set_initial_state()

        if self.use_variable_step_size_integrator:
            self.integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(self.initial_time_step,
                                                                                            self.current_coefficient_set,
                                                                                            np.finfo(float).eps,
                                                                                            np.inf,
                                                                                            self.current_tolerance,
                                                                                            self.current_tolerance)
        else:
            self.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(self.initial_time_step,
                                                                                           self.current_coefficient_set)


    def set_dependent_variables_to_save(self):

        self.set_integration_settings()

        # Define required outputs
        self.dependent_variables_to_save = [
            propagation_setup.dependent_variable.relative_position(self.name_secondary, self.name_primary),
            propagation_setup.dependent_variable.relative_velocity(self.name_secondary, self.name_primary),
            propagation_setup.dependent_variable.relative_position(self.name_ELO, self.name_LPO),
            propagation_setup.dependent_variable.relative_velocity(self.name_ELO, self.name_LPO),
            propagation_setup.dependent_variable.total_acceleration(self.name_ELO),
            propagation_setup.dependent_variable.total_acceleration(self.name_LPO),
            propagation_setup.dependent_variable.keplerian_state(self.name_secondary, self.name_primary),
            propagation_setup.dependent_variable.keplerian_state(self.name_ELO, self.name_secondary)]

        self.dependent_variables_to_save.extend([
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, body_to_propagate, body_to_create) \
                        for body_to_propagate in self.bodies_to_propagate for body_to_create in self.bodies_to_create])

        self.dependent_variables_to_save.extend([propagation_setup.dependent_variable.body_mass(self.name_primary),
                                                 propagation_setup.dependent_variable.body_mass(self.name_secondary)])


    def set_termination_settings(self):

        self.set_dependent_variables_to_save()

        # Create termination settings
        self.termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch)


    def set_propagator_settings(self, estimated_parameter_vector=None):

        self.set_termination_settings()

        if estimated_parameter_vector is not None:
            self.initial_state = estimated_parameter_vector[:12]

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


    def get_propagation_simulator(self, estimated_parameter_vector=None, solve_variational_equations=True):

        self.set_propagator_settings(estimated_parameter_vector=estimated_parameter_vector)

        # Create simulation object and propagate dynamics.
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            self.bodies,
            self.propagator_settings)

        # Setup parameters settings to propagate the state transition matrix
        if solve_variational_equations:
            self.parameter_settings = estimation_setup.parameter.initial_states(self.propagator_settings, self.bodies)
            self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.bodies)
            variational_equations_solver = numerical_simulation.create_variational_equations_solver(
                    self.bodies,
                    self.propagator_settings,
                    self.parameters_to_estimate,
                    simulate_dynamics_on_creation=True)

            return dynamics_simulator, variational_equations_solver

        else:

            return dynamics_simulator



# custom_initial_state = np.array([0.985141349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0, \
#                                          1.1473302, 0, -0.15142308, 0, -0.21994554, 0])
# test2 = LowFidelityDynamicModel(60390, 28)
# dep_var = np.stack(list(test2.get_propagation_simulator()[0].dependent_variable_history.values()))
# states = np.stack(list(test2.get_propagation_simulator()[0].state_history.values()))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.plot(states[:,0], states[:,1], states[:,2])
# plt.plot(states[:,6], states[:,7], states[:,8])
# plt.show()


# print(np.shape(dep_var))
# ax = plt.figure()
# plt.plot(dep_var[:,18:24])
# plt.legend()
# # plt.yscale("log")
# # plt.show()

# ax = plt.figure()
# plt.plot(dep_var[:,24:30])
# plt.legend()
# # plt.yscale("log")
# # plt.show()

# # print(dep_var[0,18:24])
# # print(dep_var[0,24:30])