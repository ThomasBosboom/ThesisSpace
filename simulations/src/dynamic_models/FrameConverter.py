import numpy as np
import matplotlib.pyplot as plt
from tudatpy.kernel import constants
from scipy.interpolate import interp1d
from tudatpy.kernel.astro import time_conversion, frame_conversion
import Interpolator
import TraditionalLowFidelity


class SynodicToInertialHistoryConverter:

    def __init__(self, dynamic_model_object, step_size=0.005):

        self.dynamic_model_object = dynamic_model_object
        self.G = constants.GRAVITATIONAL_CONSTANT
        self.m1 = self.dynamic_model_object.bodies.get("Earth").mass
        self.m2 = self.dynamic_model_object.bodies.get("Moon").mass
        self.a = self.dynamic_model_object.distance_between_primaries
        self.lu_cr3bp = self.dynamic_model_object.lu_cr3bp
        self.tu_cr3bp = self.dynamic_model_object.tu_cr3bp
        self.rotation_rate = self.dynamic_model_object.rotation_rate
        self.propagation_time = self.dynamic_model_object.propagation_time
        self.step_size = step_size


    def get_total_rsw_to_inertial_rotation_matrix(self, inertial_moon_state):

        # Determine transformation matrix from Moon-fixed frame to inertial coordinates for a given epoch
        rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(inertial_moon_state[:6])

        # Determine rotation rate and direction with respect to synodic frame (so only rotation w of rsw is relevant)
        omega_w = np.dot(rsw_to_inertial_rotation_matrix[:, 2], self.rotation_rate)
        omega_w_norm = np.linalg.norm(omega_w)
        Omega = np.array([[0, -omega_w_norm, 0],[omega_w_norm, 0, 0],[0, 0, 0]])

        # Update total transformation matrix with matrix derivative element
        time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, Omega)
        total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
                        [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

        return total_rsw_to_inertial_rotation_matrix


    def convert_state_nondim_to_dim_state(self, synodic_state):
        return np.concatenate((synodic_state[:3]*self.lu_cr3bp, synodic_state[3:]*self.lu_cr3bp/self.tu_cr3bp))


    def convert_body_fixed_to_inertial_state(self, inertial_moon_state, total_rsw_to_inertial_rotation_matrix, state_body_fixed):
        return inertial_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,state_body_fixed)


    def convert_synodic_to_inertial_state(self, synodic_state, inertial_moon_state):

        total_rsw_to_inertial_rotation_matrix = self.get_total_rsw_to_inertial_rotation_matrix(inertial_moon_state)

        # Non-dimensionalize
        initial_state_lpf_moon_fixed = self.convert_state_nondim_to_dim_state(synodic_state[:6])
        initial_state_lumio_moon_fixed = self.convert_state_nondim_to_dim_state(synodic_state[6:])

        # Apply transformations and rotations
        state_history_lumio_inertial = self.convert_body_fixed_to_inertial_state(inertial_moon_state, total_rsw_to_inertial_rotation_matrix, initial_state_lumio_moon_fixed)
        state_history_lpf_inertial = self.convert_body_fixed_to_inertial_state(inertial_moon_state, total_rsw_to_inertial_rotation_matrix, initial_state_lpf_moon_fixed)
        state_history_inertial = np.concatenate((state_history_lpf_inertial, state_history_lumio_inertial))

        return state_history_inertial


    def get_results(self):

        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(self.dynamic_model_object, step_size=self.step_size).get_results()

        state_rotating_bary_lpf_0 = self.dynamic_model_object.custom_initial_state[:6]
        state_rotating_bary_lumio_0 = self.dynamic_model_object.custom_initial_state[6:]

        # Generate history of classical CRTBP
        system   = TraditionalLowFidelity.TraditionalLowFidelity(self.G, self.m1, self.m2, self.a)
        t, state_rotating_bary_lumio = system.get_state_history(state_rotating_bary_lumio_0, 0, self.propagation_time, self.step_size)
        t, state_rotating_bary_lpf = system.get_state_history(state_rotating_bary_lpf_0, 0, self.propagation_time, self.step_size)

        # Convert satellite states to moon centric frame
        state_history_moon_fixed_lumio = system.convert_state_barycentric_to_body(state_rotating_bary_lumio, "secondary", state_type="rotating")
        state_history_moon_fixed_lpf = system.convert_state_barycentric_to_body(state_rotating_bary_lpf, "secondary", state_type="rotating")
        state_history_moon_fixed_satellites = np.concatenate((state_history_moon_fixed_lpf, state_history_moon_fixed_lumio), axis=1)

        # Convert primaries states to moon centric frame
        state_rotating_bary_primary = np.multiply(np.ones((np.shape(t)[0],6)), system.state_m1)
        state_rotating_bary_secondary = np.multiply(np.ones((np.shape(t)[0],6)), system.state_m2)
        print("rotating primaries: ", state_rotating_bary_primary, state_rotating_bary_secondary)
        state_history_moon_fixed_primary = system.convert_state_barycentric_to_body(state_rotating_bary_primary, "secondary", state_type="rotating")
        print("state_rotating_bary_primary: ", state_rotating_bary_primary)
        state_history_moon_fixed_secondary = system.convert_state_barycentric_to_body(state_rotating_bary_secondary, "secondary", state_type="rotating")
        state_history_moon_fixed_primaries = np.concatenate((state_history_moon_fixed_primary, state_history_moon_fixed_secondary), axis=1)
        print("rotating moon fixed primaries 1: ", state_history_moon_fixed_primaries)

        # Looping through all epochs to convert each synodic frame element to J2000 Earth-centered
        state_history_inertial_satellites = np.empty(np.shape(state_history_moon_fixed_satellites))
        state_history_inertial_primaries = np.empty(np.shape(state_history_moon_fixed_primaries))
        for epoch, state in enumerate(state_history_moon_fixed_satellites):
            state_history_inertial_satellites[epoch] = self.convert_synodic_to_inertial_state(state_history_moon_fixed_satellites[epoch], dependent_variables_history[epoch, :6])
            state_history_inertial_primaries[epoch] = self.convert_synodic_to_inertial_state(state_history_moon_fixed_primaries[epoch], dependent_variables_history[epoch, :6])

        print("rotating moon fixed primaries[:,6:12]: ", state_history_moon_fixed_primaries[:,6:12])
        print("state_history_inertial_primaries[:,6:12]: ", state_history_inertial_primaries[:,6:12])

        return t, state_history_inertial_satellites, state_history_inertial_primaries