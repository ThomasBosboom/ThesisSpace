import numpy as np
import matplotlib.pyplot as plt
from tudatpy.kernel import constants
from scipy.interpolate import interp1d
from tudatpy.kernel.astro import time_conversion, frame_conversion
import Interpolator
import TraditionalLowFidelity


class SynodicToInertialHistoryConverter:

    def __init__(self, dynamic_model_object, step_size=0.001):

        self.dynamic_model_object = dynamic_model_object
        self.propagation_time = dynamic_model_object.propagation_time
        self.mu = dynamic_model_object.mu
        self.step_size = step_size


    def get_total_rsw_to_inertial_rotation_matrix(self, inertial_moon_state):

        # Determine transformation matrix from Moon-fixed frame to inertial coordinates for a given epoch
        rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(inertial_moon_state)

        # Determine rotation rate and direction with respect to synodic frame (so only rotation w of rsw is relevant)
        rotation_rate = self.dynamic_model_object.rotation_rate
        m = self.dynamic_model_object.bodies.get("Moon").mass
        r_norm = np.linalg.norm(inertial_moon_state[:3])
        v_norm = np.linalg.norm(inertial_moon_state[3:])
        h = m*r_norm*v_norm
        rotation_rate = h/(m*r_norm**2)

        Omega = np.array([[0, -rotation_rate, 0],[rotation_rate, 0, 0],[0, 0, 0]])

        # Update total transformation matrix with matrix derivative element
        time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, Omega)
        total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
                        [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

        return total_rsw_to_inertial_rotation_matrix


    def convert_state_nondim_to_dim_state(self, synodic_state, inertial_moon_state):

        lu_cr3bp = self.dynamic_model_object.lu_cr3bp
        tu_cr3bp = self.dynamic_model_object.lu_cr3bp

        lu_cr3bp = np.linalg.norm(inertial_moon_state[:3])
        tu_cr3bp = 1/np.sqrt((self.dynamic_model_object.gravitational_parameter_primary + \
            self.dynamic_model_object.gravitational_parameter_secondary)/lu_cr3bp**3)

        return np.concatenate((synodic_state[:3]*lu_cr3bp, synodic_state[3:]*lu_cr3bp/tu_cr3bp))


    def convert_state_barycentric_to_body(self, state_barycentric, body, state_type="rotating"):

        if state_type == "inertial":
            if body == "primary":
                return state_barycentric - np.array([-self.mu, 0, 0, 0, 0, 0])*state_barycentric
            if body == "secondary":
                return state_barycentric - np.array([1-self.mu, 0, 0, 0, 0, 0])*state_barycentric

        elif state_type == "rotating":
            state_body = state_barycentric
            if body == "primary":
                return state_body - np.array([-self.mu, 0, 0, 0, 0, 0])
            if body == "secondary":
                return state_body - np.array([1-self.mu, 0, 0, 0, 0, 0])


    def convert_secondary_fixed_to_inertial_state(self, inertial_moon_state, total_rsw_to_inertial_rotation_matrix, state_body_fixed):
        return inertial_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,state_body_fixed)


    def convert_synodic_to_inertial_state(self, synodic_state, inertial_moon_state):

        # Obtain the transformation matrix from J2000 inertial to Earth-Moon synodic frame
        total_rsw_to_inertial_rotation_matrix = self.get_total_rsw_to_inertial_rotation_matrix(inertial_moon_state)

        # Dimensionalize
        initial_state_lpf_moon_fixed   = self.convert_state_nondim_to_dim_state(synodic_state[:6], inertial_moon_state)
        initial_state_lumio_moon_fixed = self.convert_state_nondim_to_dim_state(synodic_state[6:], inertial_moon_state)

        # Apply transformations and rotations
        state_history_lumio_inertial = self.convert_secondary_fixed_to_inertial_state(inertial_moon_state, total_rsw_to_inertial_rotation_matrix, initial_state_lumio_moon_fixed)
        state_history_lpf_inertial   = self.convert_secondary_fixed_to_inertial_state(inertial_moon_state, total_rsw_to_inertial_rotation_matrix, initial_state_lpf_moon_fixed)
        state_history_inertial       = np.concatenate((state_history_lpf_inertial, state_history_lumio_inertial))

        return state_history_inertial


    def get_results(self, synodic_state_history):

        epochs, _, dependent_variables_history, _ = \
            Interpolator.Interpolator(step_size=self.step_size).get_propagator_results(self.dynamic_model_object)

        # Generate history of classical CRTBP
        state_rotating_bary_lpf, state_rotating_bary_lumio = synodic_state_history[:,:6], synodic_state_history[:,6:]

        # Convert satellite states to moon centric frame
        state_history_moon_fixed_lumio      = self.convert_state_barycentric_to_body(state_rotating_bary_lumio, "secondary", state_type="rotating")
        state_history_moon_fixed_lpf        = self.convert_state_barycentric_to_body(state_rotating_bary_lpf, "secondary", state_type="rotating")
        state_history_moon_fixed_satellites = np.concatenate((state_history_moon_fixed_lpf, state_history_moon_fixed_lumio), axis=1)

        # Convert primaries states to moon centric frame
        state_rotating_bary_primary        = np.multiply(np.ones((np.shape(state_rotating_bary_lumio)[0],6)), np.array([-self.mu, 0, 0, 0, 0, 0]))
        state_rotating_bary_secondary      = np.multiply(np.ones((np.shape(state_rotating_bary_lumio)[0],6)), np.array([1-self.mu, 0, 0, 0, 0, 0]))
        state_history_moon_fixed_primary   = self.convert_state_barycentric_to_body(state_rotating_bary_primary, "secondary", state_type="rotating")
        state_history_moon_fixed_secondary = self.convert_state_barycentric_to_body(state_rotating_bary_secondary, "secondary", state_type="rotating")
        state_history_moon_fixed_primaries = np.concatenate((state_history_moon_fixed_primary, state_history_moon_fixed_secondary), axis=1)

        # Looping through all epochs to convert each synodic frame element to J2000 Earth-centered
        state_history_inertial_satellites = np.empty(np.shape(state_history_moon_fixed_satellites))
        state_history_inertial_primaries  = np.empty(np.shape(state_history_moon_fixed_primaries))
        for epoch, state in enumerate(synodic_state_history):
            state_history_inertial_satellites[epoch] = self.convert_synodic_to_inertial_state(state_history_moon_fixed_satellites[epoch], dependent_variables_history[epoch, :6])
            state_history_inertial_primaries[epoch]  = self.convert_synodic_to_inertial_state(state_history_moon_fixed_primaries[epoch], dependent_variables_history[epoch, :6])

        return epochs, state_history_inertial_satellites, state_history_inertial_primaries




class InertialToSynodicHistoryConverter:

    def __init__(self, dynamic_model_object, step_size=0.001):

        self.dynamic_model_object = dynamic_model_object
        self.propagation_time = dynamic_model_object.propagation_time
        self.mu = dynamic_model_object.mu
        self.step_size = step_size


    def get_total_inertial_to_rsw_rotation_matrix(self, inertial_moon_state):

        # Determine transformation matrix from Moon-fixed frame to inertial coordinates for a given epoch
        rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(inertial_moon_state)

        # Determine rotation rate and direction with respect to synodic frame (so only rotation w of rsw is relevant)
        rotation_rate = self.dynamic_model_object.rotation_rate
        m = self.dynamic_model_object.bodies.get("Moon").mass
        r_norm = np.linalg.norm(inertial_moon_state[:3])
        v_norm = np.linalg.norm(inertial_moon_state[3:])
        h = m*r_norm*v_norm
        rotation_rate = h/(m*r_norm**2)

        Omega = np.array([[0, -rotation_rate, 0],[rotation_rate, 0, 0],[0, 0, 0]])

        # Update total transformation matrix with matrix derivative element
        time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, Omega)
        total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
                        [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

        return np.linalg.inv(total_rsw_to_inertial_rotation_matrix)


    def convert_state_dim_to_nondim_state(self, synodic_state, inertial_moon_state):

        lu_cr3bp = self.dynamic_model_object.lu_cr3bp
        tu_cr3bp = self.dynamic_model_object.lu_cr3bp

        lu_cr3bp = np.linalg.norm(inertial_moon_state[:3])
        tu_cr3bp = 1/np.sqrt((self.dynamic_model_object.gravitational_parameter_primary + \
            self.dynamic_model_object.gravitational_parameter_secondary)/lu_cr3bp**3)

        return np.concatenate((synodic_state[:3]/lu_cr3bp, synodic_state[3:]/(lu_cr3bp/tu_cr3bp)))


    def convert_state_body_to_barycentric(self, state_barycentric, body, state_type="rotating"):

        if state_type == "inertial":
            if body == "primary":
                return state_barycentric + np.array([-self.mu, 0, 0, 0, 0, 0])*state_barycentric
            if body == "secondary":
                return state_barycentric + np.array([1-self.mu, 0, 0, 0, 0, 0])*state_barycentric

        elif state_type == "rotating":
            state_body = state_barycentric
            if body == "primary":
                return state_body + np.array([-self.mu, 0, 0, 0, 0, 0])
            if body == "secondary":
                return state_body + np.array([1-self.mu, 0, 0, 0, 0, 0])


    def convert_inertial_to_secondary_fixed_state(self, inertial_moon_state, total_inertial_to_rsw_rotation_matrix, state_inertial):
        return np.dot(total_inertial_to_rsw_rotation_matrix, state_inertial-inertial_moon_state)


    def convert_inertial_to_synodic_state(self, inertial_state, inertial_moon_state):

        # Obtain the transformation matrix from J2000 inertial to Earth-Moon synodic frame
        total_inertial_to_rsw_rotation_matrix = self.get_total_inertial_to_rsw_rotation_matrix(inertial_moon_state)

        # Apply transformations and rotations
        initial_state_lpf_moon_fixed = self.convert_inertial_to_secondary_fixed_state(inertial_moon_state, total_inertial_to_rsw_rotation_matrix, inertial_state[:6])
        initial_state_lumio_moon_fixed = self.convert_inertial_to_secondary_fixed_state(inertial_moon_state, total_inertial_to_rsw_rotation_matrix, inertial_state[6:])

        # Non-dimensionalize
        initial_state_lpf_moon_fixed   = self.convert_state_dim_to_nondim_state(initial_state_lpf_moon_fixed, inertial_moon_state)
        initial_state_lumio_moon_fixed = self.convert_state_dim_to_nondim_state(initial_state_lumio_moon_fixed, inertial_moon_state)
        state_history_synodic       = np.concatenate((initial_state_lpf_moon_fixed, initial_state_lumio_moon_fixed))

        return state_history_synodic


    def get_results(self, inertial_state_history):

        epochs, _, dependent_variables_history, _ = \
            Interpolator.Interpolator(step_size=self.step_size).get_propagator_results(self.dynamic_model_object)

        # Split states into spacecraft states
        state_inertial_lpf, state_inertial_lumio = inertial_state_history[:,:6], inertial_state_history[:,6:]

        # Looping through all epochs to convert each synodic frame element to J2000 Earth-centered
        state_history_inertial_satellites = np.empty(np.shape(inertial_state_history))
        for epoch, state in enumerate(inertial_state_history):
            state_history_inertial_satellites[epoch] = self.convert_inertial_to_synodic_state(inertial_state_history[epoch], dependent_variables_history[epoch, :6])

        # Convert satellite states to barycentric frame
        state_history_barycentric_lpf = self.convert_state_body_to_barycentric(state_history_inertial_satellites[:,:6], "secondary", state_type="rotating")
        state_history_barycentric_lumio = self.convert_state_body_to_barycentric(state_history_inertial_satellites[:,6:], "secondary", state_type="rotating")
        state_history_barycentric_satellites = np.concatenate((state_history_barycentric_lpf, state_history_barycentric_lumio), axis=1)

        return epochs, state_history_barycentric_satellites




