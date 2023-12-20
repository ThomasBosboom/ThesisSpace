import numpy as np
import matplotlib.pyplot as plt
from tudatpy.kernel import constants
from scipy.interpolate import interp1d
from tudatpy.kernel.astro import time_conversion, frame_conversion
import Interpolator
import CRTBP_traditional


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
        self.step_size = step_size


    def convert_synodic_to_inertial_state(self, synodic_state, inertial_moon_state):

        # Determine transformation matrix from Moon-fixed frame to inertial coordinates for a given epoch
        rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(inertial_moon_state[:6])

        # Determine rotation rate and direction with respect to synodic frame (so only rotation w of rsw is relevant)
        omega_w = np.dot(rsw_to_inertial_rotation_matrix[:, 2], self.rotation_rate)
        omega_w_norm = np.linalg.norm(omega_w)
        Omega = np.array([[0, omega_w_norm, 0],[-omega_w_norm, 0, 0],[0, 0, 0]])

        # Update total transformation matrix with matrix derivative element
        time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, -Omega)
        total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
                        [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

        # Non-dimensionalize and apply transformations and rotations
        initial_state_lpf_moon_fixed = np.concatenate((synodic_state[0:3]*self.lu_cr3bp, synodic_state[3:6]*self.lu_cr3bp/self.tu_cr3bp))
        initial_state_lumio_moon_fixed = np.concatenate((synodic_state[6:9]*self.lu_cr3bp, synodic_state[9:12]*self.lu_cr3bp/self.tu_cr3bp))
        state_history_lumio_CRTBP = inertial_moon_state[:6] + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_lumio_moon_fixed)
        state_history_lpf_CRTBP = inertial_moon_state[:6] + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_lpf_moon_fixed)
        state_history_CRTBP = np.concatenate((state_history_lpf_CRTBP, state_history_lumio_CRTBP))

        return state_history_CRTBP


    def get_results(self, custom_initial_state=None):

        # self.dynamics_simulator, self.variational_equations_solver = self.dynamic_model_object.get_propagated_orbit()
        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(self.dynamic_model_object, step_size=self.step_size).get_results()
        self.propagation_time = self.dynamic_model_object.propagation_time

        # state_rotating_bary_lpf_0 = self.dynamic_model_object.custom_initial_state[:6]
        # state_rotating_bary_lumio_0 = self.dynamic_model_object.custom_initial_state[6:]
        print("FrameConverter: ", self.dynamic_model_object.custom_initial_state)

        # if custom_initial_state is None:



        if custom_initial_state is not None:

            state_rotating_bary_lpf_0 = self.dynamic_model_object.custom_initial_state[:6]
            state_rotating_bary_lumio_0 = self.dynamic_model_object.custom_initial_state[6:]

        state_rotating_bary_lumio_0 = [1.1473302, 0, -0.15142308, 0, -0.21994554, 0]
        state_rotating_bary_lpf_0   = [0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0]




        # Generate history of classical CRTBP
        system   = CRTBP_traditional.CRTBP(self.G, self.m1, self.m2, self.a)
        t, state_rotating_bary_lumio = system.get_state_history(state_rotating_bary_lumio_0, 0, self.propagation_time, self.step_size)
        t, state_rotating_bary_lpf = system.get_state_history(state_rotating_bary_lpf_0, 0, self.propagation_time, self.step_size)

        # Convert satellite states to moon centric frame
        state_history_moon_fixed_lumio = system.convert_state_barycentric_to_body(state_rotating_bary_lumio, "secondary", state_type="rotating")
        state_history_moon_fixed_lpf = system.convert_state_barycentric_to_body(state_rotating_bary_lpf, "secondary", state_type="rotating")
        state_history_moon_fixed_satellites = np.concatenate((state_history_moon_fixed_lpf, state_history_moon_fixed_lumio), axis=1)

        # Convert primaries states to moon centric frame
        state_rotating_bary_primary = np.multiply(np.ones((np.shape(t)[0],6)), system.state_m1)
        state_rotating_bary_secondary = np.multiply(np.ones((np.shape(t)[0],6)), system.state_m2)
        state_history_moon_fixed_primary = system.convert_state_barycentric_to_body(state_rotating_bary_primary, "secondary", state_type="rotating")
        state_history_moon_fixed_secondary = system.convert_state_barycentric_to_body(state_rotating_bary_secondary, "secondary", state_type="rotating")
        state_history_moon_fixed_primaries = np.concatenate((state_history_moon_fixed_primary, state_history_moon_fixed_secondary), axis=1)

        # Looping through all epochs to convert each synodic frame element to J2000 Earth-centered
        state_history_inertial_satellites = np.empty(np.shape(state_history_moon_fixed_satellites))
        state_history_inertial_primaries = np.empty(np.shape(state_history_moon_fixed_primaries))
        for epoch, state in enumerate(state_history_moon_fixed_satellites):
            state_history_inertial_satellites[epoch] = self.convert_synodic_to_inertial_state(state_history_moon_fixed_satellites[epoch], dependent_variables_history[epoch, :6])
            state_history_inertial_primaries[epoch] = self.convert_synodic_to_inertial_state(state_history_moon_fixed_primaries[epoch], dependent_variables_history[epoch, :6])
        print(state_history_moon_fixed_primaries)
        print(state_history_inertial_primaries)
        return t, state_history_inertial_satellites, state_history_inertial_primaries




    # def get_inertial_state_history(self, initial_state=None):

    #     # self.dynamics_simulator, self.variational_equations_solver = self.dynamic_model_object.get_propagated_orbit()
    #     epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
    #         Interpolator.Interpolator(self.dynamic_model_object, step_size=self.step_size*constants.JULIAN_DAY).get_results()
    #     self.propagation_time = self.dynamic_model_object.propagation_time

    #     state_rotating_bary_lumio_0 = [1.1473302, 0, -0.15142308, 0, -0.21994554, 0]
    #     # state_rotating_bary_lpf_0   = [0.98512134, 0.00147649, 0.00492546, -0.87329730, -1.61190048, 0]
    #     # state_rotating_bary

    #     system   = CRTBP_traditional.CRTBP(self.G, self.m1, self.m2, self.a)
    #     t, state_rotating_bary_lumio = system.get_state_history(state_rotating_bary_lumio_0, 0, self.propagation_time, self.step_size)
    #     # t, state_rotating_bary_lpf = system.get_state_history(state_rotating_bary_lpf_0, 0, self.propagation_time, self.step_size)
    #     state_history_moon_fixed_lumio = system.convert_state_barycentric_to_body(state_rotating_bary_lumio, "secondary", state_type="rotating")

    #     # Looping through all epochs to convert each synodic frame element to J2000 Earth-centered
    #     state_history_lumio_CRTBP = np.empty(np.shape(state_history_moon_fixed_lumio))
    #     for epoch, state in enumerate(state_history_moon_fixed_lumio):

    #         rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(dependent_variables_history[epoch, :6])

    #         omega_w = np.dot(rsw_to_inertial_rotation_matrix[:, 2], self.rotation_rate)
    #         omega_w_norm = np.linalg.norm(omega_w)
    #         Omega = np.array([[0, omega_w_norm, 0],[-omega_w_norm, 0, 0],[0, 0, 0]])

    #         time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, -Omega)
    #         total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
    #                         [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

    #         initial_state_lumio_moon_fixed = np.concatenate((state_history_moon_fixed_lumio[epoch, :3]*self.lu_cr3bp, state_history_moon_fixed_lumio[epoch, 3:]*self.lu_cr3bp/self.tu_cr3bp))
    #         state_history_lumio_CRTBP[epoch] = dependent_variables_history[epoch, :6] + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_lumio_moon_fixed)

    #     return t, state_history_lumio_CRTBP




# test = SynodicToInertialHistoryConverter([1.1473302, 0, -0.15142308, 0, -0.21994554, 0])
# print(test.generate_state_history())










# class SynodicToInertialHistoryConverter:

#     def __init__(self, state_history_synodic, dynamic_model_object):

#         self.state_history_synodic = state_history_synodic
#         self.dynamic_model_object = dynamic_model_object
#         self.interp_epochs, self.interp_state_history, self.interp_dependent_variables_history, self.interp_state_transition_matrix_history = \
#             Interpolator.Interpolator(dynamic_model_object).get_results()

#     def

























#     def convert_state_barycentric_to_body(self):

#         self.state_moon_fixed = self.state_history_synodic
#         self.state_moon_fixed[:,0] = self.state_moon_fixed[:,0] + (1-self.mu)
#         self.state_moon_fixed[:,0] = self.state_moon_fixed[:,0] + (1-self.mu)



#     def convert_nondim_to_dim(self):

#         lu = np.linalg.norm(self.interp_dependent_variables_history[:,:6], axis=1)
#         tu = 1/np.sqrt((self.dynamic_model_object.gravitational_parameter_primary + self.dynamic_model_object.gravitational_parameter_secondary)/lu**3)

#         self.state_history_barycentric_LPF_dim = np.concatenate((self.state_history_synodic[:,:3]*lu, self.state_history_synodic[:,3:6]*lu/tu), axis=1)
#         self.state_history_barycentric_LUMIO_dim = np.concatenate((self.state_history_synodic[:,6:9]*lu, self.state_history_synodic[:,9:12]*lu/tu), axis=1)
#         self.state_history_barycentric_dim = np.concatenate((self.state_history_barycentric_LPF_dim, self.state_history_barycentric_LUMIO_dim), axis=1)


#     def convert_barycentric_to_body_fixed(self):

#         self.convert_nondim_to_dim()

#         self.state_history_barycentric_primary = dependent_variables_history[:,:6]*(-dynamic_model.mu)
#         self.state_history_barycentric_secondary = dependent_variables_history[:,:6]*(1-dynamic_model.mu)
#         self.state_history_earth_fixed_LPF = np.add(self.interp_state_history[:,:6], self.state_history_barycentric_primary)
#         self.state_history_earth_fixed_LUMIO = np.add(self.interp_state_history[:,6:], self.state_history_barycentric_primary)
#         self.state_history_earth_fixed = np.concatenate((self.state_history_earth_fixed_LPF, self.state_history_earth_fixed_LUMIO), axis=1)


#     def convert_





#     def convert_body_fixed_to_barycentric(self):

#         state_history_barycentric_primary = self.interp_dependent_variables_history[:,:6]*(-self.dynamic_model_object.mu)
#         state_history_barycentric_secondary = self.interp_dependent_variables_history[:,:6]*(1-self.dynamic_model_object.mu)

#         state_history_barycentric_LPF = np.add(state_history[:,:6], state_history_barycentric_primary)
#         state_history_barycentric_LUMIO = np.add(state_history[:,6:], state_history_barycentric_primary)
#         state_history_barycentric = np.concatenate((state_history_barycentric_LPF, state_history_barycentric_LUMIO))










#     def convert_nondim_to_dim(self):

#         # self.dynamic_model_object.mu
#         lu = np.linalg.norm(self.dependent_variables_history[:,:3], axis=1)
#         tu = 1/np.sqrt((self.dynamic_model_object.gravitational_parameter_primary + self.dynamic_model_object.gravitational_parameter_secondary)/lu**3)

#         np.concatenate((state_history_lpf_moon_dim[0:3]/lu, state_history_lpf_moon_dim[3:6]/(lu/tu)))
