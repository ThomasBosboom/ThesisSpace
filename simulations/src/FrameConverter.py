# import numpy as np
# from scipy.interpolate import interp1d
# from tudatpy.kernel.astro import frame_conversion
# import Interpolator

# class SynodicToInertialHistoryConverter:

#     def __init__(self, dynamic_model, step_size=0.001):

#         self.dynamic_model = dynamic_model
#         self.propagation_time = dynamic_model.propagation_time
#         self.mu = dynamic_model.mu
#         self.step_size = step_size


#     def get_total_rsw_to_inertial_rotation_matrix(self, inertial_moon_state):

#         # Determine transformation matrix from Moon-fixed frame to inertial coordinates for a given epoch
#         rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(inertial_moon_state)

#         # Determine rotation rate and direction with respect to synodic frame (so only rotation w of rsw is relevant)
#         # rotation_rate = self.dynamic_model.rotation_rate
#         m = self.dynamic_model.bodies.get("Moon").mass
#         r_norm = np.linalg.norm(inertial_moon_state[:3])
#         v_norm = np.linalg.norm(inertial_moon_state[3:])
#         h = m*r_norm*v_norm
#         rotation_rate = h/(m*r_norm**2)

#         # print("Earth: ", self.dynamic_model.bodies.get("Earth").mass)
#         # print("Moon: ", self.dynamic_model.bodies.get("Moon").mass)

#         Omega = np.array([[0, -rotation_rate, 0],[rotation_rate, 0, 0],[0, 0, 0]])

#         # Update total transformation matrix with matrix derivative element
#         time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, Omega)
#         total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
#                         [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

#         return total_rsw_to_inertial_rotation_matrix


#     def convert_state_nondim_to_dim_state(self, synodic_state, inertial_moon_state):

#         # lu_cr3bp = self.dynamic_model.lu_cr3bp
#         # tu_cr3bp = self.dynamic_model.lu_cr3bp

#         lu_cr3bp = np.linalg.norm(inertial_moon_state[:3])
#         tu_cr3bp = 1/np.sqrt((self.dynamic_model.gravitational_parameter_primary + \
#             self.dynamic_model.gravitational_parameter_secondary)/lu_cr3bp**3)

#         return np.concatenate((synodic_state[:3]*lu_cr3bp, synodic_state[3:]*lu_cr3bp/tu_cr3bp))


#     def convert_state_barycentric_to_body(self, state_barycentric, body, state_type="rotating"):

#         if state_type == "inertial":
#             if body == "primary":
#                 return state_barycentric - np.array([-self.mu, 0, 0, 0, 0, 0])*state_barycentric
#             if body == "secondary":
#                 return state_barycentric - np.array([1-self.mu, 0, 0, 0, 0, 0])*state_barycentric

#         elif state_type == "rotating":
#             state_body = state_barycentric
#             if body == "primary":
#                 return state_body - np.array([-self.mu, 0, 0, 0, 0, 0])
#             if body == "secondary":
#                 return state_body - np.array([1-self.mu, 0, 0, 0, 0, 0])


#     def convert_secondary_fixed_to_inertial_state(self, inertial_moon_state, total_rsw_to_inertial_rotation_matrix, state_body_fixed):
#         return inertial_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,state_body_fixed)


#     def convert_synodic_to_inertial_state(self, synodic_state, inertial_moon_state):

#         # Obtain the transformation matrix from J2000 inertial to Earth-Moon synodic frame
#         total_rsw_to_inertial_rotation_matrix = self.get_total_rsw_to_inertial_rotation_matrix(inertial_moon_state)

#         # Dimensionalize
#         initial_state_lpf_moon_fixed   = self.convert_state_nondim_to_dim_state(synodic_state[:6], inertial_moon_state)
#         initial_state_lumio_moon_fixed = self.convert_state_nondim_to_dim_state(synodic_state[6:], inertial_moon_state)

#         # Apply transformations and rotations
#         state_history_lumio_inertial = self.convert_secondary_fixed_to_inertial_state(inertial_moon_state, total_rsw_to_inertial_rotation_matrix, initial_state_lumio_moon_fixed)
#         state_history_lpf_inertial   = self.convert_secondary_fixed_to_inertial_state(inertial_moon_state, total_rsw_to_inertial_rotation_matrix, initial_state_lpf_moon_fixed)
#         state_history_inertial       = np.concatenate((state_history_lpf_inertial, state_history_lumio_inertial))

#         return state_history_inertial


#     def get_results(self, synodic_state_history):

#         epochs, _, dependent_variables_history, _ = \
#             Interpolator.Interpolator(step_size=self.step_size).get_propagation_results(self.dynamic_model)

#         # Generate history of classical CRTBP
#         state_rotating_bary_lpf, state_rotating_bary_lumio = synodic_state_history[:,:6], synodic_state_history[:,6:]

#         # Convert satellite states to moon centric frame
#         state_history_moon_fixed_lumio      = self.convert_state_barycentric_to_body(state_rotating_bary_lumio, "secondary", state_type="rotating")
#         state_history_moon_fixed_lpf        = self.convert_state_barycentric_to_body(state_rotating_bary_lpf, "secondary", state_type="rotating")
#         state_history_moon_fixed_satellites = np.concatenate((state_history_moon_fixed_lpf, state_history_moon_fixed_lumio), axis=1)

#         # Convert primaries states to moon centric frame
#         state_rotating_bary_primary        = np.multiply(np.ones((np.shape(state_rotating_bary_lumio)[0],6)), np.array([-self.mu, 0, 0, 0, 0, 0]))
#         state_rotating_bary_secondary      = np.multiply(np.ones((np.shape(state_rotating_bary_lumio)[0],6)), np.array([1-self.mu, 0, 0, 0, 0, 0]))
#         state_history_moon_fixed_primary   = self.convert_state_barycentric_to_body(state_rotating_bary_primary, "secondary", state_type="rotating")
#         state_history_moon_fixed_secondary = self.convert_state_barycentric_to_body(state_rotating_bary_secondary, "secondary", state_type="rotating")
#         state_history_moon_fixed_primaries = np.concatenate((state_history_moon_fixed_primary, state_history_moon_fixed_secondary), axis=1)

#         # Looping through all epochs to convert each synodic frame element to J2000 Earth-centered
#         state_history_inertial_satellites = np.empty(np.shape(state_history_moon_fixed_satellites))
#         state_history_inertial_primaries  = np.empty(np.shape(state_history_moon_fixed_primaries))
#         for epoch, state in enumerate(synodic_state_history):
#             state_history_inertial_satellites[epoch] = self.convert_synodic_to_inertial_state(state_history_moon_fixed_satellites[epoch], dependent_variables_history[epoch, :6])
#             state_history_inertial_primaries[epoch]  = self.convert_synodic_to_inertial_state(state_history_moon_fixed_primaries[epoch], dependent_variables_history[epoch, :6])

#         return epochs, state_history_inertial_satellites, state_history_inertial_primaries




# class InertialToSynodicHistoryConverter:

#     def __init__(self, dynamic_model, step_size=0.001):

#         self.dynamic_model = dynamic_model
#         self.propagation_time = dynamic_model.propagation_time
#         self.mu = dynamic_model.mu
#         self.step_size = step_size


#     def get_total_inertial_to_rsw_rotation_matrix(self, inertial_moon_state):

#         # Determine transformation matrix from Moon-fixed frame to inertial coordinates for a given epoch
#         rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(inertial_moon_state)

#         # Determine rotation rate and direction with respect to synodic frame (so only rotation w of rsw is relevant)
#         # rotation_rate = self.dynamic_model.rotation_rate
#         m = self.dynamic_model.bodies.get("Moon").mass
#         r_norm = np.linalg.norm(inertial_moon_state[:3])
#         v_norm = np.linalg.norm(inertial_moon_state[3:])
#         h = m*r_norm*v_norm
#         rotation_rate = h/(m*r_norm**2)

#         Omega = np.array([[0, -rotation_rate, 0],[rotation_rate, 0, 0],[0, 0, 0]])

#         # Update total transformation matrix with matrix derivative element
#         time_derivative_rsw_to_inertial_rotation_matrix = np.dot(rsw_to_inertial_rotation_matrix, Omega)
#         total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
#                         [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

#         return np.linalg.inv(total_rsw_to_inertial_rotation_matrix)


#     def convert_state_dim_to_nondim_state(self, synodic_state, inertial_moon_state):

#         # lu_cr3bp = self.dynamic_model.lu_cr3bp
#         # tu_cr3bp = self.dynamic_model.lu_cr3bp

#         lu_cr3bp = np.linalg.norm(inertial_moon_state[:3])
#         tu_cr3bp = 1/np.sqrt((self.dynamic_model.gravitational_parameter_primary + \
#             self.dynamic_model.gravitational_parameter_secondary)/lu_cr3bp**3)

#         return np.concatenate((synodic_state[:3]/lu_cr3bp, synodic_state[3:]/(lu_cr3bp/tu_cr3bp)))


#     def convert_state_body_to_barycentric(self, state_barycentric, body, state_type="rotating"):

#         if state_type == "inertial":
#             if body == "primary":
#                 return state_barycentric + np.array([-self.mu, 0, 0, 0, 0, 0])*state_barycentric
#             if body == "secondary":
#                 return state_barycentric + np.array([1-self.mu, 0, 0, 0, 0, 0])*state_barycentric

#         elif state_type == "rotating":
#             state_body = state_barycentric
#             if body == "primary":
#                 return state_body + np.array([-self.mu, 0, 0, 0, 0, 0])
#             if body == "secondary":
#                 return state_body + np.array([1-self.mu, 0, 0, 0, 0, 0])


#     def convert_inertial_to_secondary_fixed_state(self, inertial_moon_state, total_inertial_to_rsw_rotation_matrix, state_inertial):
#         return np.dot(total_inertial_to_rsw_rotation_matrix, state_inertial-inertial_moon_state)


#     def convert_inertial_to_synodic_state(self, inertial_state, inertial_moon_state):

#         # Obtain the transformation matrix from J2000 inertial to Earth-Moon synodic frame
#         total_inertial_to_rsw_rotation_matrix = self.get_total_inertial_to_rsw_rotation_matrix(inertial_moon_state)

#         # Apply transformations and rotations
#         initial_state_lpf_moon_fixed = self.convert_inertial_to_secondary_fixed_state(inertial_moon_state, total_inertial_to_rsw_rotation_matrix, inertial_state[:6])
#         initial_state_lumio_moon_fixed = self.convert_inertial_to_secondary_fixed_state(inertial_moon_state, total_inertial_to_rsw_rotation_matrix, inertial_state[6:])

#         # Non-dimensionalize
#         initial_state_lpf_moon_fixed   = self.convert_state_dim_to_nondim_state(initial_state_lpf_moon_fixed, inertial_moon_state)
#         initial_state_lumio_moon_fixed = self.convert_state_dim_to_nondim_state(initial_state_lumio_moon_fixed, inertial_moon_state)
#         state_history_synodic       = np.concatenate((initial_state_lpf_moon_fixed, initial_state_lumio_moon_fixed))

#         return state_history_synodic


#     def get_results(self, inertial_state_history):

#         epochs, _, dependent_variables_history = \
#             Interpolator.Interpolator(step_size=self.step_size).get_propagation_results(self.dynamic_model, solve_variational_equations=False)

#         # Split states into spacecraft states
#         state_inertial_lpf, state_inertial_lumio = inertial_state_history[:,:6], inertial_state_history[:,6:]

#         # Looping through all epochs to convert each inertial element to the synodic frame
#         state_history_inertial_satellites = np.empty(np.shape(inertial_state_history))
#         for epoch, state in enumerate(inertial_state_history):
#             state_history_inertial_satellites[epoch] = self.convert_inertial_to_synodic_state(inertial_state_history[epoch], dependent_variables_history[epoch, :6])

#         # Convert satellite states to barycentric frame
#         state_history_barycentric_lpf = self.convert_state_body_to_barycentric(state_history_inertial_satellites[:,:6], "secondary", state_type="rotating")
#         state_history_barycentric_lumio = self.convert_state_body_to_barycentric(state_history_inertial_satellites[:,6:], "secondary", state_type="rotating")
#         state_history_barycentric_satellites = np.concatenate((state_history_barycentric_lpf, state_history_barycentric_lumio), axis=1)

#         return epochs, state_history_barycentric_satellites











# # import numpy as np
# # from scipy.interpolate import interp1d
# # from tudatpy.kernel.astro import frame_conversion
# # import Interpolator

# # class FrameConverter:

# #     def __init__(self, state_history, moon_state_history, epochs, step_size=0.001):

# #         self.moon_data_dict = {epoch: state for epoch, state in zip(epochs, moon_state_history[:, :])}
# #         self.satellite_data_dict = {epoch: state for epoch, state in zip(epochs, state_history[:, :])}
# #         # self.moon_data_dict = moon_state_history
# #         # self.satellite_data_dict = state_history

# #         # print(self.satellite_data_dict)
# #         self.G = 6.67430e-11
# #         self.m1 = 5.972e24
# #         self.m2 = 7.34767309e22
# #         self.mu = self.m2/(self.m2 + self.m1)


# #     def InertialToSynodicHistoryConverter(self):

# #         # Create the Direct Cosine Matrix based on rotation axis of Moon around Earth
# #         transformation_matrix_dict = {}
# #         for epoch, moon_state in self.moon_data_dict.items():

# #             moon_position, moon_velocity = moon_state[:3], moon_state[3:]

# #             # Define the complementary axes of the rotating frame
# #             rotation_axis = np.cross(moon_position, moon_velocity)
# #             second_axis = np.cross(moon_position, rotation_axis)

# #             # Define the rotation matrix (DCM) using the rotating frame axes
# #             first_axis = moon_position/np.linalg.norm(moon_position)
# #             second_axis = second_axis/np.linalg.norm(second_axis)
# #             third_axis = rotation_axis/np.linalg.norm(rotation_axis)
# #             transformation_matrix = np.array([first_axis, second_axis, third_axis])
# #             rotation_axis = rotation_axis*self.m2
# #             rotation_rate = rotation_axis/(self.m2*np.linalg.norm(moon_position)**2)

# #             skew_symmetric_matrix = np.array([[0, -rotation_rate[2], rotation_rate[1]],
# #                                               [rotation_rate[2], 0, -rotation_rate[0]],
# #                                               [-rotation_rate[1], rotation_rate[0], 0]])

# #             transformation_matrix_derivative =  np.dot(transformation_matrix, skew_symmetric_matrix)
# #             transformation_matrix = np.block([[transformation_matrix, np.zeros((3,3))],
# #                                               [transformation_matrix_derivative, transformation_matrix]])

# #             transformation_matrix_dict.update({epoch: transformation_matrix})


# #         # self.Generate the synodic states of the satellites
# #         synodic_satellite_states_dict = {}
# #         for epoch, state in self.satellite_data_dict.items():

# #             transformation_matrix = transformation_matrix_dict[epoch]
# #             synodic_state = np.concatenate((np.dot(transformation_matrix, state[0:6]), np.dot(transformation_matrix, state[6:12])))

# #             LU = np.linalg.norm((self.moon_data_dict[epoch][0:3]))
# #             TU = np.sqrt(LU**3/(self.G*(self.m1+self.m2)))
# #             synodic_state[0:3] = synodic_state[0:3]/LU
# #             synodic_state[6:9] = synodic_state[6:9]/LU
# #             synodic_state[3:6] = synodic_state[3:6]/(LU/TU)
# #             synodic_state[9:12] = synodic_state[9:12]/(LU/TU)
# #             synodic_state = (1-self.mu)*synodic_state

# #             synodic_satellite_states_dict.update({epoch: synodic_state})

# #         synodic_states = np.stack(list(synodic_satellite_states_dict.values()))

# #         # Generate the synodic states of the moon
# #         synodic_moon_states_dict = {}
# #         for epoch, state in self.moon_data_dict.items():

# #             transformation_matrix = transformation_matrix_dict[epoch]
# #             synodic_state = np.dot(transformation_matrix, state)
# #             LU = np.linalg.norm((self.moon_data_dict[epoch][0:3]))
# #             TU = np.sqrt(LU**3/(self.G*(self.m1+self.m2)))
# #             synodic_state[0:3] = synodic_state[0:3]/LU
# #             synodic_state[3:6] = synodic_state[3:6]/(LU/TU)
# #             synodic_state = (1-self.mu)*synodic_state
# #             synodic_moon_states_dict.update({epoch: synodic_state})

# #         synodic_states = np.stack(list(synodic_satellite_states_dict.values()))
# #         synodic_moon_states = np.stack(list(synodic_moon_states_dict.values()))

# #         return synodic_states, synodic_moon_states






import numpy as np
from scipy.interpolate import interp1d
from tudatpy.kernel.astro import frame_conversion
import Interpolator, ReferenceData



class FrameConverter():

    def __init__(self, dynamics_simulator, G=6.67430e-11, m1=5.972e24, m2=7.34767309e22):

        self.dynamics_simulator = dynamics_simulator
        self.moon_data_dict = {epoch: state[:6] for epoch, state in dynamics_simulator.dependent_variable_history.items()}
        self.G = G
        self.m1 = m1
        self.m2 = m2
        self.mu = m2/(m2 + m1)


    def interpolate_dict(self, original_dict, new_keys):
        original_keys = list(original_dict.keys())
        original_values = np.array(list(original_dict.values()))

        num_dims = original_values.shape[1]
        new_values = []
        for dim in range(num_dims):

            interpolation_function = interp1d(original_keys, original_values[:, dim], kind='linear', fill_value='extrapolate')
            new_values.append(interpolation_function(new_keys))

        new_values = np.vstack(new_values).T
        return {k: v for k, v in zip(new_keys, new_values)}


    def get_transformation_matrix_dict(self):

        # Create the transformation based on rotation axis of Moon around Earth
        transformation_matrix_dict = {}
        for epoch, moon_state in self.moon_data_dict.items():

            moon_position, moon_velocity = moon_state[:3], moon_state[3:]

            # Define the complementary axes of the rotating frame
            rotation_axis = np.cross(moon_position, moon_velocity)
            second_axis = np.cross(moon_position, rotation_axis)

            # Define the rotation matrix (DCM) using the rotating frame axes
            first_axis = moon_position/np.linalg.norm(moon_position)
            second_axis = second_axis/np.linalg.norm(second_axis)
            third_axis = rotation_axis/np.linalg.norm(rotation_axis)
            transformation_matrix = np.array([first_axis, second_axis, third_axis])
            rotation_axis = rotation_axis*self.m2
            rotation_rate = rotation_axis/(self.m2*np.linalg.norm(moon_position)**2)

            skew_symmetric_matrix = np.array([[0, -rotation_rate[2], rotation_rate[1]],
                                                [rotation_rate[2], 0, -rotation_rate[0]],
                                                [-rotation_rate[1], rotation_rate[0], 0]])

            transformation_matrix_derivative =  np.dot(transformation_matrix, skew_symmetric_matrix)
            transformation_matrix = np.block([[transformation_matrix, np.zeros((3,3))],
                                                [transformation_matrix_derivative, transformation_matrix]])

            transformation_matrix_dict.update({epoch: transformation_matrix})

        return transformation_matrix_dict


    def get_synodic_state_history(self, other_dict={}):

        transformation_matrix_dict = self.get_transformation_matrix_dict()
        inertial_state_history_dict = self.dynamics_simulator.state_history

        if other_dict:
            new_keys = np.array(list(inertial_state_history_dict.keys()))
            inertial_state_history_dict = self.interpolate_dict(other_dict, new_keys)

            # interpolator = Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.001)
            # reference_data = ReferenceData.ReferenceData(interpolator)
            # moon_state_history = reference_data.state_history_reference_lumio[1]
            # moon_data_dict = {epoch: vector for epoch, vector in zip(moon_state_history[:, :1], moon_state_history[:, 2:]*1000)}

            # self.moon_data_dict = self.interpolate_dict(moon_data_dict_reference, new_keys)
            # transformation_matrix_dict = self.get_transformation_matrix_dict()

            # print("HEREEE: ", np.array(list(other_dict.keys())), new_keys)

        # Generate the synodic states of the satellites
        synodic_full_state_history_estimated_dict = {}
        synodic_dictionaries = [synodic_full_state_history_estimated_dict]
        inertial_dictionaries = [inertial_state_history_dict]
        for index, dictionary in enumerate(inertial_dictionaries):
            for epoch, state in dictionary.items():

                transformation_matrix = transformation_matrix_dict[epoch]
                synodic_state = np.concatenate((np.dot(transformation_matrix, state[0:6]), np.dot(transformation_matrix, state[6:12])))

                LU = np.linalg.norm((self.moon_data_dict[epoch][0:3]))
                TU = np.sqrt(LU**3/(self.G*(self.m1+self.m2)))
                synodic_state[0:3] = synodic_state[0:3]/LU
                synodic_state[6:9] = synodic_state[6:9]/LU
                synodic_state[3:6] = synodic_state[3:6]/(LU/TU)
                synodic_state[9:12] = synodic_state[9:12]/(LU/TU)
                synodic_state = (1-self.mu)*synodic_state

                synodic_dictionaries[index].update({epoch: synodic_state})

        return synodic_dictionaries[0]




# a = np.array([[1, 2], [3, 4]])

# b = np.array([[5, 6], [7, 8]])

# print(np.concatenate((a, b.T), axis=1))

# moon = {60390: [-2.79077269e8,  2.52757217e8,  1.45049671e8, -7.19983882e2, -5.83630638e2, -2.97164832e2], 60391: [-2.79077269e8,  2.52757217e8,  1.45049671e8, -7.19983882e2, -5.83630638e2, -2.97164832e2]}
# state = {60390: [-2.74751546e8,  2.50414392e+08,  1.37232531e+08, -5.03244807e+02, -1.83582360e+02, -2.97163832e+02, -3.10468779e+08 , 2.49476676e+08, 1.74974583e+08, -9.93404005e+02, -7.66335485e+02 ,-5.24989115e+02],
#         60391: [-2.74751546e8,  2.50414392e+08,  1.37232531e+08, -5.03244807e+02, -1.83582360e+02, -2.97163832e+02, -3.10468779e+08 , 2.49476676e+08, 1.74974583e+08, -9.93404005e+02, -7.66335485e+02 ,-5.24989115e+02]}


# frame_converter = FrameConverter(moon)

# final = frame_converter.get_synodic_state_history(state)


# print(final)



# synodic_states_estimated = np.stack(list(synodic_full_state_history_estimated_dict.values()))