import numpy as np
from scipy.interpolate import interp1d
from tudatpy.kernel.astro import frame_conversion
import Interpolator

class FrameConverter:

    def __init__(self, state_history, moon_state_history, epochs, step_size=0.001):

        self.moon_data_dict = {epoch: state for epoch, state in zip(epochs, moon_state_history[:, :])}
        self.satellite_data_dict = {epoch: state for epoch, state in zip(epochs, state_history[:, :])}
        # self.moon_data_dict = moon_state_history
        # self.satellite_data_dict = state_history

        # print(self.satellite_data_dict)
        self.G = 6.67430e-11
        self.m1 = 5.972e24
        self.m2 = 7.34767309e22
        self.mu = self.m2/(self.m2 + self.m1)


    def InertialToSynodicHistoryConverter(self):

        # Create the Direct Cosine Matrix based on rotation axis of Moon around Earth
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


        # self.Generate the synodic states of the satellites
        synodic_satellite_states_dict = {}
        for epoch, state in self.satellite_data_dict.items():

            transformation_matrix = transformation_matrix_dict[epoch]
            synodic_state = np.concatenate((np.dot(transformation_matrix, state[0:6]), np.dot(transformation_matrix, state[6:12])))

            LU = np.linalg.norm((self.moon_data_dict[epoch][0:3]))
            TU = np.sqrt(LU**3/(self.G*(self.m1+self.m2)))
            synodic_state[0:3] = synodic_state[0:3]/LU
            synodic_state[6:9] = synodic_state[6:9]/LU
            synodic_state[3:6] = synodic_state[3:6]/(LU/TU)
            synodic_state[9:12] = synodic_state[9:12]/(LU/TU)
            synodic_state = (1-self.mu)*synodic_state

            synodic_satellite_states_dict.update({epoch: synodic_state})

        synodic_states = np.stack(list(synodic_satellite_states_dict.values()))

        # Generate the synodic states of the moon
        synodic_moon_states_dict = {}
        for epoch, state in self.moon_data_dict.items():

            transformation_matrix = transformation_matrix_dict[epoch]
            synodic_state = np.dot(transformation_matrix, state)
            LU = np.linalg.norm((self.moon_data_dict[epoch][0:3]))
            TU = np.sqrt(LU**3/(self.G*(self.m1+self.m2)))
            synodic_state[0:3] = synodic_state[0:3]/LU
            synodic_state[3:6] = synodic_state[3:6]/(LU/TU)
            synodic_state = (1-self.mu)*synodic_state
            synodic_moon_states_dict.update({epoch: synodic_state})

        synodic_states = np.stack(list(synodic_satellite_states_dict.values()))
        synodic_moon_states = np.stack(list(synodic_moon_states_dict.values()))

        return synodic_states, synodic_moon_states