import numpy as np
import sys
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt




file_path = os.path.realpath(__file__)
file_path = os.path.dirname(file_path)
cut_value = 60390+1
step_size = 0.01

state_histories = {}
for body in ["LPF", "Moon"]:

    original_file = f"{body}_states_J2000_Earth_centered.txt"
    new_file = f"{body}_states_J2000_Earth_centered.txt"

    # Load the data from the provided text file
    with open(os.path.join(file_path, original_file), "r") as file:
        lines = file.readlines()

    # Load the data excluding the title row
    data = np.loadtxt(lines[1:], delimiter=",")

    # data = np.loadtxt(os.path.join(file_path, original_file), delimiter=",")
    filtered_data = data[data[:, 0] <= cut_value]
    print(len(data))

    filtered_data_dict = {data[i, 0]: data[i, 2:8] for i in range(len(data))}

    # print(filtered_data_dict)

    state_histories[body] = filtered_data_dict


moon_data_dict = state_histories["Moon"]
satellite_data_dict = state_histories["LPF"]

print(len(moon_data_dict))

# Create the Direct Cosine Matrix based on rotation axis of Moon around Earth
def create_transformation_matrix(duration):

    transformation_matrix_dict = {}
    for epoch, moon_state in moon_data_dict.items():

        moon_position, moon_velocity = moon_state[:3], moon_state[3:]

        # Define the complementary axes of the rotating frame
        rotation_axis = np.cross(moon_position, moon_velocity)
        second_axis = np.cross(moon_position, rotation_axis)

        # Define the rotation matrix (DCM) using the rotating frame axes
        first_axis = moon_position/np.linalg.norm(moon_position)
        second_axis = second_axis/np.linalg.norm(second_axis)
        third_axis = rotation_axis/np.linalg.norm(rotation_axis)
        transformation_matrix = np.array([first_axis, second_axis, third_axis])

        transformation_matrix_dict.update({epoch: transformation_matrix})

    return transformation_matrix_dict


transformation_matrix_dict = create_transformation_matrix()

rotating_states_dict = {}
for epoch, state in satellite_data_dict.items():

    transformation_matrix = transformation_matrix_dict[epoch]

    rotating_state = np.dot(transformation_matrix, state[:3])

    # Non dimensionalize
    rotating_state = rotating_state/np.linalg.norm((moon_data_dict[epoch][:3]))

    # Shift to bary-centric frame
    mu = 7.34767309e22/(7.34767309e22 + 5.972e24)
    rotating_state = (1-mu)*rotating_state

    rotating_states_dict.update({epoch: rotating_state})




rotating_moon_states_dict = {}
for epoch, state in moon_data_dict.items():

    # Obtain the transformation matrix
    transformation_matrix = transformation_matrix_dict[epoch]

    # Convert to the inertial frame
    rotating_state = np.dot(transformation_matrix, state[:3])

    # Non dimensionalize
    rotating_state = rotating_state/np.linalg.norm(state[:3])

    # Shift to bary-centric frame
    mu = 7.34767309e22/(7.34767309e22 + 5.972e24)
    rotating_state = (1-mu)*rotating_state

    rotating_moon_states_dict.update({epoch: rotating_state})



rotating_states = np.stack(list(rotating_states_dict.values()))
rotating_moon_states = np.stack(list(rotating_moon_states_dict.values()))

fig, ax = plt.subplots(1, 3, figsize=(12, 5))
ax[0].scatter(rotating_moon_states[:, 0], rotating_moon_states[:, 2], s=10)
ax[1].scatter(rotating_moon_states[:, 1], rotating_moon_states[:, 2], s=10)
ax[2].scatter(rotating_moon_states[:, 0], rotating_moon_states[:, 1], s=10)
ax[0].plot(rotating_states[:, 0], rotating_states[:, 2], lw=0.2)
ax[1].plot(rotating_states[:, 1], rotating_states[:, 2], lw=0.2)
ax[2].plot(rotating_states[:, 0], rotating_states[:, 1], lw=0.2)

fig1_3d = plt.figure()
ax_3d = fig1_3d.add_subplot(111, projection='3d')
ax_3d.plot(rotating_states[:, 0], rotating_states[:, 1], rotating_states[:, 2], lw=0.2)
plt.show()




