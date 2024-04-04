import numpy as np
import sys
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# file_path = os.path.realpath(__file__)
# file_path = os.path.dirname(file_path)
# cut_value = 60390+60
# step_size = 0.01

# state_histories = {}

# for body in ["LPF", "Moon"]:

#     original_file = f"365days\{body}_states_J2000_Earth_centered.txt"
#     new_file = f"{body}_states_J2000_Earth_centered.txt"

#     # Load the data from the provided text file
#     with open(os.path.join(file_path, original_file), "r") as file:
#         lines = file.readlines()

#     # Load the data excluding the title row
#     data = np.loadtxt(lines[1:], delimiter=",")

#     # data = np.loadtxt(os.path.join(file_path, original_file), delimiter=",")
#     filtered_data = data[data[:, 0] <= cut_value]

#     # print(filtered_data)

#     state_histories[body] = filtered_data


# moon_data = state_histories["Moon"]
# satellite_data = state_histories["LPF"]

# # # Extracting positions and velocities
# # moon_position = moon_data[:, 2:5]
# # moon_velocity = moon_data[:, 5:8]

# # satellite_position = satellite_data[:, 2:5]
# # satellite_velocity = satellite_data[:, 5:8]

# rotating_states_dict = {}
# for data in moon_data[:100, :]:

#     # Create the Direct Cosine Matrix based on rotation axis of Moon around Earth
#     epoch = data[0]
#     moon_position = data[2:5]
#     moon_velocity = data[5:8]

#     rotation_axis = np.cross(moon_position, moon_velocity)
#     rotation_rate_magnitude = np.linalg.norm(rotation_axis) / np.linalg.norm(moon_position)**2
#     rotation_axis_unit = rotation_axis / np.linalg.norm(rotation_axis)
#     second_axis = np.cross(moon_position, rotation_axis)

#     first_axis = moon_position/np.linalg.norm(moon_position)
#     second_axis = second_axis/np.linalg.norm(second_axis)
#     third_axis = rotation_axis/np.linalg.norm(rotation_axis)

#     # print("Rotation Axis Vector:", rotation_axis_unit)
#     # print("Magnitude of Rotation Rate:", rotation_rate_magnitude)

#     # print("First axis:", moon_position/np.linalg.norm(moon_position))
#     # print("Second axis:", second_axis/np.linalg.norm(second_axis))
#     # print("Third axis:", rotation_axis/np.linalg.norm(rotation_axis))


#     # Define the rotation matrix (DCM) using the rotating frame axes
#     rotating_frame_axes = np.array([first_axis, second_axis, third_axis])


#     satellite_position = satellite_data[:, 2:5]
#     rotating_state = np.dot(rotating_frame_axes, moon_position)

#     # rotating_states_dict.update({epoch: rotating_state})

#     # Non dimensionalize
#     rotating_state = rotating_state/np.linalg.norm(rotating_state)

#     # Shift to bary-centric frame
#     mu = 7.34767309e22/(7.34767309e22 + 5.972e24)
#     rotating_state = (1-mu)*rotating_state

#     rotating_states_dict.update({epoch: rotating_state})


#     # Print the result
#     print("State in rotating frame:", rotating_state)


# rotating_states = np.stack(list(rotating_states_dict.values()))



# fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharex=True)
# ax[0].scatter(rotating_states[:, 0], rotating_states[:, 1], s=10)
# ax[1].scatter(rotating_states[:, 1], rotating_states[:, 2], s=10)
# ax[2].scatter(rotating_states[:, 2], rotating_states[:, 0], s=10)
# ax[0].set_xlim([0, 1.3])
# ax[0].set_ylim([-0.1, 0.1])
# plt.show()






file_path = os.path.realpath(__file__)
file_path = os.path.dirname(file_path)
cut_value = 60390+1
step_size = 0.01

state_histories = {}
for body in ["LPF", "Moon"]:

    original_file = "LUMIO_Halo_Cj3p09_states_J2000_Earth_centered.txt"
    if body == "Moon":
        original_file = f"{body}_states_J2000_Earth_centered.txt"

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


duration = 1
transformation_matrix_dict = create_transformation_matrix(duration=duration)

print(transformation_matrix_dict)

rotating_states_dict = {}
for epoch, state in satellite_data_dict.items():
    # print(epoch, state)

    transformation_matrix = transformation_matrix_dict[epoch]

    rotating_state = np.dot(transformation_matrix, state[:3])

    # Non dimensionalize
    rotating_state = rotating_state/np.linalg.norm((moon_data_dict[epoch][:3]))

    # Shift to bary-centric frame
    mu = 7.34767309e22/(7.34767309e22 + 5.972e24)
    rotating_state = (1-mu)*rotating_state

    rotating_states_dict.update({epoch: rotating_state})

    # Print the result
    # print("State in rotating frame:", rotating_state)



rotating_moon_states_dict = {}
for epoch, state in moon_data_dict.items():
    # print(epoch, state)

    transformation_matrix = transformation_matrix_dict[epoch]
    # print("tasdfsadf", transformation_matrix)
    # rotating_states_dict.update({epoch: rotating_state})

    rotating_state = np.dot(transformation_matrix, state[:3])

    # Non dimensionalize
    rotating_state = rotating_state/np.linalg.norm(state[:3])

    # Shift to bary-centric frame
    mu = 7.34767309e22/(7.34767309e22 + 5.972e24)
    rotating_state = (1-mu)*rotating_state

    rotating_moon_states_dict.update({epoch: rotating_state})

    # Print the result
    # print("State in rotating frame:", rotating_state)


rotating_states = np.stack(list(rotating_states_dict.values()))

rotating_moon_states = np.stack(list(rotating_moon_states_dict.values()))

fig, ax = plt.subplots(1, 3, figsize=(12, 5))
ax[0].scatter(rotating_moon_states[:, 0], rotating_moon_states[:, 2], s=10)
ax[1].scatter(rotating_moon_states[:, 1], rotating_moon_states[:, 2], s=10)
ax[2].scatter(rotating_moon_states[:, 0], rotating_moon_states[:, 1], s=10)
ax[0].plot(rotating_states[:, 0], rotating_states[:, 2], lw=0.2)
ax[1].plot(rotating_states[:, 1], rotating_states[:, 2], lw=0.2)
ax[2].plot(rotating_states[:, 0], rotating_states[:, 1], lw=0.2)
# ax[0].set_xlim([0.95, 1.01])
# ax[0].set_ylim([-0.1, 0.1])

# fig1, ax1 = plt.subplots(1, 3, figsize=(12, 5))
# ax1[0].plot(moon_states[:, 0], moon_states[:, 2], lw=0.2)
# ax1[1].plot(moon_states[:, 1], moon_states[:, 2], lw=0.2)
# ax1[2].plot(moon_states[:, 0], moon_states[:, 1], lw=0.2)

# fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
# ax2.plot(moon_states[:, 0], lw=1)
# ax2.plot(moon_states[:, 1], lw=1)
# ax2.plot(moon_states[:, 2], lw=1)

fig1_3d = plt.figure()
ax_3d = fig1_3d.add_subplot(111, projection='3d')
ax_3d.plot(rotating_states[:, 0], rotating_states[:, 1], rotating_states[:, 2], lw=0.2)
# ax_3d.set_xlim([0, 1.4])

# plt.axis('equal')
plt.show()









