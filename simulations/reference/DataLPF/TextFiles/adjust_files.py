import numpy as np
import sys
import os
from scipy.interpolate import interp1d



file_path = os.path.realpath(__file__)
file_path = os.path.dirname(file_path)
cut_value = 60390+110
step_size = 0.01

for body in ["LPF", "Moon"]:

    original_file = f"365days\{body}_states_J2000_Earth_centered.txt"
    new_file = f"{body}_states_J2000_Earth_centered.txt"

    # Load the data from the provided text file
    with open(os.path.join(file_path, original_file), "r") as file:
        lines = file.readlines()

    # Load the data excluding the title row
    data = np.loadtxt(lines[1:], delimiter=",")

    # data = np.loadtxt(os.path.join(file_path, original_file), delimiter=",")
    filtered_data = data[data[:, 0] <= cut_value]

    epoch_mjd = filtered_data[:, 0]

    # Define the range of epochs from the minimum to maximum with a step of 0.01
    new_epoch_mjd = np.arange(epoch_mjd.min(), epoch_mjd.max(), step_size)

    f = interp1d(epoch_mjd, filtered_data, axis=0, kind='cubic')
    interpolated_data = f(new_epoch_mjd)

    # Combine the interpolated epoch column with other interpolated columns
    interpolated_data_with_epoch = np.column_stack((new_epoch_mjd, interpolated_data[:,1:]))

    np.savetxt(os.path.join(file_path, new_file), interpolated_data_with_epoch, delimiter=",", fmt='%.6f', header=lines[0].strip())