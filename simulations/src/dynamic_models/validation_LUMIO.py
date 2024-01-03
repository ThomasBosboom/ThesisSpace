import os
import sys
import numpy as np
from tudatpy.kernel import constants
from scipy.interpolate import CubicSpline, interp1d
from tudatpy.kernel.astro import time_conversion
from pathlib import Path
import TraditionalLowFidelity


root_dir = Path(__file__).resolve().parent.parent.parent
reference_folder_path = root_dir / "reference"


def read_textfiles(data_type, satellite="LUMIO"):

    if satellite == "LUMIO":

        folder_path = reference_folder_path / "DataLUMIO" / "TextFiles"
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

        if data_type == "state":

            state_fixed_LUMIO_Earth_centered = np.loadtxt(fname=file_paths[1], delimiter=',', skiprows=1, usecols=tuple(range(0,8,1)), unpack=False)
            state_fixed_Moon_Earth_centered  = np.loadtxt(fname=file_paths[3], delimiter=',', skiprows=1, usecols=tuple(range(0,8,1)), unpack=False)
            state_fixed_Sun_Earth_centered   = np.loadtxt(fname=file_paths[5], delimiter=',', skiprows=1, usecols=tuple(range(0,8,1)), unpack=False)

            return np.stack([state_fixed_LUMIO_Earth_centered, state_fixed_Moon_Earth_centered, state_fixed_Sun_Earth_centered])

        if data_type == "attitude":

            attitude_fixed_Earth_LUMIO_centered = np.loadtxt(fname=file_paths[0], delimiter=',', skiprows=1, usecols=tuple(range(0,5,1)), unpack=False)
            attitude_fixed_Moon_LUMIO_centered  = np.loadtxt(fname=file_paths[2], delimiter=',', skiprows=1, usecols=tuple(range(0,5,1)), unpack=False)
            attitude_fixed_Sun_LUMIO_centered   = np.loadtxt(fname=file_paths[4], delimiter=',', skiprows=1, usecols=tuple(range(0,5,1)), unpack=False)

            return np.stack([state_fixed_LUMIO_Earth_centered, state_fixed_Moon_Earth_centered, state_fixed_Sun_Earth_centered])


    if satellite == "LPF":

        folder_path = reference_folder_path / "DataLPF" / "TextFiles"
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

        if data_type == "state":

            state_fixed_LPF_Earth_centered   = np.loadtxt(fname=file_paths[0], delimiter=',', skiprows=1, usecols=tuple(range(0,8,1)), unpack=False)
            state_fixed_Moon_Earth_centered  = np.loadtxt(fname=file_paths[1], delimiter=',', skiprows=1, usecols=tuple(range(0,8,1)), unpack=False)

            return np.stack([state_fixed_LPF_Earth_centered, state_fixed_Moon_Earth_centered])




def get_reference_state_history(simulation_start_epoch_MJD, propagation_time,  step_size=0.001, satellite="LUMIO",body="satellite", interpolation_kind='cubic', get_dict=False, get_epoch_in_array=False, get_full_history=False):

    state_history = read_textfiles("state", satellite=satellite)
    if body == "satellite":
        state_history = state_history[0]
    elif body == "moon":
        state_history = state_history[1]

    # Example: User-defined epoch for interpolation
    user_start_epoch = time_conversion.julian_day_to_seconds_since_epoch(\
        time_conversion.modified_julian_day_to_julian_day(simulation_start_epoch_MJD))+69.1826417446136475
    user_end_epoch = user_start_epoch + propagation_time*constants.JULIAN_DAY

    # Perform interpolation using SciPy's interp1d
    epochs = state_history[:, 1]
    state_vectors = state_history[:, 2:]*1000
    interp_func = interp1d(epochs, state_vectors, axis=0, kind=interpolation_kind, fill_value='extrapolate')
    interpolated_state = interp_func(user_start_epoch)

    interpolated_states = np.zeros((1,6))
    epochs = np.arange(user_start_epoch, user_end_epoch+step_size*constants.JULIAN_DAY, step_size*constants.JULIAN_DAY)
    for epoch in epochs:
        interpolated_states = np.vstack((interpolated_states, interp_func(epoch)))
    interpolated_states = np.delete(interpolated_states, 0, 0)

    # Create a dictionary with epochs as keys and vectors as values
    data_dict = {epoch: vector for epoch, vector in zip(epochs, interpolated_states) \
                    if epoch <= user_end_epoch \
                        and epoch >= user_start_epoch}

    if get_dict == False:
        if get_full_history == True:
            if get_epoch_in_array == True:
                return  np.concatenate((np.vstack(list(data_dict.keys())), np.vstack(list(data_dict.values()))), axis=1)
            return np.vstack(list(data_dict.values()))
        return interpolated_states[0]

    else:
        if get_full_history == True:
            return data_dict
        return {user_start_epoch: interpolated_states[0]}



# # print(get_reference_state_history(60390.00, 5, satellite="LPF", get_dict=False, get_full_history=True))
# # print(get_reference_state_history(60390.00, 5, satellite="LUMIO", get_dict=False, get_full_history=False))
# states = get_reference_state_history(60390.00, 28, satellite="LUMIO", body="moon", get_dict=False, get_full_history=True)
# import matplotlib.pyplot as plt
# ax = plt.figure().add_subplot(projection='3d')
# plt.plot(states[:,0], states[:,1], states[:,2])
# # plt.plot(initial_states[:,6], initial_states[:,7], initial_states[:,8])
# # plt.plot(state_history_moon[:,1], state_history_moon[:,2], state_history_moon[:,3])
# plt.show()


def get_state_history_richardson(dc_corrected=False):

    # Specify the file path
    orbit_files = reference_folder_path / "Halo_orbit_files"

    if dc_corrected == False:
        file_path = orbit_files / "Richardson.txt"
    if dc_corrected == True:
        file_path = orbit_files / "Richardson_dc.txt"

    # Open the file for reading
    states_richardson = np.empty((1, 7))
    with open(file_path, 'r') as file:
        for line in file:
            states_richardson = np.vstack((states_richardson, np.array([float(state) for state in line.strip().split('\t')])))

    return np.delete(states_richardson, 0, 0)


# def get_state_history_erdem():

#     # Specify the file path
#     root_dir = Path(__file__).resolve().parent.parent.parent
#     file_path = root_dir / "Reference" / "Halo_orbit_files" / "Erdem_old.txt"

#     # Open the file for reading
#     states_erdem = np.empty((1, 13))
#     with open(file_path, 'r') as file:
#         for line in file:
#             states_erdem = np.vstack((states_erdem, np.array([float(state) for state in line.split()])))

#     return np.delete(states_erdem[:,:],0,0)

def get_synodic_state_history_erdem():

    # Specify the file path
    root_dir = Path(__file__).resolve().parent.parent.parent
    file_path = root_dir / "reference" / "Halo_orbit_files" / "Erdem.txt"

    # Open the file for reading
    states_erdem = np.empty((1, 13))
    with open(file_path, 'r') as file:
        for line in file:
            states_erdem = np.vstack((states_erdem, np.array([float(state) for state in line.split()])))

    return np.delete(states_erdem[:,:],0,0)


custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
                                 1.147342501,      -0.0002324517381,  -0.151368318,       -0.000202046355,    -0.2199137166,      0.0002817105509])
def get_synodic_state_history(G, m1, m2, a, propagation_time, step_size, custom_initial_state=custom_initial_state):

    dynamic_model_classic = TraditionalLowFidelity.TraditionalLowFidelity(G, m1, m2, a)
    epoch_history, state_rotating_bary_lpf = dynamic_model_classic.get_state_history(custom_initial_state[:6], 0, propagation_time, step_size)
    epoch_history, state_rotating_bary_lumio = dynamic_model_classic.get_state_history(custom_initial_state[6:], 0, propagation_time, step_size)
    synodic_state_history = np.concatenate((state_rotating_bary_lpf, state_rotating_bary_lumio), axis=1)

    return epoch_history, synodic_state_history


# import matplotlib.pyplot as plt
# ax = plt.figure().add_subplot(projection='3d')
# plt.plot(get_synodic_state_history_erdem()[:,1], get_synodic_state_history_erdem()[:,2], get_synodic_state_history_erdem()[:,3])
# plt.plot(get_synodic_state_history_erdem()[:1400,7], get_synodic_state_history_erdem()[:1400,8], get_synodic_state_history_erdem()[:1400,9])
# plt.plot(get_state_history_richardson()[:,1], get_state_history_richardson()[:,2], get_state_history_richardson()[:,3])
# plt.legend()
# plt.show()
