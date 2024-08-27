# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Own
from tests import utils
import ReferenceData, Interpolator, FrameConverter
from src.dynamic_models.LF.CRTBP import *
from src.dynamic_models.HF.PM import *
from src.dynamic_models.HF.PMSRP import *
from src.dynamic_models.HF.SH import *
from src.dynamic_models.HF.SHSRP import *
from src.dynamic_models.FF.TRUTH import *


def plot_dynamic_model_comparison(mission_start_epoch, propagation_time):

    models = [
        PM01.HighFidelityDynamicModel(mission_start_epoch, propagation_time),
        PMSRP01.HighFidelityDynamicModel(mission_start_epoch, propagation_time),
        # SH01.HighFidelityDynamicModel(mission_start_epoch, propagation_time),
        # SHSRP01.HighFidelityDynamicModel(mission_start_epoch, propagation_time)
    ]
    print(models)

    dynamics_simulators = [
        model.get_propagation_simulator(solve_variational_equations=False) for model in models
    ]
    print(dynamics_simulators)

    state_history_dicts = [
        dynamics_simulator.state_history for dynamics_simulator in dynamics_simulators
    ]
    # print(state_histories)

    state_histories = [
        np.stack(list(state_history_dict.values())) for state_history_dict in state_history_dicts
    ]
    print(state_histories)

    frame_converters = [
        FrameConverter.FrameConverter(dynamics_simulator) for dynamics_simulator in dynamics_simulators
    ]
    print(frame_converters)


    # Obtain the initial state of the whole simulation once
    interpolator = Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.001)
    reference_data = ReferenceData.ReferenceData(interpolator)

    state_history_reference = reference_data.get_reference_state_history(
            mission_start_epoch, propagation_time, satellite="LUMIO", get_full_history=True)
    state_history_reference_dict_lpf= reference_data.get_reference_state_history(
        mission_start_epoch, propagation_time, satellite="LPF", get_dict=True, get_full_history=True)
    state_history_reference_dict_lumio= reference_data.get_reference_state_history(
        mission_start_epoch, propagation_time, satellite="LUMIO", get_dict=True, get_full_history=True)

    state_history_reference_dict = {}
    keys_lpf = list(sorted(state_history_reference_dict_lpf.keys()))
    keys_lumio = list(sorted(state_history_reference_dict_lumio.keys()))
    for index, key in enumerate(keys_lumio):
        state_history_reference_dict[key] = np.concatenate((state_history_reference_dict_lpf[keys_lpf[index]], state_history_reference_dict_lumio[keys_lumio[index]]))

    print(state_history_reference_dict, list(state_history_reference_dict.keys()))

    synodic_state_histories = [
        np.stack(list(frame_converter.get_synodic_state_history().values())) for frame_converter in frame_converters
    ]
    print(synodic_state_histories)

    synodic_state_histories_reference = [
        np.stack(list(frame_converter.get_synodic_state_history(state_history_reference_dict).values())) for frame_converter in frame_converters
    ]
    # print(synodic_state_histories_reference)

    # Starting the plotting work
    fig1_3d = plt.figure()
    ax_3d = fig1_3d.add_subplot(111, projection='3d')

    fig, ax = plt.subplots(2, 3, figsize=(11, 6))

    model_names = ["PM", "PMSRP", "SH", "SHSRP"]
    lss = ['dotted', 'dashed', 'loosely dotted','loosely dashed']
    colors = ["blue", "green", "orange", "yellow"]
    for model_index, model in enumerate(models):

        state_history = synodic_state_histories[model_index]
        for i in range(2):

            ls = lss[model_index]
            ls="solid"
            if i == 0:
                color = "red"
                label = "LPF" if model_index==0 else None

            else:
                # color = "blue"
                # if model_index==1:
                color = colors[model_index]
                label = f"LUMIO, {model_names[model_index]}"

            start_label = "Start" if model_index==0 else None
            ax[i][0].scatter(state_history[0, 6*i+0], state_history[0, 6*i+2], s=30, marker="X", color="black")
            ax[i][1].scatter(state_history[0, 6*i+1], state_history[0, 6*i+2], s=30, marker="X", color="black")
            ax[i][2].scatter(state_history[0, 6*i+0], state_history[0, 6*i+1], s=30, marker="X", color="black", label=start_label)

            satellite_label = model_names[model_index] if model_index == 0 else None
            ax_3d.plot(state_history[:, (6*i+0)], state_history[:, (6*i+1)], state_history[:, (6*i+2)],
                alpha=0,
                color=color,
                label=satellite_label,
                ls=ls)

            model_label = model_names[model_index] if model_index == 0 else None
            ax_3d.plot(state_history[:, (6*i+0)], state_history[:, (6*i+1)], state_history[:, (6*i+2)],
                color=color,
                label=model_label,
                ls=ls)


            # ax[i][0].scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 2], s=50, color="darkgray")
            # ax[i][1].scatter(synodic_moon_states[:, 1], synodic_moon_states[:, 2], s=50, color="darkgray")
            # ax[i][2].scatter(synodic_moon_states[:, 0], synodic_moon_states[:, 1], s=50, color="darkgray", label="Moon" if type_index==0 and i==0 else None)
            ax[i][0].plot(state_history[:, 6*i+0], state_history[:, 6*i+2], lw=0.5, color=color, ls=ls)
            ax[i][1].plot(state_history[:, 6*i+1], state_history[:, 6*i+2], lw=0.5, color=color, ls=ls)
            ax[i][2].plot(state_history[:, 6*i+0], state_history[:, 6*i+1], lw=0.5, color=color, label=label, ls=ls)
            ax[1][0].plot(state_history[:, 6*i+0], state_history[:, 6*i+2], lw=0.1, color=color, ls=ls)
            ax[1][1].plot(state_history[:, 6*i+1], state_history[:, 6*i+2], lw=0.1, color=color, ls=ls)
            ax[1][2].plot(state_history[:, 6*i+0], state_history[:, 6*i+1], lw=0.1, color=color, ls=ls)

            ax_3d.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2], lw=0.2, color="gray")
            ax_3d.plot(state_history[:, 6], state_history[:, 7], state_history[:, 8], lw=0.7, color="black")



            # if model_index==0 and i == 1:
            #     ax[i][0].plot(synodic_state_histories_reference[-1][0, 6*i+0], synodic_state_histories_reference[-1][0, 6*i+2], color="green")
            #     ax[i][1].plot(synodic_state_histories_reference[-1][0, 6*i+1], synodic_state_histories_reference[-1][0, 6*i+2], color="green")
            #     ax[i][2].plot(synodic_state_histories_reference[-1][0, 6*i+0], synodic_state_histories_reference[-1][0, 6*i+1], color="green", label="Reference")

                # ax[i].plot(synodic_state_histories_reference[-1][:, 6], synodic_state_histories_reference[-1][:, 7], synodic_state_histories_reference[-1][:, 8],
                #     color="green",
                #     label="Reference LUMIO")

    ax_3d.plot(synodic_state_histories_reference[0][:, 6], synodic_state_histories_reference[0][:, 7], synodic_state_histories_reference[0][:, 8],
        color="green",
        label="Reference LUMIO")

    # ax_3d.plot(state_history_reference[:, 0], state_history_reference[:, 1], state_history_reference[:, 2],
    #     color="green",
    #     label="Reference LUMIO")

    axes_labels = ['X [-]', 'Y [-]', 'Z [-]']
    for i in range(2):
        for j in range(3):
            ax[i][j].grid(alpha=0.3)
            ax[i][0].set_xlabel(axes_labels[0])
            ax[i][0].set_ylabel(axes_labels[2])
            ax[i][1].set_xlabel(axes_labels[1])
            ax[i][1].set_ylabel(axes_labels[2])
            ax[i][2].set_xlabel(axes_labels[0])
            ax[i][2].set_ylabel(axes_labels[1])


    ax[0][2].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")
    ax[1][2].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

    ax_3d.set_aspect('equal')
    plt.tight_layout()
    plt.show()



plot_dynamic_model_comparison(60390, 30)