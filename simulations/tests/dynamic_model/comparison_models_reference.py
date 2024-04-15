# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Third party
import pytest
import pytest_html
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Own
from tests import utils
import reference_data, Interpolator, FrameConverter
from src.dynamic_models.LF.CRTBP import *
from src.dynamic_models.HF.PM import *
from src.dynamic_models.HF.PMSRP import *
from src.dynamic_models.HF.SH import *
from src.dynamic_models.HF.SHSRP import *
from src.estimation_models import EstimationModel

# Load spice kernels.
from tudatpy.kernel.interface import spice
spice.load_standard_kernels()


def comparison_models_reference(simulation_start_epoch_MJD, propagation_time, step_size=0.001):


    custom_model_dict = {"LF": ["CRTBP"], "HF": ["PM", "PMSRP", "SH", "SHSRP"], "FF": ["FF"]}
    custom_model_dict = {"HF": ["PMSRP"]}

    dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD,
                                                            propagation_time,
                                                            custom_model_dict=custom_model_dict,
                                                            get_only_first=False)

    fig1_3d = plt.figure()
    ax_3d = fig1_3d.add_subplot(111, projection='3d')
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), layout="constrained")
    for i, (model_types, model_names) in enumerate(dynamic_model_objects.items()):
        for j, (model_name, dynamic_models) in enumerate(model_names.items()):
            for k, dynamic_model in enumerate(dynamic_models):

                if k in range(1):
                    # print(dynamic_model)

                    gravitational_parameter_primary = spice.get_body_gravitational_parameter("Earth")
                    gravitational_parameter_secondary = spice.get_body_gravitational_parameter("Moon")

                    for gravitational_parameter_primary in np.linspace(gravitational_parameter_primary*(1-0.07), gravitational_parameter_primary*(1+0.07), 10):

                        # print(area)

                        for gravitational_parameter_secondary in np.linspace(gravitational_parameter_secondary*(1-0.07), gravitational_parameter_secondary*(1+0.07), 10):

                            # print(pressure_coefficient)

                            # for mass in np.linspace(18, 24, 6):

                            # dynamic_model.bodies_reference_area_radiation = [3.0, area]
                            # dynamic_model.bodies_radiation_pressure_coefficient = [1.8, pressure_coefficient]
                            # dynamic_model.bodies_mass = [280, mass]
                            dynamic_model.gravitational_parameter_primary = gravitational_parameter_primary
                            dynamic_model.gravitational_parameter_secondary = gravitational_parameter_secondary

                            # spice.get_body_gravitational_parameter("Earth")

                            # 0.5 0.9 20.4


                            # self.bodies_mass = [280, 22.3]
                            # self.bodies_reference_area_radiation = [3.0, 0.41064]
                            # self.bodies_radiation_pressure_coefficient = [1.8, 1.08]
                            # self.gravitational_parameter_primary = spice.get_body_gravitational_parameter(self.name_primary)
                            # self.gravitational_parameter_secondary = spice.get_body_gravitational_parameter(self.name_secondary)

                            # Extract simulation histories tudatpy solution
                            epochs, state_history, dependent_variables_history = \
                                Interpolator.Interpolator(epoch_in_MJD=True, step_size=step_size).get_propagation_results(dynamic_model,
                                                                                                    solve_variational_equations=False)


                            # Get the reference orbit states
                            reference_state_history = list()
                            for body in dynamic_model.bodies_to_propagate:
                                reference_state_history.append(reference_data.get_reference_state_history(simulation_start_epoch_MJD,
                                                                                                        propagation_time,
                                                                                                        satellite=body,
                                                                                                        step_size=step_size,
                                                                                                        get_full_history=True))

                            reference_state_history = np.concatenate(reference_state_history, axis=1)

                            # Convert back to synodic
                            epochs_synodic, state_history_synodic = \
                                FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=step_size).get_results(state_history)

                            # Plot comparison figures
                            reference_deviation_history = state_history - reference_state_history
                            # axs[j].plot(epochs-epochs[0], np.linalg.norm(reference_deviation_history[:, 6:9], axis=1), label=str(area))
                            # axs[j].set_yscale("log")
                            # axs[j].set_title(model_name)
                            # axs[-1].set_xlabel(f"Days since {simulation_start_epoch_MJD}")
                            # axs[j].set_ylabel(r"||$\mathbf{r}-\mathbf{r}_{ref}$||")
                            print(gravitational_parameter_primary, gravitational_parameter_secondary, max(np.linalg.norm(reference_deviation_history[:, 6:9], axis=1)))
                            ax_3d.scatter(gravitational_parameter_primary, gravitational_parameter_secondary, max(np.linalg.norm(reference_deviation_history[:, 6:9], axis=1)))


    axs[0].legend()

    utils.save_figure_to_folder([fig], [])

    plt.show()

# comparison_models_reference(60390, 5, step_size=0.001)


import FrameConverter2

def continuities_reference(simulation_start_epoch_MJD, propagation_time, step_size=0.001):

    # Get the reference orbit states
    reference_state_history = list()
    for body in ["LPF", "LUMIO"]:
        reference_state_history.append(reference_data.get_reference_state_history(simulation_start_epoch_MJD,
                                                                                propagation_time,
                                                                                satellite=body,
                                                                                step_size=step_size,
                                                                                get_full_history=True,
                                                                                interpolation_kind="cubic",
                                                                                get_epoch_in_array=False,
                                                                                get_dict=False))

    reference_state_history = np.concatenate(reference_state_history, axis=1)

    reference_state_history_moon = list()
    reference_state_history_moon.append(reference_data.get_reference_state_history(simulation_start_epoch_MJD,
                                                                                propagation_time,
                                                                                body="moon",
                                                                                step_size=step_size,
                                                                                get_full_history=True,
                                                                                interpolation_kind="cubic",
                                                                                get_epoch_in_array=True,
                                                                                get_dict=False))

    reference_state_history_moon = np.concatenate(reference_state_history_moon, axis=1)

    reference_state_history = reference_state_history
    epochs = reference_state_history_moon[:, 0]
    reference_state_history_moon = reference_state_history_moon[:, 1:]

    # Get example model
    dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD,
                                                            propagation_time,
                                                            custom_model_dict={"HF": ["PM"]},
                                                            get_only_first=True)

    for i, (model_types, model_names) in enumerate(dynamic_model_objects.items()):
        for j, (model_name, dynamic_models) in enumerate(model_names.items()):
            for k, dynamic_model in enumerate(dynamic_models):

                epochs, state_history, dependent_variables_history = \
                    Interpolator.Interpolator(epoch_in_MJD=True, step_size=step_size).get_propagation_results(dynamic_model,
                                                                                        solve_variational_equations=False)


                frame_converter = FrameConverter2.FrameConverter(state_history, dependent_variables_history[:, :6], epochs)
                synodic_state_history, synodic_state_history_moon = frame_converter.InertialToSynodicHistoryConverter()



    frame_converter = FrameConverter2.FrameConverter(reference_state_history, reference_state_history_moon, epochs)
    synodic_state_history_ref, synodic_state_history_ref_moon = frame_converter.InertialToSynodicHistoryConverter()

    # Get difference between epochs
    diff_reference_state_history = np.diff(reference_state_history, 3, axis=0)

    fig, ax = plt.subplots(2, 3, figsize=(12, 5))

    ax[1][0].plot(synodic_state_history_ref[:, 0], synodic_state_history_ref[:, 2], color="red", lw=0.1)
    ax[1][1].plot(synodic_state_history_ref[:, 1], synodic_state_history_ref[:, 2], color="red", lw=0.1)
    ax[1][2].plot(synodic_state_history_ref[:, 0], synodic_state_history_ref[:, 1], color="red", lw=0.1)

    ax[1][0].plot(synodic_state_history[:, 6], synodic_state_history[:, 8], color="darkblue", lw=0.3)
    ax[1][1].plot(synodic_state_history[:, 7], synodic_state_history[:, 8], color="darkblue", lw=0.3)
    ax[1][2].plot(synodic_state_history[:, 6], synodic_state_history[:, 7], color="darkblue", lw=0.3, label="LUMIO\nuncorrected")

    for i in range(2):

        axes_labels = ['X [-]', 'Y [-]', 'Z [-]']
        for j in range(3):
            ax[i][j].grid(alpha=0.3)
            ax[i][0].set_xlabel(axes_labels[0])
            ax[i][0].set_ylabel(axes_labels[2])
            ax[i][1].set_xlabel(axes_labels[1])
            ax[i][1].set_ylabel(axes_labels[2])
            ax[i][2].set_xlabel(axes_labels[0])
            ax[i][2].set_ylabel(axes_labels[1])

        lw = 0.3
        if i == 0:
            label="LPF"
            color="red"
        else:
            label="LUMIO"
            color="blue"

        ax[i][0].plot(synodic_state_history_ref[:, 6*i+0], synodic_state_history_ref[:, 6*i+2], color=color, lw=lw)
        ax[i][1].plot(synodic_state_history_ref[:, 6*i+1], synodic_state_history_ref[:, 6*i+2], color=color, lw=lw)
        ax[i][2].plot(synodic_state_history_ref[:, 6*i+0], synodic_state_history_ref[:, 6*i+1], color=color, lw=lw, label=label)
        ax[i][0].scatter(synodic_state_history_ref[0, 6*i+0], synodic_state_history_ref[0, 6*i+2], color=color, lw=lw, s=80, marker="X")
        ax[i][1].scatter(synodic_state_history_ref[0, 6*i+1], synodic_state_history_ref[0, 6*i+2], color=color, lw=lw, s=80, marker="X")
        ax[i][2].scatter(synodic_state_history_ref[0, 6*i+0], synodic_state_history_ref[0, 6*i+1], color=color, lw=lw, s=80, marker="X", label="Start")
        ax[i][0].scatter(synodic_state_history_ref_moon[:, 0], synodic_state_history_ref_moon[:, 2], s=50, color="darkgray")
        ax[i][1].scatter(synodic_state_history_ref_moon[:, 1], synodic_state_history_ref_moon[:, 2], s=50, color="darkgray")
        ax[i][2].scatter(synodic_state_history_ref_moon[:, 0], synodic_state_history_ref_moon[:, 1], s=50, color="darkgray", label="Moon")

        ax[i][2].legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize="small")

    fig.suptitle("Reference and uncorrected orbit history, synodic frame")
    plt.tight_layout()

    utils.save_figure_to_folder([fig], [])

    plt.show()



continuities_reference(60390, 30, step_size=0.01)


