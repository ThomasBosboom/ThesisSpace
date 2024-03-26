# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(parent_dir)

# Third party
import pytest
import pytest_html
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Own
from tests import utils
from src.dynamic_models import validation
from src.dynamic_models import Interpolator, FrameConverter, TraditionalLowFidelity
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model




def comparison_high_fidelity(simulation_start_epoch_MJD, propagation_time, durations, step_size=0.001):


    custom_model_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"], "full_fidelity": ["full_fidelity"]}
    custom_model_dict = {"high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]}

    dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch_MJD,
                                                            propagation_time,
                                                            custom_model_dict=custom_model_dict,
                                                            get_only_first=True)

    fig1_3d = plt.figure()
    ax_3d = fig1_3d.add_subplot(111, projection='3d')
    fig, ax = plt.subplots(len(durations), 3, figsize=(14, 3*(len(durations))), layout="constrained")
    for i, (model_types, model_names) in enumerate(dynamic_model_objects.items()):
        for j, (model_name, dynamic_models) in enumerate(model_names.items()):
            for dynamic_model in dynamic_models:
                print(dynamic_model)

                # Extract simulation histories tudatpy solution
                epochs, state_history, dependent_variables_history = \
                    Interpolator.Interpolator(step_size=step_size).get_propagation_results(dynamic_model,
                                                                                           solve_variational_equations=False)

                # Convert back to synodic
                epochs_synodic, state_history_synodic = \
                    FrameConverter.InertialToSynodicHistoryConverter(dynamic_model, step_size=step_size).get_results(state_history)

                satellites = ["LPF", "LUMIO"]
                plot_labels = [['X [m]', 'Y [m]', 'Z [m]'], ['X [-]', 'Y [-]', 'Z [-]']]
                legend_labels = [["PM", "PMSRP", "SH", "SHSRP"]]
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                colors = [["darkred", "red", "indianred", "lightsalmon"], ["navy", "blue", "lightskyblue", "lightcyan"]]
                for l in range(len(durations)):
                    n = int(durations[l]/(step_size))
                    for m in range(3):
                        ax[l][0].title.set_text(str(durations[l])+ " days")
                        for k in range(len(satellites)):
                            ax[l][m].plot(state_history_synodic[0,6*k+m%3], state_history_synodic[0,6*k+(m+1)%3], marker="o", color=colors[k][j], label=None)
                            ax[l][m].plot(state_history_synodic[:n,6*k+m%3], state_history_synodic[:n,6*k+(m+1)%3], color=colors[k][j], label=legend_labels[i][j] if (k==0 and l == 0 and m == 2) else None)
                        ax[l][m].set_xlabel(plot_labels[1][m%3])
                        ax[l][m].set_ylabel(plot_labels[1][(m+1)%3])
                        ax[l][m].grid(alpha=0.5, linestyle='--')
                        ax[0][-1].legend(loc='upper right')

                ax_3d.plot(state_history_synodic[:,0], state_history_synodic[:,1], state_history_synodic[:,2], color=colors[0][j], label=legend_labels[i][j])
                ax_3d.plot(state_history_synodic[:,6], state_history_synodic[:,7], state_history_synodic[:,8], color=colors[1][j], label=legend_labels[i][j])


    utils.save_figure_to_folder([fig], [])

    # plt.show()

comparison_high_fidelity(60390, 28, [7, 14, 21, 28])
comparison_high_fidelity(60390, 7, [4, 6])