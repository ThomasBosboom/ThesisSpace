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
import reference_data, Interpolator, FrameConverter
from src.dynamic_models.LF.CRTBP import *
from src.dynamic_models.HF.PM import *
from src.dynamic_models.HF.PMSRP import *
from src.dynamic_models.HF.SH import *
from src.dynamic_models.HF.SHSRP import *
from src.dynamic_models.FF.TRUTH import *
from src import EstimationModel


###############################
#### Acceleration terms #######
###############################

def acceleration_norms():

    # Use the FF dynamic model to generate most sophisticated acceleration terms
    model_type = "FF"
    model_name = "TRUTH"
    model_number = 0

    # Use point_mass model to generate the point_mass terms of the Earth and Moon also
    model_type_PM = "HF"
    model_name_PM = "PM"
    model_number_PM = 7

    # Defining dynamic model setup
    simulation_start_epoch_MJD = 60390
    propagation_time = 14
    custom_model_dict = {model_type_PM: [model_name_PM], model_type: [model_name]}
    step_size = 0.01
    epoch_in_MJD = True
    solve_variational_equations = False
    specific_model_list = [model_number_PM, model_number]
    return_dynamic_model_objects = True

    # Initialize dictionaries to store accumulated values
    dynamic_model_objects_results = utils.get_dynamic_model_results(simulation_start_epoch_MJD,
                                                                    propagation_time,
                                                                    custom_model_dict=custom_model_dict,
                                                                    step_size=step_size,
                                                                    epoch_in_MJD=epoch_in_MJD,
                                                                    solve_variational_equations=solve_variational_equations,
                                                                    specific_model_list=specific_model_list,
                                                                    return_dynamic_model_objects=return_dynamic_model_objects)

    epochs, state_history, dependent_variable_history, run_time, dynamic_model_object = dynamic_model_objects_results[model_type][model_name][model_number]
    epochs_PM, state_history_PM, dependent_variable_history_PM, run_time_PM, dynamic_model_object_PM = dynamic_model_objects_results[model_type_PM][model_name_PM][model_number_PM]

    bodies_to_create = dynamic_model_object.bodies_to_create
    bodies_to_create_PM = dynamic_model_object_PM.bodies_to_create
    new_bodies_to_create = dynamic_model_object.new_bodies_to_create

    epochs = epochs - epochs[0]

    satellites = ["LPF", "LUMIO"]
    subtitles = [ "Net acceleration", "Point mass", "Spherical harmonics", "Radiation pressure", "Relativistic corrections"]
    ylabels = [r"$||\mathbf{a}_{net}||$  $[m/s^{2}]$", r"$||\mathbf{a}_{PM}||$  $[m/s^{2}]$", r"$||\mathbf{a}_{SH}||$  $[m/s^{2}]$", r"$||\mathbf{a}_{SRP}||$  $[m/s^{2}]$", r"$||\mathbf{a}_{RC}||$  $[m/s^{2}]$"]
    xlabel = f"Time since MJD {simulation_start_epoch_MJD} [days]"
    plot_labels = [None,
                    bodies_to_create_PM,
                    ["Earth J2,0", "Earth J2,2", "Moon J2,0", "Moon J2,2"],
                    ["Earth", "Moon", "Sun"],
                    bodies_to_create]

    for l in range(2):

        fig, ax = plt.subplots(5, 1, figsize=(8, 9), sharex=True)
        fig.suptitle(f'Absolute accelerations acting on {satellites[l]}', fontsize=14)

        n_net = 1
        n_point_mass = 10
        n_spherical_harmonics = 4
        n_radiation_pressure = 3
        n_relativistic_correction = 10

        data_to_plot = [dependent_variable_history[:,12+n_net*l:12+n_net*(l+1)],
                        dependent_variable_history_PM[:,14+n_point_mass*l:14+n_point_mass*(l+1)],
                        dependent_variable_history[:,30+n_spherical_harmonics*l:30+n_spherical_harmonics*(l+1)],
                        dependent_variable_history[:,38+n_radiation_pressure*l:38+n_radiation_pressure*(l+1)],
                        dependent_variable_history[:,44+n_relativistic_correction*l:44+n_relativistic_correction*(l+1)]]

        # Small plots
        for i in range(5):
            if i == 0:
                ax[i].plot(epochs, data_to_plot[i])
            else:
                ax[i].plot(epochs, data_to_plot[i], label=[label for label in plot_labels[i]])
                ax[i].legend(loc="lower right", ncol=2, fontsize="x-small")
            ax[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax[i].set_title(subtitles[i], fontsize=10)
            if i == 4:
                ax[i].set_xlabel(xlabel, fontsize=8)
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_yscale("log")
            ax[i].grid(alpha=0.5, linestyle='--')

        plt.tight_layout()

        utils.save_figure_to_folder(figs=[fig], labels=[satellites[l]])

    # plt.show()


acceleration_norms()