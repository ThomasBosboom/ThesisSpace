# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

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


###############################
#### Acceleration terms #######
###############################

def test_plot_acceleration_norms():

    # Use the full_fidelity dynamic model to generate most sophisticated acceleration terms
    model_type = "full_fidelity"
    model_name = "full_fidelity"
    model_number = 0

    # Use point_mass model to generate the point_mass terms of the Earth and Moon also
    model_type_PM = "high_fidelity"
    model_name_PM = "point_mass"
    model_number_PM = 7

    # Defining dynamic model setup
    simulation_start_epoch_MJD = 60390
    propagation_time = 7
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

        utils.save_figures_to_folder(figs=[fig], labels=[satellites[l]])

    plt.show()



#################################
#### Dynamic model run times ####
#################################

def test_dynamic_model_run_times(self):

    custom_model_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"], "full_fidelity": ["full_fidelity"]}
    # custom_model_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}
    propagation_time = 1
    get_only_first = False
    start_epoch = 60390
    end_epoch = 60404
    n = 5
    run_cases = np.linspace(start_epoch, end_epoch, n)

    # Initialize dictionaries to store accumulated values
    accumulator_dict = {
        fidelity_key: {subkey: [[] for value in values]
            for subkey, values in sub_dict.items()
        }
        for fidelity_key, sub_dict in utils.get_dynamic_model_objects(start_epoch, propagation_time, custom_model_dict=custom_model_dict, get_only_first=get_only_first).items()
    }

    start_time = time.time()
    for run_case in run_cases:
        params = (run_case, propagation_time, custom_model_dict, get_only_first)
        run_times_dict = utils.get_dynamic_model_results(*params, step_size=0.1, entry_list=[-1])

        # Accumulate values during the loop
        for fidelity_key, sub_dict in run_times_dict.items():
            for i, (subkey, subvalue_list) in enumerate(sub_dict.items()):
                for j, subvalue in enumerate(subvalue_list):
                    for k, entry in enumerate(subvalue):
                        accumulator_dict[fidelity_key][subkey][j].append(entry)

    total_run_time = time.time()-start_time

    # Calculate averages and standard deviations
    result_dict = {
        fidelity_key: {
            subkey: [
                {
                    "average": statistics.mean(sublist),
                    "std_dev": statistics.stdev(sublist)
                }
                for sublist in sublists
            ]
            for subkey, sublists in sub_dict.items()
        }
        for fidelity_key, sub_dict in accumulator_dict.items()
    }

    ### Plot run times for each model
    keys_list = [["CRTBP"], ["PM", "PMSRP", "SH", "SHSRP"], ["FF"]]
    key_count = sum(len(sublist) for sublist in custom_model_dict.values()) #0.75*key_count
    fig, axs = plt.subplots(1, key_count, figsize=(6.4, 0.75*5), sharey=True)
    index = 0
    for i, (model_types, model_names) in enumerate(result_dict.items()):
        for j, (key, values) in enumerate(model_names.items()):
            averages = []
            std_devs = []
            for subvalue in values:
                averages.append(subvalue["average"])
                std_devs.append(subvalue["std_dev"])
            axs[index].grid(alpha=0.5, linestyle='--')
            axs[index].bar(range(1, len(values)+1), averages, yerr=std_devs, ecolor="black", capsize=4, label=key)
            axs[index].set_xlabel(keys_list[i][j])
            axs[index].set_xticks(range(1, 1+max([len(value) for value in model_names.values()])))
            axs[index].set_yscale("log")
            index += 1

    axs[0].set_ylabel('Run time [s]')
    legend_handles = [plt.Line2D([0], [0], color='black', markersize=1, label=r'1$\sigma$ Std Dev')]
    fig.legend(handles=[legend_handles[0]], loc='upper right', fontsize="x-small")
    fig.suptitle(f"Run time dynamic models for {propagation_time} day, n={len(run_cases)*10}, start MJD [{start_epoch}, {end_epoch}] \nProcessor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz, 2208 Mhz", fontsize=8)

    utils.save_figures_to_folder([fig], [run_cases[0], run_cases[-1], n*10])

    plt.show()




















def get_Gamma(delta_t):

    # Construct the submatrices
    submatrix1 = (delta_t**2 / 2) * np.eye(3)
    submatrix2 = delta_t * np.eye(3)

    # Concatenate the submatrices to form Gamma
    Gamma = np.block([[submatrix1, np.zeros((3, 3))],
                    [submatrix2, np.zeros((3, 3))],
                    [np.zeros((3, 3)), submatrix1],
                    [np.zeros((3, 3)), submatrix2]])
    return Gamma


def get_process_noise_matrix(delta_t, Q_c1, Q_c2):

    Gamma = get_Gamma(delta_t)

    Q_c_diag = np.block([[Q_c1*np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), Q_c2*np.eye(3)]])
    result = Gamma @ Q_c_diag @ Gamma.T
    return result


for days in range(1, 8, 1):

    point_mass_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(60390, days)
    point_mass_srp_model = high_fidelity_point_mass_srp_01.HighFidelityDynamicModel(60390, days)
    spherical_harmonics_srp_model = high_fidelity_spherical_harmonics_srp_01_2_2_2_2.HighFidelityDynamicModel(60390, days)


    epochs_1, state_history_1, dependent_variables_history_1 = \
        Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.01).get_propagation_results(point_mass_srp_model, solve_variational_equations=False)

    # plt.plot(epochs, state_history[:, :3])

    epochs_2, state_history_2, dependent_variables_history_2 = \
        Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.01).get_propagation_results(spherical_harmonics_srp_model, solve_variational_equations=False)



    k = 0
    var_list = []
    rmse_list = []
    argument_list = []
    for i in np.logspace(start=-22, stop=-10, num=300):
        for j in np.logspace(start=-22, stop=-10, num=300):


            process_noise_matrix = get_process_noise_matrix(days*86400, i, j)
            # print(np.diag(process_noise_matrix))

            sqrt = np.sqrt(np.diag(process_noise_matrix))
            # print("sqrt: ", sqrt)

            diff = state_history_2-state_history_1
            # plt.plot(epochs_1, diff[:,6:9])

            # print(diff[-1,:])

            var1 = np.abs(diff[-1,:])-sqrt

            # var1 = var1/np.linalg.norm(var1)

            var1 = np.abs((np.abs(diff[-1,:])-sqrt)/sqrt)
            # print(var1)

            # print("normalized: ", var1/np.linalg.norm(var1))

            # plt.show()

            rmse = np.sqrt(np.mean(var1[:] ** 2))
            # rmse = np.min(var1)

            # print(rmse)

            rmse_list.append(rmse)
            argument_list.append((i, j))
            var_list.append(var1)

            k += 1

    print(days, "================")

    print(np.argmin(rmse_list))

    res = argument_list[np.argmin(rmse_list)]
    print(res)



process_noise_matrix = get_process_noise_matrix(days*86400, res[0], res[1])
# print(np.diag(process_noise_matrix))

sqrt = np.sqrt(np.diag(process_noise_matrix))
# print("sqrt: ", sqrt)

diff = state_history_2-state_history_1
# plt.plot(epochs_1, diff[:,6:9])


print(sqrt)
print(diff[-1,:])
print(var_list[np.argmin(rmse_list)])

