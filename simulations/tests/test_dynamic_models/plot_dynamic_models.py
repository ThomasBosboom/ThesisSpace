# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define path to import src files
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Third party
import pytest
from tudatpy.kernel import constants
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Own
from src.dynamic_models import validation_LUMIO
from src.dynamic_models.low_fidelity import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import EstimationModel

from src.dynamic_models import Interpolator

class PlotOutputsDynamicalModels:

    def __init__(self, simulation_start_epoch_MJD, propagation_time):

        self.simulation_start_epoch_MJD = simulation_start_epoch_MJD
        self.propagation_time = propagation_time


    def get_dynamic_model_objects(self, package_list=["point_mass"]):

        # package_list = ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]
        # Get a list of all relevant packages within 'high_fidelity'
        packages_dir = os.path.join(parent_dir, 'src', 'dynamic_models', 'high_fidelity')
        # full_package_list = [d for d in os.listdir(packages_dir) if d != "__pycache__" if os.path.isdir(os.path.join(packages_dir, d))]
        # package_list = [package for package in full_package_list if package not in remove_list]

        # Loop through each package and obtain HighFidelityDynamicModel derived from each file
        dynamic_model_objects = []
        for package_name in package_list:

            package_module_path = f'dynamic_models.high_fidelity.{package_name}'
            package_module = __import__(package_module_path, fromlist=[package_name])
            package_files = os.listdir(os.path.join(packages_dir, package_name))

            print("package_files, ", package_files)

            # Loop through each file and call HighFidelityDynamicModel
            for file_name in package_files:
                if file_name.endswith('.py') and not file_name.startswith('__init__'):
                    module_path = f'{package_module_path}.{os.path.splitext(file_name)[0]}'
                    module = __import__(module_path, fromlist=[file_name])
                    dynamic_model_objects.append(module.HighFidelityDynamicModel(self.simulation_start_epoch_MJD, self.propagation_time))

                    print("dynamic model: ", file_name)

        return dynamic_model_objects


    def get_estimation_model_objects(self, package_list=["point_mass"], get_dynamic_model_objects=False):

        estimation_model_objects = []
        dynamic_model_objects = self.get_dynamic_model_objects(package_list=package_list)
        for dynamic_model_object in dynamic_model_objects:
            estimation_model_objects.append(EstimationModel.EstimationModel(dynamic_model_object))

        if get_dynamic_model_objects:
            return dynamic_model_objects, estimation_model_objects
        else:
            return estimation_model_objects


    def get_interpolated_propagation_results(self, dynamic_model_object):

        interp_epochs, interp_state_history, interp_dependent_variables_history, interp_state_transition_matrix_history = Interpolator(dynamic_model_object)

        # dynamics_simulator, variational_equations_solver = dynamic_model_object.get_propagated_orbit()

        # # Extract the time vector
        # epochs                          = np.stack(list(variational_equations_solver.state_history.keys()))
        # state_history                   = np.stack(list(variational_equations_solver.state_history.values()))
        # state_transition_matrix_history = np.stack(list(variational_equations_solver.state_transition_matrix_history.values()))

        # # Define updated time vector
        # interp_epochs = np.arange(np.min(epochs), np.max(epochs), step_size)

        # # Perform interpolation using on the results from variational_equations_solver
        # interp_func = interp1d(epochs, state_history, axis=0, kind=kind, fill_value='extrapolate')
        # interp_state_history = interp_func(interp_epochs)

        # interp_state_transition_matrix_history = np.zeros((len(interp_epochs), *state_transition_matrix_history.shape[1:]))
        # for i in range(state_transition_matrix_history.shape[1]):
        #     interp_func = interp1d(epochs, state_transition_matrix_history[:, i, :], axis=0, kind=kind, fill_value='extrapolate')
        #     interp_state_transition_matrix_history[:, i, :] = interp_func(interp_epochs)

        # # Perform interpolation using on the results from dynamics_simulator
        # epochs                          = np.stack(list(dynamics_simulator.dependent_variable_history.keys()))
        # dependent_variables_history     = np.stack(list(dynamics_simulator.dependent_variable_history.values()))
        # interp_func = interp1d(epochs, dependent_variables_history, axis=0, kind=kind, fill_value='extrapolate')
        # interp_dependent_variables_history = interp_func(interp_epochs)

        # if epoch_in_MJD:

        #     interp_epochs = np.array([time_conversion.julian_day_to_modified_julian_day(\
        #         time_conversion.seconds_since_epoch_to_julian_day(interp_epoch)) for interp_epoch in interp_epochs])

        # return interp_epochs, interp_state_history, interp_dependent_variables_history, interp_state_transition_matrix_history


    def plot_3D_state_history(self, dynamic_model_objects):

        ax = plt.figure(figsize=(6.5,6)).add_subplot(projection='3d')
        for index, dynamic_model_object in enumerate(dynamic_model_objects):

            epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
                    self.get_interpolated_propagation_results(dynamic_model_object, step_size=100)

            ax.plot(state_history[:,0], state_history[:,1], state_history[:,2], color="red", label="LPF" if index  == 0 else None)
            ax.plot(state_history[:,6], state_history[:,7], state_history[:,8], color="blue", label="LUMIO" if index == 0 else None)

            print("Plotted ", dynamic_model_object)

        plt.title("LUMIO and LPF trajectories | Start epoch (MJD): "+str(self.simulation_start_epoch_MJD)+" Propagation time (days): "+str(self.propagation_time))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        plt.axis('equal')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()


    def plot_2D_state_history(self, dynamic_model_objects):

        ax = plt.figure(figsize=(6.5,6))
        for index, dynamic_model_object in enumerate(dynamic_model_objects):

            epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
                    self.get_interpolated_propagation_results(dynamic_model_object)

            plt.plot(epochs, state_history[:,0:3], color="red", label=[r'$x_{1}$', r'$y_{1}$', r'$y_{1}$'] if index  == 0 else None)
            plt.plot(epochs, state_history[:,6:9], color="blue", label=[r'$x_{2}$', r'$y_{2}$', r'$y_{2}$'] if index  == 0 else None)

            print("Plotted ", dynamic_model_object)

        plt.title("LUMIO and LPF trajectories | Start epoch (MJD): "+str(self.simulation_start_epoch_MJD)+" Propagation time (days): "+str(self.propagation_time))
        plt.xlabel("Time in MJD")
        plt.ylabel("x [m]")
        ax.legend()
        plt.tight_layout()
        plt.show()





    def plot_observability_effectiveness(self):

        low_fidelity_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(self.simulation_start_epoch_MJD, self.propagation_time)
        high_fidelity_model = high_fidelity_point_mass_08.HighFidelityDynamicModel(self.simulation_start_epoch_MJD, self.propagation_time)
        # low_fidelity_model = LowFidelityDynamicModel.LowFidelityDynamicModel(simulation_start_epoch_MJD, propagation_time)

        estimation_model_high = EstimationModel.EstimationModel(high_fidelity_model)
        estimation_model_low = EstimationModel.EstimationModel(low_fidelity_model)

        epoch, state_history_high, dependent_variables_history, state_transition_matrix_history = estimation_model_high.get_propagated_orbit_from_estimator()
        epoch, state_history_low, dependent_variables_history, state_transition_matrix_history = estimation_model_low.get_propagated_orbit_from_estimator()
        estimation_results_high = estimation_model_high.get_estimation_results()
        estimation_results_low = estimation_model_low.get_estimation_results()

        # Plot the observability history and compare them to the different models
        plt.figure(figsize=(9,6))

        plt.suptitle("Observability effectiveness comparison of high and low fidelity models. Time step: "+str(self.propagation_time)+" days")

        plt.subplot(2, 1, 1)
        epoch = np.stack(list(estimation_results_high[-1][observation.one_way_range_type].keys()), axis=0)
        information_matrix_history_high = np.stack(list(estimation_results_high[-1][observation.one_way_range_type].values()), axis=0)
        information_matrix_history_low = np.stack(list(estimation_results_low[-1][observation.one_way_range_type].values()), axis=0)
        plt.plot(epoch, np.max(np.linalg.eigvals(information_matrix_history_low[:,6:9,6:9]), axis=1, keepdims=True), color="blue", label="LUMIO, low", linestyle='dashed')
        plt.plot(epoch, np.max(np.linalg.eigvals(information_matrix_history_high[:,6:9,6:9]), axis=1, keepdims=True), color="blue", label="LUMIO, high")
        plt.plot(epoch, np.max(np.linalg.eigvals(information_matrix_history_low[:,:3,:3]), axis=1, keepdims=True), color="red", label="LPF, low", linestyle='dashed')
        plt.plot(epoch, np.max(np.linalg.eigvals(information_matrix_history_high[:,:3,:3]), axis=1, keepdims=True), color="red", label="LPF, high")
        plt.title("Range measurements")
        plt.xlabel(r"Time since $t_0$"+ " [days] ("+str(self.simulation_start_epoch_MJD)+" MJD)")
        plt.ylabel(r"$\sqrt{\max(\lambda_r(t))}$ [-]")
        plt.xlim(min(epoch), max(epoch))
        plt.yscale("log")
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        epoch = np.stack(list(estimation_results_high[-1][observation.one_way_instantaneous_doppler_type].keys()), axis=0)
        information_matrix_history_high = np.stack(list(estimation_results_high[-1][observation.one_way_instantaneous_doppler_type].values()), axis=0)
        information_matrix_history_low = np.stack(list(estimation_results_low[-1][observation.one_way_instantaneous_doppler_type].values()), axis=0)
        plt.plot(epoch, np.max(np.linalg.eigvals(information_matrix_history_low[:,6:9,6:9]), axis=1, keepdims=True), color="blue", label="LUMIO, low", linestyle='dashed')
        plt.plot(epoch, np.max(np.linalg.eigvals(information_matrix_history_high[:,6:9,6:9]), axis=1, keepdims=True), color="blue", label="LUMIO, high")
        plt.plot(epoch, np.max(np.linalg.eigvals(information_matrix_history_low[:,:3,:3]), axis=1, keepdims=True), color="red", label="LPF, low", linestyle='dashed')
        plt.plot(epoch, np.max(np.linalg.eigvals(information_matrix_history_high[:,:3,:3]), axis=1, keepdims=True), color="red", label="LPF, high")
        plt.title("Range-rate measurements")
        plt.xlabel(r"Time since $t_0$"+ " [days] ("+str(self.simulation_start_epoch_MJD)+" MJD)")
        plt.ylabel(r"$\sqrt{\max(\lambda_r(t))}$ [-]")
        plt.xlim(min(epoch), max(epoch))
        plt.yscale("log")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()



    def reference_convergence_history(self):

        # Sample size for Monte Carlo simulation
        sample_size = 100

        # Ellipsoid parameters (covariance matrices) for both satellites
        # estimation_model_objects = self.get_estimation_model_objects()

        covariance_history = dict()
        dynamic_model_objects, estimation_model_objects = self.get_estimation_model_objects(get_dynamic_model_objects=True)
        for estimation_model_object in estimation_model_objects:
            estimation_results = estimation_model_object.get_estimation_results()
            covariance_history[estimation_model_object] = estimation_results[-3][1]
            dynamics_simulator, variational_equations_solver = estimation_model_object.get_propagated_orbit_from_estimator()




        np.stack(covariance_history[estimation_model_objects[0]])

        get_propagated_orbit_from_estimator




        # mean_position_lumio = state[6:9]
        # mean_position_reference = np.array([1,0,0])
        # covariance_satellite1 = np.eye(3)

        # initial_state_LPF = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD,
        #                                                                  self.propagation_time,
        #                                                                  satellite="LPF",
        #                                                                  get_full_history=True,
        #                                                                  get_dict=False)

        # initial_state_LUMIO = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD,
        #                                                                  self.propagation_time,
        #                                                                  satellite="LUMIO",
        #                                                                  get_full_history=True,
        #                                                                  get_dict=False)







        # # Monte Carlo simulation
        # samples_lumio = np.random.multivariate_normal(mean_position_lumio, covariance_satellite1, sample_size)

        # # Generate samples for satellite 2 within a sphere of radius 1
        # theta = 2 * np.pi * np.random.rand(sample_size)
        # phi = np.arccos(2 * np.random.rand(sample_size) - 1)
        # radius = 10*np.ones(sample_size)  # Sphere radius

        # # Convert spherical coordinates to Cartesian coordinates
        # x = radius * np.sin(phi) * np.cos(theta)
        # y = radius * np.sin(phi) * np.sin(theta)
        # z = radius * np.cos(phi)

        # # Stack the Cartesian coordinates to form the samples for satellite 2
        # samples_reference = np.column_stack((x, y, z)) + mean_position_reference

        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter3D(samples_lumio[:,0], samples_lumio[:,1], samples_lumio[:,2])
        # ax.scatter3D(samples_reference[:,0], samples_reference[:,1], samples_reference[:,2])
        # plt.legend()
        # plt.show()

        # # Check intersection for each pair of samples
        # distances = np.linalg.norm(samples_lumio, axis=1)
        # points_outside_sphere = np.sum(distances > radius)
        # percentage_outside_sphere = (points_outside_sphere / sample_size) * 100

        # print(f"Percentage of outside sphere: {percentage_outside_sphere}%")



test = PlotOutputsDynamicalModels(60390, 10)
dynamic_model_objects = test.get_dynamic_model_objects()
# print(test.get_interpolated_propagation_results(dynamic_model_objects[0]))
# print(test.plot_2D_state_history(dynamic_model_objects))
# get_estimation_model_objects = test.get_estimation_model_objects()
# reference_convergence_history = test.plot_observability_effectiveness()
print(dynamic_model_objects)