# Standard
import os
import sys
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# Tudatpy
from tudatpy.kernel import constants

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from tests import utils
import reference_data, Interpolator, StationKeeping, NavigationSimulator
from src.dynamic_models.full_fidelity.full_fidelity import *
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model


class OptimizationModel():

    def __init__(self, dynamic_model, angle_treshold=30, step_size=0.001):

        self.dynamic_model = dynamic_model
        self.angle_treshold = angle_treshold
        self.step_size = step_size

        self.apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2

    def generate_boundary_tuples(self, input_dict):

        boundaries = []
        start = None
        for i, (key, value) in enumerate(input_dict.items()):
            if i == 0 and value:  # Handle the first element separately
                start = key
            elif value and not input_dict[list(input_dict.keys())[i - 1]]:  # Transition from False to True
                start = key
            elif not value and input_dict[list(input_dict.keys())[i - 1]]:  # Transition from True to False
                boundaries.append((start, list(input_dict.keys())[i - 1]))
                start = None

        # If the last region continues until the end, add it
        if start is not None:
            boundaries.append((start, list(input_dict.keys())[-1]))

        return boundaries

    # def objective(self, )

    def distance_constraint(self, relative_state, distance_threshold):

        return np.linalg.norm(relative_state) < distance_threshold


    # def station_keeping_constraint(self, input_dict, station_keeping_threshold):

    #     epochs, values = utils.convert_dictionary_to_array(input_dict)

    #     start_epoch = epochs[0]
    #     for epoch in epochs:
    #         if int((epoch-start_epoch)%station_keeping_threshold) == 0:
    #             if input_dict[epoch]:



        # if
        # return




    def get_initial_observation_windows(self):


        epochs, state_history, dependent_variables_history, state_transition_history = \
            Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(self.dynamic_model)

        # Define the relative state of LPF with respect to LUMIO
        relative_state_history = dependent_variables_history[:,6:12]


        propagated_covariance_dict = dict()
        for i in range(len(epochs)):
            propagated_covariance = state_transition_history[i] @ self.apriori_covariance @ state_transition_history[i].T
            propagated_covariance_dict.update({epochs[i]: propagated_covariance})

        # Generate history of eigenvectors
        eigenvectors_dict = dict()
        for key, matrix in propagated_covariance_dict.items():
            eigenvalues, eigenvectors = np.linalg.eigh(matrix[6:9, 6:9])
            # eigenvalues, eigenvectors = np.linalg.eigh(matrix[0:3, 0:3])
            max_eigenvalue_index = np.argmax(eigenvalues)
            eigenvector_largest = eigenvectors[:, max_eigenvalue_index]
            eigenvectors_dict.update({key: eigenvector_largest})

        # Initialize an empty list to store the angles
        angles_dict = dict()
        for i, (key, value) in enumerate(eigenvectors_dict.items()):
            vec1 = relative_state_history[i,:3]
            vec2 = value
            dot_product = np.dot(vec1, vec2)
            magnitude_vec1 = np.linalg.norm(vec1)
            magnitude_vec2 = np.linalg.norm(vec2)
            cosine_angle = dot_product / (magnitude_vec1 * magnitude_vec2)
            angle_radians = np.arccos(cosine_angle)
            angle_degrees = np.degrees(angle_radians)
            angles_dict.update({key: angle_degrees if angle_degrees<90 else (180-angle_degrees)})

        # Generate boolans for when treshold condition holds to generate estimation window
        filtered_dict = dict()
        relative_state_history_dict = dict(zip(epochs, relative_state_history))
        for key, value in angles_dict.items():

            # Check if any element of the value vector is below the angle_treshold
            below_angle_treshold = value < self.angle_treshold
            below_distance_treshold = np.linalg.norm(relative_state_history_dict[key][0:3]) < 9e7
            filtered_dict[key] = below_angle_treshold and below_distance_treshold

        observation_windows = self.generate_boundary_tuples(filtered_dict)

        fig = plt.figure()
        plt.plot((epochs-epochs[0]), np.stack(list(angles_dict.values())), label="angles in degrees")
        # plt.plot((epochs-epochs[0]), np.linalg.norm(relative_state_history[:,0:3], axis=1), label="distance")
        for i, gap in enumerate(observation_windows):
            plt.axvspan(
                xmin=(gap[0]-epochs[0]),
                xmax=(gap[1]-epochs[0]),
                color="gray",
                alpha=0.1,
                label="Observation window" if i == 0 else None)
        plt.xlabel("Time since MJD 60390 [days]")
        plt.legend()



        return observation_windows


    # Objective function: to minimize the total length of time intervals
    def objective_function(self, start_times):

        step_size = 0.01
        station_keeping_epoch = 4
        # observation_windows =

        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows, [model_type, model_name, k], truth_model_list, step_size=step_size, station_keeping_epoch=station_keeping_epoch)
        navigation_results = navigation_simulator.perform_navigation(include_station_keeping=True)
        print("been here")
        full_propagated_formal_errors_dict = navigation_results[3]
        full_propagated_formal_errors_epochs = navigation_results[3][0]
        full_propagated_formal_errors_history = navigation_results[3][1]

        total_uncertainty = []
        for epoch, value in full_propagated_formal_errors_dict.items():
            epoch_MJD = utils.convert_epochs_to_MJD(epoch, full_array=False)
            if int((epoch_MJD-60390)%station_keeping_epoch) == 0 and int(epoch_MJD) != 0:
                total_uncertainty.append(value[6:9])

        return np.sum(total_uncertainty)



    # def objective_function(self, start_times):

    #     return np.linalg.norm(start_times)

    # Constraint function: ensure that time between start times is at least 1 day
    def inequality_constraint1(self, start_times):
        return np.diff(start_times)

    def inequality_constraint1(self, start_times):
        return np.diff(start_times) - 2.08  # Ensure the difference between consecutive start times is at least 1 day

    def inequality_constraint2(self, start_times):
        return start_times - 60390



    # Optimization function
    def optimize_intervals(self):

        iteration_history = []
        # Initial guess for start times
        initial_start_times = np.arange(60390,60404,1)
        initial_end_times = np.arange(60390,60404,1)+1
        print(initial_end_times)
        initial_start_times = np.array([60390, 60391, 60392, 60395])

        # Define constraints
        constraints = (
            {'type': 'ineq', 'fun': self.inequality_constraint1},
            {'type': 'ineq', 'fun': self.inequality_constraint2}
                    )

        # Define a callback function to record iteration history
        def callback(xk):
            iteration_history.append(self.objective_function(xk))

        # Minimize the objective function subject to constraints
        result = minimize(self.objective_function, initial_start_times,
                          constraints=constraints,
                          bounds=(lower_bounds, upper_bounds),
                          options={'gtol': 1e-6, 'disp': True},
                          callback=callback)

        # Extract optimized start times
        optimized_start_times = result.x

        # Calculate corresponding end times
        optimized_end_times = optimized_start_times + 1

        plt.plot(iteration_history)
        plt.show()

        return optimized_start_times, optimized_end_times, iteration_history







custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
                                1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])
dynamic_model = low_fidelity.LowFidelityDynamicModel(60390, 14, custom_initial_state=custom_initial_state, use_synodic_state=True)


model_type = "high_fidelity"
model_name = "spherical_harmonics_srp"
model_number = 4

model_type = "low_fidelity"
model_name = "three_body_problem"
model_number = 0

# Define dynamic models and select one to test the estimation on
dynamic_model_objects = utils.get_dynamic_model_objects(60390, 28)
dynamic_model = dynamic_model_objects[model_type][model_name][model_number]

optimization_model = OptimizationModel(dynamic_model, angle_treshold=40, step_size=0.01)
observation_windows = optimization_model.get_initial_observation_windows()

print(observation_windows)



# # Example usage
# optimized_start_times, optimized_end_times, iteration_history = optimization_model.optimize_intervals()
# print("Optimized start times:", optimized_start_times)
# print("Corresponding end times:", optimized_end_times)


plt.show()