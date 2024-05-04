# Standard
import os
import sys
import numpy as np
import time
from matplotlib import pyplot as plt

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Own
from tests import utils
import reference_data, Interpolator, StationKeeping, EstimationModel
from NavigationSimulatorBase import NavigationSimulatorBase
from tudatpy.kernel import constants

class NavigationSimulator(NavigationSimulatorBase):

    def __init__(self, observation_windows, **kwargs):
        super().__init__()

        # Flexible initialization using optional parameters and default values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Adjusting decimals based on the step size used
        num_str = "{:.15f}".format(self.step_size).rstrip('0')
        self.decimal_places = len(num_str) - num_str.index('.') - 1

        # Rounding values
        self.observation_windows = observation_windows
        self.observation_windows = [(np.round(tup[0], self.decimal_places), np.round(tup[1], self.decimal_places)) for tup in observation_windows]
        self.initial_station_keeping_epochs = [np.round(observation_window[1], self.decimal_places) for observation_window in observation_windows]
        self.batch_start_times = np.array([t[0] for t in self.observation_windows])
        self.station_keeping_epochs = []

        # Managing the time vector to define all arcs
        self.times = list([self.observation_windows[0][0]] + [item for sublist in self.observation_windows for item in sublist] + [self.observation_windows[0][0]] + self.initial_station_keeping_epochs)
        self.times = list(set(self.times))
        self.times = sorted(self.times)

        self.navigation_arc_durations = np.round(np.diff(self.times), self.decimal_places)
        self.estimation_arc_durations = np.round(np.array([tup[1] - tup[0] for tup in self.observation_windows]), self.decimal_places)


    def get_Gamma(self, delta_t):

        # Construct the submatrices
        submatrix1 = (delta_t**2 / 2) * np.eye(3)
        submatrix2 = delta_t * np.eye(3)

        # Concatenate the submatrices to form Gamma
        Gamma = np.block([[submatrix1, np.zeros((3, 3))],
                        [submatrix2, np.zeros((3, 3))],
                        [np.zeros((3, 3)), submatrix1],
                        [np.zeros((3, 3)), submatrix2]])
        return Gamma


    def get_process_noise_matrix(self, delta_t, Q_c1, Q_c2):

        Gamma = self.get_Gamma(delta_t)

        Q_c_diag = np.block([[Q_c1*np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), Q_c2*np.eye(3)]])
        result = Gamma @ Q_c_diag @ Gamma.T

        return result


    def perform_navigation(self):

        estimation_arc = 0
        navigation_arc = 0
        self.full_estimation_error_dict = dict()
        self.full_reference_state_deviation_dict = dict()
        self.full_propagated_covariance_dict = dict()
        self.full_propagated_formal_errors_dict = dict()
        self.full_state_history_reference_dict = dict()
        self.full_state_history_truth_dict = dict()
        self.full_state_history_initial_dict = dict()
        self.full_state_history_final_dict = dict()
        self.delta_v_dict = dict()
        self.full_dependent_variables_history_initial = dict()
        self.full_state_transition_matrix_history_initial = dict()
        self.estimation_arc_results_dict = dict()
        for t, time in enumerate(self.times):

            navigation_arc_duration = self.navigation_arc_durations[navigation_arc]

            # print(f"Start of navigation arc {navigation_arc} at {time} for {navigation_arc_duration} days")

            # print("=============================")
            # print("time: \n", time)
            # print("navigation_arc_duration", navigation_arc_duration)
            # print("custom_initial_state: \n", self.custom_initial_state)
            # print("custom_initial_state_truth: \n", self.custom_initial_state_truth)
            # print("============================")

            # Define dynamic and truth models to calculate the relevant histories
            dynamic_model_objects = utils.get_dynamic_model_objects(time, navigation_arc_duration, custom_initial_state=self.custom_initial_state)
            dynamic_model = dynamic_model_objects[self.model_type][self.model_name][self.model_number]

            truth_model_objects = utils.get_dynamic_model_objects(time, navigation_arc_duration, custom_initial_state=self.custom_initial_state_truth)
            truth_model = truth_model_objects[self.model_type_truth][self.model_name_truth][self.model_number_truth]


            ##############################################################
            #### PROPAGATIONS OF STATE HISTORIES #########################
            ##############################################################

            # Obtain the initial state of the whole simulation once
            state_history_reference = list()
            for body in dynamic_model.bodies_to_propagate:
                state_history_reference.append(reference_data.get_reference_state_history(time, navigation_arc_duration,
                                                                                          satellite=body,
                                                                                          step_size=self.step_size,
                                                                                          get_full_history=True))
            state_history_reference = np.concatenate(state_history_reference, axis=1)

            if navigation_arc == 0:
                self.custom_initial_state_truth = state_history_reference[0,:] + self.orbit_insertion_error
                self.custom_initial_state = self.custom_initial_state_truth + self.initial_estimation_error

            dynamic_model.custom_initial_state = self.custom_initial_state
            truth_model.custom_initial_state = self.custom_initial_state_truth

            print(dynamic_model.propagation_time, truth_model.propagation_time)

            epochs, state_history_initial, dependent_variables_history_initial, state_transition_matrix_history_initial = \
                Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model,
                                                                                            custom_initial_state=self.custom_initial_state,
                                                                                            custom_propagation_time=navigation_arc_duration,
                                                                                            solve_variational_equations=True)

            epochs, state_history_truth, dependent_variables_history_truth = \
                Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(truth_model,
                                                                                            custom_initial_state=self.custom_initial_state_truth,
                                                                                            custom_propagation_time=navigation_arc_duration,
                                                                                            solve_variational_equations=False)

            # Save the propagated histories of the uncertainties
            propagated_covariance_initial = dict()
            propagated_formal_errors_initial = dict()
            for i in range(len(epochs)):
                propagated_covariance = state_transition_matrix_history_initial[i] @ self.apriori_covariance @ state_transition_matrix_history_initial[i].T
                propagated_covariance_initial.update({epochs[i]: propagated_covariance})
                propagated_formal_errors_initial.update({epochs[i]: np.sqrt(np.diagonal(propagated_covariance))})


            ##############################################################
            #### LOOP FOR ESTIMATION ARCS ################################
            ##############################################################
            estimation_arc_activated = False
            if time in self.batch_start_times:

                estimation_arc_activated = True
                estimation_arc_duration = self.estimation_arc_durations[estimation_arc]

                # print(f"Start of estimation arc {estimation_arc} at navigation arc {navigation_arc}, at {time} for {estimation_arc_duration} days")

                print(f"Estimation error for arc {estimation_arc}")

                # Define dynamic models with a duration equal to the arc length
                dynamic_model_objects = utils.get_dynamic_model_objects(time, estimation_arc_duration,
                                                                        custom_initial_state=self.custom_initial_state)
                dynamic_model = dynamic_model_objects[self.model_type][self.model_name][self.model_number]

                truth_model.custom_propagation_time = estimation_arc_duration

                # Obtain estimation results for given batch and extract results of the estimation arc
                estimation_model = EstimationModel.EstimationModel(dynamic_model,
                                                                   truth_model,
                                                                   apriori_covariance=self.apriori_covariance,
                                                                   initial_estimation_error=self.initial_estimation_error,
                                                                   observation_step_size_range=self.observation_step_size_range,
                                                                   noise_range=self.noise_range,
                                                                   maximum_iterations=self.maximum_iterations)

                estimation_model_result = estimation_model.get_estimation_results()
                estimation_output = estimation_model_result.estimation_output
                parameter_history = estimation_output.parameter_history
                final_covariance = estimation_output.covariance
                best_iteration = estimation_output.best_iteration

                epochs_final, state_history_final, dependent_variables_history_final, state_transition_history_matrix_final = \
                    Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model,
                                                                                                custom_initial_state=parameter_history[:, best_iteration],
                                                                                                custom_propagation_time=estimation_arc_duration,
                                                                                                solve_variational_equations=True)

                propagated_covariance_final = dict()
                propagated_formal_errors_final = dict()
                for i in range(len(epochs_final)):
                    propagated_covariance = state_transition_history_matrix_final[i] @ final_covariance @ state_transition_history_matrix_final[i].T
                    propagated_covariance_final.update({epochs_final[i]: propagated_covariance})
                    propagated_formal_errors_final.update({epochs_final[i]: np.sqrt(np.diagonal(propagated_covariance))})

                # Save the results relevant to the estimation arc
                self.full_state_history_final_dict.update(dict(zip(epochs_final, state_history_final)))
                self.estimation_arc_results_dict.update({estimation_arc: estimation_model_result})

                estimation_arc += 1


            # Update the values for the next navigation arc
            self.custom_initial_state_truth = state_history_truth[-1,:]

            if estimation_arc_activated:
                self.custom_initial_state = state_history_final[-1,:]
                self.initial_estimation_error = state_history_final[-1,:]-state_history_truth[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_final.values()))[-1]
                self.apriori_covariance += self.get_process_noise_matrix(navigation_arc_duration*constants.JULIAN_DAY,
                                                                            self.state_noise_compensation,
                                                                            self.state_noise_compensation)

                print("Estimation error after estimation: \n", self.initial_estimation_error)
                print("LUMIO pos 3D OD error: \n", np.linalg.norm(self.initial_estimation_error[6:9]))
            else:
                self.custom_initial_state = state_history_initial[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_initial.values()))[-1]

            # Save histories for reading out later
            self.full_estimation_error_dict.update(dict(zip(epochs, state_history_initial-state_history_truth)))
            self.full_propagated_covariance_dict.update(propagated_covariance_initial)
            self.full_propagated_formal_errors_dict.update(propagated_formal_errors_initial)
            self.full_state_history_reference_dict.update(dict(zip(epochs, state_history_reference)))
            self.full_state_history_truth_dict.update(dict(zip(epochs, state_history_truth)))
            self.full_state_history_initial_dict.update(dict(zip(epochs, state_history_initial)))
            self.full_dependent_variables_history_initial.update(dict(zip(epochs, dependent_variables_history_initial)))
            self.full_state_transition_matrix_history_initial.update(dict(zip(epochs, state_transition_matrix_history_initial)))
            self.full_reference_state_deviation_dict.update(dict(zip(epochs, state_history_initial-state_history_reference)))


            ##############################################################
            #### STATION KEEPING #########################################
            ##############################################################

            print("state_deviation_history sim: \n", epochs[-1], state_history_initial[-1, :] - state_history_reference[-1, :])

            if self.include_station_keeping:
                if self.times[navigation_arc+1] in self.initial_station_keeping_epochs:

                    params = [0, self.target_point_epochs]

                    dynamic_model_objects = utils.get_dynamic_model_objects(self.times[navigation_arc+1], max(params[1]), custom_initial_state=self.custom_initial_state)
                    dynamic_model = dynamic_model_objects[self.model_type][self.model_name][self.model_number]

                    station_keeping = StationKeeping.StationKeeping(dynamic_model, step_size=self.step_size)
                    delta_v = station_keeping.get_corrected_state_vector(cut_off_epoch=params[0],
                                                                        correction_epoch=params[0],
                                                                        target_point_epochs=params[1])

                    # Generate random noise to simulate station-keeping errors
                    if self.model_type == "HF":

                        delta_v_noise_sigma = np.abs(self.station_keeping_error*delta_v)
                        delta_v_noise = np.random.normal(loc=0, scale=delta_v_noise_sigma, size=delta_v.shape)

                        if np.linalg.norm(delta_v) >= self.delta_v_min:
                            self.custom_initial_state[9:12] += delta_v
                            self.custom_initial_state_truth[9:12] += delta_v + delta_v_noise

                            delta_v_noise_covariance = np.zeros(12)
                            delta_v_noise_covariance[9:12] = delta_v_noise_sigma
                            self.apriori_covariance += np.outer(delta_v_noise_covariance, delta_v_noise_covariance)

                            self.station_keeping_epochs.append(self.times[navigation_arc+1])
                            self.delta_v_dict.update({self.times[navigation_arc+1]: delta_v})

                            print(f"Correction at {self.times[navigation_arc+1]}: \n", delta_v, np.linalg.norm(delta_v))

            # print("=============================")
            # print("time: \n: ", time)
            # print("FINAL custom_initial_state: \n", self.custom_initial_state)
            # print("FINAL custom_initial_state_truth: \n", self.custom_initial_state_truth)
            # print("============================")

            od_error = np.stack(list(self.full_state_history_initial_dict.values()))-np.stack(list(self.full_state_history_truth_dict.values()))
            od_error_epochs = np.stack(list(self.full_state_history_initial_dict.keys()))
            plt.plot(od_error_epochs, od_error)


            if navigation_arc < len(self.times)-2:
                navigation_arc += 1

            else:
                print("End navigation simulation")
                break

        plt.plot()

        self.navigation_result_dicts = [self.full_estimation_error_dict, self.full_reference_state_deviation_dict, self.full_propagated_covariance_dict, self.full_propagated_formal_errors_dict,\
                        self.full_state_history_reference_dict, self.full_state_history_truth_dict, self.full_state_history_initial_dict, self.full_state_history_final_dict, self.delta_v_dict,\
                        self.full_dependent_variables_history_initial, self.full_state_transition_matrix_history_initial, self.estimation_arc_results_dict]

        self.navigation_output = NavigationResults(self)

        return self.navigation_output


class NavigationResults():

    def __init__(self, navigation_simulator):

        self.navigation_simulator = navigation_simulator
        navigation_result_dicts = navigation_simulator.navigation_result_dicts

        self.navigation_results = []
        for i, navigation_result_dict in enumerate(navigation_result_dicts):

            if navigation_result_dict:
                if i in range(0, 11):
                    self.navigation_results.append(utils.convert_dictionary_to_array(navigation_result_dict))
                else:
                    self.navigation_results.append(navigation_result_dict)
            else:
                self.navigation_results.append(([],[]))