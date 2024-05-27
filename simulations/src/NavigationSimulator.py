# Standard
import os
import sys
import numpy as np
import time
from matplotlib import pyplot as plt
import copy

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

    def __init__(self, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


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


    def perform_navigation(self, observation_windows, seed=0):

        np.random.seed(seed)

        # Adjusting decimals based on the step size used
        num_str = "{:.15f}".format(self.step_size).rstrip('0')
        self.decimal_places = len(num_str) - num_str.index('.') - 1

        # Update mission start epoch based on given observation window
        self.mission_start_epoch = observation_windows[0][0]

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

        print("=========================")
        # print("Start navigation simulation")

        estimation_arc = 0
        navigation_arc = 0
        self.full_estimation_error_dict = dict()
        self.full_reference_state_deviation_dict = dict()
        self.full_propagated_covariance_dict = dict()
        self.full_propagated_formal_errors_dict = dict()
        self.full_state_history_reference_dict = dict()
        self.full_state_history_truth_dict = dict()
        self.full_state_history_estimated_dict = dict()
        self.full_state_history_final_dict = dict()
        self.delta_v_dict = dict()
        self.full_dependent_variables_history_estimated = dict()
        self.full_state_transition_matrix_history_estimated = dict()
        self.estimation_arc_results_dict = dict()
        for t, time in enumerate(self.times):

            navigation_arc_duration = self.navigation_arc_durations[navigation_arc]

            # print(f"Start of navigation arc {navigation_arc} at {time} for {navigation_arc_duration} days")

            # Define dynamic and truth models to calculate the relevant histories
            dynamic_model_objects = utils.get_dynamic_model_objects(time, navigation_arc_duration,
                                                                    custom_initial_state=self.custom_initial_state,
                                                                    custom_model_dict={self.model_type: [self.model_name]},
                                                                    custom_model_list=[self.model_number])
            dynamic_model = dynamic_model_objects[self.model_type][self.model_name][0]

            truth_model_objects = utils.get_dynamic_model_objects(time, navigation_arc_duration,
                                                                  custom_initial_state=self.custom_initial_state_truth,
                                                                  custom_model_dict={self.model_type_truth: [self.model_name_truth]},
                                                                  custom_model_list=[self.model_number_truth])
            truth_model = truth_model_objects[self.model_type_truth][self.model_name_truth][0]


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
                self.custom_initial_state_truth = truth_model.initial_state + self.orbit_insertion_error
                self.custom_initial_state = self.custom_initial_state_truth + self.initial_estimation_error

            dynamic_model.custom_initial_state = self.custom_initial_state
            truth_model.custom_initial_state = self.custom_initial_state_truth

            epochs, state_history_estimated, dependent_variables_history_estimated, state_transition_matrix_history_estimated = \
                Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model, solve_variational_equations=True)

            epochs, state_history_truth, dependent_variables_history_truth = \
                Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(truth_model, solve_variational_equations=False)

            # print("difference initial and truth: \n", state_history_estimated[0, :]-state_history_truth[0, :])
            od_error = state_history_estimated[0, :]-state_history_truth[0, :]
            # print("state_history_estimated: \n", state_history_estimated[0, :])
            # print("state_history_truth: \n", state_history_truth[0, :])
            # print("epoch before: ", epochs[-1])
            # print("LUMIO OD error before arc:  ", np.linalg.norm(od_error[6:9]))

            if self.propagate_dynamics_linearly:
                state_history_estimated = state_history_truth + np.dot(state_transition_matrix_history_estimated, self.custom_initial_state-self.custom_initial_state_truth)
                # print("length: \n", epochs[0], epochs[-1], len(state_history_estimated))

            # Save the propagated histories of the uncertainties
            propagated_covariance_initial = dict()
            propagated_formal_errors_initial = dict()
            for i in range(len(epochs)):
                propagated_covariance = state_transition_matrix_history_estimated[i] @ self.apriori_covariance @ state_transition_matrix_history_estimated[i].T
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

                # print("dynamic_model: \n", dynamic_model.simulation_start_epoch_MJD, dynamic_model.propagation_time, dynamic_model.custom_initial_state)
                # print("truth_model: \n", truth_model.simulation_start_epoch_MJD, truth_model.propagation_time, truth_model.custom_initial_state)

                # print("3D OD errors before: \n", \
                #     np.linalg.norm(dynamic_model.custom_initial_state[0:3]-truth_model.custom_initial_state[0:3]), "\n",
                #     np.linalg.norm(dynamic_model.custom_initial_state[3:6]-truth_model.custom_initial_state[3:6]), "\n",
                #     np.linalg.norm(dynamic_model.custom_initial_state[6:9]-truth_model.custom_initial_state[6:9]), "\n",
                #     np.linalg.norm(dynamic_model.custom_initial_state[9:12]-truth_model.custom_initial_state[9:12])
                #     )

                maximum_iterations = copy.deepcopy(self.maximum_iterations)
                if estimation_arc == 0:
                    self.maximum_iterations = self.maximum_iterations_first_arc

                estimation_model = EstimationModel.EstimationModel(dynamic_model, truth_model, **vars(self))
                # print(vars(estimation_model))
                estimation_model_result = estimation_model.get_estimation_results()
                estimation_output = estimation_model_result.estimation_output
                parameter_history = estimation_output.parameter_history
                final_covariance = estimation_output.covariance
                best_iteration = estimation_output.best_iteration
                final_residuals = estimation_output.final_residuals

                dynamic_model.custom_initial_state = parameter_history[:, best_iteration]
                epochs_final, state_history_final, dependent_variables_history_final, state_transition_history_matrix_final = \
                    Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model, solve_variational_equations=True)

                if self.propagate_dynamics_linearly:
                    state_history_final = state_history_truth + np.dot(state_transition_history_matrix_final, dynamic_model.custom_initial_state-state_history_truth[0,:])
                    # print("length: \n", epochs_final[0], epochs_final[-1], len(state_history_final))

                # print("3D OD errors after: \n", \
                #     np.linalg.norm(parameter_history[:, best_iteration][0:3]-self.custom_initial_state_truth[0:3]), "\n",
                #     np.linalg.norm(parameter_history[:, best_iteration][3:6]-self.custom_initial_state_truth[3:6]), "\n",
                #     np.linalg.norm(parameter_history[:, best_iteration][6:9]-self.custom_initial_state_truth[6:9]), "\n",
                #     np.linalg.norm(parameter_history[:, best_iteration][9:12]-self.custom_initial_state_truth[9:12])
                #     )

                # print("estimated: \n", parameter_history[:, best_iteration])
                # print("initial model: \n", self.custom_initial_state)
                # print("diff update: ", parameter_history[:, best_iteration]-self.custom_initial_state)
                # print("OD error: \n", parameter_history[:, best_iteration]-self.custom_initial_state_truth)
                # print("epoch before: ", epochs[-1])
                # print("LUMIO OD error before arc:  ", np.linalg.norm(od_error[6:9]))

                # print("difference best and final before: ", parameter_history[:, best_iteration]-parameter_history[:, -1])
                # dynamic_model.custom_initial_state = parameter_history[:, -1]
                # epochs_final_test, state_history_final_test, dependent_variables_history_final_test, state_transition_history_matrix_final_test = \
                #     Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model, solve_variational_equations=True)

                # print("epoch final: ", epochs_final_test[-1])
                # print("difference best and best and truth after: ", np.linalg.norm(state_history_final[-1,:] - state_history_truth[-1,:]))
                # print("difference best and final and truth after: ", np.linalg.norm(state_history_final_test[-1,:] - state_history_truth[-1,:]))


                propagated_covariance_final = dict()
                propagated_formal_errors_final = dict()
                for i in range(len(epochs_final)):
                    propagated_covariance = state_transition_history_matrix_final[i] @ final_covariance @ state_transition_history_matrix_final[i].T
                    propagated_covariance_final.update({epochs_final[i]: propagated_covariance})
                    propagated_formal_errors_final.update({epochs_final[i]: np.sqrt(np.diagonal(propagated_covariance))})

                # Save the results relevant to the estimation arc
                self.full_state_history_final_dict.update(dict(zip(epochs_final, state_history_final)))
                self.estimation_arc_results_dict.update({estimation_arc: estimation_model_result})

                self.maximum_iterations = maximum_iterations

                estimation_arc += 1


            # Update the values for the next navigation arc
            self.custom_initial_state_truth = state_history_truth[-1,:]

            if estimation_arc_activated:
                self.custom_initial_state = state_history_final[-1,:]
                self.initial_estimation_error = state_history_final[-1,:]-state_history_truth[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_final.values()))[-1]
                self.apriori_covariance += self.get_process_noise_matrix(navigation_arc_duration*constants.JULIAN_DAY,
                                                                            self.state_noise_compensation_lpf,
                                                                            self.state_noise_compensation_lumio)

                # arc_final_od_error = self.custom_initial_state[0:3]-self.custom_initial_state_truth[0:3]
                # print("3D OD errors propagated: \n", \
                #     np.linalg.norm(self.custom_initial_state[0:3]-self.custom_initial_state_truth[0:3]), "\n",
                #     np.linalg.norm(self.custom_initial_state[3:6]-self.custom_initial_state_truth[3:6]), "\n",
                #     np.linalg.norm(self.custom_initial_state[6:9]-self.custom_initial_state_truth[6:9]), "\n",
                #     np.linalg.norm(self.custom_initial_state[9:12]-self.custom_initial_state_truth[9:12])
                #     )


            else:
                self.custom_initial_state = state_history_estimated[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_initial.values()))[-1]


            # Save histories for reading out later
            self.full_estimation_error_dict.update(dict(zip(epochs, state_history_estimated-state_history_truth)))
            self.full_propagated_covariance_dict.update(propagated_covariance_initial)
            self.full_propagated_formal_errors_dict.update(propagated_formal_errors_initial)
            self.full_state_history_reference_dict.update(dict(zip(epochs, state_history_reference)))
            self.full_state_history_truth_dict.update(dict(zip(epochs, state_history_truth)))
            self.full_state_history_estimated_dict.update(dict(zip(epochs, state_history_estimated)))
            self.full_dependent_variables_history_estimated.update(dict(zip(epochs, dependent_variables_history_estimated)))
            self.full_state_transition_matrix_history_estimated.update(dict(zip(epochs, state_transition_matrix_history_estimated)))
            self.full_reference_state_deviation_dict.update(dict(zip(epochs, state_history_truth-state_history_reference)))


            ##############################################################
            #### STATION KEEPING #########################################
            ##############################################################

            if self.include_station_keeping:
                if self.times[navigation_arc+1] in self.initial_station_keeping_epochs:

                    params = [0, self.target_point_epochs]

                    dynamic_model_objects = utils.get_dynamic_model_objects(self.times[navigation_arc+1], max(params[1]),
                                                                            custom_initial_state=self.custom_initial_state,
                                                                            custom_model_dict={self.model_type: [self.model_name]},
                                                                            custom_model_list=[self.model_number])
                    dynamic_model_station_keeping = dynamic_model_objects[self.model_type][self.model_name][0]

                    station_keeping = StationKeeping.StationKeeping(dynamic_model_station_keeping, step_size=self.step_size)
                    delta_v, dispersion = station_keeping.get_corrected_state_vector(cut_off_epoch=params[0],
                                                                                    correction_epoch=params[0],
                                                                                    target_point_epochs=params[1])
                    # print(dispersion, self.initial_estimation_error)
                    # ratio = np.linalg.norm(self.initial_estimation_error[6:9])/np.linalg.norm(dispersion[:3])
                    # print(np.linalg.norm(dispersion[:3]), np.linalg.norm(dispersion[3:]))
                    # print("ratio: ", ratio)

                    # print(dynamic_model.custom_initial_state[0:3], parameter_history[:, best_iteration], truth_model.custom_initial_state[0:3])
                    # print("Final residuals: ", np.sqrt(np.mean(final_residuals**2)), "Dispersion: ", np.linalg.norm(dispersion[:3]), "xhat0: ", np.linalg.norm(parameter_history[:, best_iteration][6:9]-truth_model.custom_initial_state[6:9]),  "Estimation error: xhatf", np.linalg.norm(self.initial_estimation_error[6:9]))

                    # Generate random noise to simulate station-keeping errors
                    if self.model_type == "HF":

                        delta_v_noise_sigma = np.abs(self.station_keeping_error*delta_v)
                        delta_v_noise = np.random.normal(loc=0, scale=delta_v_noise_sigma, size=delta_v.shape)

                        od_update = np.linalg.norm(state_history_estimated[0, 6:9]-state_history_truth[0, 6:9])-np.linalg.norm(self.initial_estimation_error[6:9])

                        if np.linalg.norm(delta_v) >= self.delta_v_min:
                            self.custom_initial_state[9:12] += delta_v
                            self.custom_initial_state_truth[9:12] += delta_v + delta_v_noise

                            delta_v_noise_covariance = np.zeros(12)
                            delta_v_noise_covariance[9:12] = delta_v_noise_sigma
                            self.apriori_covariance += np.outer(delta_v_noise_covariance, delta_v_noise_covariance)

                            self.station_keeping_epochs.append(self.times[navigation_arc+1])
                            self.delta_v_dict.update({self.times[navigation_arc+1]: delta_v})

                            # print(f"Correction at {self.times[navigation_arc+1]}: \n", delta_v, np.linalg.norm(delta_v))
                            print(f"Correction at {self.times[navigation_arc+1]}: ", np.linalg.norm(delta_v), " m/s")


            if navigation_arc < len(self.times)-2:
                navigation_arc += 1

            else:
                break

        # print("End navigation simulation")
        print("=========================")

        self.navigation_result_dicts = [self.full_estimation_error_dict, self.full_reference_state_deviation_dict, self.full_propagated_covariance_dict, self.full_propagated_formal_errors_dict,\
                        self.full_state_history_reference_dict, self.full_state_history_truth_dict, self.full_state_history_estimated_dict, self.full_state_history_final_dict, self.delta_v_dict,\
                        self.full_dependent_variables_history_estimated, self.full_state_transition_matrix_history_estimated, self.estimation_arc_results_dict]

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