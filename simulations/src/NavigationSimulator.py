# Standard
import os
import sys
import numpy as np
import time

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Own
from tests import utils
import reference_data, Interpolator, StationKeeping
from tudatpy.kernel import constants


# np.random.seed(0)

class NavigationSimulator():

    def __init__(self, observation_windows,
                       dynamic_model_list,
                       truth_model_list,
                       step_size=1e-2,
                       station_keeping_epochs=[],
                       target_point_epochs=[3],
                       custom_station_keeping_error=None,
                       custom_initial_estimation_error=None,
                       custom_apriori_covariance=None,
                       custom_orbit_insertion_error=None,
                       mission_start_time = 60390):

        # Miscellaneous
        self.step_size = step_size

        # Managing the dynamic model specifications
        self.model_type, self.model_name, self.model_number = dynamic_model_list[0], dynamic_model_list[1], dynamic_model_list[2]
        self.model_type_truth, self.model_name_truth, self.model_number_truth = truth_model_list[0], truth_model_list[1], truth_model_list[2]

        # Customability of estimation arc attributes
        self.custom_range_noise = None
        self.custom_observation_step_size_range = None
        self.custom_initial_state = None
        self.custom_initial_state_truth = None

        # Defining mission start
        self.mission_start_time = mission_start_time
        self.initial_station_keeping_epochs = station_keeping_epochs
        self.target_point_epochs = target_point_epochs

        # Initial state and uncertainty parameters
        self.station_keeping_error = 1e-20
        self.initial_estimation_error = np.array([5e-0, 5e-0, 5e-0, 1e-5, 1e-5, 1e-5, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])*1e0
        self.apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
        self.orbit_insertion_error = np.array([1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e1, 1e1, 1e1, 1e-4, 1e-4, 1e-4])*1e2

        if custom_station_keeping_error is not None:
            self.station_keeping_error = custom_station_keeping_error

        if custom_initial_estimation_error is not None:
            self.initial_estimation_error = custom_initial_estimation_error

        if custom_apriori_covariance is not None:
            self.apriori_covariance = custom_apriori_covariance

        if custom_orbit_insertion_error is not None:
            self.orbit_insertion_error = custom_orbit_insertion_error


        # Adjusting decimals based on the step size used
        num_str = "{:.15f}".format(step_size).rstrip('0')
        self.decimal_places = len(num_str) - num_str.index('.') - 1

        # Rounding values
        self.observation_windows = observation_windows
        self.observation_windows = [(np.round(tup[0], self.decimal_places), np.round(tup[1], self.decimal_places)) for tup in observation_windows]
        self.initial_station_keeping_epochs = [np.round(station_keeping_epoch, self.decimal_places) for station_keeping_epoch in station_keeping_epochs]
        self.batch_start_times = np.array([t[0] for t in self.observation_windows])
        self.station_keeping_epochs = []

        # Managing the time vector to define all arcs
        self.times = list([self.observation_windows[0][0]] + [item for sublist in self.observation_windows for item in sublist] + [self.observation_windows[0][0]] + self.initial_station_keeping_epochs)
        self.times = list(set(self.times))
        self.times = sorted(self.times)

        self.navigation_arc_durations = np.round(np.diff(self.times), self.decimal_places)
        self.estimation_arc_durations = np.round(np.array([tup[1] - tup[0] for tup in self.observation_windows]), self.decimal_places)

        print("Start navigation simulation")
        print("Observation windows \n ", self.observation_windows)
        print("Station keeping epochs \n ", self.initial_station_keeping_epochs)


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

            # navigation_arc_duration = np.round(np.diff(self.times)[navigation_arc], self.decimal_places)
            navigation_arc_duration = self.navigation_arc_durations[navigation_arc]

            # print(f"Start of navigation arc {navigation_arc} at {time} for {navigation_arc_duration} days")

            if navigation_arc_duration == 0 or time == self.times[-1]:
                continue

            # Define dynamic and truth models to calculate the relevant histories
            dynamic_model_objects = utils.get_dynamic_model_objects(time, navigation_arc_duration, custom_initial_state=self.custom_initial_state)
            dynamic_model = dynamic_model_objects[self.model_type][self.model_name][self.model_number]

            truth_model_objects = utils.get_dynamic_model_objects(time, navigation_arc_duration, custom_initial_state=self.custom_initial_state_truth)
            truth_model = truth_model_objects[self.model_type_truth][self.model_name_truth][self.model_number_truth]


            ##############################################################
            #### PROPAGATIONS OF STATE HISTORIES #########################
            ##############################################################

            # Obtain the initial state of the whole simulation once
            # if navigation_arc == 0:
            #     # epochs, state_history_initialize, dependent_variables_history_initialize = \
            #     #     Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model, solve_variational_equations=False)

            #     state_history_reference = list()
            #     for body in dynamic_model.bodies_to_propagate:
            #         state_history_reference.append(reference_data.get_reference_state_history(time,
            #                                                                                   navigation_arc_duration,
            #                                                                                   satellite=body,
            #                                                                                   step_size=self.step_size,
            #                                                                                   get_full_history=False))
            #     state_history_reference = np.concatenate(state_history_reference, axis=1)


            #     self.custom_initial_state_truth = state_history_reference + self.initial_estimation_error + self.orbit_insertion_error
            #     self.custom_initial_state = self.custom_initial_state_truth

            # dynamic_model.custom_initial_state = self.custom_initial_state
            # truth_model.custom_initial_state = self.custom_initial_state_truth

            # Update the reference orbit that the estimated orbit should follow
            state_history_reference = list()
            for body in dynamic_model.bodies_to_propagate:
                state_history_reference.append(reference_data.get_reference_state_history(time,
                                                                                      navigation_arc_duration,
                                                                                        satellite=body,
                                                                                        step_size=self.step_size,
                                                                                        get_full_history=True))
            state_history_reference = np.concatenate(state_history_reference, axis=1)

            if navigation_arc == 0:

                self.custom_initial_state_truth = state_history_reference[0,:] + self.initial_estimation_error + self.orbit_insertion_error
                self.custom_initial_state = state_history_reference[0,:]

                # print(self.initial_estimation_error)
                # print(self.orbit_insertion_error)
                # print(self.initial_estimation_error + self.orbit_insertion_error)

                # print("Initial offset: \n", self.custom_initial_state - self.custom_initial_state_truth)

            dynamic_model.custom_initial_state = self.custom_initial_state
            truth_model.custom_initial_state = self.custom_initial_state_truth

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

                # Define dynamic models and select one to test the estimation on
                dynamic_model_objects = utils.get_dynamic_model_objects(time,
                                                                        estimation_arc_duration,
                                                                        custom_model_dict=None,
                                                                        get_only_first=False,
                                                                        custom_initial_state=self.custom_initial_state)
                dynamic_model = dynamic_model_objects[self.model_type][self.model_name][self.model_number]

                truth_model.custom_propagation_time = estimation_arc_duration

                # Obtain estimation results for given batch and extract results of the estimation arc
                estimation_model_results = utils.get_estimation_model_results({self.model_type: {self.model_name: [dynamic_model]}},
                                                                                    get_only_first=False,
                                                                                    custom_truth_model=truth_model,
                                                                                    apriori_covariance=self.apriori_covariance,
                                                                                    initial_estimation_error=self.initial_estimation_error,
                                                                                    custom_range_noise=self.custom_range_noise,
                                                                                    custom_observation_step_size_range=self.custom_observation_step_size_range)

                estimation_model_result = estimation_model_results[self.model_type][self.model_name][0]
                estimation_output = estimation_model_result.estimation_output
                parameter_history = estimation_output.parameter_history
                final_covariance = estimation_output.covariance
                formal_errors = estimation_output.formal_errors
                best_iteration = estimation_output.best_iteration

                epochs, state_history_final, dependent_variables_history_final, state_transition_history_matrix_final = \
                    Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model,
                                                                                                    custom_initial_state=parameter_history[:, best_iteration],
                                                                                                    custom_propagation_time=estimation_arc_duration,
                                                                                                    solve_variational_equations=True)

                propagated_covariance_final = dict()
                propagated_formal_errors_final = dict()
                for i in range(len(epochs)):
                    propagated_covariance = state_transition_history_matrix_final[i] @ estimation_output.covariance @ state_transition_history_matrix_final[i].T
                    propagated_covariance_final.update({epochs[i]: propagated_covariance})
                    propagated_formal_errors_final.update({epochs[i]: np.sqrt(np.diagonal(propagated_covariance))})

                # Save the history
                self.full_state_history_final_dict.update(dict(zip(epochs, state_history_final)))

                # Save the estimation arc results to a dict
                self.estimation_arc_results_dict.update({estimation_arc: estimation_model_result})

                estimation_arc += 1

            # Update the values for the next navigation arc
            self.custom_initial_state_truth = state_history_truth[-1,:]

            if estimation_arc_activated:
                self.custom_initial_state = state_history_final[-1,:]
                self.initial_estimation_error = state_history_final[-1,:]-state_history_truth[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_final.values()))[-1]
            else:
                self.custom_initial_state = state_history_initial[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_initial.values()))[-1]

            # Adding process noise
            delta_t = navigation_arc_duration*constants.JULIAN_DAY
            if self.model_name != self.model_name_truth:
                self.apriori_covariance += self.get_process_noise_matrix(delta_t, 5.415871378079487e-12, 3.4891012134067807e-14)

            # Save histories for reading out later
            self.full_estimation_error_dict.update(dict(zip(epochs, state_history_initial-state_history_truth)))
            self.full_propagated_covariance_dict.update(propagated_covariance_initial)
            self.full_propagated_formal_errors_dict.update(propagated_formal_errors_initial)
            self.full_state_history_reference_dict.update(dict(zip(epochs, state_history_reference)))
            self.full_state_history_truth_dict.update(dict(zip(epochs, state_history_truth)))
            self.full_state_history_initial_dict.update(dict(zip(epochs, state_history_initial)))
            self.full_dependent_variables_history_initial.update(dict(zip(epochs, dependent_variables_history_initial)))
            self.full_state_transition_matrix_history_initial.update(dict(zip(epochs, state_transition_matrix_history_initial)))

            # if len(state_history_reference) != len(state_history_initial):
            #     state_history_initial = state_history_initial[:-1,:]
            self.full_reference_state_deviation_dict.update(dict(zip(epochs, state_history_reference-state_history_truth)))


            ##############################################################
            #### STATION KEEPING #########################################
            ##############################################################

            if self.times[navigation_arc+1] in self.initial_station_keeping_epochs:

                params = [0, self.target_point_epochs]
                # print("PARAMS: ", params)
                dynamic_model.simulation_start_epoch_MJD = self.times[navigation_arc+1]
                # print("dynamic_model.simulation_start_epoch_MJD:", dynamic_model.simulation_start_epoch_MJD)
                station_keeping = StationKeeping.StationKeeping(dynamic_model, custom_initial_state=self.custom_initial_state, custom_propagation_time=max(params[1]), step_size=self.step_size)
                delta_v = station_keeping.get_corrected_state_vector(cut_off_epoch=params[0], correction_epoch=params[0], target_point_epochs=params[1])

                # Generate random noise to simulate station-keeping errors
                if self.model_type == "HF":
                    print(np.linalg.norm(delta_v))
                    delta_v_noise_sigma = self.station_keeping_error*np.abs(delta_v)
                    delta_v_noise = np.random.normal(loc=0, scale=delta_v_noise_sigma, size=delta_v.shape)

                    if np.linalg.norm(delta_v) >= 0.02:
                        self.custom_initial_state[9:12] += delta_v
                        self.custom_initial_state_truth[9:12] += delta_v + delta_v_noise

                        delta_v_noise_covariance = np.zeros(12)
                        delta_v_noise_covariance[9:12] = delta_v_noise_sigma
                        self.apriori_covariance += np.outer(delta_v_noise_covariance, delta_v_noise_covariance)

                        self.station_keeping_epochs.append(self.times[navigation_arc+1])

                        print(f"Correction at {self.times[navigation_arc+1]}: \n", delta_v, np.linalg.norm(delta_v))

                        self.delta_v_dict.update({self.times[navigation_arc+1]: delta_v})

            if navigation_arc<len(self.times)-2:
                navigation_arc += 1
            else:
                print("End navigation simulation")
                break

        # result_dicts = self.full_estimation_error_dict, self.full_reference_state_deviation_dict, self.full_propagated_covariance_dict, self.full_propagated_formal_errors_dict,\
        #                 self.full_state_history_reference_dict, self.full_state_history_truth_dict, self.full_state_history_initial_dict, self.full_state_history_final_dict, self.delta_v_dict,\
        #                 self.full_dependent_variables_history_initial, self.full_state_transition_matrix_history_initial, self.estimation_arc_results_dict


        self.navigation_result_dicts = [self.full_estimation_error_dict, self.full_reference_state_deviation_dict, self.full_propagated_covariance_dict, self.full_propagated_formal_errors_dict,\
                        self.full_state_history_reference_dict, self.full_state_history_truth_dict, self.full_state_history_initial_dict, self.full_state_history_final_dict, self.delta_v_dict,\
                        self.full_dependent_variables_history_initial, self.full_state_transition_matrix_history_initial, self.estimation_arc_results_dict]

        self.navigation_output = NavigationOutput(self)

        return self.navigation_output


class NavigationOutput():

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