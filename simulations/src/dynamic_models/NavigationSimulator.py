# Standard
import os
import sys
import numpy as np

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from tests import utils
from src.dynamic_models import validation
from src.dynamic_models import Interpolator, StationKeeping
from src.estimation_models import estimation_model


np.random.seed(0)

class NavigationSimulator():

    def __init__(self, observation_windows, dynamic_model_list, truth_model_list, step_size=1e-2, custom_station_keeping_epochs=None, target_point_epochs=[3], include_station_keeping=True, exclude_first_manouvre=False):

        # Miscellaneous
        self.step_size = step_size

        # Managing the dynamic model specifications
        self.model_type, self.model_name, self.model_number = dynamic_model_list[0], dynamic_model_list[1], dynamic_model_list[2]
        self.model_type_truth, self.model_name_truth, self.model_number_truth = truth_model_list[0], truth_model_list[1], truth_model_list[2]

        # Station keeping parameters
        self.include_station_keeping = include_station_keeping
        self.custom_station_keeping_epochs = custom_station_keeping_epochs
        self.target_point_epochs = target_point_epochs
        self.exclude_first_manouvre = exclude_first_manouvre

        # Initial state and uncertainty parameters
        self.custom_initial_state = None
        self.custom_initial_state_truth = None
        self.initial_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])*1e-1
        self.apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
        self.initial_reference_error = np.array([1e1, 1e1, 1e1, 1e-4, 1e-4, 1e-4, 1e1, 1e1, 1e1, 1e-4, 1e-4, 1e-4])*1e0

        # Adjusting decimals based on the step size used
        num_str = "{:.15f}".format(step_size).rstrip('0')
        self.decimal_places = len(num_str) - num_str.index('.') - 1
        # print("DECIMAL PLACES: ", self.decimal_places)

        # Managing the timing aspect of the navigation arcs
        self.observation_windows = observation_windows
        self.mission_start_time = 60390
        self.mission_time = self.observation_windows[-1][-1]-self.mission_start_time
        self.mission_end_time = self.mission_start_time + self.mission_time

        self.batch_start_times = np.array([t[0] for t in self.observation_windows])
        self.batch_end_times = np.array([t[1] for t in self.observation_windows])

        # print(self.observation_windows)

        self.times = list(set([self.mission_start_time] + [item for sublist in self.observation_windows for item in sublist] + [self.mission_end_time]))

        self.station_keeping_epochs = []
        if custom_station_keeping_epochs is None:
            if self.include_station_keeping:
                for station_keeping_epoch in range(int(self.mission_start_time), int(self.mission_end_time)+4, 4):
                    if station_keeping_epoch != self.mission_start_time:
                        self.station_keeping_epochs.append(station_keeping_epoch)
                    # print(self.station_keeping_epochs)
                if self.exclude_first_manouvre:
                    if 60394 in self.station_keeping_epochs:
                        self.station_keeping_epochs.remove(60394)
        else:
            self.station_keeping_epochs = custom_station_keeping_epochs

        # print("times: ", self.times)
        self.times.extend(self.station_keeping_epochs)
        self.times = np.round(sorted(list(set(self.times))), self.decimal_places)

        # print("station keeping epochs: ", self.station_keeping_epochs)



    def perform_navigation(self):

        estimation_arc = 0
        navigation_arc = 0
        full_estimation_error_dict = dict()
        full_reference_state_deviation_dict = dict()
        full_propagated_covariance_dict = dict()
        full_propagated_formal_errors_dict = dict()
        full_state_history_reference_dict = dict()
        full_state_history_truth_dict = dict()
        full_state_history_initial_dict = dict()
        full_state_history_final_dict = dict()
        delta_v_dict = dict()
        full_dependent_variables_history_initial = dict()
        full_state_transition_matrix_history_initial = dict()
        print("Start navigation simulation ======================")
        for t, time in enumerate(self.times):

            # print("time: ", time, self.observation_windows, self.station_keeping_epochs)

            navigation_arc_duration = np.round(np.diff(self.times)[navigation_arc], self.decimal_places)

            # print(f"Start of navigation arc {navigation_arc} at {time} for {navigation_arc_duration} days")

            # print(f"EPOCH AND VALUE OF FIRST CUSTOM STATE OF ARC {navigation_arc}: ", self.custom_initial_state)

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
            if navigation_arc == 0:
                epochs, state_history_initialize, dependent_variables_history_initialize = \
                    Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model, solve_variational_equations=False)

                self.custom_initial_state_truth = state_history_initialize[0,:]
                self.custom_initial_state = self.custom_initial_state_truth + self.initial_estimation_error + self.initial_reference_error

            dynamic_model.custom_initial_state = self.custom_initial_state
            truth_model.custom_initial_state = self.custom_initial_state_truth

            # Update the reference orbit that the estimated orbit should follow
            state_history_reference = list()
            for body in dynamic_model.bodies_to_propagate:
                state_history_reference.append(validation.get_reference_state_history(time,
                                                                                    navigation_arc_duration,
                                                                                        satellite=body,
                                                                                        step_size=self.step_size,
                                                                                        get_full_history=True))
            state_history_reference = np.concatenate(state_history_reference, axis=1)

            epochs, state_history_initial, dependent_variables_history_initial, state_transition_history_initial = \
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
                propagated_covariance = state_transition_history_initial[i] @ self.apriori_covariance @ state_transition_history_initial[i].T
                propagated_covariance_initial.update({epochs[i]: propagated_covariance})
                propagated_formal_errors_initial.update({epochs[i]: np.sqrt(np.diagonal(propagated_covariance))})


            ##############################################################
            #### LOOP FOR ESTIMATION ARCS ################################
            ##############################################################
            estimation_arc_activated = False
            if time in self.batch_start_times:

                estimation_arc_activated = True

                observation_window = self.observation_windows[estimation_arc]
                estimation_arc_duration = np.around(observation_window[1]-observation_window[0], self.decimal_places)

                # print("ESTIMATION ARC OBS WINDOW: ", observation_window)
                # print("ESTIMATION ARC DURATION: ", estimation_arc_duration)


                # print(f"Start of estimation arc {estimation_arc} at navigation arc {navigation_arc},  at {time} for {estimation_arc_duration} days")

                # Define dynamic models and select one to test the estimation on
                dynamic_model_objects = utils.get_dynamic_model_objects(time,
                                                                        estimation_arc_duration,
                                                                        custom_model_dict=None,
                                                                        get_only_first=False,
                                                                        custom_initial_state=self.custom_initial_state)
                dynamic_model = dynamic_model_objects[self.model_type][self.model_name][self.model_number]

                # print(f"USE INITIAL STATE IN ESTIMATION ARC {estimation_arc}: \n", self.custom_initial_state)
                # print(f"CHECK WITH VALUE IN DYNAMIC MODEL {dynamic_model}: \n", self.custom_initial_state==dynamic_model.custom_initial_state)
                # print(f"CHECK WITH VALUE IN TRUTH MODEL {truth_model}: \n", self.custom_initial_state_truth==truth_model.custom_initial_state)
                # print("START THE FORMER ESTIMATOR")

                # Obtain estimation results for given batch and extract results of the estimation arc
                estimation_model_objects_results = utils.get_estimation_model_results({self.model_type: {self.model_name: [dynamic_model]}},
                                                                                    get_only_first=False,
                                                                                    custom_truth_model=truth_model,
                                                                                    apriori_covariance=self.apriori_covariance,
                                                                                    initial_estimation_error=self.initial_estimation_error)
                # print("START THE LATTER ESTIMATOR")
                # test = estimation_model.EstimationModel(dynamic_model, truth_model, apriori_covariance=self.apriori_covariance, initial_estimation_error=self.initial_estimation_error).get_estimation_results()
                # print(test)
                # plt.show()

                estimation_model_objects_result = estimation_model_objects_results[self.model_type][self.model_name][0]
                estimation_output = estimation_model_objects_result[0]
                parameter_history = estimation_output.parameter_history
                final_covariance = estimation_output.covariance
                formal_errors = estimation_output.formal_errors
                best_iteration = estimation_output.best_iteration

                # print(f"Diff initial/final parameter arc {estimation_arc}: \n", parameter_history[:,best_iteration]-parameter_history[:,0])

                epochs, state_history_final, dependent_variables_history_final, state_transition_history_matrix_final = \
                    Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(dynamic_model,
                                                                                                    custom_initial_state=parameter_history[:,best_iteration],
                                                                                                    custom_propagation_time=estimation_arc_duration,
                                                                                                    solve_variational_equations=True)

                # print("difference updated trajectory and truth:")
                # print("Start: ", state_history_final[0,:]-state_history_truth[0,:])
                # print("Final: ", state_history_final[-1,:]-state_history_truth[-1,:])

                propagated_covariance_final = dict()
                propagated_formal_errors_final = dict()
                for i in range(len(epochs)):
                    propagated_covariance = state_transition_history_matrix_final[i] @ estimation_output.covariance @ state_transition_history_matrix_final[i].T
                    propagated_covariance_final.update({epochs[i]: propagated_covariance})
                    propagated_formal_errors_final.update({epochs[i]: np.sqrt(np.diagonal(propagated_covariance))})

                # Save the history
                full_state_history_final_dict.update(dict(zip(epochs, state_history_final)))

                estimation_arc += 1

            # Update the values for the next navigation arc
            self.custom_initial_state_truth = state_history_truth[-1,:]

            if estimation_arc_activated:
                self.custom_initial_state = state_history_final[-1,:]
                self.initial_estimation_error = state_history_final[-1,:]-state_history_truth[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_final.values()))[-1]
            else:
                # print("state history inital final entry: ", state_history_initial[-1,:])
                self.custom_initial_state = state_history_initial[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_initial.values()))[-1]

            process_noise_sigmas = np.array([1e2, 1e2, 1e2, 1e-3, 1e-3, 1e-3, 1e2, 1e2, 1e2, 1e-3, 1e-3, 1e-3])*1e-2
            process_noise = np.random.normal(scale=process_noise_sigmas, size=len(process_noise_sigmas))
            # process_noise = process_noise_sigmas
            # print(process_noise)

            if estimation_arc_activated:
                # self.custom_initial_state += process_noise
                self.apriori_covariance += np.outer(process_noise, process_noise)



            # Save histories for reading out later
            full_estimation_error_dict.update(dict(zip(epochs, state_history_initial-state_history_truth)))
            full_propagated_covariance_dict.update(propagated_covariance_initial)
            full_propagated_formal_errors_dict.update(propagated_formal_errors_initial)
            full_state_history_reference_dict.update(dict(zip(epochs, state_history_reference)))
            full_state_history_truth_dict.update(dict(zip(epochs, state_history_truth)))
            full_state_history_initial_dict.update(dict(zip(epochs, state_history_initial)))

            full_dependent_variables_history_initial.update(dict(zip(epochs, dependent_variables_history_initial)))
            full_state_transition_matrix_history_initial.update(dict(zip(epochs, state_transition_history_initial)))

            if len(state_history_reference) != len(state_history_initial):
                state_history_initial = state_history_initial[:-1,:]
            full_reference_state_deviation_dict.update(dict(zip(epochs, state_history_initial-state_history_reference)))


            # print("Dict of estimation error: ", full_estimation_error_dict)

            # print(f"end of navigation arc {navigation_arc}")

            ##############################################################
            #### STATION KEEPING #########################################
            ##############################################################

            if self.include_station_keeping:
                # print("self.times: ", self.times[navigation_arc+1])
                # print("Station keeping epochs: ", self.custom_station_keeping_epochs)
                if self.times[navigation_arc+1] in np.round(self.station_keeping_epochs, self.decimal_places):


                    params = [0, self.target_point_epochs]
                    dynamic_model.simulation_start_epoch_MJD = self.times[navigation_arc+1]
                    # print("In simulator: ", dynamic_model.simulation_start_epoch_MJD)
                    station_keeping = StationKeeping.StationKeeping(dynamic_model, custom_initial_state=self.custom_initial_state, custom_propagation_time=max(params[1]), step_size=self.step_size)
                    delta_v = station_keeping.get_corrected_state_vector(cut_off_epoch=params[0], correction_epoch=params[0], target_point_epochs=params[1])

                    # Generate random noise to simulate station-keeping errors
                    delta_v_noise = np.random.normal(loc=0, scale=2e-10*np.abs(delta_v), size=delta_v.shape)
                    self.custom_initial_state[9:12] += delta_v
                    self.custom_initial_state_truth[9:12] += delta_v + delta_v_noise

                    print(f"Correction at {self.times[navigation_arc+1]}: \n", delta_v, np.linalg.norm(delta_v))

                    delta_v_dict.update({self.times[navigation_arc+1]: delta_v})

                    # delta_v_uncertainty = np.zeros((12,))
                    # delta_v_uncertainty[9:12] = delta_v_noise
                    # self.apriori_covariance += np.outer(delta_v_uncertainty, delta_v_uncertainty)

            # print("CURRENT LENGTH OF DICT: ", len(full_state_history_initial_dict.keys()))
            if navigation_arc<len(self.times)-2:
                navigation_arc += 1
            else:
                print("End navigation simulation ======================")
                break

        return full_estimation_error_dict, full_reference_state_deviation_dict, full_propagated_covariance_dict, full_propagated_formal_errors_dict,\
               full_state_history_reference_dict, full_state_history_truth_dict, full_state_history_initial_dict, full_state_history_final_dict, delta_v_dict,\
               full_dependent_variables_history_initial, full_state_transition_matrix_history_initial