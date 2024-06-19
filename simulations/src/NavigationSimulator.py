# Standard
import os
import sys
import numpy as np
import copy
import tracemalloc
from memory_profiler import profile

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Own
from tests import utils
import ReferenceData, Interpolator, StationKeeping, EstimationModel
from NavigationSimulatorBase import NavigationSimulatorBase
from tudatpy.kernel import constants

import gc

class NavigationSimulator(NavigationSimulatorBase):

    def __init__(self, **kwargs):
        super().__init__()

        self._initial_attrs = {**self.__dict__}
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self._initial_attrs.update({key: value})

        self.interpolator = Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size)
        self.reference_data = ReferenceData.ReferenceData(self.interpolator)

        self._initial_attrs.update({"interpolator": self.interpolator, "reference_data": self.reference_data})


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


    def reset_attributes(self):

        for attr, value in self._initial_attrs.items():
            setattr(self, attr, value)
        for key, value in vars(self).copy().items():
            if key != "_initial_attrs":
                if key not in self._initial_attrs.keys():
                    delattr(self, key)

        # gc.collect()


    def perform_navigation(self, observation_windows, seed=0):

        # tracemalloc.start()

        # snapshot1 = tracemalloc.take_snapshot()

        print("step size here: ", self.step_size)

        self.seed = seed
        rng = np.random.default_rng(seed=self.seed)

        # Adjusting decimals based on the step size used
        if self.run_optimization_version:
            self.step_size = self.step_size_optimization_version
        num_str = "{:.15f}".format(self.step_size).rstrip('0')
        decimal_places = len(num_str) - num_str.index('.') - 1

        # Observation window settings
        self.mission_start_epoch = observation_windows[0][0]
        self.observation_windows = observation_windows
        self.observation_windows = [(np.round(tup[0], decimal_places), np.round(tup[1], decimal_places)) for tup in observation_windows]
        self.initial_station_keeping_epochs = [np.round(observation_window[1], decimal_places) for observation_window in observation_windows]
        self.station_keeping_epochs = []

        # Managing the time vector to define all arcs
        self.batch_start_times = np.array([t[0] for t in self.observation_windows])
        times = list([self.observation_windows[0][0]] + [item for sublist in self.observation_windows for item in sublist] + [self.observation_windows[0][0]] + self.initial_station_keeping_epochs)
        times = list(set(times))
        times = sorted(times)

        # print(self.delta_v_min)

        self.navigation_arc_durations = np.round(np.diff(times), decimal_places)
        self.estimation_arc_durations = np.round(np.array([tup[1] - tup[0] for tup in self.observation_windows]), decimal_places)

        # print("=========================")
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
        for t, time in enumerate(times):

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
                state_history_reference.append(self.reference_data.get_reference_state_history(
                    time, navigation_arc_duration, satellite=body, get_full_history=True))
            state_history_reference = np.concatenate(state_history_reference, axis=1)

            if navigation_arc == 0:
                self.custom_initial_state_truth = truth_model.initial_state + self.orbit_insertion_error
                self.custom_initial_state = self.custom_initial_state_truth + self.initial_estimation_error

            dynamic_model.custom_initial_state = self.custom_initial_state
            truth_model.custom_initial_state = self.custom_initial_state_truth

            epochs, state_history_estimated, dependent_variables_history_estimated, state_transition_matrix_history_estimated = \
                self.interpolator.get_propagation_results(dynamic_model, solve_variational_equations=True)

            epochs, state_history_truth, dependent_variables_history_truth = \
                self.interpolator.get_propagation_results(truth_model, solve_variational_equations=False)

            if self.propagate_dynamics_linearly:
                state_history_estimated = state_history_truth + np.dot(state_transition_matrix_history_estimated, self.custom_initial_state-self.custom_initial_state_truth)

            # Save the propagated histories of the uncertainties
            if self.apriori_covariance is not None:
                propagated_covariance_estimated = dict()
                propagated_formal_errors_estimated = dict()
                for i in range(len(epochs)):
                    propagated_covariance = state_transition_matrix_history_estimated[i] @ self.apriori_covariance @ state_transition_matrix_history_estimated[i].T
                    propagated_covariance_estimated.update({epochs[i]: propagated_covariance})
                    propagated_formal_errors_estimated.update({epochs[i]: np.sqrt(np.diagonal(propagated_covariance))})


            ##############################################################
            #### LOOP FOR ESTIMATION ARCS ################################
            ##############################################################
            estimation_arc_activated = False
            if time in self.batch_start_times:

                estimation_arc_activated = True
                estimation_arc_duration = self.estimation_arc_durations[estimation_arc]

                maximum_iterations = copy.deepcopy(self.maximum_iterations)
                if estimation_arc == 0:
                    self.maximum_iterations = self.maximum_iterations_first_arc

                estimation_model = EstimationModel.EstimationModel(dynamic_model, truth_model, **vars(self))
                estimation_model_result = estimation_model.get_estimation_results()
                estimation_output = estimation_model_result.estimation_output
                parameter_history = estimation_output.parameter_history
                final_covariance = estimation_output.covariance
                best_iteration = estimation_output.best_iteration
                residual_history = estimation_output.residual_history

                dynamic_model.custom_initial_state = parameter_history[:, best_iteration]
                epochs_final, state_history_final, dependent_variables_history_final, state_transition_history_matrix_final = \
                    self.interpolator.get_propagation_results(dynamic_model, solve_variational_equations=True)

                if self.propagate_dynamics_linearly:
                    state_history_final = state_history_truth + np.dot(state_transition_history_matrix_final, dynamic_model.custom_initial_state-state_history_truth[0,:])

                if self.apriori_covariance is not None:
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

            else:
                self.custom_initial_state = state_history_estimated[-1,:]
                self.apriori_covariance = np.stack(list(propagated_covariance_estimated.values()))[-1]

            # Save histories for reading out later
            self.full_propagated_covariance_dict.update(propagated_covariance_estimated)

            if not self.run_optimization_version:
                self.full_reference_state_deviation_dict.update(dict(zip(epochs, state_history_truth-state_history_reference)))
                self.full_estimation_error_dict.update(dict(zip(epochs, state_history_estimated-state_history_truth)))
                self.full_propagated_formal_errors_dict.update(propagated_formal_errors_estimated)
                self.full_state_history_reference_dict.update(dict(zip(epochs, state_history_reference)))
                self.full_state_history_truth_dict.update(dict(zip(epochs, state_history_truth)))
                self.full_state_history_estimated_dict.update(dict(zip(epochs, state_history_estimated)))
                self.full_dependent_variables_history_estimated.update(dict(zip(epochs, dependent_variables_history_estimated)))
                self.full_state_transition_matrix_history_estimated.update(dict(zip(epochs, state_transition_matrix_history_estimated)))


            ##############################################################
            #### STATION KEEPING #########################################
            ##############################################################

            if self.include_station_keeping:
                if times[navigation_arc+1] in self.initial_station_keeping_epochs:

                    params = [0, self.target_point_epochs]

                    dynamic_model_objects = utils.get_dynamic_model_objects(times[navigation_arc+1], max(params[1]),
                                                                            custom_initial_state=self.custom_initial_state,
                                                                            custom_model_dict={self.model_type: [self.model_name]},
                                                                            custom_model_list=[self.model_number])
                    dynamic_model_station_keeping = dynamic_model_objects[self.model_type][self.model_name][0]

                    station_keeping = StationKeeping.StationKeeping(dynamic_model_station_keeping,
                                                                    self.reference_data,
                                                                    self.interpolator,
                                                                    )
                    delta_v, dispersion = station_keeping.get_corrected_state_vector(cut_off_epoch=params[0],
                                                                                    correction_epoch=params[0],
                                                                                    target_point_epochs=params[1])

                    # Generate random noise to simulate station-keeping errors
                    if self.model_type == "HF":

                        delta_v_noise_sigma = np.abs(self.station_keeping_error*delta_v)
                        delta_v_noise = rng.normal(loc=0, scale=delta_v_noise_sigma, size=delta_v.shape)

                        # od_update = np.linalg.norm(state_history_estimated[0, 6:9]-state_history_truth[0, 6:9])-np.linalg.norm(self.initial_estimation_error[6:9])

                        if np.linalg.norm(delta_v) >= self.delta_v_min:
                            self.custom_initial_state[9:12] += delta_v
                            self.custom_initial_state_truth[9:12] += delta_v + delta_v_noise

                            delta_v_noise_covariance = np.zeros(12)
                            delta_v_noise_covariance[9:12] = delta_v_noise_sigma
                            self.apriori_covariance += np.outer(delta_v_noise_covariance, delta_v_noise_covariance)

                            self.station_keeping_epochs.append(times[navigation_arc+1])
                            self.delta_v_dict.update({times[navigation_arc+1]: delta_v})

                            if self.show_corrections_in_terminal:
                                print(f"Correction at {times[navigation_arc+1]}: ", \
                                    np.sqrt(np.mean(residual_history[:, best_iteration]**2)), "m   ", \
                                    np.linalg.norm(delta_v), "m/s   ", \
                                    np.linalg.norm(self.initial_estimation_error[6:9]), "m   ", \
                                    # np.linalg.norm(dynamic_model.custom_initial_state[6:9]-state_history_truth[0,:][6:9]), "m   ", \
                                    np.linalg.norm(dispersion[:3]), "m"
                                )


            if navigation_arc < len(times)-2:
                navigation_arc += 1
            else:
                break

        return NavigationOutput(self)


class NavigationOutput():

    def __init__(self, navigation_simulator, **required_attributes):
        self.navigation_simulator = navigation_simulator


if __name__ == "__main__":

    # tracemalloc.start()
    # snapshot1 = tracemalloc.take_snapshot()
    navigation_simulator = NavigationSimulator(run_optimization_version=True, step_size=0.5)
    #     # Take another snapshot after the function call

    # # Compare the two snapshots
    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    # print("[ Top 5 differences ]")
    # for stat in top_stats[:5]:
    #     print(stat)
    # total_memory = sum(stat.size for stat in top_stats)
    # print(f"Total memory used after iteration: {total_memory / (1024 ** 2):.2f} MB")

    observation_windows = [(60390, 60391.0), (60394.0, 60395.0), (60398.0, 60399.0), (60402.0, 60403.0), (60406.0, 60407.0), (60410.0, 60411.0), (60414.0, 60415.0)]
    # observation_windows = [(60390, 60391.0), (60394.0, 60395.0)]

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    cost_list = []
    for i in [0, 1]:

        navigation_output = navigation_simulator.perform_navigation(observation_windows, seed=i)
        navigation_simulator = navigation_output.navigation_simulator
        full_propagated_covariance_dict = navigation_simulator.full_propagated_covariance_dict
        print(len(full_propagated_covariance_dict))

        delta_v_dict = navigation_simulator.delta_v_dict
        delta_v_epochs = np.stack(list(delta_v_dict.keys()))
        delta_v_history = np.stack(list(delta_v_dict.values()))
        delta_v = sum(np.linalg.norm(value) for key, value in delta_v_dict.items() if key > navigation_simulator.mission_start_epoch+0)

        cost_list.append(delta_v)
        print(cost_list)

        # navigation_simulator.reset_attributes()

        # # Take another snapshot after the function call
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        print("[ Top 5 differences ]")
        for stat in top_stats[:10]:
            print(stat)
        total_memory = sum(stat.size for stat in top_stats)
        print(f"Total memory used after iteration: {total_memory / (1024 ** 2):.2f} MB")