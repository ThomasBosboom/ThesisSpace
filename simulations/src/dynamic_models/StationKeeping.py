# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Third party

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from tests import utils
from src.dynamic_models import validation
from src.dynamic_models import Interpolator


class StationKeeping:

    def __init__(self, dynamic_model_object, estimated_parameter_vector=None, custom_propagation_time=14, step_size=0.001):

        self.dynamic_model_object = dynamic_model_object
        self.dynamic_model_object.propagation_time = custom_propagation_time
        self.simulation_start_epoch = self.dynamic_model_object.simulation_start_epoch
        self.simulation_end_epoch = self.dynamic_model_object.simulation_end_epoch
        self.propagation_time = self.dynamic_model_object.propagation_time

        self.estimated_parameter_vector = estimated_parameter_vector
        self.custom_propagation_time = custom_propagation_time
        self.step_size = step_size


    def get_corrected_state_vector(self, correction_epoch, target_point_epoch, cut_off_epoch=0):

        # Propagate the results of the dynamic model to generate target points
        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(epoch_in_MJD=False, step_size=self.step_size).get_propagation_results(self.dynamic_model_object,
                                                                                                            estimated_parameter_vector=self.estimated_parameter_vector,
                                                                                                            custom_propagation_time=self.custom_propagation_time)

        # Get the reference orbit states
        reference_state_history = list()
        for body in self.dynamic_model_object.bodies_to_propagate:
            reference_state_history.append(validation.get_reference_state_history(self.dynamic_model_object.simulation_start_epoch_MJD,
                                                                                    self.propagation_time,
                                                                                    satellite=body,
                                                                                    step_size=self.step_size,
                                                                                    get_full_history=True))

        reference_state_history = np.concatenate(reference_state_history, axis=1)

        # Perform target point method algorithm
        state_deviation_history = state_history - reference_state_history

        print(np.shape(state_deviation_history))
        # plt.plot(state_deviation_history[:,0:3])
        plt.plot(state_deviation_history[:,6:9])
        # plt.plot(state_history[:,6:9])
        # plt.plot(reference_state_history[:,6:9])
        plt.show()

        cut_off_epoch_index = int(cut_off_epoch/self.step_size)
        correction_epoch_index = int(correction_epoch/self.step_size)
        target_point_epoch_index = int(target_point_epoch/self.step_size)
        dr_tc = state_deviation_history[correction_epoch_index,6:9]
        dv_tc = state_deviation_history[correction_epoch_index,9:12]
        dr_ti = state_deviation_history[target_point_epoch_index,6:9]

        Phi = state_transition_matrix_history
        # Phi_tcti = np.linalg.inv(Phi[target_point_epoch_index]) @ Phi[cut_off_epoch_index]
        # Phi_tvti = np.linalg.inv(Phi[target_point_epoch_index]) @ Phi[cut_off_epoch_index]
        Phi_tcti = np.linalg.inv(Phi[cut_off_epoch_index]) @ Phi[target_point_epoch_index]
        Phi_tvti = np.linalg.inv(Phi[correction_epoch_index]) @ Phi[target_point_epoch_index]
        Phi_tvti_rr = Phi_tvti[6:9,6:9]
        Phi_tvti_rv = Phi_tvti[6:9,9:12]
        Phi_tcti_rr = Phi_tcti[6:9,6:9]
        Phi_tcti_rv = Phi_tcti[6:9,9:12]

        print(Phi_tvti_rr)
        print(Phi_tvti_rv)
        print(Phi_tcti_rr)
        print(Phi_tcti_rv)

        R_i = 1e-2*np.eye(3)
        Q = 1e-1*np.eye(3)

        A = -np.linalg.inv(np.add((Q.T+Q), Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tvti_rv))
        alpha_i = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rr
        beta_i  = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rv

        delta_v = A @ (alpha_i @ dr_tc + beta_i @ dv_tc)

        print(A)
        print(alpha_i)
        print(beta_i)
        print("=======")
        print(dr_tc)
        print(alpha_i @ dr_tc)
        print(dv_tc)
        print(beta_i @ dv_tc)
        print(delta_v, np.linalg.norm(delta_v))

        corrected_initial_state = state_history[correction_epoch_index, 9:12] + delta_v

        return corrected_initial_state



dynamic_model_objects = utils.get_dynamic_model_objects(60390,
                                                        1,
                                                        package_dict=None,
                                                        get_only_first=False,
                                                        custom_initial_state=None)

dynamic_model_object = dynamic_model_objects["high_fidelity"]["point_mass"][0]


station_keeping = StationKeeping(dynamic_model_object, custom_propagation_time=14, step_size=0.01)

lists = [[0, 7]]
for list1 in lists:
    print(list1)
    corrected_state_vector = station_keeping.get_corrected_state_vector(correction_epoch=list1[0], target_point_epoch=list1[1], cut_off_epoch=0)


