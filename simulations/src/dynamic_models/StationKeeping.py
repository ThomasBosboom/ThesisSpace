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

        # fig = plt.figure()
        # # print(np.shape(state_deviation_history))
        # # plt.plot(state_deviation_history[:,0:3])
        # plt.plot(state_deviation_history[:,6:9])
        # # plt.plot(state_history[:,6:9])
        # # plt.plot(reference_state_history[:,6:9])
        # plt.show()

        i_tc = int(cut_off_epoch/self.step_size)
        i_tv = int(correction_epoch/self.step_size)
        i_ti = int(target_point_epoch/self.step_size)
        dr_tc = state_deviation_history[i_tv,6:9]
        dv_tc = state_deviation_history[i_tv,9:12]
        dr_ti = state_deviation_history[i_ti,6:9]

        Phi = state_transition_matrix_history
        Phi_tcti = np.linalg.inv(Phi[i_tc]) @ Phi[i_ti]
        Phi_tvti = np.linalg.inv(Phi[i_tv]) @ Phi[i_ti]
        Phi_tvti_rr = Phi_tvti[6:9,6:9]
        Phi_tvti_rv = Phi_tvti[6:9,9:12]
        Phi_tcti_rr = Phi_tcti[6:9,6:9]
        Phi_tcti_rv = Phi_tcti[6:9,9:12]

        R_i = 1e-2*np.eye(3)
        Q = 1e-1*np.eye(3)

        A = -np.linalg.inv(np.add((Q.T+Q), Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tvti_rv))
        alpha_i = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rr
        beta_i  = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rv

        delta_v = A @ (alpha_i @ dr_tc + beta_i @ dv_tc)

        # print(A)
        # print(alpha_i)
        # print(beta_i)
        # print("=======")
        # print(dr_tc)
        # print(alpha_i @ dr_tc)
        # print(dv_tc)
        # print(beta_i @ dv_tc)
        # print(delta_v, np.linalg.norm(delta_v))

        delta_v = -np.linalg.inv(Phi_tvti_rv) @ Phi_tvti_rr @ dr_tc - dv_tc

        state_history[i_tv, 9:12] += delta_v





        # return state_history[i_tv, :]
        return delta_v



# dynamic_model_objects = utils.get_dynamic_model_objects(60390,
#                                                         1,
#                                                         package_dict=None,
#                                                         get_only_first=False,
#                                                         custom_initial_state=None)

# dynamic_model_object = dynamic_model_objects["high_fidelity"]["point_mass"][0]


# estimated_parameter_vector = np.array([-2.80124757e+08,  2.53324773e+08,  1.46943725e+08, -1.61475000e+03,
#  -2.23501900e+03, -2.97165000e+02, -3.10469279e+08,  2.49476176e+08,
#   1.74974083e+08, -9.93405005e+02, -7.66336485e+02, -5.24990115e+02])
# # estimated_parameter_vector = np.array([-3.34034721e+08,  1.91822727e+08,  1.11599990e+08, -1.22117098e+02,
# #  -7.02102813e+02, -9.74268692e+02, -3.83003556e+08,  1.80616085e+08,
# #   1.22244565e+08, -6.84093312e+02, -8.16786564e+02, -6.68493579e+02])
# station_keeping = StationKeeping(dynamic_model_object, estimated_parameter_vector=estimated_parameter_vector, custom_propagation_time=28, step_size=0.01)

# lists = [[0, 20]]
# for list1 in lists:
#     print(list1)
#     corrected_state_vector = station_keeping.get_corrected_state_vector(correction_epoch=list1[0], target_point_epoch=list1[1], cut_off_epoch=0)
#     print("corrected_state_vector:", corrected_state_vector)








