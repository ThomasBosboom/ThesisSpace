# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

# Own
from tests import utils
import reference_data, Interpolator


class StationKeeping:

    def __init__(self, dynamic_model_object, custom_initial_state=None, custom_propagation_time=14, step_size=1e-4):

        print("START STATION KEEPING ======")

        self.dynamic_model_object = dynamic_model_object
        self.dynamic_model_object.custom_initial_state = custom_initial_state
        self.dynamic_model_object.custom_propagation_time = custom_propagation_time
        self.step_size = step_size

        # print("custom_initial_state: ", self.dynamic_model_object.custom_initial_state)


    def get_corrected_state_vector(self, correction_epoch, target_point_epochs, cut_off_epoch=0):

        # Propagate the results of the dynamic model to generate target points


        # print("INITIAL TIME IN STATTION KEEPING: ", self.dynamic_model_object.simulation_start_epoch_MJD)
        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            Interpolator.Interpolator(epoch_in_MJD=True, step_size=self.step_size).get_propagation_results(self.dynamic_model_object,
                                                                                                            custom_initial_state=self.dynamic_model_object.custom_initial_state,
                                                                                                            custom_propagation_time=self.dynamic_model_object.custom_propagation_time)

        # Get the reference orbit states
        reference_state_history = list()
        for body in self.dynamic_model_object.bodies_to_propagate:
            reference_state_history.append(reference_data.get_reference_state_history(self.dynamic_model_object.simulation_start_epoch_MJD,
                                                                                    self.dynamic_model_object.custom_propagation_time,
                                                                                    satellite=body,
                                                                                    step_size=self.step_size,
                                                                                    get_full_history=True))

        reference_state_history = np.concatenate(reference_state_history, axis=1)

        # Perform target point method algorithm
        state_deviation_history = state_history - reference_state_history

        # print("step size: ", self.step_size)
        # print("start epoch: \n", self.dynamic_model_object.simulation_start_epoch_MJD)
        # print("propagation time: \n", self.dynamic_model_object.custom_propagation_time)
        # print("custom_initial_state:", self.dynamic_model_object.custom_initial_state)
        # print("reference state at start epoch: \n" , reference_state_history[0, :])
        # print("state at start epoch: \n" , state_history[0, :])
        # print("deviation from reference: \n" , state_deviation_history[0, :])
        # print("initial abs diff deviation: \n LPF: " , np.linalg.norm(state_deviation_history[0, 0:3]), "LUMIO: ", np.linalg.norm(state_deviation_history[0, 6:9]))

        R_i = 1e-2*np.eye(3)
        Q = 1e-1*np.eye(3)
        # print(R_i.T + R_i, Q.T + Q)
        Phi = state_transition_matrix_history

        final_sum = np.empty((3,))
        total_sum = np.empty((3,3))
        for target_point_epoch in target_point_epochs:

            # Define the indexes
            i_tc = int(cut_off_epoch/self.step_size)
            i_tv = int(correction_epoch/self.step_size)
            i_ti = int(target_point_epoch/self.step_size)
            dr_tc = state_deviation_history[i_tc,6:9]
            dv_tc = state_deviation_history[i_tc,9:12]
            dr_ti = state_deviation_history[i_ti,6:9]

            # Define the STMs at the right epochs
            Phi_tcti = Phi[i_ti] @ np.linalg.inv(Phi[i_tc])
            Phi_tvti = Phi[i_ti] @ np.linalg.inv(Phi[i_tv])
            Phi_tvti_rr = Phi_tvti[6:9,6:9]
            Phi_tvti_rv = Phi_tvti[6:9,9:12]
            Phi_tcti_rr = Phi_tcti[6:9,6:9]
            Phi_tcti_rv = Phi_tcti[6:9,9:12]

            # Calculate the TPM equations using the STMs for a given target point
            total_sum = np.add(total_sum, Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tvti_rv)

            alpha_i = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rr
            beta_i  = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rv

            final_sum = np.add(final_sum, alpha_i @ dr_tc + beta_i @ dv_tc)

        A = -np.linalg.inv(np.add((Q.T+Q), total_sum))

        delta_v = A @ final_sum

        # print(r"delta_v: ", delta_v, "Norm: ", np.linalg.norm(delta_v))

        # delta_v = -np.linalg.inv(Phi_tvti_rv) @ Phi_tvti_rr @ dr_tc - dv_tc

        # print("delta_v:", delta_v)

        state_history[i_tv, 9:12] += delta_v

        print("END STATION KEEPING ======")

        return delta_v



# dynamic_model_objects = utils.get_dynamic_model_objects(60391,
#                                                         1,
#                                                         custom_model_dict=None,
#                                                         get_only_first=False,
#                                                         custom_initial_state=None)

# dynamic_model_object = dynamic_model_objects["HF"]["PMSRP"][0]
# # dynamic_model_object = dynamic_model_objects["LF"]["CRTBP"][0]

# # custom_initial_state = np.array([-3.34034638e+08,  1.91822560e+08,  1.11600187e+08, -1.22100520e+02,
# #                                  -7.02130739e+02, -9.74257591e+02, -3.83004013e+08,  1.80617292e+08,
# #                                   1.22243914e+08, -6.84094596e+02, -8.16779163e+02, -6.68497305e+02])

# import time

# # lists = [[7, [21]], [7, [21, 28]]]
# lists = [[0, [0]]]
# for i, list1 in enumerate(lists):
#     print(list1)
#     start_time = time.time()
#     station_keeping = StationKeeping(dynamic_model_object, custom_initial_state=None, custom_propagation_time=max(list1[1]), step_size=0.01)
#     delta_v = station_keeping.get_corrected_state_vector(cut_off_epoch=list1[0], correction_epoch=list1[0], target_point_epochs=list1[1])
#     print("delta_v:", delta_v)
#     lists[i].extend([delta_v, time.time()-start_time])
# print(lists)
# plt.show()








#