# Standard
import os
import sys
import numpy as np

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

class StationKeeping:

    def __init__(self, dynamic_model, reference_data, interpolator):

        self.dynamic_model = dynamic_model
        self.reference_data = reference_data
        self.interpolator = interpolator


    def get_corrected_state_vector(self, correction_epoch, target_point_epochs, cut_off_epoch):

        # print("start stationkeeping: ", self.dynamic_model.simulation_start_epoch_MJD)
        # Get the reference orbit states
        reference_state_history = list()
        for body in self.dynamic_model.bodies_to_propagate:
            reference_state_history.append(self.reference_data.get_reference_state_history(self.dynamic_model.simulation_start_epoch_MJD,
                                                                                            self.dynamic_model.propagation_time,
                                                                                            satellite=body,
                                                                                            get_full_history=True))
        reference_state_history = np.concatenate(reference_state_history, axis=1)
        # print("reference_state_history: ", np.shape(reference_state_history))

        # Propagate the results of the dynamic model to generate target points
        epochs, state_history, dependent_variables_history, state_transition_matrix_history = \
            self.interpolator.get_propagation_results(self.dynamic_model,
                                                custom_initial_state=self.dynamic_model.custom_initial_state,
                                                custom_propagation_time=self.dynamic_model.propagation_time)

        # print("Epoch in station_keeping: ", epochs[0], epochs[-1], len(epochs))

        # Perform target point method algorithm
        state_deviation_history = state_history - reference_state_history

        # print("StationKeeping: ", \
        #     epochs[0], epochs[-1], \
        #     np.linalg.norm(state_history[0, 6:9]), "m   ", \
        #     np.linalg.norm(reference_state_history[0, 6:9]), "m   ", \
        #     np.linalg.norm(state_deviation_history[0, 6:9]), "m   "
        # )

        R_i = 1e-2*np.eye(3)
        Q = 1e-1*np.eye(3)
        # R_i = 1e-0*np.eye(3)
        # Q = 3.9e11*np.eye(3)
        # print(R_i.T + R_i, Q.T + Q)
        Phi = state_transition_matrix_history

        final_sum = np.zeros((3,))
        total_sum = np.zeros((3,3))

        cut_off_epoch = epochs[0] + cut_off_epoch
        correction_epoch = epochs[0] + correction_epoch
        for target_point_epoch in target_point_epochs:

            # Define the indexes
            target_point_epoch = epochs[0] + target_point_epoch
            i_tc = self.interpolator.get_closest_index(epochs[0], epochs[-1], cut_off_epoch)
            i_tv = self.interpolator.get_closest_index(epochs[0], epochs[-1], correction_epoch)
            i_ti = self.interpolator.get_closest_index(epochs[0], epochs[-1], target_point_epoch)

            # Extract the dispersion elements
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
            total_sum += Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tvti_rv

            alpha_i = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rr
            beta_i  = Phi_tvti_rv.T @ (R_i.T + R_i) @ Phi_tcti_rv

            final_sum += alpha_i @ dr_tc + beta_i @ dv_tc

        A = -np.linalg.inv(np.add((Q.T+Q), total_sum))

        delta_v = A @ final_sum
        dispersion = np.concatenate((dr_tc, dv_tc))

        return delta_v, dispersion


if __name__ == "__main__":

    import time
    import Interpolator, ReferenceData
    from tests import utils

    dynamic_models = utils.get_dynamic_model_objects(60390,
                                                    4,
                                                    custom_model_dict=None,
                                                    get_only_first=False,
                                                    custom_initial_state=None)

    dynamic_model = dynamic_models["HF"]["PMSRP"][0]

    interpolator = Interpolator.Interpolator(epoch_in_MJD=True, step_size=0.001)
    reference_data = ReferenceData.ReferenceData(interpolator)


    station_keeping = StationKeeping(dynamic_model, reference_data, interpolator)
    delta_v, dispersion = station_keeping.get_corrected_state_vector(0.5, [35], 0.5)
    print("delta_v:", np.linalg.norm(delta_v), np.linalg.norm(dispersion[:3]))

    delta_v, dispersion = station_keeping.get_corrected_state_vector(0.5, [35, 42], 0.5)
    print("delta_v:", np.linalg.norm(delta_v), np.linalg.norm(dispersion[:3]))








