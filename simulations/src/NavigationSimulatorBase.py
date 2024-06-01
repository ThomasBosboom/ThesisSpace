#################################################################################
###### Data class for base setting navigation routine  ##########################
#################################################################################

import numpy as np

class NavigationSimulatorBase():

    def __init__(self):

        self.mission_start_epoch = 60390
        self.custom_initial_state = None
        self.custom_initial_state_truth = None
        self.model_type, self.model_name, self.model_number = "HF", "PMSRP", 0
        self.model_type_truth, self.model_name_truth, self.model_number_truth = "HF", "PMSRP", 0
        self.step_size = 1e-2
        self.target_point_epochs = [3]
        self.delta_v_min = 0.00
        self.include_station_keeping = True
        self.station_keeping_error = 0.00
        self.state_noise_compensation_lpf = 1e-21 # 1e-21
        self.state_noise_compensation_lumio = 1e-18 # 1e-18
        self.lpf_estimation_error = np.array([5e1, 5e1, 5e1, 1e-4, 1e-4, 1e-4])*10
        self.lumio_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
        self.initial_estimation_error = np.concatenate((self.lpf_estimation_error, self.lumio_estimation_error))
        self.apriori_covariance = np.diag(self.initial_estimation_error**2)
        self.orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0
        self.propagate_dynamics_linearly = False

        # Estimation settings
        self.bias_range = 0
        self.noise_range = 1
        self.observation_step_size_range = 300
        self.total_observation_count = None
        self.retransmission_delay = 0
        self.integration_time = 1e-20
        self.time_drift_bias = 6.9e-20
        self.maximum_iterations = 5
        self.maximum_iterations_first_arc = 10
        self.margin = 120
        self.redirect_out = True
