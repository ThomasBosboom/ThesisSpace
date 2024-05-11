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
        self.state_noise_compensation = 1e-25
        self.apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
        self.orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0
        self.initial_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
        self.propagate_dynamics_linearly = True

        # Estimation settings
        self.bias_range = 0
        self.noise_range = 1
        self.observation_step_size_range = 600
        self.retransmission_delay = 0.5e-10
        self.integration_time = 1
        self.time_drift_bias = 6.9e-20
        self.maximum_iterations = 5
        self.margin = 120
        self.redirect_out = True




