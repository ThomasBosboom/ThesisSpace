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
        self.observation_step_size_range = 600
        self.range_noise = 2.98
        self.delta_v_min = 0.00
        self.include_station_keeping = True
        self.station_keeping_error = 0
        self.state_noise_compensation = 1e-18
        self.apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
        self.initial_estimation_error_sigmas = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
        self.orbit_insertion_error_sigmas = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])
        # self.initial_estimation_error = np.random.normal(loc=0, scale=self.initial_estimation_error_sigmas, size=self.initial_estimation_error_sigmas.shape)
        # self.orbit_insertion_error = np.random.normal(loc=0, scale=self.orbit_insertion_error_sigmas, size=self.orbit_insertion_error_sigmas.shape)

        self.orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])
        self.initial_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])




