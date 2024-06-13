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
        self.state_noise_compensation_lpf = 1e-22
        self.state_noise_compensation_lumio = 1e-20
        self.propagate_dynamics_linearly = False

        # Initial error settings
        self.lpf_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
        self.lumio_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
        self.initial_estimation_error = np.concatenate((self.lpf_estimation_error, self.lumio_estimation_error))
        # self.apriori_covariance = np.diag(np.array([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2)
        self.apriori_covariance = np.diag(self.initial_estimation_error**2)
        self.orbit_insertion_error = np.array([0, 0, 0, 0, 0, 0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])*0

        # Estimation settings
        self.bias = 0
        self.noise = 2.98
        self.observation_interval = 300
        self.total_observation_count = None
        self.retransmission_delay = 0
        self.integration_time = 1e-20
        self.time_drift_bias = 6.9e-20
        self.maximum_iterations = 5
        self.maximum_iterations_first_arc = 10
        self.margin = 0
        self.redirect_out = True
        self.show_corrections_in_terminal = True

        self.run_optimization_version = False
        self.step_size_optimization_version = 0.01
