# Own
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class EstimationModel:

    def __init__(self):

        # Defining basis for observations
        self.bias_range = 10.0
        self.bias_doppler = 0.001
        self.noise_range = 2.98
        self.noise_doppler = 0.00097
        self.observation_step_size_range = 600
        self.observation_step_size_doppler = 600