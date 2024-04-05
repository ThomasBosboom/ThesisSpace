# Standard
import os
import sys
import numpy as np

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)


class NoiseDataClass:

    def __init__(self):

        np.random.seed(0)

        # Measurement noise
        self.noise_range = 2.98 #102.44/50

        # Station keeping noise
        self.relative_station_keeping_error = 1e-2

        # Initial orbit uncertainties
        self.initial_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])*1e-2
        self.apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
        self.orbit_insertion_error = np.array([1e1, 1e1, 1e1, 1e-1, 1e-1, 1e-1, 1e1, 1e1, 1e1, 1e-1, 1e-1, 1e-1])*1e-1
        # self.initial_estimation_error = np.random.normal(loc=0, scale=np.abs(np.sqrt(np.diag(self.apriori_covariance))), size=np.abs(np.sqrt(np.diag(self.apriori_covariance))).shape)
        # self.orbit_insertion_error = np.random.normal(loc=0, scale=np.abs(self.orbit_insertion_error), size=self.orbit_insertion_error.shape)



