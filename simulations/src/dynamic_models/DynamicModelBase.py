from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel import constants

import sys
import os
import numpy as np

file_directory = os.path.realpath(__file__)
for _ in range(4):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from src import reference_data


# Load spice kernels.
spice.load_standard_kernels()

class DynamicModelBase:

    def __init__(self, simulation_start_epoch_MJD, propagation_time):

        # Define standard naming
        self.name_primary = "Earth"
        self.name_secondary = "Moon"
        self.name_ELO = "LPF"
        self.name_LPO = "LUMIO"

        # Define timing parameters
        self.simulation_start_epoch_MJD = simulation_start_epoch_MJD
        self.propagation_time = propagation_time
        self.simulation_start_epoch = time_conversion.julian_day_to_seconds_since_epoch(\
            time_conversion.modified_julian_day_to_julian_day(self.simulation_start_epoch_MJD))
        # self.simulation_end_epoch   = time_conversion.julian_day_to_seconds_since_epoch(\
        #     time_conversion.modified_julian_day_to_julian_day(self.simulation_start_epoch_MJD+self.propagation_time))
        self.simulation_end_epoch = self.simulation_start_epoch + propagation_time*constants.JULIAN_DAY

        # Define constant environment settings
        self.global_frame_origin = self.name_primary
        self.global_frame_orientation = 'J2000'
        self.central_bodies = [self.name_primary, self.name_primary]
        self.bodies_to_create = [self.name_primary, self.name_secondary]
        self.bodies_to_propagate = [self.name_ELO, self.name_LPO]
        self.bodies_mass = [280, 22.3]
        self.bodies_reference_area_radiation = [3.0, 0.41064]
        self.bodies_radiation_pressure_coefficient = [1.8, 1.08]
        self.gravitational_parameter_primary = spice.get_body_gravitational_parameter(self.name_primary)
        self.gravitational_parameter_secondary = spice.get_body_gravitational_parameter(self.name_secondary)
        self.mu = self.gravitational_parameter_secondary/(self.gravitational_parameter_primary+self.gravitational_parameter_secondary)

        # Define integrator settings
        self.use_variable_step_size_integrator = True
        self.current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_45
        self.current_tolerance = 1e-18*constants.JULIAN_DAY
        self.initial_time_step = 1e-3*constants.JULIAN_DAY

        # Initial state based on reference orbit
        initial_state_LPF = reference_data.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_ELO)
        initial_state_LUMIO = reference_data.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_LPO)
        self.initial_state = np.concatenate((initial_state_LPF, initial_state_LUMIO))

        # Custom parameters
        self.custom_initial_state = None
        self.custom_propagation_time = None
