# Tudatpy imports
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel import constants


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
        self.simulation_end_epoch   = time_conversion.julian_day_to_seconds_since_epoch(\
            time_conversion.modified_julian_day_to_julian_day(self.simulation_start_epoch_MJD+self.propagation_time))

        # Define constant environment settings
        self.bodies_to_create = [self.name_primary, self.name_secondary]
        self.central_bodies = [self.name_primary, self.name_primary]
        self.bodies_to_propagate = [self.name_ELO, self.name_LPO]
        self.global_frame_origin = self.name_primary
        self.global_frame_orientation = 'J2000'
        self.bodies_mass = [280, 22.3]
        self.bodies_reference_area_radiation = [3.0, 0.41064]
        self.bodies_radiation_pressure_coefficient = [1.8, 1.08]
        self.gravitational_parameter_primary = spice.get_body_gravitational_parameter(self.name_primary)
        self.gravitational_parameter_secondary = spice.get_body_gravitational_parameter(self.name_secondary)
        self.mu = self.gravitational_parameter_secondary/(self.gravitational_parameter_primary+self.gravitational_parameter_secondary)

        # Define integrator settings
        self.use_variable_step_size_integrator = True
        self.current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_45
        self.current_tolerance = 1e-15*constants.JULIAN_DAY
        self.initial_time_step = 1e-3*constants.JULIAN_DAY


    def set_environment_settings(self):
        pass


    def set_acceleration_settings(self):
        pass


    def set_initial_state(self):
        pass


    def set_integration_settings(self):
        pass


    def set_dependent_variables_to_save(self):
        pass


    def set_termination_settings(self):
        pass


    def set_propagator_settings(self):
        pass


    def get_propagated_orbit(self):
        pass