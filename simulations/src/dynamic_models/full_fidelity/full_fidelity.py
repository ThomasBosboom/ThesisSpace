# General imports
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from pathlib import Path

# Tudatpy imports
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup, environment_setup, estimation_setup
from tudatpy.kernel.astro import element_conversion, time_conversion
from tudatpy.kernel.interface import spice

# Define path to import src files
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from dynamic_models import validation
from DynamicModelBase import DynamicModelBase


def read_coeffs(scaled=True):
    # DLAM-1 coefficients are in GEODYN format
    # See https://earth.gsfc.nasa.gov/sites/default/files/volume3.pdf

    # Specify the file path
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    file_path = root_dir / "reference" / "lunar_albedo" / "DLAM-1.txt"

    with open(file_path) as f:
        lines = f.readlines()
    lines = [line[8:-1] for line in lines[5:]]

    cos_coeffs = np.zeros((16, 16))
    sin_coeffs = np.zeros((16, 16))

    for line in lines:
        l = int(line[:2])
        m = int(line[2:4])

        cos_coeffs[l, m] = float(line[20:36].replace("D", "E"))
        sin_coeffs[l, m] = float(line[36:].replace("D", "E"))

    if scaled:
        cos_coeffs = cos_coeffs / 1.3
        sin_coeffs = sin_coeffs / 1.3

    return cos_coeffs, sin_coeffs


class HighFidelityDynamicModel(DynamicModelBase):

    def __init__(self, simulation_start_epoch_MJD, propagation_time, custom_initial_state=None):
        super().__init__(simulation_start_epoch_MJD, propagation_time)

        self.custom_initial_state = custom_initial_state

        self.new_bodies_to_create = ["Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        for new_body in self.new_bodies_to_create:
            self.bodies_to_create.append(new_body)


    def set_environment_settings(self):

        # Create default body settings
        self.body_settings = environment_setup.get_default_body_settings(
            self.bodies_to_create, self.global_frame_origin, self.global_frame_orientation)

        # Create radiation model for Earth and Moon
        for body in [self.name_primary, self.name_secondary]:

            if body == self.name_primary:

                surface_radiosity_models = [
                    environment_setup.radiation_pressure.variable_albedo_surface_radiosity(
                        albedo_distribution_settings = environment_setup.radiation_pressure.predefined_knocke_type_surface_property_distribution(environment_setup.radiation_pressure.albedo_knocke),
                        original_source_name = "Sun"),
                    environment_setup.radiation_pressure.thermal_emission_blackbody_variable_emissivity(
                        emissivity_distribution_model = environment_setup.radiation_pressure.predefined_knocke_type_surface_property_distribution(environment_setup.radiation_pressure.emissivity_knocke),
                        original_source_name = "Sun")]

            else:

                cos_coeffs, sin_coeffs = read_coeffs()

                surface_radiosity_models = [
                    environment_setup.radiation_pressure.variable_albedo_surface_radiosity(
                        albedo_distribution_settings = environment_setup.radiation_pressure.spherical_harmonic_surface_property_distribution(cos_coeffs, sin_coeffs),
                        original_source_name = "Sun"),
                        # albedo_distribution_settings = environment_setup.radiation_pressure.constant_surface_property_distribution(0.150),
                        # original_source_name = "Sun"),
                    environment_setup.radiation_pressure.thermal_emission_blackbody_variable_emissivity(
                        emissivity_distribution_model = environment_setup.radiation_pressure.constant_surface_property_distribution(0.95),
                        original_source_name = "Sun")]

            self.body_settings.get(body).radiation_source_settings = environment_setup.radiation_pressure.panelled_extended_radiation_source(
                surface_radiosity_models, [6, 12, 18])

        # Create environment
        self.bodies = environment_setup.create_system_of_bodies(self.body_settings)

        # Create update to spacecraft bodies
        occulting_bodies_dict = dict()
        occulting_bodies_dict[ "Sun" ] = [self.name_primary, self.name_secondary]
        for index, body in enumerate(self.bodies_to_propagate):
            self.bodies.create_empty_body(body)
            self.bodies.get_body(body).mass = self.bodies_mass[index]
            vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
                self.bodies_reference_area_radiation[index], self.bodies_radiation_pressure_coefficient[index], occulting_bodies_dict)
            environment_setup.add_radiation_pressure_target_model(self.bodies, body, vehicle_target_settings)


    def set_acceleration_settings(self):

        self.set_environment_settings()

        # Define accelerations acting on vehicle.
        self.acceleration_settings_on_spacecrafts = dict()
        for index, spacecraft in enumerate([self.name_ELO, self.name_LPO]):
            acceleration_settings_on_spacecraft = {
                    self.name_primary: [propagation_setup.acceleration.spherical_harmonic_gravity(10,10), #10,10
                                        propagation_setup.acceleration.relativistic_correction(),
                                        propagation_setup.acceleration.radiation_pressure()],
                    self.name_secondary: [propagation_setup.acceleration.spherical_harmonic_gravity(50,50), #50,50
                                          propagation_setup.acceleration.relativistic_correction(),
                                          propagation_setup.acceleration.radiation_pressure()]}
            for body in self.new_bodies_to_create:
                acceleration_settings_on_spacecraft[body] = [propagation_setup.acceleration.point_mass_gravity(),
                                                             propagation_setup.acceleration.relativistic_correction()]
                if body == "Sun":
                    acceleration_settings_on_spacecraft[body].append(propagation_setup.acceleration.radiation_pressure())

            self.acceleration_settings_on_spacecrafts[spacecraft] = acceleration_settings_on_spacecraft

        # Create global accelerations dictionary.
        self.acceleration_settings = self.acceleration_settings_on_spacecrafts

        # Create acceleration models.
        self.acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, self.acceleration_settings, self.bodies_to_propagate, self.central_bodies)


    def set_initial_state(self):

        self.set_acceleration_settings()

        if self.custom_initial_state is not None:
            self.initial_state = self.custom_initial_state

        else:

            # Define the initial state of LPF
            moon_initial_state = spice.get_body_cartesian_state_at_epoch(
                target_body_name = self.name_secondary,
                observer_body_name = self.name_primary,
                reference_frame_name = self.global_frame_orientation,
                aberration_corrections = 'NONE',
                ephemeris_time = self.simulation_start_epoch)

            initial_state_lpf_moon = element_conversion.keplerian_to_cartesian_elementwise(
                gravitational_parameter=self.gravitational_parameter_secondary,
                semi_major_axis=5737.4E3,
                eccentricity=0.61,
                inclination=np.deg2rad(57.83),
                argument_of_periapsis=np.deg2rad(90),
                longitude_of_ascending_node=np.deg2rad(61.55),
                true_anomaly=np.deg2rad(0))

            initial_state_LPF = np.add(initial_state_lpf_moon, moon_initial_state)

            # Define the initial state of LUMIO
            initial_state_LUMIO = validation.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_LPO)

            # Combine the initial states
            self.initial_state = np.concatenate((initial_state_LPF, initial_state_LUMIO))


    def set_integration_settings(self):

        self.set_initial_state()

        if self.use_variable_step_size_integrator:
            self.integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(self.initial_time_step,
                                                                                            self.current_coefficient_set,
                                                                                            np.finfo(float).eps,
                                                                                            np.inf,
                                                                                            self.current_tolerance,
                                                                                            self.current_tolerance)
        else:
            self.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(self.initial_time_step,
                                                                                           self.current_coefficient_set)


    def set_dependent_variables_to_save(self):

        self.set_integration_settings()

        # Define required outputs
        self.dependent_variables_to_save = [
            propagation_setup.dependent_variable.relative_position(self.name_secondary, self.name_primary),
            propagation_setup.dependent_variable.relative_velocity(self.name_secondary, self.name_primary),
            propagation_setup.dependent_variable.relative_position(self.name_ELO, self.name_LPO),
            propagation_setup.dependent_variable.relative_velocity(self.name_ELO, self.name_LPO),
            propagation_setup.dependent_variable.total_acceleration(self.name_ELO),
            propagation_setup.dependent_variable.total_acceleration(self.name_LPO),
            propagation_setup.dependent_variable.keplerian_state(self.name_secondary, self.name_primary),
            propagation_setup.dependent_variable.keplerian_state(self.name_ELO, self.name_secondary)]

        self.dependent_variables_to_save.extend([
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, body_to_propagate, new_body_to_create) \
                        for body_to_propagate in self.bodies_to_propagate for new_body_to_create in self.new_bodies_to_create])

        self.dependent_variables_to_save.extend([
            propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm(body_to_propagate, body_to_create, [(2,0), (2,1), (2,2)]) \
                        for body_to_propagate in self.bodies_to_propagate for body_to_create in [self.name_primary, self.name_secondary] ])

        self.dependent_variables_to_save.extend([
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.radiation_pressure_type, body_to_propagate, body) \
                        for body_to_propagate in self.bodies_to_propagate for body in [self.name_primary, self.name_secondary, "Sun"]])

        self.dependent_variables_to_save.extend([
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.relativistic_correction_acceleration_type, body_to_propagate, body_to_create) \
                        for body_to_propagate in self.bodies_to_propagate for body_to_create in self.bodies_to_create])

        self.dependent_variables_to_save.extend([propagation_setup.dependent_variable.body_mass(self.name_primary),
                                                 propagation_setup.dependent_variable.body_mass(self.name_secondary)])


    def set_termination_settings(self):

        self.set_dependent_variables_to_save()

        # Create termination settings
        self.termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch)


    def set_propagator_settings(self, estimated_parameter_vector=None):

        self.set_termination_settings()

        if estimated_parameter_vector is not None:
            self.initial_state = estimated_parameter_vector[:12]

        # Create propagation settings
        self.propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            self.acceleration_models,
            self.bodies_to_propagate,
            self.initial_state,
            self.simulation_start_epoch,
            self.integrator_settings,
            self.termination_settings,
            output_variables= self.dependent_variables_to_save)


    def get_propagation_simulator(self, estimated_parameter_vector=None, solve_variational_equations=True):

        self.set_propagator_settings(estimated_parameter_vector=estimated_parameter_vector)

        # Create simulation object and propagate dynamics.
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            self.bodies,
            self.propagator_settings)

        # Setup parameters settings to propagate the state transition matrix
        if solve_variational_equations:
            self.parameter_settings = estimation_setup.parameter.initial_states(self.propagator_settings, self.bodies)
            self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.bodies)
            variational_equations_solver = numerical_simulation.create_variational_equations_solver(
                    self.bodies,
                    self.propagator_settings,
                    self.parameters_to_estimate,
                    simulate_dynamics_on_creation=True)

            return dynamics_simulator, variational_equations_solver

        else:

            return dynamics_simulator

# test = HighFidelityDynamicModel(60390, 365)
# dynamics_simulator = test.get_propagation_simulator()

# state_history = np.array([(time_conversion.julian_day_to_modified_julian_day(time_conversion.seconds_since_epoch_to_julian_day(key)), key, *value/1000) for key, value in dynamics_simulator.state_history.items()], dtype=object)
# moon_state_history = np.array([(time_conversion.julian_day_to_modified_julian_day(time_conversion.seconds_since_epoch_to_julian_day(key)), key, *value/1000) for key, value in dynamics_simulator.dependent_variable_history.items()], dtype=object)

# header = "epoch (MJD), epoch (seconds TDB), x [km], y [km], z [km], vx [km/s], vy [km/s], vz [km/s]"
# np.savetxt("C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/reference/DataLPF/TextFiles/LPF_states_J2000_Earth_centered.txt", state_history[:,:8], delimiter=',', fmt='%f', header=header)
# np.savetxt("C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/reference/DataLPF/TextFiles/Moon_states_J2000_Earth_centered.txt", moon_state_history[:,:8], delimiter=',', fmt='%f', header=header)

# header = "epoch (MJD), epoch (seconds TDB), x [km], y [km], z [km], vx [km/s], vy [km/s], vz [km/s]"
# np.savetxt("C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/reference/DataLPF/TextFiles/LPF_states_J2000_Earth_centered_test.txt", state_history[:,:8], delimiter=',', fmt='%f', header=header)
# np.savetxt("C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/reference/DataLPF/TextFiles/Moon_states_J2000_Earth_centered_test.txt", moon_state_history[:,:8], delimiter=',', fmt='%f', header=header)


# ax = plt.figure().add_subplot(projection='3d')
# plt.plot(moon_state_history[:,2], moon_state_history[:,3], moon_state_history[:,4])
# plt.plot(state_history[:,2], state_history[:,3], state_history[:,4])
# plt.legend()
# plt.show()

# dependent_variables_history = np.vstack(list(dynamics_simulator.dependent_variable_history.values()))
# print(np.shape(dependent_variables_history))
# ax = plt.figure()
# plt.plot(dependent_variables_history[:,-6:-3], label="ELO")
# plt.plot(dependent_variables_history[:,-3:], label="LPO")
# plt.show()