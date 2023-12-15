# General imports
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

# Tudatpy imports
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup, environment_setup, estimation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice

# Define path to import src files
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from dynamic_models import validation_LUMIO
from DynamicModelBase import DynamicModelBase

class HighFidelityDynamicModel(DynamicModelBase):

    def __init__(self, simulation_start_epoch_MJD, propagation_time):
        super().__init__(simulation_start_epoch_MJD, propagation_time)
        
        self.new_bodies_to_create = ["Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        for new_body in self.new_bodies_to_create:
            self.bodies_to_create.append(new_body)


    def set_environment_settings(self):
                
        # Create default body settings        
        self.body_settings = environment_setup.get_default_body_settings(
            self.bodies_to_create, self.global_frame_origin, self.global_frame_orientation)

        # Create environment
        self.bodies = environment_setup.create_system_of_bodies(self.body_settings)

        # Create spacecraft bodies
        for index, body in enumerate(self.bodies_to_propagate):
            self.bodies.create_empty_body(body)
            self.bodies.get_body(body).mass = self.bodies_mass[index]


    def set_acceleration_settings(self):

        self.set_environment_settings()
    
        # Define accelerations acting on vehicle.
        self.acceleration_settings_on_spacecrafts = dict()
        for index, spacecraft in enumerate([self.name_ELO, self.name_LPO]):
            acceleration_settings_on_spacecraft = {
                    self.name_primary: [propagation_setup.acceleration.spherical_harmonic_gravity(20,20)],
                    self.name_secondary: [propagation_setup.acceleration.spherical_harmonic_gravity(20,20)]}
            for body in self.new_bodies_to_create:
                acceleration_settings_on_spacecraft[body] = [propagation_setup.acceleration.point_mass_gravity()]
            self.acceleration_settings_on_spacecrafts[spacecraft] = acceleration_settings_on_spacecraft

        # Create global accelerations dictionary.
        self.acceleration_settings = self.acceleration_settings_on_spacecrafts

        # Create acceleration models.
        self.acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, self.acceleration_settings, self.bodies_to_propagate, self.central_bodies)


    def set_initial_state(self):

        self.set_acceleration_settings()

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
        initial_state_LUMIO = validation_LUMIO.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, fixed_step_size=0.01)[0][0]

        # Combine the initial states
        self.initial_state = np.concatenate((initial_state_LPF, initial_state_LUMIO))


    def set_integration_settings(self):

        self.set_initial_state()

        current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkdp_87
        current_tolerance = 1e-10*constants.JULIAN_DAY
        initial_time_step = 1e-3*constants.JULIAN_DAY
        self.integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(initial_time_step,
                                                                                        current_coefficient_set,
                                                                                        np.finfo(float).eps, 
                                                                                        np.inf,
                                                                                        current_tolerance, 
                                                                                        current_tolerance)


    def set_dependent_variables_to_save(self):

        self.set_integration_settings()

        # Define required outputs
        self.dependent_variables_to_save = [
            propagation_setup.dependent_variable.relative_position(self.name_primary, self.name_secondary),
            propagation_setup.dependent_variable.relative_velocity(self.name_primary, self.name_secondary),
            propagation_setup.dependent_variable.relative_distance(self.name_primary, self.name_secondary),
            propagation_setup.dependent_variable.total_acceleration(self.name_ELO),
            propagation_setup.dependent_variable.total_acceleration(self.name_LPO)
        ]
        

    def set_termination_settings(self):

        self.set_dependent_variables_to_save()

        # Create termination settings
        self.termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch)


    def set_propagator_settings(self):

        self.set_termination_settings()

        # Create propagation settings
        self.propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            self.acceleration_models,
            self.bodies_to_propagate,
            self.initial_state,
            self.simulation_start_epoch,
            self.integrator_settings,
            self.termination_settings,
            output_variables= self.dependent_variables_to_save
        )

    

    def get_propagated_orbit(self):

        self.set_propagator_settings()

        # Create simulation object and propagate dynamics.
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            self.bodies, 
            self.propagator_settings)

        # Setup parameters settings to propagate the state transition matrix
        self.parameter_settings = estimation_setup.parameter.initial_states(self.propagator_settings, self.bodies)
        self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.bodies)
        variational_equations_solver = numerical_simulation.create_variational_equations_solver(
                self.bodies,
                self.propagator_settings,
                self.parameters_to_estimate,
                simulate_dynamics_on_creation=True)

        # Extract the simulation results
        self.state_history                   = np.vstack(list(variational_equations_solver.state_history.values()))
        self.dependent_variables_history     = np.vstack(list(dynamics_simulator.dependent_variable_history.values()))
        self.state_transition_matrix_history = np.vstack(list(variational_equations_solver.state_transition_matrix_history.values())).reshape((np.shape(self.state_history)[0], np.shape(self.state_history)[1], np.shape(self.state_history)[1]))

        return self.state_history, self.dependent_variables_history, self.state_transition_matrix_history


# test2 = HighFidelityDynamicModel(60390, 28)
# states2 = test2.get_propagated_orbit()[0]

# ax = plt.figure().add_subplot(projection='3d')
# # plt.plot(states[:,0], states[:,1], states[:,2])
# # plt.plot(states[:,6], states[:,7], states[:,8])
# plt.plot(states2[:,0], states2[:,1], states2[:,2])
# plt.plot(states2[:,6], states2[:,7], states2[:,8])
# plt.legend()
# plt.show()