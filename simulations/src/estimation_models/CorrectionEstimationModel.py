# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

# Tudatpy imports
import tudatpy
from tudatpy import util
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import estimation
from tudatpy.kernel.numerical_simulation import propagation_setup, environment_setup, estimation_setup
from tudatpy.kernel.astro import time_conversion, element_conversion, frame_conversion
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Own
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dynamic_models import validation_LUMIO
from dynamic_models.low_fidelity import LowFidelityDynamicModel
from dynamic_models.high_fidelity.point_mass import *
from dynamic_models.high_fidelity.point_mass_srp import *
from dynamic_models.high_fidelity.spherical_harmonics import *
from dynamic_models.high_fidelity.spherical_harmonics_srp import *


class CorrectionEstimationModel:

    def __init__(self, parent_instance):

        self.parent_instance = parent_instance


    def set_observation_settings(self):

        # Define the uplink link ends for one-way observable
        link_ends_lumio = dict()
        link_ends_lumio[estimation_setup.observation.observed_body] = estimation_setup.observation.\
            body_origin_link_end_id(name_spacecraft)
        link_definition_lumio = estimation_setup.observation.LinkDefinition(link_ends_lumio)

        link_definition_dict = {
            name_spacecraft: link_definition_lumio
        }

        position_observation_settings = [estimation_setup.observation.cartesian_position(link_definition_lumio)
                                        ]

        # Define epochs at which the ephemerides shall be checked
        observation_times = np.arange(simulation_start_epoch, simulation_start_epoch+propagation_time*86400, 30)

        # Create the observation simulation settings per moon
        observation_simulation_settings = list()
        for body in link_definition_dict.keys():
            observation_simulation_settings.append(estimation_setup.observation.tabulated_simulation_settings(
                estimation_setup.observation.position_observable_type,
                link_definition_dict[body],
                observation_times,
                reference_link_end_type=estimation_setup.observation.observed_body))


    def set_viability_settings(self):

        self.set_observation_settings()

        pass


    def set_simulated_observations(self):

        self.set_viability_settings()
        self.parent_instance.set_propagator_settings()

        # Create observation simulators
        ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
            position_observation_settings, bodies)

        # Get ephemeris states as ObservationCollection
        print('Checking ephemerides...')
        ephemeris_satellite_states = estimation.simulate_observations(
            observation_simulation_settings,
            ephemeris_observation_simulators,
            bodies)

        # Setup parameters settings to propagate the state transition matrix
        parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
        parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)
        original_parameter_vector = parameters_to_estimate.parameter_vector



    def get_estimation_results(self):

        self.set_simulated_observations()

        print('Running propagation...')
        with util.redirect_std():
            estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate,
                                                    position_observation_settings, propagator_settings)


        # Create input object for the estimation
        convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=5)
        estimation_input = estimation.EstimationInput(ephemeris_satellite_states, convergence_checker=convergence_checker)
        # Set methodological options
        estimation_input.define_estimation_settings(save_state_history_per_iteration=True)
        # Perform the estimation
        print('Performing the estimation...')
        print(f'Original initial states: {original_parameter_vector}')


        with util.redirect_std(redirect_out=False):
            estimation_output = estimator.perform_estimation(estimation_input)
        initial_states_updated = parameters_to_estimate.parameter_vector
        print('Done with the estimation...')
        print(f'Updated initial states: {initial_states_updated}')


        return initial_states_updated



    def get_propagated_orbit_from_estimator(self):

        self.set_simulated_observations()

        # # Extract the simulation results
        # self.epochs                          = np.vstack(list(self.estimator.variational_solver.state_history.keys()))
        # self.state_history                   = np.vstack(list(self.estimator.variational_solver.state_history.values()))
        # self.dependent_variables_history     = np.vstack(list(self.estimator.variational_solver.dynamics_simulator.dependent_variable_history.values()))
        # self.state_transition_matrix_history = np.vstack(list(self.estimator.variational_solver.state_transition_matrix_history.values())).reshape((np.shape(self.state_history)[0], np.shape(self.state_history)[1], np.shape(self.state_history)[1]))


        print(np.shape(self.epochs), np.shape(self.dependent_variables_history))
        return self.estimator.variational_solver, self.estimator.variational_solver.dynamics_simulator


model = high_fidelity_point_mass_srp_01.HighFidelityDynamicModel(60390, 28)
# model = LowFidelityDynamicModel.LowFidelityDynamicModel(60390, 14)
estimation_model = EstimationModel(model)

covariance_dict = estimation_model.get_estimation_results()

print(covariance_dict)

covariance_history = np.stack(covariance_dict.values())

print(covariance_history)
# estimation_result = estimation_model.get_estimation_results()

# information_matrix_history  = estimation_result[-1]

ax = plt.figure(figsize=(6.5,6))
plt.plot(covariance_history)
plt.plot()
plt.show()