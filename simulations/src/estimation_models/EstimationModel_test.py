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
from dynamic_models.low_fidelity import low_fidelity
from dynamic_models.high_fidelity.point_mass import *
from dynamic_models.high_fidelity.point_mass_srp import *
from dynamic_models.high_fidelity.spherical_harmonics import *
from dynamic_models.high_fidelity.spherical_harmonics_srp import *

class EstimationModel:

    def __init__(self, parent_instance):

        self.parent_instance = parent_instance

    def update_environment_settings(self):

        self.parent_instance.set_environment_settings()

        # Create default body settings
        self.parent_instance.body_settings = environment_setup.get_default_body_settings(
                                                self.parent_instance.bodies_to_create,
                                                self.parent_instance.global_frame_origin,
                                                self.parent_instance.global_frame_orientation)

        self.parent_instance.body_settings.add_empty_settings(self.parent_instance.name_ELO)
        self.parent_instance.body_settings.add_empty_settings(self.parent_instance.name_LPO)

        # Create spacecraft bodies
        for index, body in enumerate(self.parent_instance.bodies_to_propagate):
            self.parent_instance.bodies.create_empty_body(body)
            self.parent_instance.bodies.get_body(body).mass = self.parent_instance.bodies_mass[index]
            self.parent_instance.body_settings.get(body).ephemeris_settings = environment_setup.ephemeris.tabulated(
                validation_LUMIO.get_reference_state_history(self.parent_instance.simulation_start_epoch_MJD,
                                                             self.parent_instance.propagation_time,
                                                             get_dict=True,
                                                             get_full_history=True),
                self.parent_instance.global_frame_origin,
                self.parent_instance.global_frame_orientation)

        # Update environment with reference states as ephermeris settings
        self.parent_instance.bodies = environment_setup.create_system_of_bodies(self.parent_instance.body_settings)



    def set_observation_settings(self):

        self.update_environment_settings()

        # Define the uplink link ends for one-way observable
        self.link_ends_lpf = dict()
        self.link_ends_lpf[estimation_setup.observation.observed_body] = estimation_setup.observation.\
            body_origin_link_end_id(self.parent_instance.name_ELO)
        self.link_definition_lpf = estimation_setup.observation.LinkDefinition(self.link_ends_lpf)

        self.link_ends_lumio = dict()
        self.link_ends_lumio[estimation_setup.observation.observed_body] = estimation_setup.observation.\
            body_origin_link_end_id(self.parent_instance.name_LPO)
        self.link_definition_lumio = estimation_setup.observation.LinkDefinition(self.link_ends_lumio)

        self.link_definition_dict = {
            self.parent_instance.name_ELO: self.link_definition_lpf,
            self.parent_instance.name_LPO: self.link_definition_lumio,
        }

        print(self.link_definition_dict)

        self.position_observation_settings = [estimation_setup.observation.cartesian_position(self.link_definition_lpf),
                                         estimation_setup.observation.cartesian_position(self.link_definition_lumio)
                                        ]

        # Define epochs at which the ephemerides shall be checked
        self.observation_times = np.arange(self.parent_instance.simulation_start_epoch, self.parent_instance.simulation_start_epoch+self.parent_instance.propagation_time*86400, 30)

        # Create the observation simulation settings per moon
        self.observation_simulation_settings = list()
        for body in self.link_definition_dict.keys():
            self.observation_simulation_settings.append(estimation_setup.observation.tabulated_simulation_settings(
                estimation_setup.observation.position_observable_type,
                self.link_definition_dict[body],
                self.observation_times,
                reference_link_end_type=estimation_setup.observation.observed_body))

        print("Observation settings: ", self.observation_simulation_settings)


    def set_viability_settings(self):

        self.set_observation_settings()

        pass


    def set_simulated_observations(self):

        self.set_viability_settings()
        self.parent_instance.set_propagator_settings()

        # Create observation simulators
        self.ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
            self.position_observation_settings, self.parent_instance.bodies)
        # Get ephemeris states as ObservationCollection
        print('Checking ephemerides...')
        self.ephemeris_satellite_states = estimation.simulate_observations(
            self.observation_simulation_settings,
            self.ephemeris_observation_simulators,
            self.parent_instance.bodies)


        self.parameters_to_estimate_settings = estimation_setup.parameter.initial_states(self.parent_instance.propagator_settings, self.parent_instance.bodies)
        self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameters_to_estimate_settings, self.parent_instance.bodies)
        self.original_parameter_vector = self.parameters_to_estimate.parameter_vector



        # self.single_observation_set_lists = [self.simulated_observations.get_single_link_and_type_observations(observation.one_way_range_type, self.link_definition),
        #                                      self.simulated_observations.get_single_link_and_type_observations(observation.one_way_instantaneous_doppler_type, self.link_definition),
        #                                     ]

        # self.observations = dict()
        # for single_observation_set_list in self.single_observation_set_lists:
        #     for single_observation_set in single_observation_set_list:
        #         self.observations[single_observation_set.observable_type] = dict(zip(single_observation_set.observation_times, single_observation_set.concatenated_observations))

        # # ax = plt.figure()
        # # plt.plot(self.observations[observation.one_way_instantaneous_doppler_type].keys(), self.observations[observation.one_way_instantaneous_doppler_type].values(), color="red", marker="o")
        # # plt.show()

        # return self.observations


    def get_estimation_results(self):

        self.set_simulated_observations()

        print('Running propagation...')
        with util.redirect_std():
            estimator = numerical_simulation.Estimator(self.parent_instance.bodies, self.parameters_to_estimate,
                                                    self.position_observation_settings, self.parent_instance.propagator_settings)


        # Create input object for the estimation
        convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=5)
        estimation_input = estimation.EstimationInput(self.ephemeris_satellite_states, convergence_checker=convergence_checker)
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

covariance_dict = estimation_model.get_estimation_results()[-3]

print(covariance_dict)

covariance_history = np.stack(covariance_dict.values())

print(covariance_history)
# estimation_result = estimation_model.get_estimation_results()

# information_matrix_history  = estimation_result[-1]

ax = plt.figure(figsize=(6.5,6))
plt.plot(covariance_history)
plt.plot()
plt.show()