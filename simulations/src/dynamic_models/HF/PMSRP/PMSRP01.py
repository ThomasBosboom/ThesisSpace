# General imports
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

# Tudatpy imports
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup, environment_setup, estimation_setup

# Define path to import src files
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# Own
import reference_data
from DynamicModelBase import DynamicModelBase


class HighFidelityDynamicModel(DynamicModelBase):

    def __init__(self, simulation_start_epoch_MJD, propagation_time, **kwargs):
        super().__init__(simulation_start_epoch_MJD, propagation_time)

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.new_bodies_to_create = ["Sun"]
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

        # Create radiation pressure settings, and add to vehicle
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
                    self.name_primary: [propagation_setup.acceleration.point_mass_gravity()],
                    self.name_secondary: [propagation_setup.acceleration.point_mass_gravity()]}
            for body in self.new_bodies_to_create:
                acceleration_settings_on_spacecraft[body] = [propagation_setup.acceleration.point_mass_gravity()]
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

        self.simulation_initial_state = self.initial_state
        if self.custom_initial_state is not None:
            self.simulation_initial_state = self.custom_initial_state


    def set_integration_settings(self):

        self.set_initial_state()

        if self.use_variable_step_size_integrator:
            self.integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(self.initial_time_step,
                                                                                            self.current_coefficient_set,
                                                                                            np.finfo(float).eps,
                                                                                            np.inf,
                                                                                            self.relative_error_tolerance,
                                                                                            self.absolute_error_tolerance)
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
            propagation_setup.dependent_variable.relative_velocity(self.name_ELO, self.name_LPO)]

        self.dependent_variables_to_save.extend([propagation_setup.dependent_variable.total_acceleration_norm(self.name_ELO),
                                                 propagation_setup.dependent_variable.total_acceleration_norm(self.name_LPO)])

        self.dependent_variables_to_save.extend([
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, body_to_propagate, body_to_create) \
                        for body_to_propagate in self.bodies_to_propagate for body_to_create in self.bodies_to_create])

        self.dependent_variables_to_save.extend([
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.radiation_pressure_type, body_to_propagate, "Sun") \
                        for body_to_propagate in self.bodies_to_propagate])

        self.dependent_variables_to_save.extend([
            propagation_setup.dependent_variable.total_acceleration(body_to_propagate) \
                for body_to_propagate in self.bodies_to_propagate])


    def set_termination_settings(self):

        self.set_dependent_variables_to_save()

        # Create termination settings
        if self.custom_propagation_time is not None:
            self.simulation_end_epoch = self.simulation_start_epoch + self.custom_propagation_time*constants.JULIAN_DAY

        self.termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch)


    def set_propagator_settings(self):

        self.set_termination_settings()

        # Create propagation settings
        self.propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            self.acceleration_models,
            self.bodies_to_propagate,
            self.simulation_initial_state,
            self.simulation_start_epoch,
            self.integrator_settings,
            self.termination_settings,
            output_variables= self.dependent_variables_to_save
        )


    def get_propagation_simulator(self, solve_variational_equations=True):

        self.set_propagator_settings()

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


# from src import Interpolator


# start = 60390
# prop = 0.2
# step_size = 0.01
# custom_initial_state = np.array([-2.80124257e+08,  2.53325273e+08,  1.46944225e+08 ,-1.61474900e+03,
#                                 -2.23501800e+03, -2.97164000e+02, -3.10468779e+08, 2.49476676e+08,
#                                 1.74974583e+08, -9.93404005e+02, -7.66335485e+02, -5.24989115e+02])
# model = HighFidelityDynamicModel(start, prop, custom_initial_state=custom_initial_state)
# _, state_history, _, _ = \
#     Interpolator.Interpolator(epoch_in_MJD=True, step_size=step_size).get_propagation_results(model, solve_variational_equations=True)


# custom_initial_state = np.array([
#     -2.80124837e+08, 2.53324810e+08, 1.46943682e+08, -1.61474821e+03,
#     -2.23501636e+03, -2.97163878e+02, -3.10468232e+08, 2.49475875e+08,
#     1.74975213e+08, -9.93403510e+02, -7.66335369e+02, -5.24989596e+02
# ])
# model_estimated = HighFidelityDynamicModel(start, prop, custom_initial_state=custom_initial_state)
# _, state_history_estimated, _, _ = \
#     Interpolator.Interpolator(epoch_in_MJD=True, step_size=step_size).get_propagation_results(model_estimated, solve_variational_equations=True)

# custom_initial_state = np.array([-2.80124257e+08,  2.53325273e+08,  1.46944225e+08 ,-1.61474900e+03,
#                                 -2.23501800e+03, -2.97164000e+02, -3.10468779e+08, 2.49476676e+08,
#                                 1.74974583e+08, -9.93404005e+02, -7.66335485e+02, -5.24989115e+02])
# model_truth = HighFidelityDynamicModel(start, prop, custom_initial_state=custom_initial_state)
# _, state_history_truth, _, _ = \
#     Interpolator.Interpolator(epoch_in_MJD=True, step_size=step_size).get_propagation_results(model_truth, solve_variational_equations=True)

# print("initial states: \n", state_history[0,:], state_history_estimated[0,:]-state_history_truth[0,:])

# ax = plt.figure()
# # plt.plot(states[:,0], states[:,1], states[:,2])
# # plt.plot(states[:,6], states[:,7], states[:,8])
# plt.plot(state_history[:,6:9]-state_history_truth[:,6:9], color="red")
# plt.plot(state_history_estimated[:,6:9]-state_history_truth[:,6:9], color="blue")
# # plt.plot(states[:,6], states[:,7])
# # plt.plot(dep_var[:,:6])
# # plt.plot(dep_var[:,-8:-6])
# plt.legend()
# plt.show()