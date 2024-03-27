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
sys.path.append(os.path.dirname(parent_dir))

# Own
import reference_data
from DynamicModelBase import DynamicModelBase


class HighFidelityDynamicModel(DynamicModelBase):

    def __init__(self, simulation_start_epoch_MJD, propagation_time, custom_initial_state=None, custom_propagation_time=None):
        super().__init__(simulation_start_epoch_MJD, propagation_time)

        self.custom_initial_state = custom_initial_state
        self.custom_propagation_time = custom_propagation_time

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
            self.acceleration_settings_on_spacecrafts[spacecraft] = acceleration_settings_on_spacecraft

        # Create global accelerations dictionary.
        self.acceleration_settings = self.acceleration_settings_on_spacecrafts

        # Create acceleration models.
        self.acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, self.acceleration_settings, self.bodies_to_propagate, self.central_bodies)


    def set_initial_state(self):

        self.set_acceleration_settings()

        initial_state_LPF = reference_data.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_ELO)
        initial_state_LUMIO = reference_data.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, satellite=self.name_LPO)

        if self.custom_initial_state is not None:
            self.initial_state = self.custom_initial_state
        else:
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
            propagation_setup.dependent_variable.relative_velocity(self.name_ELO, self.name_LPO)]

        self.dependent_variables_to_save.extend([propagation_setup.dependent_variable.total_acceleration_norm(self.name_ELO),
                                                 propagation_setup.dependent_variable.total_acceleration_norm(self.name_LPO)])

        self.dependent_variables_to_save.extend([
            propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, body_to_propagate, body_to_create) \
                        for body_to_propagate in self.bodies_to_propagate for body_to_create in self.bodies_to_create])


    def set_termination_settings(self):

        self.set_dependent_variables_to_save()

        # Create termination settings
        if self.custom_propagation_time is not None:
            self.simulation_end_epoch = self.simulation_start_epoch + self.custom_propagation_time*constants.JULIAN_DAY

        self.termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch)


    def set_propagator_settings(self):

        self.set_termination_settings()

        if self.custom_initial_state is not None:
            self.initial_state = self.custom_initial_state

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





# from dynamic_models import Interpolator



# step_size = 0.001

# test0 = HighFidelityDynamicModel(60390, 28)
# epochs0 = np.stack(list(test0.get_propagation_simulator()[0].state_history.keys()))
# states0 = np.stack(list(test0.get_propagation_simulator()[0].state_history.values()))
# print(epochs0[0], epochs0[-1], epochs0[-1]-epochs0[0])
# print(states0[0,:], states0[-1,:])

# epochs0, state_history0, dependent_variables_history0 = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(test0,
#                                                                                             custom_initial_state=None,
#                                                                                             solve_variational_equations=False)

# print("STARTING HERE ===========")
# test1 = HighFidelityDynamicModel(60390, 1)
# epochs1 = np.stack(list(test1.get_propagation_simulator()[0].state_history.keys()))
# states1 = np.stack(list(test1.get_propagation_simulator()[0].state_history.values()))
# print("1 =================")
# # print(epochs1[0], epochs1[-1], epochs1[-1]-epochs1[0])
# # print(states1[0,:], states1[-1,:])
# # print("Difference: ", states1[0,:]-states0[-1,:])


# epochs, state_history1, dependent_variables_history1 = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(test1,
#                                                                                             custom_initial_state=None,
#                                                                                             custom_propagation_time=1,
#                                                                                             solve_variational_equations=False)

# print(epochs[0], epochs[-1], epochs[-1]-epochs[0])
# print(state_history1[0,:], state_history1[-1,:])


# # custom_initial_state = np.array([-2.94494412e+08,  2.09254412e+08,  1.19603970e+08, -1.20598462e+02,
# #                                  -3.97387177e+02, -1.18349658e+03, -3.34395442e+08,  1.96596530e+08,
# #                                   1.38362397e+08, -8.43070472e+02, -8.68868628e+02, -6.65684845e+02])
# test2 = HighFidelityDynamicModel(60391, 1, custom_initial_state=state_history1[-1,:])
# epochs2 = np.stack(list(test2.get_propagation_simulator()[0].state_history.keys()))
# states2 = np.stack(list(test2.get_propagation_simulator()[0].state_history.values()))
# print("2 =================")
# # print(epochs2[0], epochs2[-1], epochs2[-1]-epochs2[0])
# # print(states2[0,:], states2[-1,:])
# print("Difference: ", states2[0,:]-state_history1[-1,:])

# epochs, state_history2, dependent_variables_history2 = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(test2,
#                                                                                             custom_initial_state=None,
#                                                                                             custom_propagation_time=1,
#                                                                                             solve_variational_equations=False)

# print(epochs[0], epochs[-1], epochs[-1]-epochs[0])
# print(state_history2[0,:], state_history2[-1,:])

# # custom_initial_state = np.array([-3.04827426e+08,  1.98342823e+08,  9.38530195e+07, -3.39162394e+02,
# #                                  -3.53289967e+02, -3.67877005e+02, -3.67269391e+08,  1.58881973e+08,
# #                                   1.08554131e+08, -6.80914768e+02, -8.76074687e+02, -7.08399431e+02])
# test3 = HighFidelityDynamicModel(60392, 1, custom_initial_state=state_history2[-1,:])
# epochs3 = np.stack(list(test3.get_propagation_simulator()[0].state_history.keys()))
# states3 = np.stack(list(test3.get_propagation_simulator()[0].state_history.values()))
# print("3 =================")
# # print(epochs3[0], epochs3[-1], epochs3[-1]-epochs3[0])
# # print(states3[0,:], states3[-1,:])
# print("Difference: ", states3[0,:]-state_history2[-1,:])

# from dynamic_models import Interpolator
# epochs, state_history3, dependent_variables_history3 = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(test3,
#                                                                                             custom_initial_state=None,
#                                                                                             custom_propagation_time=1,
#                                                                                             solve_variational_equations=False)

# print(epochs[0], epochs[-1], epochs[-1]-epochs[0])
# print(state_history3[0,:], state_history3[-1,:])

# # custom_initial_state = np.array([-3.20733730e+08,  1.78164330e+08,  8.09048108e+07, -3.81352146e+02,
# #                                  -5.72246215e+02, -2.61197463e+02, -3.93597685e+08,  1.20914657e+08,
# #                                   7.75677652e+07, -5.39727735e+02, -8.80967175e+02, -7.23126700e+02])
# test4 = HighFidelityDynamicModel(60393, 1, custom_initial_state=state_history3[-1,:])
# states4 = np.stack(list(test4.get_propagation_simulator()[0].state_history.values()))

# from dynamic_models import Interpolator
# epochs, state_history4, dependent_variables_history4 = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(test4,
#                                                                                             custom_initial_state=None,
#                                                                                             custom_propagation_time=1,
#                                                                                             solve_variational_equations=False)

# print(epochs[0], epochs[-1], epochs[-1]-epochs[0])
# print(state_history4[0,:], state_history4[-1,:])


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.plot(states0[:,0], states0[:,1], states0[:,2], color="gray")
# plt.plot(states0[:,6], states0[:,7], states0[:,8], color="gray", label="states0")
# plt.plot(dependent_variables_history0[:,0], dependent_variables_history0[:,1], dependent_variables_history0[:,2], color="orange")
# plt.plot(states1[:,0], states1[:,1], states1[:,2], color="blue")
# plt.plot(states1[:,6], states1[:,7], states1[:,8], color="blue", label="states1")
# plt.plot(dependent_variables_history1[:,0], dependent_variables_history1[:,1], dependent_variables_history1[:,2], color="black")
# plt.plot(states2[:,0], states2[:,1], states2[:,2], color="orange")
# plt.plot(states2[:,6], states2[:,7], states2[:,8], color="orange", label="states2")
# plt.plot(dependent_variables_history2[:,0], dependent_variables_history2[:,1], dependent_variables_history2[:,2], color="black")
# plt.plot(states3[:,0], states3[:,1], states3[:,2], color="green")
# plt.plot(states3[:,6], states3[:,7], states3[:,8], color="green", label="states3")
# plt.plot(dependent_variables_history3[:,0], dependent_variables_history3[:,1], dependent_variables_history3[:,2], color="black")
# plt.plot(states4[:,0], states4[:,1], states4[:,2], color="red")
# plt.plot(states4[:,6], states4[:,7], states4[:,8], color="red", label="states4")
# plt.plot(dependent_variables_history4[:,0], dependent_variables_history4[:,1], dependent_variables_history4[:,2], color="black")
# plt.legend()
# plt.show()