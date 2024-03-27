# General imports
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

# Tudatpy imports
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup, environment_setup, estimation_setup
from tudatpy.kernel.astro import element_conversion, frame_conversion
from tudatpy.kernel.interface import spice

# Define path to import src files
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(parent_dir))

# Own
import reference_data
from DynamicModelBase import DynamicModelBase


class LowFidelityDynamicModel(DynamicModelBase):

    def __init__(self, simulation_start_epoch_MJD, propagation_time, custom_initial_state=None, custom_propagation_time=None, use_synodic_state=False):
        super().__init__(simulation_start_epoch_MJD, propagation_time)

        self.custom_initial_state = custom_initial_state
        self.custom_propagation_time = custom_propagation_time

        self.use_synodic_state = use_synodic_state

        # Get CRTBP characteristics
        self.distance_between_primaries = 3.84747963e8
        self.eccentricity = 0
        self.bodies_mass = [0,0]
        self.lu_cr3bp = self.distance_between_primaries
        self.tu_cr3bp = 1/np.sqrt((self.gravitational_parameter_primary + self.gravitational_parameter_secondary)/self.distance_between_primaries**3)
        self.rotation_rate = 1/self.tu_cr3bp


    def set_initial_cartesian_moon_state(self):

        # Adjust the orbit of the moon to perfectly circular
        moon_initial_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name = self.name_secondary,
            observer_body_name = self.name_primary,
            reference_frame_name = self.global_frame_orientation,
            aberration_corrections = 'NONE',
            ephemeris_time = self.simulation_start_epoch)

        self.central_body_gravitational_parameter = self.gravitational_parameter_primary + self.gravitational_parameter_secondary
        self.initial_keplerian_moon_state = element_conversion.cartesian_to_keplerian(moon_initial_state,
                                                                                      self.central_body_gravitational_parameter)
        self.initial_keplerian_moon_state[0] = self.distance_between_primaries
        self.initial_keplerian_moon_state[1] = self.eccentricity

        self.initial_cartesian_moon_state = element_conversion.keplerian_to_cartesian(self.initial_keplerian_moon_state,
                                                                                      self.central_body_gravitational_parameter)

        # print("initial_keplerian_moon_state: ", self.simulation_start_epoch, self.initial_keplerian_moon_state)



    def convert_synodic_to_inertial_state(self, initial_state_barycenter_fixed):

        self.set_initial_cartesian_moon_state()

        # Convert barycentric CRTBP non-dimensionalized state into moon fixed frame.
        initial_state_moon_fixed = initial_state_barycenter_fixed
        initial_state_moon_fixed[0] = initial_state_moon_fixed[0] - (1-self.mu)
        initial_state_moon_fixed[6] = initial_state_moon_fixed[6] - (1-self.mu)

        # Compute the transformation matrix from CRTBP synodic frame to Earth-centered J2000 frame
        rsw_to_inertial_rotation_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(self.initial_cartesian_moon_state)

        # Compute time derivative element for the conversion of the velocity state elements and combine into final matrix
        omega_w_norm = self.rotation_rate
        m = self.gravitational_parameter_secondary/constants.GRAVITATIONAL_CONSTANT
        r_norm = np.linalg.norm(self.initial_cartesian_moon_state[:3])
        v_norm = np.linalg.norm(self.initial_cartesian_moon_state[3:])
        h = m*r_norm*v_norm
        rotation_rate = h/(m*r_norm**2)
        omega_w_norm = rotation_rate
        Omega = np.array([[0, omega_w_norm, 0],[-omega_w_norm, 0, 0],[0, 0, 0]])

        time_derivative_rsw_to_inertial_rotation_matrix = -np.dot(rsw_to_inertial_rotation_matrix, Omega)
        total_rsw_to_inertial_rotation_matrix = np.block([[rsw_to_inertial_rotation_matrix, np.zeros((3,3))],
                        [time_derivative_rsw_to_inertial_rotation_matrix, rsw_to_inertial_rotation_matrix]])

        # Convert initial state defined in CRTBP synodic frame to Earth-centered J2000 frame
        initial_state_moon_fixed_lpf = np.concatenate((initial_state_moon_fixed[:3]*self.lu_cr3bp, initial_state_moon_fixed[3:6]*self.lu_cr3bp/self.tu_cr3bp))
        initial_state_moon_fixed_lumio = np.concatenate((initial_state_moon_fixed[6:9]*self.lu_cr3bp, initial_state_moon_fixed[9:12]*self.lu_cr3bp/self.tu_cr3bp))
        initial_state_lpf = self.initial_cartesian_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_moon_fixed_lpf)
        initial_state_lumio = self.initial_cartesian_moon_state + np.dot(total_rsw_to_inertial_rotation_matrix,initial_state_moon_fixed_lumio)
        initial_state = np.concatenate((initial_state_lpf, initial_state_lumio))

        return initial_state


    def get_closest_initial_state(self, step_size=0.05):

        self.set_initial_cartesian_moon_state()

        _, state_history_synodic = reference_data.get_synodic_state_history(constants.GRAVITATIONAL_CONSTANT,
                                                                            self.bodies.get("Earth").mass,
                                                                            self.bodies.get("Moon").mass,
                                                                            self.distance_between_primaries,
                                                                            14, # max 14 days, to save run time, full period halo orbit approximate
                                                                            step_size)
        reference_state_LUMIO = reference_data.get_reference_state_history(self.simulation_start_epoch_MJD, self.propagation_time, step_size=step_size, satellite=self.name_LPO, get_full_history=True, get_dict=False)
        distance_array = np.empty((0, 1))
        initial_state_history = np.empty((0, 12))
        for initial_state_barycenter_fixed in state_history_synodic:

            self.initial_state = self.convert_synodic_to_inertial_state(initial_state_barycenter_fixed)

            distance = np.linalg.norm(self.initial_state[6:12]-reference_state_LUMIO[0])

            distance_array = np.vstack((distance_array, distance))
            initial_state_history = np.vstack((initial_state_history, self.initial_state))

        min_distance_index = np.argmin(distance_array)
        closest_initial_state = initial_state_history[min_distance_index]


        # ax = plt.figure().add_subplot(projection='3d')
        # plt.plot(initial_state_history[:, 6], initial_state_history[:, 7], initial_state_history[:, 8])
        # plt.plot(reference_state_LUMIO[0, 0], reference_state_LUMIO[0,1], reference_state_LUMIO[0,2], marker='o', color="red")
        # plt.plot(closest_initial_state[6], closest_initial_state[7], closest_initial_state[8], marker='o', color="blue")
        # ax.set_xlabel('X [m]')
        # ax.set_ylabel('Y [m]')
        # ax.set_zlabel('Z [m]')
        # plt.show()

        # plt.plot(distance_array)
        # plt.show()

        return closest_initial_state


    def set_environment_settings(self):

        # Create default body settings
        self.body_settings = environment_setup.get_default_body_settings(
            self.bodies_to_create, self.global_frame_origin, self.global_frame_orientation)

        # Create spacecraft bodies
        for index, body in enumerate(self.bodies_to_propagate):
            self.body_settings.add_empty_settings(body)
            self.body_settings.get(body).constant_mass = self.bodies_mass[index]

        # Adjust the orbit of the moon to perfectly circular
        self.set_initial_cartesian_moon_state()

        # Add ephemeris settings to body settings of the Moon
        self.body_settings.get(self.name_secondary).ephemeris_settings = environment_setup.ephemeris.keplerian(
            [3.84747963e+08, 0.00000000e+00, 4.98236095e-01, 5.86774306e+00, 5.08985834e-02, 2.70556877e+00],
            764251200.0,
            self.central_body_gravitational_parameter,
            self.global_frame_origin,
            self.global_frame_orientation)

        # # Folder to save the dictionary
        # folder_path = "C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/src/dynamic_models/low_fidelity/three_body_problem"
        # # File path for saving the JSON file
        # import json
        # file_path = os.path.join(folder_path, "moon_history.json")

        # # Read the JSON file and load it into a dictionary
        # with open(file_path, 'r') as json_file:
        #     data_dict_opened = json.load(json_file)

        # for key, value in data_dict_opened.items():
        #     # print(type(key))
        #     key = float(key)
        #     # print(type(key))
        # data_dict_opened = {float(key): value for key, value in data_dict_opened.items()}
        # self.body_settings.get(self.name_secondary).ephemeris_settings = environment_setup.ephemeris.tabulated(data_dict_opened,
        #                                                                                         self.global_frame_origin,
        #                                                                                         self.global_frame_orientation)

        # define parameters describing the synchronous rotation model for Moon fixed frame w.r.t. central body Earth
        self.body_settings.get(self.name_secondary).rotation_model_settings = environment_setup.rotation_model.synchronous(
            self.name_primary, self.global_frame_orientation, "Moon_fixed")

        # Update the body settings
        self.bodies = environment_setup.create_system_of_bodies(self.body_settings)


    def set_acceleration_settings(self):

        self.set_environment_settings()

        # Define accelerations acting on vehicle.
        self.acceleration_settings_on_spacecrafts = dict()
        for index, spacecraft in enumerate([self.name_ELO, self.name_LPO]):
            acceleration_settings_on_spacecraft = {
                    self.name_primary: [propagation_setup.acceleration.point_mass_gravity()],
                    self.name_secondary: [propagation_setup.acceleration.point_mass_gravity()]
            }
            self.acceleration_settings_on_spacecrafts[spacecraft] = acceleration_settings_on_spacecraft

        # Create global accelerations dictionary.
        self.acceleration_settings = self.acceleration_settings_on_spacecrafts

        # Create acceleration models.
        self.acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, self.acceleration_settings, self.bodies_to_propagate, self.central_bodies)


    def set_initial_state(self):

        self.set_acceleration_settings()

        if self.custom_initial_state is None:
            self.initial_state = self.get_closest_initial_state()
        else:
            if self.use_synodic_state:
                self.initial_state = self.convert_synodic_to_inertial_state(self.custom_initial_state)
                self.custom_initial_state[0] = self.custom_initial_state[0] + (1-self.mu)
                self.custom_initial_state[6] = self.custom_initial_state[6] + (1-self.mu)
            else:
                self.initial_state = self.custom_initial_state
                # print("initial state used: ", self.initial_state)


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


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
#                                 1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])
# test1 = LowFidelityDynamicModel(60390, 28, custom_initial_state=custom_initial_state, use_synodic_state=True)
# test1 = LowFidelityDynamicModel(60390, 28)
# states1 = np.stack(list(test1.get_propagation_simulator()[0].state_history.values()))

# custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
#                                 1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])
# test2 = LowFidelityDynamicModel(60396, 28, custom_initial_state=custom_initial_state, use_synodic_state=True)
# test2 = LowFidelityDynamicModel(60396, 28)
# states2 = np.stack(list(test2.get_propagation_simulator()[0].state_history.values()))



# plt.plot(states1[:,0], states1[:,1], states1[:,2])
# plt.plot(states1[:,6], states1[:,7], states1[:,8])
# plt.plot(states2[:,0], states2[:,1], states2[:,2])
# plt.plot(states2[:,6], states2[:,7], states2[:,8])
# plt.show()


# from dynamic_models import Interpolator



# step_size = 0.01

# test0 = LowFidelityDynamicModel(60390, 28)
# epochs0 = np.stack(list(test0.get_propagation_simulator()[0].state_history.keys()))
# states0 = np.stack(list(test0.get_propagation_simulator()[0].state_history.values()))
# print(epochs0[0], epochs0[-1], epochs0[-1]-epochs0[0])
# print(states0[0,:], states0[-1,:])

# epochs0, state_history0, dependent_variables_history0 = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(test0,
#                                                                                             custom_initial_state=None,
#                                                                                             solve_variational_equations=False)

# print("STARTING HERE ===========")
# test1 = LowFidelityDynamicModel(60390, 1)
# epochs1 = np.stack(list(test1.get_propagation_simulator()[0].state_history.keys()))
# states1 = np.stack(list(test1.get_propagation_simulator()[0].state_history.values()))
# print("1 =================")
# # print(epochs1[0], epochs1[-1], epochs1[-1]-epochs1[0])
# # print(states1[0,:], states1[-1,:])
# # print("Difference: ", states1[0,:]-states0[-1,:])


# epochs, state_history1, dependent_variables_history1 = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(test1,
#                                                                                             custom_initial_state=None,
#                                                                                             # custom_propagation_time=1,
#                                                                                             solve_variational_equations=False)

# # print(epochs[0], epochs[-1], epochs[-1]-epochs[0])
# # print(state_history1[0,:], state_history1[-1,:])


# # custom_initial_state = np.array([-2.94494412e+08,  2.09254412e+08,  1.19603970e+08, -1.20598462e+02,
# #                                  -3.97387177e+02, -1.18349658e+03, -3.34395442e+08,  1.96596530e+08,
# #                                   1.38362397e+08, -8.43070472e+02, -8.68868628e+02, -6.65684845e+02])
# test2 = LowFidelityDynamicModel(60391, 1, custom_initial_state=state_history1[-1,:])
# epochs2 = np.stack(list(test2.get_propagation_simulator()[0].state_history.keys()))
# states2 = np.stack(list(test2.get_propagation_simulator()[0].state_history.values()))
# print("2 =================")
# # print(epochs2[0], epochs2[-1], epochs2[-1]-epochs2[0])
# # print(states2[0,:], states2[-1,:])
# # print("Difference: ", states2[0,:]-state_history1[-1,:])

# epochs, state_history2, dependent_variables_history2 = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(test2,
#                                                                                             custom_initial_state=state_history1[-1,:],
#                                                                                             # custom_propagation_time=1,
#                                                                                             solve_variational_equations=False)
# print(state_history2[0,:], state_history1[-1,:])

# # custom_initial_state = np.array([-3.04827426e+08,  1.98342823e+08,  9.38530195e+07, -3.39162394e+02,
# #                                  -3.53289967e+02, -3.67877005e+02, -3.67269391e+08,  1.58881973e+08,
# #                                   1.08554131e+08, -6.80914768e+02, -8.76074687e+02, -7.08399431e+02])
# test3 = LowFidelityDynamicModel(60392, 1, custom_initial_state=state_history2[-1,:])
# epochs3 = np.stack(list(test3.get_propagation_simulator()[0].state_history.keys()))
# states3 = np.stack(list(test3.get_propagation_simulator()[0].state_history.values()))
# print("3 =================")
# # print(epochs3[0], epochs3[-1], epochs3[-1]-epochs3[0])
# # print(states3[0,:], states3[-1,:])
# # print("Difference: ", states3[0,:]-state_history2[-1,:])

# epochs, state_history3, dependent_variables_history3 = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(test3,
#                                                                                             custom_initial_state=state_history2[-1,:],
#                                                                                             # custom_propagation_time=1,
#                                                                                             solve_variational_equations=False)
# print(state_history3[0,:], state_history2[-1,:])

# # custom_initial_state = np.array([-3.20733730e+08,  1.78164330e+08,  8.09048108e+07, -3.81352146e+02,
# #                                  -5.72246215e+02, -2.61197463e+02, -3.93597685e+08,  1.20914657e+08,
# #                                   7.75677652e+07, -5.39727735e+02, -8.80967175e+02, -7.23126700e+02])
# test4 = LowFidelityDynamicModel(60393, 1, custom_initial_state=state_history3[-1,:])
# states4 = np.stack(list(test4.get_propagation_simulator()[0].state_history.values()))

# epochs, state_history4, dependent_variables_history4 = \
#     Interpolator.Interpolator(epoch_in_MJD=False, step_size=step_size).get_propagation_results(test4,
#                                                                                             custom_initial_state=state_history3[-1,:],
#                                                                                             # custom_propagation_time=1,
#                                                                                             solve_variational_equations=False)
# print(state_history4[0,:], state_history3[-1,:])



# plt.plot(state_history0[:,0], state_history0[:,1], state_history0[:,2], color="gray")
# plt.plot(state_history0[:,6], state_history0[:,7], state_history0[:,8], color="gray", label="state_history0")
# plt.plot(dependent_variables_history0[:,0], dependent_variables_history0[:,1], dependent_variables_history0[:,2], color="orange")
# plt.plot(state_history1[:,0], state_history1[:,1], state_history1[:,2], color="blue")
# plt.plot(state_history1[:,6], state_history1[:,7], state_history1[:,8], color="blue", label="state_history1")
# plt.plot(dependent_variables_history1[:,0], dependent_variables_history1[:,1], dependent_variables_history1[:,2], color="blue", ls="--")
# plt.plot(state_history2[:,0], state_history2[:,1], state_history2[:,2], color="yellow")
# plt.plot(state_history2[:,6], state_history2[:,7], state_history2[:,8], color="yellow", label="state_history2")
# plt.plot(dependent_variables_history2[:,0], dependent_variables_history2[:,1], dependent_variables_history2[:,2], color="yellow", ls="--")
# plt.plot(state_history3[:,0], state_history3[:,1], state_history3[:,2], color="green")
# plt.plot(state_history3[:,6], state_history3[:,7], state_history3[:,8], color="green", label="state_history3")
# plt.plot(dependent_variables_history3[:,0], dependent_variables_history3[:,1], dependent_variables_history3[:,2], color="green", ls="--")
# plt.plot(state_history4[:,0], state_history4[:,1], state_history4[:,2], color="red")
# plt.plot(state_history4[:,6], state_history4[:,7], state_history4[:,8], color="red", label="state_history4")
# plt.plot(dependent_variables_history4[:,0], dependent_variables_history4[:,1], dependent_variables_history4[:,2], color="red", ls="--")




# # Create a dictionary
# data_dict = {epoch: state for epoch, state in zip(epochs0, dependent_variables_history0[:,:6])}

# # Convert NumPy arrays to lists in the dictionary
# for key, value in data_dict.items():
#     if isinstance(value, np.ndarray):
#         data_dict[float(key)] = value.tolist()

# print(data_dict)

# # Folder to save the dictionary
# folder_path = "C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/src/dynamic_models/low_fidelity/three_body_problem"

# # Check if the folder exists, if not create it
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)

# # File path for saving the JSON file
# file_path = os.path.join(folder_path, "moon_history.json")

# # import json

# # # Save the dictionary as a JSON file
# # with open(file_path, 'w') as json_file:
# #     json.dump(data_dict, json_file, indent=4)


# # Read the JSON file and load it into a dictionary
# # import json
# # with open(file_path, 'r') as json_file:
# #     data_dict_opened = json.load(json_file)


# # epoch_dict = np.stack(list(data_dict_opened.keys()))
# # values_dict = np.stack(list(data_dict_opened.values()))


# # plt.plot(values_dict[:,0], values_dict[:,1], values_dict[:,2], color="blue", ls="--", label="moon from dict")



# plt.legend()
# plt.show()