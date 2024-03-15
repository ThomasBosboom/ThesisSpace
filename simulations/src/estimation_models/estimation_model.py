# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

# Tudatpy imports
from tudatpy import util
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Own
from dynamic_models.full_fidelity import *
from dynamic_models.low_fidelity.three_body_problem import *
from dynamic_models.high_fidelity.point_mass import *
from dynamic_models.high_fidelity.point_mass_srp import *
from dynamic_models.high_fidelity.spherical_harmonics import *
from dynamic_models.high_fidelity.spherical_harmonics_srp import *


np.random.seed(0)

class EstimationModel:

    def __init__(self, dynamic_model, truth_model, apriori_covariance=None, initial_estimation_error=None, include_consider_parameters=False):

        # Loading dynamic model
        self.dynamic_model = dynamic_model
        self.truth_model = truth_model

        # Loading apriori covariance
        self.apriori_covariance = apriori_covariance
        self.include_consider_parameters = include_consider_parameters

        # Setting up initial state error
        self.initial_estimation_error = initial_estimation_error

        # Defining basis for observations
        self.bias_range = 0
        self.bias_doppler = 0
        self.noise_range = 102.44
        self.noise_doppler = 0.00097
        self.observation_step_size_range = 600
        self.observation_step_size_doppler = 600
        self.retransmission_delay = 6
        self.integration_time = 0.5
        self.time_drift_bias = 6.9e-10

        # Creating observation time vector
        margin = 60
        self.observation_times_range = np.arange(self.dynamic_model.simulation_start_epoch+margin, self.dynamic_model.simulation_end_epoch-margin, self.observation_step_size_range)
        self.observation_times_doppler = np.arange(self.dynamic_model.simulation_start_epoch+margin, self.dynamic_model.simulation_end_epoch-margin, self.observation_step_size_doppler)

        self.arc_start_epoch = self.dynamic_model.simulation_start_epoch
        self.arc_end_epoch = self.dynamic_model.simulation_end_epoch

    def set_observation_model_settings(self):

        self.dynamic_model.set_environment_settings()

        # Define the uplink link ends for one-way observable
        link_ends = {observation.transmitter: observation.body_origin_link_end_id(self.dynamic_model.name_LPO),
                     observation.retransmitter: observation.body_origin_link_end_id(self.dynamic_model.name_ELO),
                     observation.receiver: observation.body_origin_link_end_id(self.dynamic_model.name_LPO)}
        self.link_definition = observation.LinkDefinition(link_ends)

        self.link_definition = {"two_way_system": self.link_definition}

        # Define settings for light-time calculations
        light_time_correction_settings = list()
        correcting_bodies = list(set(self.truth_model.bodies_to_create).intersection(self.dynamic_model.bodies_to_create))
        for correcting_body in correcting_bodies:
            light_time_correction_settings.append(observation.first_order_relativistic_light_time_correction(correcting_bodies))

        # Define settings for range and doppler bias
        range_bias_settings = observation.combined_bias([observation.absolute_bias([self.bias_range]),
                                                         observation.time_drift_bias(bias_value = np.array([self.time_drift_bias]),
                                                                                     time_link_end = observation.transmitter,
                                                                                     ref_epoch = self.dynamic_model.simulation_start_epoch)])

        # Create observation settings for each link/observable
        self.observation_settings_list = list()
        for link in self.link_definition.keys():
            self.observation_settings_list.extend([observation.two_way_range(self.link_definition[link],
                                                                        light_time_correction_settings = light_time_correction_settings,
                                                                        bias_settings = range_bias_settings)])


    def set_observation_simulation_settings(self):

        self.set_observation_model_settings()

        # Define observation simulation times for each link
        self.observation_simulation_settings = list()
        for link in self.link_definition.keys():
            self.observation_simulation_settings.extend([observation.tabulated_simulation_settings(observation.n_way_range_type,
                                                                                                self.link_definition[link],
                                                                                                self.observation_times_range,
                                                                                                reference_link_end_type = observation.transmitter)])

        # Add noise levels to observations
        observation.add_gaussian_noise_to_observable(
            self.observation_simulation_settings,
            self.noise_range,
            observation.n_way_range_type)

        # Provide ancillary settings for n-way observables
        observation.two_way_range_ancilliary_settings(retransmission_delay = self.retransmission_delay)

        # Create viability settings
        viability_setting_list = [observation.body_occultation_viability([self.dynamic_model.name_ELO, self.dynamic_model.name_LPO], self.dynamic_model.name_secondary)]

        observation.add_viability_check_to_all(
            self.observation_simulation_settings,
            viability_setting_list)


    def set_observation_simulators(self):

        self.set_observation_simulation_settings()
        self.truth_model.set_propagator_settings()

        # Create or update the ephemeris of all propagated bodies (here: LPF and LUMIO) to match the propagated results
        self.truth_model.propagator_settings.processing_settings.set_integrated_result = True

        # Run propagation
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(self.truth_model.bodies, self.truth_model.propagator_settings)

        # Create observation simulators
        self.observation_simulators = estimation_setup.create_observation_simulators(
            self.observation_settings_list, self.truth_model.bodies)


    def set_simulated_observations(self):

        self.set_observation_simulators()

        # Get LPF and LUMIO simulated observations as ObservationCollection
        self.simulated_observations = estimation.simulate_observations(
            self.observation_simulation_settings,
            self.observation_simulators,
            self.truth_model.bodies)

        # Get sorted observation sets
        self.sorted_observation_sets = self.simulated_observations.sorted_observation_sets


    def set_parameters_to_estimate(self):

        self.set_simulated_observations()
        self.dynamic_model.set_propagator_settings()

        # Setup parameters settings to propagate the state transition matrix
        self.parameter_settings = estimation_setup.parameter.initial_states(self.dynamic_model.propagator_settings, self.dynamic_model.bodies)

        if self.include_consider_parameters:

            # Add estimated parameters to the sensitivity matrix that will be propagated
            self.parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(self.link_definition["two_way_system"], observation.n_way_range_type))
            self.parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(self.link_definition["two_way_system"], observation.n_way_averaged_doppler_type))
            self.parameter_settings.append(estimation_setup.parameter.gravitational_parameter(self.dynamic_model.name_primary))
            self.parameter_settings.append(estimation_setup.parameter.gravitational_parameter(self.dynamic_model.name_secondary))

            # Depending on the dynamic model
            # self.parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient(self.dynamic_model.name_ELO))
            # self.parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient(self.dynamic_model.name_LPO))

        # Create the parameters that will be estimated
        self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.dynamic_model.bodies)


    def set_estimator_settings(self, maximum_iterations=6):

        self.set_parameters_to_estimate()

        # Create the estimator
        self.estimator = numerical_simulation.Estimator(
            self.dynamic_model.bodies,
            self.parameters_to_estimate,
            self.observation_settings_list,
            self.dynamic_model.propagator_settings)

        # Save the true parameters to later analyse the error
        self.truth_parameters = self.parameters_to_estimate.parameter_vector

        # Perturb the initial state estimate from the truth (500 m in position, 0.001 m/s in velocity)
        import copy
        self.perturbed_parameters = self.truth_parameters[:12]
        if self.initial_estimation_error is not None:
            for i in range(3):
                for j in range(2):
                    self.perturbed_parameters[i+6*j] += self.initial_estimation_error[i+6*j]
                    self.perturbed_parameters[i+6*j+3] += self.initial_estimation_error[i+6*j+3]
        self.parameters_to_estimate.parameter_vector[:12] = self.perturbed_parameters

        # Create input object for the estimation
        convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=maximum_iterations)
        if self.apriori_covariance is None:
            self.estimation_input = estimation.EstimationInput(observations_and_times=self.simulated_observations,
                                                               convergence_checker=convergence_checker)
        else:
            self.estimation_input = estimation.EstimationInput(observations_and_times=self.simulated_observations,
                                                               convergence_checker=convergence_checker,
                                                               inverse_apriori_covariance=np.linalg.inv(self.apriori_covariance))

        # Set methodological options
        self.estimation_input.define_estimation_settings(reintegrate_variational_equations=False,
                                                         save_state_history_per_iteration=True)

        # Define weighting of the observations in the inversion
        weights_per_observable = {estimation_setup.observation.n_way_range_type: self.noise_range**-2}
        self.estimation_input.set_constant_weight_per_observable(weights_per_observable)


    def get_estimation_results(self, redirect_out=True):

        self.set_estimator_settings()

        # Run the estimation
        with util.redirect_std(redirect_out=redirect_out):
            estimation_output = self.estimator.perform_estimation(self.estimation_input)

        # Generate information and covariance histories based on all the combinations of observables and link definitions
        total_information_dict = dict()
        total_covariance_dict = dict()
        total_single_information_dict = dict()
        len_obs_list = []
        for i, (observable_type, observation_sets) in enumerate(self.sorted_observation_sets.items()):
            total_information_dict[observable_type] = dict()
            total_covariance_dict[observable_type] = dict()
            total_single_information_dict[observable_type] = dict()
            for j, observation_set in enumerate(observation_sets.values()):
                total_information_dict[observable_type][j] = list()
                total_covariance_dict[observable_type][j] = list()
                total_single_information_dict[observable_type][j] = list()
                for k, single_observation_set in enumerate(observation_set):

                    epochs = single_observation_set.observation_times
                    len_obs_list.append(len(epochs))

                    weighted_design_matrix_history = np.stack([estimation_output.weighted_design_matrix[sum(len_obs_list[:-1]):sum(len_obs_list), :]], axis=1)

                    information_dict = dict()
                    single_information_dict = dict()
                    information_vector_dict = dict()
                    total_information = 0
                    total_information_vector = 0
                    for index, weighted_design_matrix in enumerate(weighted_design_matrix_history):

                        epoch = epochs[index]
                        weighted_design_matrix_product = np.dot(weighted_design_matrix.T, weighted_design_matrix)

                        # Calculate the information matrix
                        current_information = total_information + weighted_design_matrix_product
                        single_information_dict[epoch] = weighted_design_matrix_product
                        information_dict[epoch] = current_information
                        total_information = current_information

                    covariance_dict = dict()
                    for key in information_dict:
                        if self.apriori_covariance is not None:
                            information_dict[key] = information_dict[key] + np.linalg.inv(self.apriori_covariance)
                        covariance_dict[key] = np.linalg.inv(information_dict[key])

                    total_information_dict[observable_type][j].append(information_dict)
                    total_covariance_dict[observable_type][j].append(covariance_dict)
                    total_single_information_dict[observable_type][j].append(single_information_dict)

        plot_residuals = False
        if plot_residuals:
            print("ESTIMATION PROPERTIES FOR THE ESTIMATION ARC")
            parameter_history = estimation_output.parameter_history
            residual_history = estimation_output.residual_history
            covariance = estimation_output.covariance
            formal_errors = estimation_output.formal_errors
            weighted_design_matrix = estimation_output.weighted_design_matrix
            residual_history = estimation_output.residual_history

            print("best iteration: ", estimation_output.best_iteration)
            # print("First: ", parameter_history[:,0])
            # print("Last: ", parameter_history[:,-1])
            # print("Diff: ", parameter_history[:,-1]-parameter_history[:,0])
            for i, (observable_type, information_sets) in enumerate(self.sorted_observation_sets.items()):
                for j, observation_set in enumerate(information_sets.values()):
                    for k, single_observation_set in enumerate(observation_set):

                        residual_history = estimation_output.residual_history

                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
                        subplots_list = [ax1, ax2, ax3, ax4]

                        index = int(len(single_observation_set.observation_times))
                        for l in range(4):
                            subplots_list[l].scatter(single_observation_set.observation_times, residual_history[i*index:(i+1)*index, l])
                            subplots_list[l].set_ylabel("Observation Residual")
                            subplots_list[l].set_title("Iteration "+str(l+1))

                        ax3.set_xlabel("Time since J2000 [s]")
                        ax4.set_xlabel("Time since J2000 [s]")

                        # plt.figure(figsize=(9,5))
                        # plt.hist(residual_history[i*index:(i+1)*index, 0], 25)
                        # plt.xlabel('Final iteration range-rate residual')
                        # plt.ylabel('Occurences [-]')
                        # plt.title('Histogram of residuals on final iteration')

                        plt.tight_layout()
                        # plt.show()


        return estimation_output, total_single_information_dict, \
               total_covariance_dict, total_information_dict, \
               self.sorted_observation_sets

test=False
if test:
    # custom_initial_state = np.array([0.985121349979458, 0.001476496155141, 0.004925468520363, -0.873297306080392, -1.611900486933861, 0,	\
    #                                 1.147342501,	-0.0002324517381, -0.151368318,	-0.000202046355,	-0.2199137166,	0.0002817105509])
    # dynamic_model = low_fidelity.LowFidelityDynamicModel(60390, 2, custom_initial_state=None, use_synodic_state=False)
    # truth_model = low_fidelity.LowFidelityDynamicModel(60390, 2, custom_initial_state=None, use_synodic_state=False)

    custom_initial_state =  np.array([-3.71540223e+08,  1.28764473e+08,  7.45403636e+07, -8.10536637e+01,
                                      -4.46441097e+02, -5.58911322e+02, -4.29595776e+08,  1.09374240e+08,
                                       6.28862754e+07, -4.01822951e+02, -8.28967368e+02, -6.92699430e+02])

    custom_initial_state_truth =  np.array([-3.71489884e+08,  1.28827089e+08,  7.45231487e+07, -8.51869762e+01,
                                            -4.42230343e+02, -5.49851439e+02, -4.29596784e+08,  1.09373730e+08,
                                            6.28861326e+07, -4.01829733e+02, -8.28967754e+02, -6.92696484e+02])

    dynamic_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(60392, 2, custom_initial_state=custom_initial_state)
    truth_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(60392, 2, custom_initial_state=custom_initial_state_truth)
    apriori_covariance = np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2
    initial_estimation_error = np.array([5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3, 5e2, 5e2, 5e2, 1e-3, 1e-3, 1e-3])
    estimation_model = EstimationModel(dynamic_model, truth_model, apriori_covariance=apriori_covariance, initial_estimation_error=initial_estimation_error)

    # [-3.19221439e+08  1.79045089e+08  1.01064488e+08 -2.03558866e+02
    #  -2.78869032e+02 -8.54302136e+02 -3.67010775e+08  1.58923594e+08
    #   1.08399977e+08 -6.68962205e+02 -8.73864980e+02 -7.14321553e+02]
    results = estimation_model.get_estimation_results(redirect_out=False)
    estimation_output = results[0]
    parameter_history = estimation_output.parameter_history
    residual_history = estimation_output.residual_history
    covariance = estimation_output.covariance
    formal_errors = estimation_output.formal_errors
    weighted_design_matrix = estimation_output.weighted_design_matrix
    residual_history = estimation_output.residual_history

    print("First: ", parameter_history[:,0])
    print("Last: ", parameter_history[:,-1])
    print("Diff: ", parameter_history[:,-1]-parameter_history[:,0])
    for i, (observable_type, information_sets) in enumerate(results[-1].items()):
        for j, observation_set in enumerate(information_sets.values()):
            for k, single_observation_set in enumerate(observation_set):

                residual_history = estimation_output.residual_history

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
                subplots_list = [ax1, ax2, ax3, ax4]

                index = int(len(single_observation_set.observation_times))
                for l in range(4):
                    subplots_list[l].scatter(single_observation_set.observation_times, residual_history[i*index:(i+1)*index, l])
                    subplots_list[l].set_ylabel("Observation Residual")
                    subplots_list[l].set_title("Iteration "+str(l+1))

                ax3.set_xlabel("Time since J2000 [s]")
                ax4.set_xlabel("Time since J2000 [s]")

                plt.figure(figsize=(9,5))
                plt.hist(residual_history[i*index:(i+1)*index, 0], 25)
                plt.xlabel('Final iteration range-rate residual')
                plt.ylabel('Occurences [-]')
                plt.title('Histogram of residuals on final iteration')

                plt.tight_layout()
                plt.show()



# from matplotlib.lines import Line2D
# import matplotlib.cm as cm
# # Corellation can be retrieved using the CovarianceAnalysisInput class:
# # covariance_input = estimation.CovarianceAnalysisInput(observation_collection)
# covariance_output = estimation_output.covariance

# correlations = estimation_output.correlations
# estimated_param_names = [r"$x_{1}$", r"$y_{1}$", r"$z_{1}$", r"$\dot{x}_{1}$", r"$\dot{y}_{1}$", r"$\dot{z}_{1}$",
#                          r"$x_{2}$", r"$y_{2}$", r"$z_{2}$", r"$\dot{x}_{2}$", r"$\dot{y}_{2}$", r"$\dot{z}_{2}$"]


# fig, ax = plt.subplots(1, 1, figsize=(9, 7))

# im = ax.imshow(correlations, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)

# ax.set_xticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)
# ax.set_yticks(np.arange(len(estimated_param_names)), labels=estimated_param_names)

# # add numbers to each of the boxes
# for i in range(len(estimated_param_names)):
#     for j in range(len(estimated_param_names)):
#         text = ax.text(
#             j, i, round(correlations[i, j], 2), ha="center", va="center", color="black"
#         )

# cb = plt.colorbar(im)

# ax.set_xlabel("Estimated Parameter")
# ax.set_ylabel("Estimated Parameter")
# fig.suptitle(f"Correlations for estimated parameters for LPF and LUMIO")
# fig.set_tight_layout(True)
# plt.show()