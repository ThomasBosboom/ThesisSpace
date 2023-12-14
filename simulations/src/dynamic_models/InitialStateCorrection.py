
class InitialStateCorrection(DynamicModelSetup):

    def set_ephemeris_settings(self):

        with open("lumio_crtbp_j2000.txt", 'r') as file:
            my_dict = json.load(file)
            new_dict = {float(key): value for key, value in my_dict.items()}

        print(new_dict)
            
        self.body_settings.get(self.name_spacecraft).ephemeris_settings = environment_setup.ephemeris.tabulated(new_dict,
            self.global_frame_origin,
            self.global_frame_orientation)


    def set_observation_settings(self):


        # Define the uplink link ends for one-way observable
        link_ends_lpf = {observation.observed_body: observation.body_origin_link_end_id(self.name_ELO)}
        self.link_definition_lpf = observation.LinkDefinition(link_ends_lpf)

        link_ends_lumio = {observation.observed_body: observation.body_origin_link_end_id(self.name_LPO)}
        self.link_definition_lumio = observation.LinkDefinition(link_ends_lumio)

        self.link_definition_dict = {
            self.name_ELO: self.link_definition_lpf,
            self.name_LPO: self.link_definition_lumio
        }

        self.position_observation_settings = [observation.cartesian_position(self.link_definition_lpf),
                                              observation.cartesian_position(self.link_definition_lumio),
                                             ]

        # Define settings for light-time calculations
        # light_time_correction_settings = observation.first_order_relativistic_light_time_correction(self.bodies_to_create)

        # Define settings for range and doppler bias
        # self.bias_level_range = 1.0E1
        # range_bias_settings = observation.absolute_bias([self.bias_level_range])

        # self.bias_level_doppler = 1.0E-3
        # doppler_bias_settings = observation.absolute_bias([self.bias_level_doppler])

        # Define observation simulation times for each link
        observation_times = np.arange(self.simulation_start_epoch+1000, self.simulation_end_epoch, 1000)

        self.observation_simulation_settings = [observation.tabulated_simulation_settings(observation.position_observable_type,
                                                                                          self.link_definition_lpf,
                                                                                          observation_times,
                                                                                          reference_link_end_type=estimation_setup.observation.observed_body),
                                                observation.tabulated_simulation_settings(observation.position_observable_type,
                                                                                          self.link_definition_lumio,
                                                                                          observation_times,
                                                                                          reference_link_end_type=estimation_setup.observation.observed_body),
        ]                  

        # Add noise levels of roughly 1.0E-3 [m/s] and add this as Gaussian noise to the observation
        # self.noise_level_range = 1.0E3
        # observation.add_gaussian_noise_to_observable(
        #     self.observation_simulation_settings,
        #     self.noise_level_range,
        #     observation.one_way_range_type
        # )

        # self.noise_level_doppler = 1.0E-3
        # observation.add_gaussian_noise_to_observable(
        #     self.observation_simulation_settings,
        #     self.noise_level_doppler,
        #     observation.one_way_instantaneous_doppler_type
        # )


    # def set_viability_settings(self):

    #     self.set_observation_settings()
        
    #     # Create viability settings
    #     viability_setting_list = [observation.body_occultation_viability([self.name_ELO, self.name_LPO], self.name_secondary)]
                            
    #     observation.add_viability_check_to_all(
    #         self.observation_simulation_settings,
    #         viability_setting_list
    #     )
        


    def set_simulated_observations(self):

        # self.set_viability_settings()
        self.set_propagator_settings()
        self.set_observation_settings()

        # Create observation simulators
        self.ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
            self.position_observation_settings, self.bodies)

        # Simulate required observations
        self.ephemeris_satellite_states = estimation.simulate_observations(
            self.observation_simulation_settings,
            self.ephemeris_observation_simulators,
            self.bodies)

        # Setup parameters settings to propagate the state transition matrix
        self.parameter_settings = estimation_setup.parameter.initial_states(self.propagator_settings, self.bodies)

        # Create the parameters that will be estimated
        self.parameters_to_estimate = estimation_setup.create_parameter_set(self.parameter_settings, self.bodies)
        self.original_parameter_vector = parameters_to_estimate.parameter_vector

        # Create the estimator
        self.estimator = numerical_simulation.Estimator(
            self.bodies,
            self.parameters_to_estimate,
            self.position_observation_settings,
            self.propagator_settings)



        # single_observation_set_lists = [self.simulated_observations.get_single_link_and_type_observations(observation.one_way_range_type, self.link_definition),
        #                                 self.simulated_observations.get_single_link_and_type_observations(observation.one_way_instantaneous_doppler_type, self.link_definition)
        #                                 ]

        # # self.observations = np.empty()
        # for single_observation_set_list in single_observation_set_lists:
        #     for single_observation_set in single_observation_set_list:
        #         # print("observations: ", single_observation_set.concatenated_observations, np.shape(single_observation_set.concatenated_observations))
        #         self.observations = single_observation_set.concatenated_observations
        #         times = single_observation_set.observation_times

        # # ax = plt.figure()
        # # plt.plot(times, self.observations, color="red", marker="o")
        # # plt.show()

        # return self.observations

    def perform_estimation(self):

        self.set_simulated_observations()

        print('Running propagation...')
        with util.redirect_std():
            estimator = numerical_simulation.Estimator(self.bodies, self.parameters_to_estimate,
                                                    self.position_observation_settings, self.propagator_settings)


        # Create input object for the estimation
        estimation_input = estimation.EstimationInput(self.ephemeris_satellite_states)
        # Set methodological options
        estimation_input.define_estimation_settings(save_state_history_per_iteration=True)
        # Perform the estimation
        print('Performing the estimation...')
        print(f'Original initial states: {self.original_parameter_vector}')


        with util.redirect_std(redirect_out=False):
            estimation_output = estimator.perform_estimation(estimation_input)
        initial_states_updated = self.parameters_to_estimate.parameter_vector
        print('Done with the estimation...')
        print(f'Updated initial states: {initial_states_updated}')

    
    def get_propagated_orbit_from_estimator(self):

        self.set_simulated_observations()

        # print("variational solver state history: ", self.estimator.variational_solver.state_history)
        # print("variational solver stm history: ", estimator.variational_solver.state_transition_matrix_history)
        # print("variational solver dynamics simulator dep var: ", estimator.variational_solver.dynamics_simulator.dependent_variable_history)

        return self.estimator.variational_solver.state_history, self.estimator.variational_solver.dynamics_simulator.dependent_variable_history, self.estimator.variational_solver.state_transition_matrix_history