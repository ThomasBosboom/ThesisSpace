import numpy as np
from scipy.interpolate import interp1d
from tudatpy.kernel import constants
from tudatpy.kernel.astro import time_conversion


class Interpolator:

    def __init__(self, step_size=0.005, kind='cubic', epoch_in_MJD=True):

        self.step_size = step_size
        self.kind = kind
        self.epoch_in_MJD = epoch_in_MJD


    def interp_function(self, epochs, interp_epochs, history):

        if history.ndim == 2:
            interp_function = interp1d(epochs, history, axis=0, kind=self.kind, fill_value='extrapolate')
            return interp_function(interp_epochs)
        else:
            interpolated_history = np.zeros((len(interp_epochs), *history.shape[1:]))
            for i in range(history.shape[1]):
                interp_function = interp1d(epochs, history[:, i, :], axis=0, kind=self.kind, fill_value='extrapolate')
                interpolated_history[:, i, :] = interp_function(interp_epochs)
            return interpolated_history


    def get_propagation_results(self, dynamic_model, solve_variational_equations=True, **kwargs):

        for key, value in kwargs.items():
            if hasattr(dynamic_model, key):
                setattr(dynamic_model, key, value)

        step_size = self.step_size*constants.JULIAN_DAY

        # Get simulation results from each dynamic model
        if solve_variational_equations:
            dynamics_simulator, variational_equations_solver = dynamic_model.get_propagation_simulator(solve_variational_equations=solve_variational_equations)
        else:
            dynamics_simulator = dynamic_model.get_propagation_simulator(solve_variational_equations=solve_variational_equations)

        simulation_start_epoch = dynamic_model.simulation_start_epoch
        simulation_end_epoch = dynamic_model.simulation_end_epoch

        # Define updated time vector that is the same for all dynamic models irrespective of their own time vector
        interp_epochs = np.arange(simulation_start_epoch, simulation_end_epoch+step_size, step_size)
        # print("Interpolator: ", simulation_start_epoch, simulation_end_epoch+step_size, len(interp_epochs))

        if solve_variational_equations:
             # Extract the variational_equations_solver results
            epochs                          = np.stack(list(variational_equations_solver.state_transition_matrix_history.keys()))
            state_transition_matrix_history = np.stack(list(variational_equations_solver.state_transition_matrix_history.values()))

            # Perform interpolation
            interp_state_transition_matrix_history = self.interp_function(epochs, interp_epochs, state_transition_matrix_history)

        # Perform interpolation using on the results from dynamics_simulator
        epochs                          = np.stack(list(dynamics_simulator.state_history.keys()))
        state_history                   = np.stack(list(dynamics_simulator.state_history.values()))
        dependent_variables_history     = np.stack(list(dynamics_simulator.dependent_variable_history.values()))

        # Perform interpolation
        interp_state_history = self.interp_function(epochs, interp_epochs, state_history)
        interp_dependent_variables_history = self.interp_function(epochs, interp_epochs, dependent_variables_history)

        if self.epoch_in_MJD:
            interp_epochs = np.array([time_conversion.julian_day_to_modified_julian_day(\
                time_conversion.seconds_since_epoch_to_julian_day(interp_epoch)) for interp_epoch in interp_epochs])

        if solve_variational_equations:
            return interp_epochs, interp_state_history, interp_dependent_variables_history, interp_state_transition_matrix_history
        else:
            return interp_epochs, interp_state_history, interp_dependent_variables_history