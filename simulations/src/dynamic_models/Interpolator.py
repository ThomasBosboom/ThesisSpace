import numpy as np
from scipy.interpolate import interp1d
from tudatpy.kernel import constants
from tudatpy.kernel.astro import time_conversion


class Interpolator:

    def __init__(self, dynamic_model_object, step_size=0.005, kind='cubic', epoch_in_MJD=True):

        self.dynamic_model_object = dynamic_model_object
        self.custom_initial_state = self.dynamic_model_object.custom_initial_state
        print("Interpolator first: ", self.dynamic_model_object.custom_initial_state)
        self.step_size = step_size*constants.JULIAN_DAY
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


    def get_results(self):

        # Get simulation results from each dynamic model
        print("Interpolator second before: ", self.dynamic_model_object.custom_initial_state)
        print("dynamic_model_object: ", self.dynamic_model_object)
        self.dynamics_simulator, self.variational_equations_solver = self.dynamic_model_object.get_propagated_orbit()
        self.simulation_start_epoch = self.dynamic_model_object.simulation_start_epoch
        self.simulation_end_epoch = self.dynamic_model_object.simulation_end_epoch
        print("Interpolator second after: ", self.custom_initial_state)

        # Extract the variational_equations_solver results
        epochs                          = np.stack(list(self.variational_equations_solver.state_history.keys()))
        state_history                   = np.stack(list(self.variational_equations_solver.state_history.values()))
        state_transition_matrix_history = np.stack(list(self.variational_equations_solver.state_transition_matrix_history.values()))

        # Define updated time vector that is the same for all dynamic models irrespective of their own time vector
        interp_epochs = np.arange(self.simulation_start_epoch, self.simulation_end_epoch+self.step_size, self.step_size)

        # Perform interpolation using on the results from self.variational_equations_solver
        interp_state_history = self.interp_function(epochs, interp_epochs, state_history)
        interp_state_transition_matrix_history = self.interp_function(epochs, interp_epochs, state_transition_matrix_history)

        # Perform interpolation using on the results from dynamics_simulator
        epochs                          = np.stack(list(self.dynamics_simulator.dependent_variable_history.keys()))
        dependent_variables_history     = np.stack(list(self.dynamics_simulator.dependent_variable_history.values()))
        interp_dependent_variables_history = self.interp_function(epochs, interp_epochs, dependent_variables_history)

        if self.epoch_in_MJD:
            interp_epochs = np.array([time_conversion.julian_day_to_modified_julian_day(\
                time_conversion.seconds_since_epoch_to_julian_day(interp_epoch)) for interp_epoch in interp_epochs])

        return interp_epochs, interp_state_history, interp_dependent_variables_history, interp_state_transition_matrix_history