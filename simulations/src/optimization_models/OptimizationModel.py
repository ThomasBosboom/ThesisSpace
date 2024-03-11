# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import scipy as sp
from warnings import warn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Own
from dynamic_models import NavigationSimulator, PlotNavigationResults
from tests import utils


class OptimizationModel():

    def __init__(self, dynamic_model_list, truth_model_list, threshold=8, duration=14, options=None):

        # Managing the dynamic model specifications
        self.model_type, self.model_name, self.model_number = dynamic_model_list[0], dynamic_model_list[1], dynamic_model_list[2]
        self.model_type_truth, self.model_name_truth, self.model_number_truth = truth_model_list[0], truth_model_list[1], truth_model_list[2]

        # Timing parameters
        self.mission_start_time = 60390
        self.threshold = threshold + self.mission_start_time
        self.duration = duration + self.mission_start_time

        self.options = options
        if options is None:
            self.options = {
                "t_od": {
                    "use_in_vec": True,
                    "min_t_to_skm": 0.2,
                    "max_t_to_skm": 1.5,
                    "t_to_skm": 1, # must be bigger than min_t_to_skm
                    "t_cut_off": 0
                },
                "t_skm": {
                    "use_in_vec": False,
                    "max_var": 0.5,
                    "skm_freq": 4,
                    "skm_at_threshold": True,
                    "custom_skms": None #[60394.5, 60395, 60455]
                }
            }


    def get_initial_design_vector_dict(self):

        t_to_skm = self.options["t_od"]["t_to_skm"]
        min_t_to_skm = self.options["t_od"]["min_t_to_skm"]
        max_t_to_skm = self.options["t_od"]["max_t_to_skm"]
        skm_freq = self.options["t_skm"]["skm_freq"]

        skms = np.arange(self.mission_start_time, self.duration, skm_freq)

        # Remove values smaller than the threshold in the first list
        design_vector_dict = dict()
        for key, value in self.options.items():
            if key == "t_od":
                list = skms-t_to_skm
            if key == "t_skm":
                list = skms
            design_vector_dict[key] = [x for x in list if x >= self.threshold]

            if key == "t_skm":
                if value["custom_skms"] is not None:

                    skm_list = value["custom_skms"]
                    if not all(skm_list[i] <= skm_list[i + 1] for i in range(len(skm_list) - 1)):
                        warn(f'Custom SKMs in list are not chronological order, automatically sorted', RuntimeWarning)
                        skm_list = sorted(skm_list)
                    design_vector_dict[key] = skm_list
                    design_vector_dict["t_od"] = [t_skm-t_to_skm for t_skm in skm_list]

                if not value["skm_at_threshold"]:
                    for i, epoch in enumerate(design_vector_dict[key]):
                        if design_vector_dict[key][i] == self.threshold:
                            design_vector_dict[key].remove(design_vector_dict[key][i])

                for key, value in design_vector_dict.items():
                    for epoch in value:
                        if epoch < self.mission_start_time:
                            warn(f'Epoch {epoch} of {key} has value that is before minimum start epoch of MJD {self.mission_start_time}', RuntimeWarning)
                        if epoch > self.duration:
                            warn(f'Epoch {epoch} of {key} has value that is after final duration epoch of MJD {self.duration}', RuntimeWarning)

                    design_vector_dict[key] = [x for x in design_vector_dict[key] if x >= self.threshold and x<=self.duration]

        # Some fault handling
        if t_to_skm > skm_freq:
            raise ValueError('Orbit determination of next SKM happens before current SKM')

        if t_to_skm < min_t_to_skm:
            raise ValueError('OD time to next SKM is smaller than required minimum')

        if max_t_to_skm < t_to_skm:
            raise ValueError('Maximum time to next SKM is smaller than currently set time to next SKM')

        # for i in range(len(design_vector_dict["t_skm"])):
        #     if design_vector_dict["t_skm"][i] > design_vector_dict["t_od"][i+1]:
        #         raise ValueError('Current t_od is smaller than previous SKM epoch')

        return design_vector_dict


    def get_initial_design_vector(self):

        design_vector_dict = self.get_initial_design_vector_dict()

        design_vector = []
        for key, value in self.options.items():

            if value["use_in_vec"]:
                design_vector.extend(design_vector_dict[key])

        return design_vector


    def get_design_vector_bounds(self):

        design_vector_dict = self.get_initial_design_vector_dict()

        bounds_dict = dict()
        for key, value in self.options.items():
            skm_list = np.array(design_vector_dict["t_skm"])
            t_od_list = np.array(design_vector_dict["t_od"])
            if value["use_in_vec"]:
                if key == "t_skm":
                    bounds_dict[key] = list(zip(skm_list-self.options[key]["max_var"], skm_list+self.options[key]["max_var"]))

                if key == "t_od":
                    upper_bounds = np.array([abs(x - y) for x, y in zip(t_od_list, skm_list)])
                    bounds_dict[key] = list(zip(skm_list-self.options[key]["max_t_to_skm"], skm_list-self.options[key]["min_t_to_skm"]))

        if self.threshold in skm_list:
            bounds_dict["t_od"] = bounds_dict["t_od"][1:]

        return bounds_dict


    def get_initial_observation_windows(self):

        design_vector_dict = self.get_initial_design_vector_dict()

        observation_windows = [(60390, self.threshold)]
        a = 0
        if len(design_vector_dict["t_skm"]) > len(design_vector_dict["t_od"]):
            a = 1
        observation_windows.extend([(design_vector_dict["t_od"][i], design_vector_dict["t_skm"][i+a]) for i in range(len(design_vector_dict["t_od"]))])

        return observation_windows


    def objective_function(self, x):

        print("Design vector: ", x)

        # observation_windows = self.get_initial_observation_windows()
        station_keeping_epochs = self.get_initial_design_vector_dict()["t_skm"]
        if self.threshold in station_keeping_epochs:
            station_keeping_epochs = station_keeping_epochs[1:]

        observation_windows = [(self.mission_start_time, self.threshold)]
        observation_windows.extend([(x[i], station_keeping_epochs[i]) for i in range(len(station_keeping_epochs))])
        station_keeping_epochs = self.get_initial_design_vector_dict()["t_skm"]
        print("observation windows in objective: \n", observation_windows)
        print("custom station keeping windows: \n", station_keeping_epochs)

        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                       [self.model_type, self.model_name, self.model_number],
                                                                       [self.model_type_truth, self.model_name_truth, self.model_number_truth],
                                                                       custom_station_keeping_epochs=station_keeping_epochs
                                                                       )

        # print("station keeping epochs: ", navigation_simulator.station_keeping_epochs)
        results = []
        for result_dict in navigation_simulator.perform_navigation():
            if result_dict:
                results.append(utils.convert_dictionary_to_array(result_dict))
            else: # if there is no station keeping, so empty delta_v_dict (last entry)
                results.append(([],[]))
        results.append((navigation_simulator))

        results_dict = {self.model_type: {self.model_name: [results]}}
        # print("RESULTS_DICT: ", results_dict)
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_formal_error_history()
        PlotNavigationResults.PlotNavigationResults(results_dict).plot_uncertainty_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_reference_deviation_history()
        PlotNavigationResults.PlotNavigationResults(results_dict).plot_estimation_error_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_full_state_history()
        # plt.show()

        delta_v = results[8][1]
        objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
        print(f"objective value at {x}: ", objective_value)

        return objective_value




    def optimize(self):

        # Initial guess for the design vector
        x0 = self.get_initial_design_vector()
        print("Initial state: \n", x0)

        # Define boundaries for the design vector entries
        # xl = x0 + a*np.ones(len(x0))
        # xu = x0 + b*np.ones(len(x0))
        # print("Lower bound: \n", xl)
        # print("Upper bound: \n", xu)

        # Define constraints
        # constraints = (
            # {'type': 'ineq', 'fun': self.inequality_constraint1},
            # {'type': 'ineq', 'fun': self.inequality_constraint2},
            # {'type': 'ineq', 'fun': self.inequality_constraint3}
        # )

        # Define bounds
        bounds_dict = self.get_design_vector_bounds()
        bounds_list = []
        for key, value in bounds_dict.items():
            bounds_list.extend(value)
        print(bounds_list)

        # Define a callback function to record iteration history
        iteration_history = []
        design_vectors = []
        def callback(xk):
            iteration_history.append(self.objective_function(xk))
            design_vectors.append(xk)

        # Minimize the objective function subject to constraints
        result = sp.optimize.minimize(self.objective_function, x0,
                                    # constraints=constraints,
                                    bounds=bounds_list,
                                    method='Nelder-Mead',
                                    options={'maxiter': 3, "xatol": 1e-2, 'disp': True},
                                    callback=callback)

        # Extract optimized start times
        x_optim = result.x
        print("x_optim:", x_optim)
        print("iteration history: \n", iteration_history)
        print("design vector history: \n", np.array(design_vectors))

        plt.plot(iteration_history)
        plt.yscale("log")


        return iteration_history, np.array(design_vectors)





dynamic_model_list = ["low_fidelity", "three_body_problem", 0]
truth_model_list = ["low_fidelity", "three_body_problem", 0]
# dynamic_model_list = ["high_fidelity", "point_mass", 0]
# truth_model_list = ["high_fidelity", "point_mass", 0]



optimization_model = OptimizationModel(dynamic_model_list, truth_model_list,
                                threshold=8, duration=14, options=None)


print(optimization_model.get_initial_observation_windows())
print(optimization_model.get_initial_design_vector())
print(optimization_model.get_design_vector_bounds())

# iteration_history, design_vectors = optimization_model.optimize()
# print(iteration_history, design_vectors)

# plt.show()


