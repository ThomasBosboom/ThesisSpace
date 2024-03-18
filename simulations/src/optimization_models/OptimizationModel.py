# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import scipy as sp
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Own
from dynamic_models import NavigationSimulator, PlotNavigationResults
from tests import utils

class OptimizationModel():

    def __init__(self, dynamic_model_list, truth_model_list, threshold=8, duration=14, skm_to_od_duration=3, od_duration=1, bounds=(0.5, 1.5)):

        # Specify the dynamic and truth model used for the estimation arcs
        self.model_type, self.model_name, self.model_number = dynamic_model_list[0], dynamic_model_list[1], dynamic_model_list[2]
        self.model_type_truth, self.model_name_truth, self.model_number_truth = truth_model_list[0], truth_model_list[1], truth_model_list[2]

        # Timing parameters
        self.mission_start_time = 60390
        self.threshold = threshold + self.mission_start_time
        self.duration = duration + self.mission_start_time

        self.skm_to_od_duration = skm_to_od_duration
        self.od_duration = od_duration

        skm_epochs = [self.threshold]
        while skm_epochs[-1]<self.duration:
            skm_epochs.append(skm_epochs[-1] + self.skm_to_od_duration + self.od_duration)
        self.skm_epochs = skm_epochs[:-1]

        self.observation_windows = []
        for i, skm_epoch in enumerate(self.skm_epochs):
            window = (skm_epoch-self.od_duration, skm_epoch)
            if i == 0:
                window = (self.mission_start_time, self.threshold)

            self.observation_windows.append(window)

        self.vec_len = len(self.skm_epochs)-1

        self.initial_design_vector = self.od_duration*np.ones(self.vec_len)

        self.design_vector_bounds = list(zip(bounds[0]*np.ones(self.vec_len), bounds[-1]*np.ones(self.vec_len)))

        # if od_duration > skm_to_od_duration:
        #     print("Warning: OD duration is larger than time between SKMs")


    def get_updated_skm_epochs(self, x):

        new_skm_epochs = [self.threshold]
        for i, epoch in enumerate(self.skm_epochs[1:]):
            new_skm_epochs.append(new_skm_epochs[-1]+self.skm_to_od_duration+x[i])

        return new_skm_epochs


    def get_updated_observation_windows(self, x):

        new_skm_epochs = self.get_updated_skm_epochs(x)
        new_observation_windows = [(self.mission_start_time, self.threshold)]
        for i, skm_epoch in enumerate(new_skm_epochs[1:]):
            new_observation_windows.append((skm_epoch-x[i], skm_epoch))

        return new_observation_windows


    def objective_function(self, x):

        station_keeping_epochs = self.get_updated_skm_epochs(x)
        observation_windows = self.get_updated_observation_windows(x)
        target_point_epochs = [self.skm_to_od_duration]
        # target_point_epochs = [7, 14]

        print("design vector in objective: \n", x)
        print("observation windows in objective: \n", observation_windows)
        print("station keeping windows in objective: \n", station_keeping_epochs)
        print("target_point_epochs in objective: \n", target_point_epochs)

        navigation_simulator = NavigationSimulator.NavigationSimulator(observation_windows,
                                                                    [self.model_type, self.model_name, self.model_number],
                                                                    [self.model_type_truth, self.model_name_truth, self.model_number_truth],
                                                                    custom_station_keeping_epochs=station_keeping_epochs,
                                                                    target_point_epochs=target_point_epochs)

        results = []
        for result_dict in navigation_simulator.perform_navigation():
            if result_dict:
                results.append(utils.convert_dictionary_to_array(result_dict))
            else: # if there is no station keeping, so empty delta_v_dict (last entry)
                results.append(([],[]))
        results.append((navigation_simulator))

        results_dict = {self.model_type: {self.model_name: [results]}}
        # print("RESULTS_DICT: ", results_dict)
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_estimation_error_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_uncertainty_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_reference_deviation_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_full_state_history()
        # PlotNavigationResults.PlotNavigationResults(results_dict).plot_formal_error_history()

        # plt.show()

        delta_v = results[8][1]
        objective_value = np.sum(np.linalg.norm(delta_v, axis=1))
        print(f"objective value at {x}: \n", delta_v, objective_value)

        return objective_value


    def optimize(self):

        # Initial guess for the design vector
        x0 = self.initial_design_vector
        print("Initial design vector: \n", x0)

        # Define a callback function to record iteration history
        iteration_history = []
        design_vectors = []

        def callback(xk):
            callback.iteration += 1
            print("Iteration:", callback.iteration, xk, self.objective_function(xk))
            design_vectors.append(xk)
            iteration_history.append(self.objective_function(xk))

        callback.iteration = 0

        # Minimize the objective function subject to constraints
        result = sp.optimize.minimize(self.objective_function, x0,
                                        bounds=self.design_vector_bounds,
                                        method='Nelder-Mead',
                                        options={'maxiter': 8, "return_all": True, 'disp': True},
                                        callback=callback)

        # Extract optimized start times
        x_optim = result.x
        print("x_optim:", x_optim)
        # print("iteration history: \n", iteration_history)
        # print("design vector history: \n", np.array(design_vectors))

        plt.plot(iteration_history)
        plt.yscale("log")
        plt.show()

        result_dict = dict(zip(iteration_history, np.array(design_vectors)))
        print(result_dict)

        return result_dict




dynamic_model_list = ["low_fidelity", "three_body_problem", 0]
truth_model_list = ["low_fidelity", "three_body_problem", 0]
dynamic_model_list = ["high_fidelity", "point_mass", 0]
truth_model_list = ["high_fidelity", "point_mass", 0]
# dynamic_model_list = ["high_fidelity", "point_mass_srp", 0]
# truth_model_list = ["high_fidelity", "point_mass_srp", 0]
# dynamic_model_list = ["high_fidelity", "spherical_harmonics_srp", 1]
# truth_model_list = ["high_fidelity", "spherical_harmonics_srp", 1]


model = "point_mass"
threshold = 7
skm_to_od_duration = 3
duration = 28
od_duration = 1

optimization_model = OptimizationModel(["high_fidelity", model, 0], ["high_fidelity", model, 0], threshold=threshold, skm_to_od_duration=skm_to_od_duration, duration=duration, od_duration=od_duration)

result_dict = optimization_model.optimize()

print(result_dict)



# threshold = 7
# duration = 28
# od_duration = 1
# delta_v_per_model_dict = dict()
# for model in ["point_mass", "point_mass_srp", "spherical_harmonics", "spherical_harmonics_srp"]:
# # for model in ["point_mass", "point_mass_srp"]:
#     delta_v_dict = dict()
#     for i in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
#     # for i in [3, 5]:

#         optimization_model = OptimizationModel(["high_fidelity", model, 0], ["high_fidelity", model, 0], threshold=threshold, skm_to_od_duration=i, duration=duration, od_duration=od_duration)
#         x = optimization_model.initial_design_vector

#         start_time = time.time()
#         delta_v = optimization_model.objective_function(x)
#         run_time = time.time() - start_time
#         delta_v_dict[i] = [delta_v, run_time]

#         print("ITERATION: ", x, i, delta_v)
#         print(delta_v_dict)

#     delta_v_per_model_dict[model] = delta_v_dict

# print(delta_v_per_model_dict)

# data = delta_v_per_model_dict

# # Save the dictionary to a JSON file
# import json
# file_path = os.path.join(os.path.dirname(__file__), "delta_v_per_model_dict.json")
# with open(file_path, 'w') as json_file:
#     json.dump(data, json_file, indent=4)

# # Plot bar chart of delta v per model per skm frequency
# groups = list(data.keys())
# inner_keys = list(data[groups[0]].keys())
# num_groups = len(groups)

# fig, ax = plt.subplots(figsize=(8, 3))
# index = np.arange(len(inner_keys))
# bar_width = 0.2



# ax.set_xlabel('SKM interval [days]')
# ax.set_ylabel(r'$\Delta V$ [m/s]')
# ax.set_title(f'Station keeping costs, OD of {od_duration} [day], simulation of {duration} [days]')

# # Center the bars around each xtick
# bar_offsets = np.arange(-(num_groups-1)/2, (num_groups-1)/2 + 1, 1) * bar_width
# for i in range(num_groups):
#     values = [data[groups[i]][inner_key][0] for inner_key in inner_keys]
#     ax.bar(index + bar_offsets[i], values, bar_width, label=str(groups[i]))

# ax.set_yscale("log")
# ax.set_axisbelow(True)
# ax.grid(alpha=0.3)
# ax.set_xticks(index)
# ax.set_xticklabels([key + od_duration for key in inner_keys])
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")
# plt.tight_layout()
# plt.show()
