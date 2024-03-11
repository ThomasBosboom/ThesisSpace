# General imports
import numpy as np
import os
import sys
from warnings import warn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class OptimizationModelBase():

    def __init__(self, threshold=8, duration=14, options=None):

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
                    "custom_skms": None #[60393.1, 60393.2, 60455]
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

        observation_windows = [(self.mission_start_time, self.threshold)]
        a = 0
        if len(design_vector_dict["t_skm"]) > len(design_vector_dict["t_od"]):
            a = 1
        observation_windows.extend([(design_vector_dict["t_od"][i], design_vector_dict["t_skm"][i+a]) for i in range(len(design_vector_dict["t_od"]))])

        return observation_windows