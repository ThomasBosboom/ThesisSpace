import numpy as np
import os
import sys
import numpy as np
import copy
# import tracemalloc
from memory_profiler import profile

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from src import NavigationSimulator

class Interpolator:
    def __init__(self, step_size):
        self.step_size = step_size


class ReferenceData:
    def __init__(self, step_size):
        self.step_size = step_size
        self.large_data_set = np.zeros((10000, 10))


class Estimator:

    def __init__(self, model_attribute1, model_attribute2):

        self.model_attribute1 = model_attribute1
        self.model_attribute2 = model_attribute2

    def process_a_lot(self):

        # print('processing estimation')
        self.large_observation_history = np.random.rand(1000, 12, 12)
        self.large_estimation_properties = 2*np.random.rand(1000, 12, 12)

        value = 10000
        # ndarray = self.large_observation_history.copy()
        self.large_observation_history0 = {i: np.random.rand(12, 12) for i in range(value)}
        self.large_observation_history1 = self.large_observation_history.copy()
        self.large_observation_history2 = self.large_observation_history.copy()

        return self




### TEST SETUP TO SHOW WORKINGS OF TRACEMALLOC
class DataProcessor:
    def __init__(self, data_efficient_mode=False):
        self.data = None
        self.step_size = 0.01
        self.data_efficient_mode = data_efficient_mode

        self.interpolator = Interpolator(self.step_size)
        self.reference_data = ReferenceData(self.step_size)

        # self.initial_attributes.update({"interpolator": self.interpolator, "reference_data": self.reference_data})

        self.initial_attributes = {**self.__dict__}

    # @profile
    def process(self, value):
        self.data = 1000
        self.new_data = {i: i * 2 for i in range(value)}

        self.estimator_objects = {}
        for i in range(value):
            # print("processing iteration")
            model_1 = True
            model_2 = True
            estimator = Estimator(model_1, model_2)
            estimator = estimator.process_a_lot()
            self.estimator_objects.update({i: estimator})

        if not self.data_efficient_mode:
            self.more_data = {i: i * 3 for i in range(100)}

        return DataOutput(self)


    def reset_attributes(self):

        for key, value in self.initial_attributes.items():
            setattr(self, key, value)
        for key, value in vars(self).copy().items():
            if key != "initial_attributes":
                if key not in self.initial_attributes.keys():
                    delattr(self, key)

    def get_total_size(self):
        total_size = 0
        for key, value in vars(self).items():
            total_size += asizeof.asizeof(value)
            print(key, total_size)
        return total_size


class DataOutput():

    def __init__(self, data_processor):

        self.data_processor = data_processor
        # self.data_processor.reset_attributes()
        # print(self.data_processor.more_data)

import psutil
from pympler import asizeof

if __name__ == "__main__":

    # @profile
    def example_processor():

        # Create an instance of DataProcessor
        processor = DataProcessor()

        saved_data = []
        # Simulate a loop that processes data multiple times
        for i, value in enumerate(range(10)):

            # Example method that generates a lot of data
            data_output = processor.process(value)
            extracted_data = list(data_output.data_processor.new_data.values())
            saved_data.append(extracted_data)

            data_output.data_processor.reset_attributes()

            print(data_output.data_processor.get_total_size())



            # print("been here, memtory: ", psutil.virtual_memory())


        print(saved_data)


    example_processor()




        # Take a snapshot after each iteration
        # snapshot = tracemalloc.take_snapshot()

    #     # Compare the snapshot to the initial snapshot
    #     top_stats = snapshot.compare_to(initial_snapshot, 'lineno')

    #     if top_stats:
    #         # Calculate the total memory difference
    #         total_memory_diff = sum(stat.size_diff for stat in top_stats)
    #         print(f"[Iteration {i + 1}] Top memory usage difference:")
    #         print(top_stats[0])  # Print only the top 1 memory usage difference
    #         print(f"Total memory difference: {total_memory_diff / 1024:.2f} KiB")

    # # Stop tracing memory allocations
    # tracemalloc.stop()
