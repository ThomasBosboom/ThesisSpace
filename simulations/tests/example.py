import numpy as np
import os
import sys
import numpy as np
import copy
# import tracemalloc
# from memory_profiler import profile

# Define path to import src files
file_directory = os.path.realpath(__file__)
for _ in range(2):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from src import NavigationSimulator

navigation_simulator = NavigationSimulator.NavigationSimulator()
observation_windows = observation_windows = [(60390, 60391.0), (60394.0, 60395.0), (60398.0, 60399.0), (60402.0, 60403.0), (60406.0, 60407.0), (60410.0, 60411.0), (60414.0, 60415.0)]

for i in range(3):
    navigation_output = navigation_simulator.perform_navigation(observation_windows, seed=i)
    navigation_simulator = navigation_output.navigation_simulator
    print(navigation_simulator.delta_v_dict)


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

        self.large_observation_history = np.zeros((100000, 10))
        self.large_estimation_properties = np.zeros((100000, 10))

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

    def process(self, value):
        self.data = 1000
        self.new_data = {i: i * 2 for i in range(value)}

        self.estimator_objects = {}
        for i in range(value):
            model_1 = True
            model_2 = True
            self.estimator_objects.update({i: Estimator(model_1, model_2)})

        if not self.data_efficient_mode:
            self.more_data = {i: i * 3 for i in range(value*10)}

        return DataOutput(self)


    def reset_attributes(self):

        for key, value in self.initial_attributes.items():
            setattr(self, key, value)
        for key, value in vars(self).copy().items():
            if key is not "initial_attributes":
                if key not in self.initial_attributes.keys():
                    delattr(self, key)

class DataOutput():

    def __init__(self, data_processor):

        self.data_processor = data_processor
        # self.data_processor.reset_attributes()
        # print(self.data_processor.more_data)




# Create an instance of DataProcessor
processor = DataProcessor()

# # Start tracing memory allocations
# tracemalloc.start()

# # Take an initial snapshot before the loop
# initial_snapshot = tracemalloc.take_snapshot()

# # Initialize a variable to accumulate memory usage differences
# total_memory_diff = 0


saved_data = []
# Simulate a loop that processes data multiple times
for i, value in enumerate(range(10)):

    # Example method that generates a lot of data
    data_output = processor.process(value*1000)
    extracted_data = list(data_output.data_processor.new_data.values())
    saved_data.append(extracted_data)

    data_output.data_processor.reset_attributes()

    print("been here")
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
