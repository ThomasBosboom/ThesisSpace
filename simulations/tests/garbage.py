# # # # # # # # # # # import tracemalloc

# # # # # # # # # # # class DataGenerator:
# # # # # # # # # # #     def generate_large_list(self, size):
# # # # # # # # # # #         return [i ** 2 for i in range(size)]

# # # # # # # # # # #     def generate_large_dict(self, size):
# # # # # # # # # # #         return {i: i ** 2 for i in range(size)}

# # # # # # # # # # # class DataProcessor:
# # # # # # # # # # #     def __init__(self):
# # # # # # # # # # #         self.data = None

# # # # # # # # # # #     def process_data(self, data):
# # # # # # # # # # #         self.data = data
# # # # # # # # # # #         # Simulate some processing
# # # # # # # # # # #         processed_data = [x * 2 for x in self.data]
# # # # # # # # # # #         return processed_data

# # # # # # # # # # # def main():
# # # # # # # # # # #     # Start tracing memory allocations
# # # # # # # # # # #     tracemalloc.start()

# # # # # # # # # # #     generator = DataGenerator()
# # # # # # # # # # #     processor = DataProcessor()

# # # # # # # # # # #     # Generate large datasets
# # # # # # # # # # #     large_list = generator.generate_large_list(10000)
# # # # # # # # # # #     large_dict = generator.generate_large_dict(10000)

# # # # # # # # # # #     # Process the large list
# # # # # # # # # # #     processed_list = processor.process_data(large_list)

# # # # # # # # # # #     # Process the large dict (just converting values to list for simplicity)
# # # # # # # # # # #     processed_dict_values = processor.process_data(list(large_dict.values()))

# # # # # # # # # # #     # Take a snapshot
# # # # # # # # # # #     snapshot = tracemalloc.take_snapshot()

# # # # # # # # # # #     # Display top memory allocations
# # # # # # # # # # #     # top_stats = snapshot.statistics('lineno')
# # # # # # # # # # #     top_stats = snapshot.statistics('traceback')

# # # # # # # # # # #     print("[ Top 10 memory allocations ]")
# # # # # # # # # # #     for stat in top_stats[:10]:
# # # # # # # # # # #         print(stat)

# # # # # # # # # # #     # Stop tracing memory allocations
# # # # # # # # # # #     tracemalloc.stop()

# # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # #     main()



# # # # # # # # # # import tracemalloc
# # # # # # # # # # import time

# # # # # # # # # # # Start tracing memory allocations
# # # # # # # # # # tracemalloc.start()

# # # # # # # # # # # Simulate a function that gradually increases memory usage
# # # # # # # # # # def simulate_memory_usage():
# # # # # # # # # #     data = []
# # # # # # # # # #     for i in range(10):
# # # # # # # # # #         data.append([i] * 1000000)  # Increase memory usage
# # # # # # # # # #         time.sleep(1)
# # # # # # # # # #         print(f"Snapshot {i + 1}")
# # # # # # # # # #         snapshot = tracemalloc.take_snapshot()
# # # # # # # # # #         top_stats = snapshot.statistics('lineno')
# # # # # # # # # #         for stat in top_stats[:10]:
# # # # # # # # # #             print(stat)
# # # # # # # # # #         total_memory = sum(stat.size for stat in top_stats)
# # # # # # # # # #         print(f"Total memory used after iteration {i + 1}: {total_memory / (1024 ** 2):.2f} MB")
# # # # # # # # # #     return data

# # # # # # # # # # simulate_memory_usage()

# # # # # # # # # # # Optional: Stop tracing memory allocations
# # # # # # # # # # tracemalloc.stop()

# # # # # # # # # from memory_profiler import profile

# # # # # # # # # class MyClass:
# # # # # # # # #     def __init__(self):
# # # # # # # # #         self.attribute1 = "initial_value1"
# # # # # # # # #         self.attribute2 = "initial_value2"
# # # # # # # # #         self.attribute3 = "initial_value3"
# # # # # # # # #         self.initial_state = vars(self).copy()  # Store initial state

# # # # # # # # #     @profile
# # # # # # # # #     def generate_additional_attributes(self):
# # # # # # # # #         # Method that generates additional attributes
# # # # # # # # #         self.additional_attribute1 = "new_value1"
# # # # # # # # #         self.additional_attribute2 = "new_value2"

# # # # # # # # #     @profile
# # # # # # # # #     def reset_to_init_state(self):

# # # # # # # # #         current_state = vars(self)
# # # # # # # # #         for attr_name in list(current_state.keys()):
# # # # # # # # #             if attr_name is not str("initial_state"):
# # # # # # # # #                 if attr_name not in self.initial_state.keys():
# # # # # # # # #                     delattr(self, attr_name)


# # # # # # # # # # Example usage:
# # # # # # # # # obj = MyClass()

# # # # # # # # # print(vars(obj))
# # # # # # # # # # Call the method to generate additional attributes
# # # # # # # # # obj.generate_additional_attributes()

# # # # # # # # # # Print the attributes after generating additional attributes
# # # # # # # # # print(vars(obj))

# # # # # # # # # # Call the method to reset attributes to their initial state
# # # # # # # # # obj.reset_to_init_state()

# # # # # # # # # # Print the attributes after resetting
# # # # # # # # # print(vars(obj))



# # # # # # # # #             self.full_estimation_error_dict = dict()
# # # # # # # # #             self.full_reference_state_deviation_dict = dict()
# # # # # # # # #             self.full_propagated_covariance_dict = dict()
# # # # # # # # #             self.full_propagated_formal_errors_dict = dict()
# # # # # # # # #             self.full_state_history_reference_dict = dict()
# # # # # # # # #             self.full_state_history_truth_dict = dict()
# # # # # # # # #             self.full_state_history_estimated_dict = dict()
# # # # # # # # #             self.full_state_history_final_dict = dict()
# # # # # # # # #             self.delta_v_dict = dict()
# # # # # # # # #             self.full_dependent_variables_history_estimated = dict()
# # # # # # # # #             self.full_state_transition_matrix_history_estimated = dict()
# # # # # # # # #             self.estimation_arc_results_dict = dict()
# # # # # # # # import numpy as np

# # # # # # # # # Example input dictionary
# # # # # # # # data = {
# # # # # # # #     '0.0': {
# # # # # # # #         60391: {0: 0.0012819549366386333, 1: 0.001185896799597702},
# # # # # # # #         60395: {0: 0.005411142569208645, 1: 0.006770971960928585},
# # # # # # # #         60399: {0: 0.003109677105139794, 1: 0.002999612884133889},
# # # # # # # #         60403: {0: 0.004734675217803918, 1: 0.004341626884436936}
# # # # # # # #     },
# # # # # # # #     '0.03': {
# # # # # # # #         60399: {0: 0.0672568474862809, 1: 0.06727010238539159}
# # # # # # # #     }
# # # # # # # # }

# # # # # # # # def calculate_stats(data, evaluation_threshold=14):
# # # # # # # #     result_dict = {}

# # # # # # # #     # Iterate through each test case
# # # # # # # #     for case_type, epochs in data.items():


# # # # # # # #         # Iterate through each epoch
# # # # # # # #         epoch_stats = {}
# # # # # # # #         combined_per_run = {}
# # # # # # # #         combined_per_run_with_threshold = {}
# # # # # # # #         for epoch, runs in epochs.items():
# # # # # # # #             keys = list(runs.keys())
# # # # # # # #             values = list(runs.values())
# # # # # # # #             epoch_stats[epoch] = {'mean': np.mean(values), 'std': np.std(values)}

# # # # # # # #             for key in keys:
# # # # # # # #                 if key not in combined_per_run:
# # # # # # # #                     combined_per_run[key] = []
# # # # # # # #                     combined_per_run_with_threshold[key] = []

# # # # # # # #                 combined_per_run[key].append(runs[key])
# # # # # # # #                 if epoch >= 60390 + evaluation_threshold:
# # # # # # # #                     combined_per_run_with_threshold[key].append(runs[key])


# # # # # # # #         total = []
# # # # # # # #         total_with_threshold = []
# # # # # # # #         for run, combined in combined_per_run.items():
# # # # # # # #             total.append(np.sum(combined))
# # # # # # # #         for run, combined in combined_per_run_with_threshold.items():
# # # # # # # #             total_with_threshold.append(np.sum(combined))

# # # # # # # #         total_stats = {'mean': np.mean(total), 'std': np.std(total)}
# # # # # # # #         total_stats_with_threshold = {'mean': np.mean(total_with_threshold), 'std': np.std(total_with_threshold)}

# # # # # # # #         # Store statistics in the result dictionary
# # # # # # # #         result_dict[case_type] = {'epoch_stats': epoch_stats, 'total_stats': total_stats, 'total_stats_with_threshold': total_stats_with_threshold}

# # # # # # # #     return result_dict

# # # # # # # import json

# # # # # # # def create_design_vector_table(data, caption='', label='', file_name='design_vector_table.tex', decimals=4):
# # # # # # #     table_str = r'% Please add the following required packages to your document preamble:' + '\n'
# # # # # # #     table_str += r'% \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # # #     table_str += r'% Beamer presentation requires \usepackage{colortbl} instead of \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # # #     table_str += r'\begin{table}[]' + '\n'
# # # # # # #     table_str += r'\centering' + '\n'
# # # # # # #     table_str += r'\begin{tabular}{lll}' + '\n'
# # # # # # #     table_str += r'\textbf{}      & \cellcolor[HTML]{EFEFEF}\textbf{Design vector} & \textbf{}          \\' + '\n'
# # # # # # #     table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
# # # # # # #     table_str += r'\textbf{State} & \textbf{Initial}                               & \textbf{Final} \\' + '\n'

# # # # # # #     states = ['x', 'y', 'z', 'v_{x}', 'v_{y}', 'v_{z}']
# # # # # # #     initial_values = data.get('initial', [])
# # # # # # #     final_values = data.get('final', [])

# # # # # # #     for state, initial, final in zip(states, initial_values, final_values):
# # # # # # #         table_str += f"${state}$ & {initial:.{decimals}f} & {final:.{decimals}f} \\" + '\n'

# # # # # # #     table_str += r'\end{tabular}' + '\n'
# # # # # # #     table_str += r'\caption{' + caption + '}' + '\n'
# # # # # # #     table_str += r'\label{' + label + '}' + '\n'
# # # # # # #     table_str += r'\end{table}'

# # # # # # #     if file_name:
# # # # # # #         with open(file_name, 'w') as f:
# # # # # # #             f.write(table_str)

# # # # # # #     return table_str

# # # # # # # # Example usage
# # # # # # # data = {
# # # # # # #     "initial": [1, 1, 1, 1, 1, 1],
# # # # # # #     "final": [2, 5, 1, 5, 2, 6]
# # # # # # # }

# # # # # # # # Generate the Overleaf table with custom caption, label, and decimals
# # # # # # # file_name = "design_vector_table.tex"
# # # # # # # overleaf_table = create_design_vector_table(data, caption="Design Vector comparison before and after optimization", label="tab:DesignVectorOptimization", file_name=file_name, decimals=4)

# # # # # # # # Print the Overleaf table
# # # # # # # print(overleaf_table)



# # # # # # def create_design_vector_table(data, caption='', label='', file_name='design_vector_table.tex', decimals=4):
# # # # # #     table_str = r'% Please add the following required packages to your document preamble:' + '\n'
# # # # # #     table_str += r'% \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # #     table_str += r'% Beamer presentation requires \usepackage{colortbl} instead of \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # #     table_str += r'\begin{table}[]' + '\n'
# # # # # #     table_str += r'\centering' + '\n'
# # # # # #     table_str += r'\begin{tabular}{llll}' + '\n'
# # # # # #     table_str += r'\textbf{}      & \cellcolor[HTML]{EFEFEF}\textbf{Design vector} & \textbf{}          & \textbf{}             \\' + '\n'
# # # # # #     table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
# # # # # #     table_str += r'\textbf{State} & \textbf{Initial}                               & \textbf{Optimized} & \textbf{\% Difference} \\' + '\n'

# # # # # #     states = ['x', 'y', 'z', 'v_{x}', 'v_{y}', 'v_{z}']
# # # # # #     initial_values = data.get('initial', {}).get('values', [])
# # # # # #     final_values = data.get('final', {}).get('values', [])

# # # # # #     for state, initial, final in zip(states, initial_values, final_values):
# # # # # #         if initial != 0:
# # # # # #             percentage_diff = ((final - initial) / initial) * 100
# # # # # #         else:
# # # # # #             percentage_diff = 0  # Handle division by zero case
# # # # # #         table_str += f"${state}$ & {initial:.{decimals}f} & {final:.{decimals}f} & {round(percentage_diff, 2)}\%" + r' \\ ' + '\n'

# # # # # #     initial_cost = data.get('initial', {}).get('cost', 0)
# # # # # #     final_cost = data.get('final', {}).get('cost', 0)

# # # # # #     if initial_cost != 0:
# # # # # #         cost_percentage_diff = ((final_cost - initial_cost) / initial_cost) * 100
# # # # # #     else:
# # # # # #         cost_percentage_diff = 0  # Handle division by zero case

# # # # # #     table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
# # # # # #     table_str += f"\\textbf{{Cost}}  & {initial_cost:.{decimals}f} & {final_cost:.{decimals}f} & {round(cost_percentage_diff, 2)}\%" + r' \\ ' + '\n'

# # # # # #     table_str += r'\end{tabular}' + '\n'
# # # # # #     table_str += r'\caption{' + caption + '}' + '\n'
# # # # # #     table_str += r'\label{' + label + '}' + '\n'
# # # # # #     table_str += r'\end{table}'

# # # # # #     if file_name:
# # # # # #         with open(file_name, 'w') as f:
# # # # # #             f.write(table_str)

# # # # # #     return table_str

# # # # # # # Example usage
# # # # # # data = {
# # # # # #     "initial": {
# # # # # #         "values": [1, 1, 1, 1, 1, 1],
# # # # # #         "cost": 10
# # # # # #     },
# # # # # #     "final": {
# # # # # #         "values": [2, 5, 1, 5, 2, 6],
# # # # # #         "cost": 8
# # # # # #     }
# # # # # # }

# # # # # # # Generate the Overleaf table with custom caption, label, and decimals
# # # # # # file_name = "design_vector_table.tex"
# # # # # # overleaf_table = create_design_vector_table(data, caption="Design vector comparison before and after optimization", label="tab:DesignVectorOptimization", file_name=file_name, decimals=4)

# # # # # # # Print the Overleaf table
# # # # # # print(overleaf_table)

# # # # # import copy

# # # # # class NavigationSimulatorBase:
# # # # #     def __init__(self):
# # # # #         # Initialize some default attributes for the base class
# # # # #         self.default_attr1 = "default_value1"
# # # # #         self.default_attr2 = "default_value2"

# # # # # class NavigationSimulator(NavigationSimulatorBase):

# # # # #     def __init__(self, **kwargs):
# # # # #         super().__init__()

# # # # #         self.attr1 = 1
# # # # #         self.attr2 = 2
# # # # #         for key, value in kwargs.items():
# # # # #             if hasattr(self, key):
# # # # #                 setattr(self, key, value)


# # # # #     def modify_attributes(self):

# # # # #         # Example modifications
# # # # #         if hasattr(self, 'attr1'):
# # # # #             self.attr1 += 1
# # # # #         if hasattr(self, 'attr2'):
# # # # #             self.attr2 += 1

# # # # #             self.new_attribute = self.attr1 + self.attr2

# # # # #         return self.new_attribute


# # # # #     def reset(self):
# # # # #         # Reset attributes to their initial values
# # # # #         self.__dict__ = copy.deepcopy(self.__dict__)

# # # # # # Usage example
# # # # # initial_instance = NavigationSimulator(attr1=1, attr2=2)

# # # # # print(vars(initial_instance))

# # # # # for _ in range(5):  # Example: call method 5 times
# # # # #     new_attribute = initial_instance.modify_attributes()
# # # # #     initial_instance.reset()
# # # # #     print(new_attribute, vars(initial_instance))



# # # # import numpy as np

# # # # class RandomNumberGenerator:
# # # #     def __init__(self):
# # # #         self.seed = 0

# # # #     def generate_normal(self, mean=0.0, std=1.0, size=1):
# # # #         # Increment the seed for each call
# # # #         self.seed = 1
# # # #         # Create a new random generator with the updated seed
# # # #         rng = np.random.default_rng(self.seed)
# # # #         for _ in range(3):
# # # #             print(rng.normal(mean, std, size))
# # # #         # Generate the normal random numbers
# # # #         # return rng.normal(mean, std, size)

# # # # # Example usage
# # # # rng_gen = RandomNumberGenerator()

# # # # # Generate normal random numbers with different seeds
# # # # print(rng_gen.generate_normal(mean=0, std=1, size=5))


# # # class MyClass:
# # #     def __init__(self, attr1, attr2):
# # #         self.attr1 = attr1
# # #         self.attr2 = attr2
# # #         # Store the initial state of attributes
# # #         self._initial_attrs = {attr: getattr(self, attr) for attr in vars(self)}
# # #         print(self._initial_attrs)

# # #     def modify_attributes(self):
# # #         # Method that modifies attributes
# # #         self.attr1 = "modified"
# # #         self.attr2 = "modified"
# # #         self.attr3 = "newattr"
# # #         # Reset attributes to their initial state after modification
# # #         self.reset_attributes()

# # #     def reset_attributes(self):

# # #         # Reset attributes to their initial state
# # #         _initial_attrs = self._initial_attrs
# # #         for attr, value in self._initial_attrs.items():
# # #             setattr(self, attr, value)
# # #         # Delete any newly created attributes
# # #         for attr in list(vars(self)):
# # #             if attr is not "_initial_attrs" and attr in self._initial_attrs:
# # #                 delattr(self, attr)

# # # # Example usage
# # # obj = MyClass("initial_value1", "initial_value2")
# # # print("Before modification:", obj.attr1, obj.attr2)  # Output: Before modification: initial_value1 initial_value2

# # # obj.modify_attributes()
# # # print("After modification:", obj.attr1, obj.attr2, getattr(obj, "attr3", None))  # Output: After modification: initial_value1 initial_value2 None


# # import gc

# # class A:
# #     def __init__(self):
# #         self.b = None

# # class B:
# #     def __init__(self):
# #         self.a = None

# # # Create instances of A and B
# # a = A()
# # b = B()

# # # Create a cyclic reference
# # a.b = b
# # b.a = a

# # # Manually break the reference cycle (uncomment to test cleanup)
# # # a.b = None
# # # b.a = None

# # # Delete the references to a and b
# # del a
# # del b

# # # Force garbage collection
# # gc.collect()

# # # Check for unreachable objects
# # unreachable = gc.garbage
# # print("Unreachable objects:", unreachable)


# import tracemalloc

# def large_function_call(data):
#     # Simulate a large memory-consuming operation
#     result = [x * 2 for x in data]
#     return result

# def process_data(dataset):
#     results = []

#     # Start tracing memory allocations
#     tracemalloc.start()

#     # Take an initial snapshot
#     initial_snapshot = tracemalloc.take_snapshot()

#     for index, data in enumerate(dataset):
#         result = large_function_call(data)
#         results.append(result)

#         # Optionally take periodic snapshots to monitor memory usage
#         if index % 100 == 0:  # Adjust the modulus value based on your use case
#             intermediate_snapshot = tracemalloc.take_snapshot()
#             top_stats = intermediate_snapshot.compare_to(initial_snapshot, 'lineno')
#             print(f"Memory usage after {index + 1} iterations:")
#             for stat in top_stats[:5]:
#                 print(stat)

#     # Take a final snapshot
#     final_snapshot = tracemalloc.take_snapshot()
#     top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')

#     # Display the top memory usage differences
#     print("[ Top 10 differences ]")
#     for stat in top_stats[:10]:
#         print(stat)

#     # Stop tracing memory allocations
#     tracemalloc.stop()

#     return results

# # Example usage
# if __name__ == "__main__":
#     dataset = [list(range(1000)) for _ in range(1000)]  # Example dataset
#     process_data(dataset)


# import tracemalloc

# class Base:

#     def __init__(self):

#         self.base_data = [list(range(1000)) for _ in range(1000)]


# class DataProcessor(Base):
#     def __init__(self):
#         super().__init__()
#         self.data = None

#     def process_data(self, data):
#         self.data = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation
#         self.data1 = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation
#         self.data2 = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation
#         self.data3 = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation

#     def cleanup(self):
#         self.data = None  # Release the reference to the dictionary

# # Create an instance of DataProcessor
# processor = DataProcessor()

# # Start tracing memory allocations
# tracemalloc.start()

# # Take an initial snapshot before the loop
# initial_snapshot = tracemalloc.take_snapshot()

# # Simulate a loop that processes data multiple times
# list = []
# for i in range(10):
#     processor.process_data(10000)
#     # processor.cleanup()  # Clean up after each iteration

#     list.append(processor.data[0])

#     # Take a snapshot after each iteration
#     snapshot = tracemalloc.take_snapshot()

#     # Compare the snapshot to the initial snapshot
#     top_stats = snapshot.compare_to(initial_snapshot, 'lineno')

#     print(f"[Iteration {i + 1}] Top 1 memory usage differences:")
#     for stat in top_stats[:1]:
#         print(stat)

# print(list)

# # Stop tracing memory allocations
# tracemalloc.stop()



