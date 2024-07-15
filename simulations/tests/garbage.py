# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # import tracemalloc

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # class DataGenerator:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def generate_large_list(self, size):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         return [i ** 2 for i in range(size)]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def generate_large_dict(self, size):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         return {i: i ** 2 for i in range(size)}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # class DataProcessor:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.data = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def process_data(self, data):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.data = data
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Simulate some processing
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         processed_data = [x * 2 for x in self.data]
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         return processed_data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # def main():
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Start tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     tracemalloc.start()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     generator = DataGenerator()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     processor = DataProcessor()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Generate large datasets
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     large_list = generator.generate_large_list(10000)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     large_dict = generator.generate_large_dict(10000)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Process the large list
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     processed_list = processor.process_data(large_list)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Process the large dict (just converting values to list for simplicity)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     processed_dict_values = processor.process_data(list(large_dict.values()))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Take a snapshot
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     snapshot = tracemalloc.take_snapshot()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Display top memory allocations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     # top_stats = snapshot.statistics('lineno')
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     top_stats = snapshot.statistics('traceback')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     print("[ Top 10 memory allocations ]")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     for stat in top_stats[:10]:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         print(stat)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Stop tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     tracemalloc.stop()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     main()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # import tracemalloc
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # import time

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Start tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # tracemalloc.start()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Simulate a function that gradually increases memory usage
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # def simulate_memory_usage():
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     data = []
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     for i in range(10):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         data.append([i] * 1000000)  # Increase memory usage
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         time.sleep(1)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         print(f"Snapshot {i + 1}")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         snapshot = tracemalloc.take_snapshot()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         top_stats = snapshot.statistics('lineno')
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         for stat in top_stats[:10]:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             print(stat)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         total_memory = sum(stat.size for stat in top_stats)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         print(f"Total memory used after iteration {i + 1}: {total_memory / (1024 ** 2):.2f} MB")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     return data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # simulate_memory_usage()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Optional: Stop tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # tracemalloc.stop()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # from memory_profiler import profile

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # class MyClass:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.attribute1 = "initial_value1"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.attribute2 = "initial_value2"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.attribute3 = "initial_value3"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.initial_state = vars(self).copy()  # Store initial state

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     @profile
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def generate_additional_attributes(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Method that generates additional attributes
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.additional_attribute1 = "new_value1"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.additional_attribute2 = "new_value2"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     @profile
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def reset_to_init_state(self):

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         current_state = vars(self)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         for attr_name in list(current_state.keys()):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             if attr_name is not str("initial_state"):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                 if attr_name not in self.initial_state.keys():
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                     delattr(self, attr_name)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Example usage:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # obj = MyClass()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # print(vars(obj))
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Call the method to generate additional attributes
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # obj.generate_additional_attributes()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Print the attributes after generating additional attributes
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # print(vars(obj))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Call the method to reset attributes to their initial state
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # obj.reset_to_init_state()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Print the attributes after resetting
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # print(vars(obj))



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.full_estimation_error_dict = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.full_reference_state_deviation_dict = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.full_propagated_covariance_dict = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.full_propagated_formal_errors_dict = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.full_state_history_reference_dict = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.full_state_history_truth_dict = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.full_state_history_estimated_dict = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.full_state_history_final_dict = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.delta_v_dict = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.full_dependent_variables_history_estimated = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.full_state_transition_matrix_history_estimated = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.estimation_arc_results_dict = dict()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Example input dictionary
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # data = {
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     '0.0': {
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         60391: {0: 0.0012819549366386333, 1: 0.001185896799597702},
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         60395: {0: 0.005411142569208645, 1: 0.006770971960928585},
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         60399: {0: 0.003109677105139794, 1: 0.002999612884133889},
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         60403: {0: 0.004734675217803918, 1: 0.004341626884436936}
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     },
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     '0.03': {
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         60399: {0: 0.0672568474862809, 1: 0.06727010238539159}
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # }

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # def calculate_stats(data, evaluation_threshold=14):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     result_dict = {}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Iterate through each test case
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     for case_type, epochs in data.items():


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Iterate through each epoch
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         epoch_stats = {}
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         combined_per_run = {}
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         combined_per_run_with_threshold = {}
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         for epoch, runs in epochs.items():
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             keys = list(runs.keys())
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             values = list(runs.values())
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             epoch_stats[epoch] = {'mean': np.mean(values), 'std': np.std(values)}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             for key in keys:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                 if key not in combined_per_run:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                     combined_per_run[key] = []
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                     combined_per_run_with_threshold[key] = []

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                 combined_per_run[key].append(runs[key])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                 if epoch >= 60390 + evaluation_threshold:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                     combined_per_run_with_threshold[key].append(runs[key])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         total = []
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         total_with_threshold = []
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         for run, combined in combined_per_run.items():
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             total.append(np.sum(combined))
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         for run, combined in combined_per_run_with_threshold.items():
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             total_with_threshold.append(np.sum(combined))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         total_stats = {'mean': np.mean(total), 'std': np.std(total)}
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         total_stats_with_threshold = {'mean': np.mean(total_with_threshold), 'std': np.std(total_with_threshold)}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Store statistics in the result dictionary
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         result_dict[case_type] = {'epoch_stats': epoch_stats, 'total_stats': total_stats, 'total_stats_with_threshold': total_stats_with_threshold}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     return result_dict

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # import json

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # def create_design_vector_table(data, caption='', label='', file_name='design_vector_table.tex', decimals=4):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str = r'% Please add the following required packages to your document preamble:' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'% \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'% Beamer presentation requires \usepackage{colortbl} instead of \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\begin{table}[]' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\centering' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\begin{tabular}{lll}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\textbf{}      & \cellcolor[HTML]{EFEFEF}\textbf{Design vector} & \textbf{}          \\' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\textbf{State} & \textbf{Initial}                               & \textbf{Final} \\' + '\n'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     states = ['x', 'y', 'z', 'v_{x}', 'v_{y}', 'v_{z}']
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     initial_values = data.get('initial', [])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     final_values = data.get('final', [])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     for state, initial, final in zip(states, initial_values, final_values):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         table_str += f"${state}$ & {initial:.{decimals}f} & {final:.{decimals}f} \\" + '\n'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\end{tabular}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\caption{' + caption + '}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\label{' + label + '}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\end{table}'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     if file_name:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         with open(file_name, 'w') as f:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             f.write(table_str)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     return table_str

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Example usage
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # data = {
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     "initial": [1, 1, 1, 1, 1, 1],
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     "final": [2, 5, 1, 5, 2, 6]
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # }

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Generate the Overleaf table with custom caption, label, and decimals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # file_name = "design_vector_table.tex"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # overleaf_table = create_design_vector_table(data, caption="Design Vector comparison before and after optimization", label="tab:DesignVectorOptimization", file_name=file_name, decimals=4)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Print the Overleaf table
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # print(overleaf_table)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # def create_design_vector_table(data, caption='', label='', file_name='design_vector_table.tex', decimals=4):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str = r'% Please add the following required packages to your document preamble:' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'% \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'% Beamer presentation requires \usepackage{colortbl} instead of \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\begin{table}[]' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\centering' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\begin{tabular}{llll}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\textbf{}      & \cellcolor[HTML]{EFEFEF}\textbf{Design vector} & \textbf{}          & \textbf{}             \\' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\textbf{State} & \textbf{Initial}                               & \textbf{Optimized} & \textbf{\% Difference} \\' + '\n'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     states = ['x', 'y', 'z', 'v_{x}', 'v_{y}', 'v_{z}']
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     initial_values = data.get('initial', {}).get('values', [])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     final_values = data.get('final', {}).get('values', [])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     for state, initial, final in zip(states, initial_values, final_values):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         if initial != 0:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             percentage_diff = ((final - initial) / initial) * 100
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             percentage_diff = 0  # Handle division by zero case
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         table_str += f"${state}$ & {initial:.{decimals}f} & {final:.{decimals}f} & {round(percentage_diff, 2)}\%" + r' \\ ' + '\n'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     initial_cost = data.get('initial', {}).get('cost', 0)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     final_cost = data.get('final', {}).get('cost', 0)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     if initial_cost != 0:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         cost_percentage_diff = ((final_cost - initial_cost) / initial_cost) * 100
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         cost_percentage_diff = 0  # Handle division by zero case

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += f"\\textbf{{Cost}}  & {initial_cost:.{decimals}f} & {final_cost:.{decimals}f} & {round(cost_percentage_diff, 2)}\%" + r' \\ ' + '\n'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\end{tabular}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\caption{' + caption + '}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\label{' + label + '}' + '\n'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     table_str += r'\end{table}'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     if file_name:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         with open(file_name, 'w') as f:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             f.write(table_str)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     return table_str

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Example usage
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # data = {
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     "initial": {
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         "values": [1, 1, 1, 1, 1, 1],
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         "cost": 10
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     },
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     "final": {
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         "values": [2, 5, 1, 5, 2, 6],
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         "cost": 8
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # }

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Generate the Overleaf table with custom caption, label, and decimals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # file_name = "design_vector_table.tex"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # overleaf_table = create_design_vector_table(data, caption="Design vector comparison before and after optimization", label="tab:DesignVectorOptimization", file_name=file_name, decimals=4)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Print the Overleaf table
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # print(overleaf_table)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # import copy

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # class NavigationSimulatorBase:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Initialize some default attributes for the base class
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.default_attr1 = "default_value1"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.default_attr2 = "default_value2"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # class NavigationSimulator(NavigationSimulatorBase):

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def __init__(self, **kwargs):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         super().__init__()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.attr1 = 1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.attr2 = 2
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         for key, value in kwargs.items():
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             if hasattr(self, key):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                 setattr(self, key, value)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def modify_attributes(self):

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Example modifications
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         if hasattr(self, 'attr1'):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.attr1 += 1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         if hasattr(self, 'attr2'):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.attr2 += 1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             self.new_attribute = self.attr1 + self.attr2

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         return self.new_attribute


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def reset(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Reset attributes to their initial values
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.__dict__ = copy.deepcopy(self.__dict__)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Usage example
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # initial_instance = NavigationSimulator(attr1=1, attr2=2)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # print(vars(initial_instance))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # for _ in range(5):  # Example: call method 5 times
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     new_attribute = initial_instance.modify_attributes()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     initial_instance.reset()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     print(new_attribute, vars(initial_instance))



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # class RandomNumberGenerator:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.seed = 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def generate_normal(self, mean=0.0, std=1.0, size=1):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Increment the seed for each call
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.seed = 1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Create a new random generator with the updated seed
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         rng = np.random.default_rng(self.seed)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         for _ in range(3):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             print(rng.normal(mean, std, size))
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Generate the normal random numbers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # return rng.normal(mean, std, size)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Example usage
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # rng_gen = RandomNumberGenerator()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Generate normal random numbers with different seeds
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # print(rng_gen.generate_normal(mean=0, std=1, size=5))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # class MyClass:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def __init__(self, attr1, attr2):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.attr1 = attr1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.attr2 = attr2
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Store the initial state of attributes
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self._initial_attrs = {attr: getattr(self, attr) for attr in vars(self)}
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         print(self._initial_attrs)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def modify_attributes(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Method that modifies attributes
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.attr1 = "modified"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.attr2 = "modified"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.attr3 = "newattr"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Reset attributes to their initial state after modification
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.reset_attributes()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def reset_attributes(self):

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Reset attributes to their initial state
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         _initial_attrs = self._initial_attrs
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         for attr, value in self._initial_attrs.items():
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             setattr(self, attr, value)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Delete any newly created attributes
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         for attr in list(vars(self)):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             if attr is not "_initial_attrs" and attr in self._initial_attrs:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                 delattr(self, attr)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Example usage
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # obj = MyClass("initial_value1", "initial_value2")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # print("Before modification:", obj.attr1, obj.attr2)  # Output: Before modification: initial_value1 initial_value2

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # obj.modify_attributes()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # print("After modification:", obj.attr1, obj.attr2, getattr(obj, "attr3", None))  # Output: After modification: initial_value1 initial_value2 None


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # import gc

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # class A:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.b = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # class B:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.a = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Create instances of A and B
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # a = A()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # b = B()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Create a cyclic reference
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # a.b = b
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # b.a = a

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Manually break the reference cycle (uncomment to test cleanup)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # a.b = None
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # b.a = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Delete the references to a and b
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # del a
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # del b

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Force garbage collection
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # gc.collect()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Check for unreachable objects
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # unreachable = gc.garbage
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # print("Unreachable objects:", unreachable)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # import tracemalloc

# # # # # # # # # # # # # # # # # # # # # # # # # # # # def large_function_call(data):
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Simulate a large memory-consuming operation
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     result = [x * 2 for x in data]
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     return result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # def process_data(dataset):
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     results = []

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Start tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     tracemalloc.start()

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Take an initial snapshot
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     initial_snapshot = tracemalloc.take_snapshot()

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     for index, data in enumerate(dataset):
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         result = large_function_call(data)
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         results.append(result)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #         # Optionally take periodic snapshots to monitor memory usage
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         if index % 100 == 0:  # Adjust the modulus value based on your use case
# # # # # # # # # # # # # # # # # # # # # # # # # # # #             intermediate_snapshot = tracemalloc.take_snapshot()
# # # # # # # # # # # # # # # # # # # # # # # # # # # #             top_stats = intermediate_snapshot.compare_to(initial_snapshot, 'lineno')
# # # # # # # # # # # # # # # # # # # # # # # # # # # #             print(f"Memory usage after {index + 1} iterations:")
# # # # # # # # # # # # # # # # # # # # # # # # # # # #             for stat in top_stats[:5]:
# # # # # # # # # # # # # # # # # # # # # # # # # # # #                 print(stat)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Take a final snapshot
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     final_snapshot = tracemalloc.take_snapshot()
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Display the top memory usage differences
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     print("[ Top 10 differences ]")
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     for stat in top_stats[:10]:
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         print(stat)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Stop tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     tracemalloc.stop()

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     return results

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # Example usage
# # # # # # # # # # # # # # # # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     dataset = [list(range(1000)) for _ in range(1000)]  # Example dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     process_data(dataset)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # import tracemalloc

# # # # # # # # # # # # # # # # # # # # # # # # # # # # class Base:

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     def __init__(self):

# # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.base_data = [list(range(1000)) for _ in range(1000)]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # class DataProcessor(Base):
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.data = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     def process_data(self, data):
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.data = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.data1 = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.data2 = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.data3 = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     def cleanup(self):
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         self.data = None  # Release the reference to the dictionary

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # Create an instance of DataProcessor
# # # # # # # # # # # # # # # # # # # # # # # # # # # # processor = DataProcessor()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # Start tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # tracemalloc.start()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # Take an initial snapshot before the loop
# # # # # # # # # # # # # # # # # # # # # # # # # # # # initial_snapshot = tracemalloc.take_snapshot()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # Simulate a loop that processes data multiple times
# # # # # # # # # # # # # # # # # # # # # # # # # # # # list = []
# # # # # # # # # # # # # # # # # # # # # # # # # # # # for i in range(10):
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     processor.process_data(10000)
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     # processor.cleanup()  # Clean up after each iteration

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     list.append(processor.data[0])

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Take a snapshot after each iteration
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     snapshot = tracemalloc.take_snapshot()

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     # Compare the snapshot to the initial snapshot
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     top_stats = snapshot.compare_to(initial_snapshot, 'lineno')

# # # # # # # # # # # # # # # # # # # # # # # # # # # #     print(f"[Iteration {i + 1}] Top 1 memory usage differences:")
# # # # # # # # # # # # # # # # # # # # # # # # # # # #     for stat in top_stats[:1]:
# # # # # # # # # # # # # # # # # # # # # # # # # # # #         print(stat)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # print(list)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # Stop tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # tracemalloc.stop()



# # # # # # # # # # # # # # # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # start = 60390
# # # # # # # # # # # # # # # # # # # # # # # # # end = 60390.91428571429
# # # # # # # # # # # # # # # # # # # # # # # # # num_points = 6

# # # # # # # # # # # # # # # # # # # # # # # # # array_with_endpoints = np.linspace(start, end, num_points)
# # # # # # # # # # # # # # # # # # # # # # # # # print(array_with_endpoints)


# # # # # # # # # # # # # # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # # # # import example

# # # # # # # # # # # # # # # # # # # # # # # # Generate some sample data
# # # # # # # # # # # # # # # # # # # # # # # x = np.linspace(0, 10, 100)  # Generate 100 points from 0 to 10
# # # # # # # # # # # # # # # # # # # # # # # y = np.sin(x)  # Compute sine of x for y values

# # # # # # # # # # # # # # # # # # # # # # # # Plot the data
# # # # # # # # # # # # # # # # # # # # # # # plt.figure(figsize=(8, 6))  # Create a new figure with size 8x6 inches
# # # # # # # # # # # # # # # # # # # # # # # plt.plot(x, y, label='sin(x)')  # Plot x vs y with label for the legend
# # # # # # # # # # # # # # # # # # # # # # # plt.title('Simple Plot Example')  # Set plot title
# # # # # # # # # # # # # # # # # # # # # # # plt.xlabel('x')  # Set x-axis label
# # # # # # # # # # # # # # # # # # # # # # # plt.ylabel('sin(x)')  # Set y-axis label
# # # # # # # # # # # # # # # # # # # # # # # plt.legend()  # Display legend based on labels

# # # # # # # # # # # # # # # # # # # # # # # # Save the plot as a PNG file
# # # # # # # # # # # # # # # # # # # # # # # plt.savefig('plot_example.png')

# # # # # # # # # # # # # # # # # # # # # # # # Display the plot
# # # # # # # # # # # # # # # # # # # # # # # plt.show()


# # # # # # # # # # # # # # # # # # # # # # import multiprocessing
# # # # # # # # # # # # # # # # # # # # # # import random
# # # # # # # # # # # # # # # # # # # # # # import math

# # # # # # # # # # # # # # # # # # # # # # def monte_carlo_pi_part(iterations):
# # # # # # # # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # # # # # # # #     Perform a Monte Carlo simulation to estimate the value of π.

# # # # # # # # # # # # # # # # # # # # # #     Args:
# # # # # # # # # # # # # # # # # # # # # #         iterations (int): Number of iterations for the simulation.

# # # # # # # # # # # # # # # # # # # # # #     Returns:
# # # # # # # # # # # # # # # # # # # # # #         int: Number of points inside the quarter circle.
# # # # # # # # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # # # # # # # #     inside_circle = 0
# # # # # # # # # # # # # # # # # # # # # #     for _ in range(iterations):
# # # # # # # # # # # # # # # # # # # # # #         x = random.randn(0, 1)
# # # # # # # # # # # # # # # # # # # # # #         y = random.randn(0, 1)
# # # # # # # # # # # # # # # # # # # # # #         if math.sqrt(x**2 + y**2) <= 1:
# # # # # # # # # # # # # # # # # # # # # #             inside_circle += 1
# # # # # # # # # # # # # # # # # # # # # #     return inside_circle

# # # # # # # # # # # # # # # # # # # # # # def parallel_monte_carlo_pi(total_iterations, num_processes):
# # # # # # # # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # # # # # # # #     Perform a parallelized Monte Carlo simulation to estimate the value of π.

# # # # # # # # # # # # # # # # # # # # # #     Args:
# # # # # # # # # # # # # # # # # # # # # #         total_iterations (int): Total number of iterations for the simulation.
# # # # # # # # # # # # # # # # # # # # # #         num_processes (int): Number of processes to use for parallelization.

# # # # # # # # # # # # # # # # # # # # # #     Returns:
# # # # # # # # # # # # # # # # # # # # # #         float: Estimated value of π.
# # # # # # # # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # # # # # # # #     iterations_per_process = total_iterations // num_processes

# # # # # # # # # # # # # # # # # # # # # #     # Create a pool of processes
# # # # # # # # # # # # # # # # # # # # # #     with multiprocessing.Pool(num_processes) as pool:
# # # # # # # # # # # # # # # # # # # # # #         results = pool.map(monte_carlo_pi_part, [iterations_per_process] * num_processes)

# # # # # # # # # # # # # # # # # # # # # #     total_inside_circle = sum(results)
# # # # # # # # # # # # # # # # # # # # # #     pi_estimate = (4.0 * total_inside_circle) / total_iterations
# # # # # # # # # # # # # # # # # # # # # #     return pi_estimate

# # # # # # # # # # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # # # # # # # # # #     total_iterations = 1000000
# # # # # # # # # # # # # # # # # # # # # #     num_processes = multiprocessing.cpu_count()
# # # # # # # # # # # # # # # # # # # # # #     num_processes = 5
# # # # # # # # # # # # # # # # # # # # # #     print(num_processes)  # Use all available CPU cores

# # # # # # # # # # # # # # # # # # # # # #     pi_estimate = parallel_monte_carlo_pi(total_iterations, num_processes)
# # # # # # # # # # # # # # # # # # # # # #     print(f"Estimated value of π: {pi_estimate}")


# # # # # # # # # # # # # # # # # # # # # import datetime
# # # # # # # # # # # # # # # # # # # # # import time
# # # # # # # # # # # # # # # # # # # # # from itertools import product

# # # # # # # # # # # # # # # # # # # # # # Sensitivity analysis cases
# # # # # # # # # # # # # # # # # # # # # test_cases = {
# # # # # # # # # # # # # # # # # # # # #     "step_size": [0.00, 0.01, 0.02]
# # # # # # # # # # # # # # # # # # # # # }

# # # # # # # # # # # # # # # # # # # # # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# # # # # # # # # # # # # # # # # # # # # def generate_time_tag(parameters):
# # # # # # # # # # # # # # # # # # # # #     params_str = "_".join(f"{k}_{v:.2f}".replace('.', '_') for k, v in parameters.items())
# # # # # # # # # # # # # # # # # # # # #     return f"{current_time}_{params_str}"

# # # # # # # # # # # # # # # # # # # # # # Generate combinations of all test cases
# # # # # # # # # # # # # # # # # # # # # keys, values = zip(*test_cases.items())
# # # # # # # # # # # # # # # # # # # # # combinations = [dict(zip(keys, v)) for v in product(*values)]

# # # # # # # # # # # # # # # # # # # # # # Running the sensitivity analysis
# # # # # # # # # # # # # # # # # # # # # for case in combinations:
# # # # # # # # # # # # # # # # # # # # #     time_tag = generate_time_tag(case)
# # # # # # # # # # # # # # # # # # # # #     print(f"Run completed with time tag: {time_tag}")


# # # # # # # # # # # # # # # # # # # # # import pickle

# # # # # # # # # # # # # # # # # # # # # # class MyClass:
# # # # # # # # # # # # # # # # # # # # # #     def __init__(self, name, value):
# # # # # # # # # # # # # # # # # # # # # #         self.name = name
# # # # # # # # # # # # # # # # # # # # # #         self.value = value

# # # # # # # # # # # # # # # # # # # # # #     def display(self):
# # # # # # # # # # # # # # # # # # # # # #         print(f"Name: {self.name}, Value: {self.value}")

# # # # # # # # # # # # # # # # # # # # # # # Create an instance of the class
# # # # # # # # # # # # # # # # # # # # # # my_instance = MyClass(name="example", value=42)

# # # # # # # # # # # # # # # # # # # # # # # Save the class instance to a file
# # # # # # # # # # # # # # # # # # # # # # with open("my_instance.pkl", "wb") as f:
# # # # # # # # # # # # # # # # # # # # # #     pickle.dump(my_instance, f)

# # # # # # # # # # # # # # # # # # # # # # Load the class instance from the file
# # # # # # # # # # # # # # # # # # # # # with open("my_instance.pkl", "rb") as f:
# # # # # # # # # # # # # # # # # # # # #     loaded_instance = pickle.load(f)

# # # # # # # # # # # # # # # # # # # # # # Display the loaded instance
# # # # # # # # # # # # # # # # # # # # # loaded_instance.display()


# # # # # # # # # # # # # # # # # # # # import json

# # # # # # # # # # # # # # # # # # # # class MyClass:
# # # # # # # # # # # # # # # # # # # #     def __init__(self, name, value):
# # # # # # # # # # # # # # # # # # # #         self.name = name
# # # # # # # # # # # # # # # # # # # #         self.value = value

# # # # # # # # # # # # # # # # # # # #     def display(self):
# # # # # # # # # # # # # # # # # # # #         print(f"Name: {self.name}, Value: {self.value}")
# # # # # # # # # # # # # # # # # # # #         self.value = self.value*2

# # # # # # # # # # # # # # # # # # # #         return Output(self)



# # # # # # # # # # # # # # # # # # # # class Output(my_class):

# # # # # # # # # # # # # # # # # # # #     def __init__(self, my_class):
# # # # # # # # # # # # # # # # # # # #         self.my_class = my_class



# # # # # # # # # # # # # # # # # # # # # Custom encoder and decoder for MyClass
# # # # # # # # # # # # # # # # # # # # class MyClassEncoder(json.JSONEncoder):
# # # # # # # # # # # # # # # # # # # #     def default(self, obj):
# # # # # # # # # # # # # # # # # # # #         if isinstance(obj, MyClass):
# # # # # # # # # # # # # # # # # # # #             return obj.__dict__
# # # # # # # # # # # # # # # # # # # #         return super().default(obj)

# # # # # # # # # # # # # # # # # # # # def myclass_decoder(dct):
# # # # # # # # # # # # # # # # # # # #     if 'name' in dct and 'value' in dct:
# # # # # # # # # # # # # # # # # # # #         return MyClass(**dct)
# # # # # # # # # # # # # # # # # # # #     return dct

# # # # # # # # # # # # # # # # # # # # # Create an instance of the class
# # # # # # # # # # # # # # # # # # # # my_instance = MyClass(name="example", value=42)

# # # # # # # # # # # # # # # # # # # # # Save the class instance to a file
# # # # # # # # # # # # # # # # # # # # with open("my_instance.json", "w") as f:
# # # # # # # # # # # # # # # # # # # #     json.dump(my_instance, f, cls=MyClassEncoder)

# # # # # # # # # # # # # # # # # # # # # Load the class instance from the file
# # # # # # # # # # # # # # # # # # # # with open("my_instance.json", "r") as f:
# # # # # # # # # # # # # # # # # # # #     loaded_instance = json.load(f, object_hook=myclass_decoder)

# # # # # # # # # # # # # # # # # # # # # Display the loaded instance
# # # # # # # # # # # # # # # # # # # # loaded_instance.display()


# # # # # # # # # # # # # # # # # # # import itertools

# # # # # # # # # # # # # # # # # # # def generate_case_custom_tag(case, custom_tag=False, run=0):
# # # # # # # # # # # # # # # # # # #     params_str = "_".join(f"{run}_{k}_{v:.2f}".replace('.', '_') for k, v in case.items())
# # # # # # # # # # # # # # # # # # #     time = current_time
# # # # # # # # # # # # # # # # # # #     if custom_tag is not False:
# # # # # # # # # # # # # # # # # # #         time = custom_tag
# # # # # # # # # # # # # # # # # # #     return f"{time}_{params_str}"

# # # # # # # # # # # # # # # # # # # current_time =100000
# # # # # # # # # # # # # # # # # # # custom_input=False

# # # # # # # # # # # # # # # # # # # # Define the cases
# # # # # # # # # # # # # # # # # # # run_num = 1
# # # # # # # # # # # # # # # # # # # cases = {
# # # # # # # # # # # # # # # # # # #     "delta_v_min": [0.00, 0.01]
# # # # # # # # # # # # # # # # # # # }

# # # # # # # # # # # # # # # # # # # # Extract keys and values
# # # # # # # # # # # # # # # # # # # keys, values = zip(*cases.items())

# # # # # # # # # # # # # # # # # # # # Generate combinations of cases
# # # # # # # # # # # # # # # # # # # combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# # # # # # # # # # # # # # # # # # # # Initialize a list to store time tags
# # # # # # # # # # # # # # # # # # # time_tags = []
# # # # # # # # # # # # # # # # # # # for case in combinations:
# # # # # # # # # # # # # # # # # # #     for run in range(run_num):

# # # # # # # # # # # # # # # # # # #         time_tag = generate_case_custom_tag(case, run=run)
# # # # # # # # # # # # # # # # # # #         if custom_input:
# # # # # # # # # # # # # # # # # # #             time_tag = generate_case_custom_tag(case, custom_tag=custom_tag, run=run)
# # # # # # # # # # # # # # # # # # #         time_tags.append(time_tag)

# # # # # # # # # # # # # # # # # # # # Output the time tags
# # # # # # # # # # # # # # # # # # # print(time_tags)
# # # # # # # # # # # # # # # # # # # a


# # # # # # # # # # # # # # # # # import multiprocessing
# # # # # # # # # # # # # # # # # import time

# # # # # # # # # # # # # # # # # def process_case(case):
# # # # # # # # # # # # # # # # #     # Simulate some work with a sleep
# # # # # # # # # # # # # # # # #     time.sleep(5)
# # # # # # # # # # # # # # # # #     return f"Processed case: {case}"

# # # # # # # # # # # # # # # # # def parallel_processing(cases, num_workers):
# # # # # # # # # # # # # # # # #     with multiprocessing.Pool(processes=num_workers) as pool:
# # # # # # # # # # # # # # # # #         results = pool.map(process_case, cases)
# # # # # # # # # # # # # # # # #     return results

# # # # # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # # # # #     # Define the cases to process
# # # # # # # # # # # # # # # # #     cases = [f"Case {i}" for i in range(5)]

# # # # # # # # # # # # # # # # #     # Number of worker processes
# # # # # # # # # # # # # # # # #     num_workers = multiprocessing.cpu_count()

# # # # # # # # # # # # # # # # #     # Process cases in parallel
# # # # # # # # # # # # # # # # #     start_time = time.time()
# # # # # # # # # # # # # # # # #     results = parallel_processing(cases, num_workers)
# # # # # # # # # # # # # # # # #     end_time = time.time()

# # # # # # # # # # # # # # # # #     # Print results
# # # # # # # # # # # # # # # # #     for result in results:
# # # # # # # # # # # # # # # # #         print(result)

# # # # # # # # # # # # # # # # #     print(f"Processing took {end_time - start_time:.2f} seconds")


# # # # # # # # # # # # # # # # import concurrent.futures
# # # # # # # # # # # # # # # # import time

# # # # # # # # # # # # # # # # def task_function(x):
# # # # # # # # # # # # # # # #     # Your task implementation
# # # # # # # # # # # # # # # #     time.sleep(5)
# # # # # # # # # # # # # # # #     return x * x

# # # # # # # # # # # # # # # # def main():
# # # # # # # # # # # # # # # #     num_workers = 56  # Number of worker processes
# # # # # # # # # # # # # # # #     tasks = list(range(100))

# # # # # # # # # # # # # # # #     with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
# # # # # # # # # # # # # # # #         results = list(executor.map(task_function, tasks))

# # # # # # # # # # # # # # # #     print("Results:", results)

# # # # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # # # #     main()



# # # # # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # # # # Example data (list of lists)
# # # # # # # # # # # # # # # histories = [
# # # # # # # # # # # # # # #     [1, 2, 3],
# # # # # # # # # # # # # # #     [4, 5],
# # # # # # # # # # # # # # #     [6, 7, 8, 9],
# # # # # # # # # # # # # # #     [10, 11, 12]
# # # # # # # # # # # # # # # ]

# # # # # # # # # # # # # # # # Convert each sublist to a NumPy array
# # # # # # # # # # # # # # # arrays = [np.array(sublist) for sublist in histories]

# # # # # # # # # # # # # # # # Determine the maximum length among all arrays
# # # # # # # # # # # # # # # max_length = max(len(arr) for arr in arrays)

# # # # # # # # # # # # # # # # Pad arrays with NaNs to ensure they all have the same length
# # # # # # # # # # # # # # # padded_arrays = [np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.nan) for arr in arrays]

# # # # # # # # # # # # # # # # Stack arrays vertically to create a 2D array (NaNs will be ignored in mean calculation)
# # # # # # # # # # # # # # # stacked_array = np.vstack(padded_arrays)

# # # # # # # # # # # # # # # # Calculate column-wise mean ignoring NaNs
# # # # # # # # # # # # # # # column_means = np.nanmean(stacked_array, axis=0)

# # # # # # # # # # # # # # # print("Column-wise means:")
# # # # # # # # # # # # # # # print(column_means)


# # # # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # # # Example data (list of lists)
# # # # # # # # # # # # # # histories = [
# # # # # # # # # # # # # #     [1, 2, 3],
# # # # # # # # # # # # # #     [4, 5],
# # # # # # # # # # # # # #     [6, 7, 8, 9],
# # # # # # # # # # # # # #     [10, 11, 12]
# # # # # # # # # # # # # # ]

# # # # # # # # # # # # # # # Determine the maximum length among all sublists
# # # # # # # # # # # # # # max_length = max(len(sublist) for sublist in histories)
# # # # # # # # # # # # # # array_nan = np.full((len(histories), max_length), np.nan)
# # # # # # # # # # # # # # for i, sublist in enumerate(histories):
# # # # # # # # # # # # # #     array_nan[i, :len(sublist)] = sublist

# # # # # # # # # # # # # # # Calculate column-wise mean ignoring NaNs
# # # # # # # # # # # # # # means = np.nanmean(array_nan, axis=0)
# # # # # # # # # # # # # # stds = np.nanstd(array_nan, axis=0)

# # # # # # # # # # # # # # print(means, np.nanstd(array_nan, axis=0))




# # # # # # # # # # # # # data = {
# # # # # # # # # # # # #         "0": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.995685665947465,
# # # # # # # # # # # # #                 "reduction": 0.0
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "1": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.901855562146639,
# # # # # # # # # # # # #                 "reduction": -1.3412567156577395
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "2": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     0.9,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.895064634431089,
# # # # # # # # # # # # #                 "reduction": -1.4383297981234953
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "3": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     0.9,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.90479921217906,
# # # # # # # # # # # # #                 "reduction": -1.2991786382113848
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "4": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     0.9,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.900378550561203,
# # # # # # # # # # # # #                 "reduction": -1.3623698939216757
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "5": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     0.9,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.896974884037777,
# # # # # # # # # # # # #                 "reduction": -1.411023688359485
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "6": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     0.9,
# # # # # # # # # # # # #                     1.0
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.905816482653776,
# # # # # # # # # # # # #                 "reduction": -1.2846372405086812
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "7": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     0.9
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.893808276210644,
# # # # # # # # # # # # #                 "reduction": -1.4562888414601693
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "8": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.795208548754506,
# # # # # # # # # # # # #                 "reduction": -2.865725059214863
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "9": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.768362789666234,
# # # # # # # # # # # # #                 "reduction": -3.2494724196622844
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "1": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9620991253644315,
# # # # # # # # # # # # #                     0.9620991253644315,
# # # # # # # # # # # # #                     0.9620991253644315,
# # # # # # # # # # # # #                     0.9620991253644315,
# # # # # # # # # # # # #                     0.9620991253644315,
# # # # # # # # # # # # #                     1.09067055393586,
# # # # # # # # # # # # #                     0.9620991253644315
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.855335975394089,
# # # # # # # # # # # # #                 "reduction": -2.0062320872498427
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "2": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9512703040399835,
# # # # # # # # # # # # #                     0.9512703040399835,
# # # # # # # # # # # # #                     1.079841732611412,
# # # # # # # # # # # # #                     0.9512703040399835,
# # # # # # # # # # # # #                     0.9512703040399835,
# # # # # # # # # # # # #                     1.016576426488963,
# # # # # # # # # # # # #                     0.9512703040399835
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.850099743400205,
# # # # # # # # # # # # #                 "reduction": -2.081081533664691
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "3": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     0.9,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.907550004081844,
# # # # # # # # # # # # #                 "reduction": -1.2598573760201184
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "1": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     0.9,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.9030015899703185,
# # # # # # # # # # # # #                 "reduction": -1.3248747928784168
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "2": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     0.9,
# # # # # # # # # # # # #                     1.0
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.8990261177898216,
# # # # # # # # # # # # #                 "reduction": -1.3817022772785261
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "3": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     1.0,
# # # # # # # # # # # # #                     0.9
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.899280360548562,
# # # # # # # # # # # # #                 "reduction": -1.3780679979400796
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "4": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715,
# # # # # # # # # # # # #                     0.9714285714285715
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.8029206002443825,
# # # # # # # # # # # # #                 "reduction": -2.7554849504087224
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "5": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103,
# # # # # # # # # # # # #                     0.9673469387755103
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.764821726233478,
# # # # # # # # # # # # #                 "reduction": -3.300090237583876
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "6": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9620991253644315,
# # # # # # # # # # # # #                     0.9620991253644315,
# # # # # # # # # # # # #                     0.9620991253644315,
# # # # # # # # # # # # #                     0.9620991253644315,
# # # # # # # # # # # # #                     0.9620991253644315,
# # # # # # # # # # # # #                     1.09067055393586,
# # # # # # # # # # # # #                     0.9620991253644315
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.8574181660317866,
# # # # # # # # # # # # #                 "reduction": -1.976468162209118
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "7": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9512703040399835,
# # # # # # # # # # # # #                     0.9512703040399835,
# # # # # # # # # # # # #                     1.079841732611412,
# # # # # # # # # # # # #                     0.9512703040399835,
# # # # # # # # # # # # #                     0.9512703040399835,
# # # # # # # # # # # # #                     1.016576426488963,
# # # # # # # # # # # # #                     0.9512703040399835
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.852626021987763,
# # # # # # # # # # # # #                 "reduction": -2.044969582553809
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "8": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.957755697030999,
# # # # # # # # # # # # #                     0.957755697030999,
# # # # # # # # # # # # #                     0.9944903909085501,
# # # # # # # # # # # # #                     1.057755697030999,
# # # # # # # # # # # # #                     0.9291842684595706,
# # # # # # # # # # # # #                     0.9845778544654016,
# # # # # # # # # # # # #                     0.9291842684595706
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.812878850514271,
# # # # # # # # # # # # #                 "reduction": -2.613136498156761
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "4": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.945685896182713,
# # # # # # # # # # # # #                     0.945685896182713,
# # # # # # # # # # # # #                     0.9929162168824217,
# # # # # # # # # # # # #                     0.9742573247541415,
# # # # # # # # # # # # #                     1.0375226308765906,
# # # # # # # # # # # # #                     0.9801715271698019,
# # # # # # # # # # # # #                     0.9089512023051622
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.782473558253875,
# # # # # # # # # # # # #                 "reduction": -3.047765692667295
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "5": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9301675808063454,
# # # # # # # # # # # # #                     0.9301675808063454,
# # # # # # # # # # # # #                     0.9908922788488279,
# # # # # # # # # # # # #                     0.9669022746838962,
# # # # # # # # # # # # #                     0.9482433825556167,
# # # # # # # # # # # # #                     0.9745062492183167,
# # # # # # # # # # # # #                     1.0115086886780658
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.746981434424865,
# # # # # # # # # # # # #                 "reduction": -3.555108725556451
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "1": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9201915209215374,
# # # # # # # # # # # # #                     0.9201915209215374,
# # # # # # # # # # # # #                     0.9895911758272318,
# # # # # # # # # # # # #                     0.9621740282101672,
# # # # # # # # # # # # #                     0.9408495800635621,
# # # # # # # # # # # # #                     0.9708642848209332,
# # # # # # # # # # # # #                     1.027438501346361
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.732315399003556,
# # # # # # # # # # # # #                 "reduction": -3.7647527278977475
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "6": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9073651582124991,
# # # # # # # # # # # # #                     0.9073651582124991,
# # # # # # # # # # # # #                     0.9879183290851796,
# # # # # # # # # # # # #                     0.9560948541725156,
# # # # # # # # # # # # #                     0.9313432625737772,
# # # # # # # # # # # # #                     1.0947531877385832,
# # # # # # # # # # # # #                     0.9193482604913115
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.701062011150313,
# # # # # # # # # # # # #                 "reduction": -4.211505045620851
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "1": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.8941316093857132,
# # # # # # # # # # # # #                     0.8941316093857132,
# # # # # # # # # # # # #                     0.9861923760973483,
# # # # # # # # # # # # #                     0.949822690482875,
# # # # # # # # # # # # #                     0.9215351572271739,
# # # # # # # # # # # # #                     1.1225750717012382,
# # # # # # # # # # # # #                     0.9078265834186419
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.68085462837025,
# # # # # # # # # # # # #                 "reduction": -4.500359973429076
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "7": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9258467425684336,
# # # # # # # # # # # # #                     0.9258467425684336,
# # # # # # # # # # # # #                     1.0327029896444384,
# # # # # # # # # # # # #                     0.9904881759847821,
# # # # # # # # # # # # #                     0.9576544320272717,
# # # # # # # # # # # # #                     0.9131982103071166,
# # # # # # # # # # # # #                     0.94174269457094
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.682565435186947,
# # # # # # # # # # # # #                 "reduction": -4.475904803508781
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "8": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9294116891867243,
# # # # # # # # # # # # #                     0.9294116891867243,
# # # # # # # # # # # # #                     0.90149217012118,
# # # # # # # # # # # # #                     1.0125221035791725,
# # # # # # # # # # # # #                     0.9703072899195166,
# # # # # # # # # # # # #                     0.9577557045592007,
# # # # # # # # # # # # #                     0.9498493417613756
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.65390749837299,
# # # # # # # # # # # # #                 "reduction": -4.885556382816493
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "1": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9262890299219728,
# # # # # # # # # # # # #                     0.9262890299219728,
# # # # # # # # # # # # #                     0.8760136611940037,
# # # # # # # # # # # # #                     1.0212723606561995,
# # # # # # # # # # # # #                     0.9730268593308786,
# # # # # # # # # # # # #                     0.9493527442835203,
# # # # # # # # # # # # #                     0.9496463471501457
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.627707133038006,
# # # # # # # # # # # # #                 "reduction": -5.260078146458876
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "9": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.9139358198788443,
# # # # # # # # # # # # #                     0.9139358198788443,
# # # # # # # # # # # # #                     0.952993017619886,
# # # # # # # # # # # # #                     0.8956129001953574,
# # # # # # # # # # # # #                     1.0049197800345893,
# # # # # # # # # # # # #                     0.9796899593879387,
# # # # # # # # # # # # #                     0.9777816855390957
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.642241407587852,
# # # # # # # # # # # # #                 "reduction": -5.0523176031201
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "10": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.8838364450384926,
# # # # # # # # # # # # #                     0.8838364450384926,
# # # # # # # # # # # # #                     0.9707875360116685,
# # # # # # # # # # # # #                     0.9602784054454376,
# # # # # # # # # # # # #                     0.9722443938101648,
# # # # # # # # # # # # #                     0.9951996389845879,
# # # # # # # # # # # # #                     0.9373525580302446
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.6078538447469155,
# # # # # # # # # # # # #                 "reduction": -5.5438714619265745
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "1": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.871323284125624,
# # # # # # # # # # # # #                     0.871323284125624,
# # # # # # # # # # # # #                     0.9706959595235392,
# # # # # # # # # # # # #                     0.958685524590704,
# # # # # # # # # # # # #                     0.972360939864678,
# # # # # # # # # # # # #                     0.9985955057783042,
# # # # # # # # # # # # #                     0.9324845561161978
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.571870547333485,
# # # # # # # # # # # # #                 "reduction": -6.0582355876417235
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "11": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.888332659696611,
# # # # # # # # # # # # #                     0.888332659696611,
# # # # # # # # # # # # #                     0.9429512455981375,
# # # # # # # # # # # # #                     0.9530005663588856,
# # # # # # # # # # # # #                     0.8875327083587417,
# # # # # # # # # # # # #                     0.9917206771315012,
# # # # # # # # # # # # #                     1.0065537425282358
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.554011828784669,
# # # # # # # # # # # # #                 "reduction": -6.313517477103188
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "1": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.8801393401985964,
# # # # # # # # # # # # #                     0.8801393401985964,
# # # # # # # # # # # # #                     0.9358133925575254,
# # # # # # # # # # # # #                     0.9499638865881348,
# # # # # # # # # # # # #                     0.8661055765704775,
# # # # # # # # # # # # #                     0.9933705556974581,
# # # # # # # # # # # # #                     1.0204969625601032
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.529863629323145,
# # # # # # # # # # # # #                 "reduction": -6.658704505432218
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         },
# # # # # # # # # # # # #         "12": {
# # # # # # # # # # # # #             "0": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.8417551603675528,
# # # # # # # # # # # # #                     0.8417551603675528,
# # # # # # # # # # # # #                     0.959510939071339,
# # # # # # # # # # # # #                     0.9549443659982668,
# # # # # # # # # # # # #                     0.92878229697267,
# # # # # # # # # # # # #                     1.0119805846463497,
# # # # # # # # # # # # #                     0.9633437271392
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.508132612557949,
# # # # # # # # # # # # #                 "reduction": -6.969339056538127
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             "1": {
# # # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # # #                     0.8238134777378447,
# # # # # # # # # # # # #                     0.8238134777378447,
# # # # # # # # # # # # #                     0.9583915105421719,
# # # # # # # # # # # # #                     0.953172569887232,
# # # # # # # # # # # # #                     0.923273062429407,
# # # # # # # # # # # # #                     1.0183568197707555,
# # # # # # # # # # # # #                     0.9627718397625844
# # # # # # # # # # # # #                 ],
# # # # # # # # # # # # #                 "objective_value": 6.4612716153697605,
# # # # # # # # # # # # #                 "reduction": -7.639194727959893
# # # # # # # # # # # # #             }
# # # # # # # # # # # # #         }
# # # # # # # # # # # # #     }


# # # # # # # # # # # # # def get_best_simplex(intermediate_iterations):

# # # # # # # # # # # # #     simplex = []
# # # # # # # # # # # # #     objective_values = [999]
# # # # # # # # # # # # #     for key, values in intermediate_iterations.items():
# # # # # # # # # # # # #         sub_objective_values = []
# # # # # # # # # # # # #         for subkey, subvalue in values.items():
# # # # # # # # # # # # #             design_vector = subvalue["design_vector"]
# # # # # # # # # # # # #             n = len(design_vector)
# # # # # # # # # # # # #             objective_value = subvalue["objective_value"]
# # # # # # # # # # # # #             sub_objective_values.append(objective_value)
# # # # # # # # # # # # #             objective_value = max(sub_objective_values)

# # # # # # # # # # # # #         if objective_value < objective_values[-1]:
# # # # # # # # # # # # #             objective_values.append(objective_value)
# # # # # # # # # # # # #             simplex.append(design_vector)
# # # # # # # # # # # # #             if len(simplex) > n + 1:
# # # # # # # # # # # # #                 simplex = simplex[1:]

# # # # # # # # # # # # #     return simplex

# # # # # # # # # # # # # print(get_best_simplex(data))




# # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # Provided data structure
# # # # # # # # # # # # # intermediate_iteration_history = {
# # # # # # # # # # # # #     "0": {
# # # # # # # # # # # # #         "0": {"design_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "objective_value": 6.995685665947465, "reduction": 0.0},
# # # # # # # # # # # # #         "1": {"design_vector": [0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "objective_value": 6.901855562146639, "reduction": -1.3412567156577395},
# # # # # # # # # # # # #         "2": {"design_vector": [1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0], "objective_value": 6.895064634431089, "reduction": -1.4383297981234953},
# # # # # # # # # # # # #         "3": {"design_vector": [1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0], "objective_value": 6.90479921217906, "reduction": -1.2991786382113848},
# # # # # # # # # # # # #         "4": {"design_vector": [1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0], "objective_value": 6.900378550561203, "reduction": -1.3623698939216757},
# # # # # # # # # # # # #         "5": {"design_vector": [1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0], "objective_value": 6.896974884037777, "reduction": -1.411023688359485},
# # # # # # # # # # # # #         "6": {"design_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0], "objective_value": 6.905816482653776, "reduction": -1.2846372405086812},
# # # # # # # # # # # # #         "7": {"design_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9], "objective_value": 6.893808276210644, "reduction": -1.4562888414601693},
# # # # # # # # # # # # #         "8": {"design_vector": [0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715], "objective_value": 6.795208548754506, "reduction": -2.865725059214863},
# # # # # # # # # # # # #         "9": {"design_vector": [0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103], "objective_value": 6.768362789666234, "reduction": -3.2494724196622844}
# # # # # # # # # # # # #     },
# # # # # # # # # # # # #     "1": {"0": {"design_vector": [0.9620991253644315, 0.9620991253644315, 0.9620991253644315, 0.9620991253644315, 0.9620991253644315, 1.09067055393586, 0.9620991253644315], "objective_value": 6.855335975394089, "reduction": -2.0062320872498427}},
# # # # # # # # # # # # #     "2": {"0": {"design_vector": [0.9512703040399835, 0.9512703040399835, 1.079841732611412, 0.9512703040399835, 0.9512703040399835, 1.016576426488963, 0.9512703040399835], "objective_value": 6.850099743400205, "reduction": -2.081081533664691}},
# # # # # # # # # # # # #     "3": {
# # # # # # # # # # # # #         "0": {"design_vector": [1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0], "objective_value": 6.907550004081844, "reduction": -1.2598573760201184},
# # # # # # # # # # # # #         "1": {"design_vector": [1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0], "objective_value": 6.9030015899703185, "reduction": -1.3248747928784168},
# # # # # # # # # # # # #         "2": {"design_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0], "objective_value": 6.8990261177898216, "reduction": -1.3817022772785261},
# # # # # # # # # # # # #         "3": {"design_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9], "objective_value": 6.899280360548562, "reduction": -1.3780679979400796},
# # # # # # # # # # # # #         "4": {"design_vector": [0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715], "objective_value": 6.8029206002443825, "reduction": -2.7554849504087224},
# # # # # # # # # # # # #         "5": {"design_vector": [0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103], "objective_value": 6.764821726233478, "reduction": -3.300090237583876},
# # # # # # # # # # # # #         "6": {"design_vector": [0.9620991253644315, 0.9620991253644315, 0.9620991253644315, 0.9620991253644315, 0.9620991253644315, 1.09067055393586, 0.9620991253644315], "objective_value": 6.8574181660317866, "reduction": -1.976468162209118},
# # # # # # # # # # # # #         "7": {"design_vector": [0.9512703040399835, 0.9512703040399835, 1.079841732611412, 0.9512703040399835, 0.9512703040399835, 1.016576426488963, 0.9512703040399835], "objective_value": 6.852626021987763, "reduction": -2.044969582553809},
# # # # # # # # # # # # #         "8": {"design_vector": [0.957755697030999, 0.957755697030999, 0.9944903909085501, 1.057755697030999, 0.9291842684595706, 0.9845778544654016, 0.9291842684595706], "objective_value": 6.812878850514271, "reduction": -2.613136498156761}
# # # # # # # # # # # # #     },
# # # # # # # # # # # # #     "4": {"0": {"design_vector": [0.945685896182713, 0.945685896182713, 0.9929162168824217, 0.965685896182713, 1.065685896182713, 1.0011001985312737, 0.965685896182713], "objective_value": 6.793835996319352, "reduction": -2.879999913254434}},
# # # # # # # # # # # # #     "5": {"0": {"design_vector": [0.945993380278593, 1.054306219928386, 0.953993380278593, 0.972178188973021, 0.953993380278593, 0.972178188973021, 0.953993380278593], "objective_value": 6.806652506742244, "reduction": -2.693691417915756}},
# # # # # # # # # # # # #     "6": {"0": {"design_vector": [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0], "objective_value": 6.8990261177898216, "reduction": -1.3817022772785261}},
# # # # # # # # # # # # #     "7": {"0": {"design_vector": [0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715, 0.9714285714285715], "objective_value": 6.8029206002443825, "reduction": -2.7554849504087224}},
# # # # # # # # # # # # #     "8": {"0": {"design_vector": [0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103, 0.9673469387755103], "objective_value": 6.764821726233478, "reduction": -3.300090237583876}},
# # # # # # # # # # # # #     "9": {"0": {"design_vector": [0.9620991253644315, 0.9620991253644315, 0.9620991253644315, 0.9620991253644315, 0.9620991253644315, 1.09067055393586, 0.9620991253644315], "objective_value": 6.8574181660317866, "reduction": -1.976468162209118}}
# # # # # # # # # # # # # }

# # # # # # # # # # # # intermediate_iteration_history = {
# # # # # # # # # # # #         "0": {
# # # # # # # # # # # #             "0": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02472565840653605,
# # # # # # # # # # # #                 "reduction": 0.0
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "1": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.5,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.023477882734525397,
# # # # # # # # # # # #                 "reduction": -5.046481074416247
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "2": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.024230214743216735,
# # # # # # # # # # # #                 "reduction": -2.0037632776983902
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "3": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02304612374269096,
# # # # # # # # # # # #                 "reduction": -6.792679233169043
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "4": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.024027827195250824,
# # # # # # # # # # # #                 "reduction": -2.822295769890432
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "5": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.023166655940117294,
# # # # # # # # # # # #                 "reduction": -6.30520102148885
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "6": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5,
# # # # # # # # # # # #                     1.0
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02319551223169515,
# # # # # # # # # # # #                 "reduction": -6.188495164344805
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "7": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02289024466427567,
# # # # # # # # # # # #                 "reduction": -7.423113722930026
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "8": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.8571428571428572,
# # # # # # # # # # # #                     0.8571428571428572,
# # # # # # # # # # # #                     0.8571428571428572,
# # # # # # # # # # # #                     0.8571428571428572,
# # # # # # # # # # # #                     0.8571428571428572,
# # # # # # # # # # # #                     0.8571428571428572,
# # # # # # # # # # # #                     0.8571428571428572
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.030090258577654554,
# # # # # # # # # # # #                 "reduction": 21.696490677475396
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "9": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.9770408163265306,
# # # # # # # # # # # #                     0.9770408163265306,
# # # # # # # # # # # #                     0.9770408163265306,
# # # # # # # # # # # #                     0.9770408163265306,
# # # # # # # # # # # #                     0.9770408163265306,
# # # # # # # # # # # #                     0.9770408163265306,
# # # # # # # # # # # #                     0.9770408163265306
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.023016043506097747,
# # # # # # # # # # # #                 "reduction": -6.914335191116196
# # # # # # # # # # # #             }
# # # # # # # # # # # #         },
# # # # # # # # # # # #         "1": {
# # # # # # # # # # # #             "0": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.8505830903790088,
# # # # # # # # # # # #                     1.4934402332361516,
# # # # # # # # # # # #                     0.8505830903790088,
# # # # # # # # # # # #                     0.8505830903790088,
# # # # # # # # # # # #                     0.8505830903790088,
# # # # # # # # # # # #                     0.8505830903790088,
# # # # # # # # # # # #                     0.8505830903790088
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.03219050046837208,
# # # # # # # # # # # #                 "reduction": 30.190670513601997
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "1": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.9759865680966264,
# # # # # # # # # # # #                     0.6596600374843815,
# # # # # # # # # # # #                     0.9759865680966264,
# # # # # # # # # # # #                     0.9759865680966264,
# # # # # # # # # # # #                     0.9759865680966264,
# # # # # # # # # # # #                     0.9759865680966264,
# # # # # # # # # # # #                     0.9759865680966264
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.025634111715432308,
# # # # # # # # # # # #                 "reduction": 3.674131923848451
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "2": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.9803206997084548,
# # # # # # # # # # # #                     0.9803206997084548,
# # # # # # # # # # # #                     0.9803206997084548,
# # # # # # # # # # # #                     0.9803206997084548,
# # # # # # # # # # # #                     0.9803206997084548,
# # # # # # # # # # # #                     0.9803206997084548,
# # # # # # # # # # # #                     0.9088921282798834
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02411826528334123,
# # # # # # # # # # # #                 "reduction": -2.4565296228239575
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "3": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5714285714285714,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.9285714285714286
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02396807866970242,
# # # # # # # # # # # #                 "reduction": -3.063941612302503
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "4": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5714285714285714,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.9285714285714286
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02384294102501914,
# # # # # # # # # # # #                 "reduction": -3.5700460105182477
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "5": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5714285714285714,
# # # # # # # # # # # #                     0.9285714285714286
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.023248922146640358,
# # # # # # # # # # # #                 "reduction": -5.972485082562358
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "6": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.5714285714285714,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.9285714285714286
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.024148118100643316,
# # # # # # # # # # # #                 "reduction": -2.3357934352925627
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "7": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5714285714285714,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.9285714285714286
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02442106632352021,
# # # # # # # # # # # #                 "reduction": -1.231886641834878
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "8": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.5714285714285714,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.9285714285714286
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.025222799733072992,
# # # # # # # # # # # #                 "reduction": 2.0106292757224504
# # # # # # # # # # # #             }
# # # # # # # # # # # #         },
# # # # # # # # # # # #         "2": {
# # # # # # # # # # # #             "0": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.8719283631820074,
# # # # # # # # # # # #                     1.4229487713452729,
# # # # # # # # # # # #                     0.8719283631820074,
# # # # # # # # # # # #                     0.8719283631820074,
# # # # # # # # # # # #                     0.8719283631820074,
# # # # # # # # # # # #                     0.8719283631820074,
# # # # # # # # # # # #                     0.8004997917534362
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02619625266365598,
# # # # # # # # # # # #                 "reduction": 5.9476444790290826
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "1": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.9794170583685369,
# # # # # # # # # # # #                     0.7082800321294698,
# # # # # # # # # # # #                     0.9794170583685369,
# # # # # # # # # # # #                     0.9794170583685369,
# # # # # # # # # # # #                     0.9794170583685369,
# # # # # # # # # # # #                     0.9794170583685369,
# # # # # # # # # # # #                     0.9079884869399655
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.03061305309992266,
# # # # # # # # # # # #                 "reduction": 23.81087126816539
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "2": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.6326530612244897,
# # # # # # # # # # # #                     0.8673469387755103
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.022888320586814215,
# # # # # # # # # # # #                 "reduction": -7.430895426574955
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "3": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.6326530612244897,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.8673469387755103
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.023982720272656594,
# # # # # # # # # # # #                 "reduction": -3.0047253814809007
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "4": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.6326530612244897,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.8673469387755103
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.028673327539460024,
# # # # # # # # # # # #                 "reduction": 15.965880738206902
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "5": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.9831320283215327,
# # # # # # # # # # # #                     0.9831320283215327,
# # # # # # # # # # # #                     0.9831320283215327,
# # # # # # # # # # # #                     0.9831320283215327,
# # # # # # # # # # # #                     0.9831320283215327,
# # # # # # # # # # # #                     0.9831320283215327,
# # # # # # # # # # # #                     0.850478967097043
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02515268815450238,
# # # # # # # # # # # #                 "reduction": 1.7270712914704345
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "6": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     0.6326530612244897,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.8673469387755103
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.03019187854057477,
# # # # # # # # # # # #                 "reduction": 22.10748059430347
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "7": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.6326530612244897,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.8673469387755103
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.029785083890059464,
# # # # # # # # # # # #                 "reduction": 20.462247760350813
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "8": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.6326530612244897,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     1.0,
# # # # # # # # # # # #                     0.8673469387755103
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.029451453009615783,
# # # # # # # # # # # #                 "reduction": 19.112917138054872
# # # # # # # # # # # #             }
# # # # # # # # # # # #         },
# # # # # # # # # # # #         "3": {
# # # # # # # # # # # #             "0": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.3625275182959484,
# # # # # # # # # # # #                     0.8902243112988637,
# # # # # # # # # # # #                     0.8902243112988635,
# # # # # # # # # # # #                     0.8902243112988637,
# # # # # # # # # # # #                     0.8902243112988637,
# # # # # # # # # # # #                     0.8902243112988637,
# # # # # # # # # # # #                     0.7575712500743738
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.021637100242460103,
# # # # # # # # # # # #                 "reduction": -12.491308070726673
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "1": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.4667952978775851,
# # # # # # # # # # # #                     0.8745420700558442,
# # # # # # # # # # # #                     0.8745420700558442,
# # # # # # # # # # # #                     0.8745420700558442,
# # # # # # # # # # # #                     0.8745420700558442,
# # # # # # # # # # # #                     0.8745420700558442,
# # # # # # # # # # # #                     0.7418890088313543
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.05030714912286826,
# # # # # # # # # # # #                 "reduction": 103.46131251886068
# # # # # # # # # # # #             }
# # # # # # # # # # # #         },
# # # # # # # # # # # #         "4": {
# # # # # # # # # # # #             "0": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0987598704621377,
# # # # # # # # # # # #                     0.8588598288128249,
# # # # # # # # # # # #                     0.8588598288128246,
# # # # # # # # # # # #                     1.3311630358099094,
# # # # # # # # # # # #                     0.8588598288128249,
# # # # # # # # # # # #                     0.8588598288128246,
# # # # # # # # # # # #                     0.7262067675883349
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.0414225813422336,
# # # # # # # # # # # #                 "reduction": 67.52872931094058
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "1": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0158721220385578,
# # # # # # # # # # # #                     0.9773167582020612,
# # # # # # # # # # # #                     0.9773167582020611,
# # # # # # # # # # # #                     0.7449135928542893,
# # # # # # # # # # # #                     0.9773167582020612,
# # # # # # # # # # # #                     0.9773167582020611,
# # # # # # # # # # # #                     0.8446636969775714
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.029023698277782697,
# # # # # # # # # # # #                 "reduction": 17.382913735112073
# # # # # # # # # # # #             }
# # # # # # # # # # # #         },
# # # # # # # # # # # #         "5": {
# # # # # # # # # # # #             "0": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.103294762473154,
# # # # # # # # # # # #                     1.3246821095819268,
# # # # # # # # # # # #                     0.8523789025848421,
# # # # # # # # # # # #                     0.8909342664213391,
# # # # # # # # # # # #                     0.8523789025848423,
# # # # # # # # # # # #                     0.8523789025848421,
# # # # # # # # # # # #                     0.7197258413603523
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.027617354663967718,
# # # # # # # # # # # #                 "reduction": 11.695123381091724
# # # # # # # # # # # #             }
# # # # # # # # # # # #         },
# # # # # # # # # # # #         "6": {
# # # # # # # # # # # #             "0": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.1124005377016237,
# # # # # # # # # # # #                     1.0792656558557456,
# # # # # # # # # # # #                     0.8393656142064326,
# # # # # # # # # # # #                     1.187740866014778,
# # # # # # # # # # # #                     0.8393656142064327,
# # # # # # # # # # # #                     0.8393656142064326,
# # # # # # # # # # # #                     0.7067125529819429
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.0241036498543311,
# # # # # # # # # # # #                 "reduction": -2.515639996225642
# # # # # # # # # # # #             }
# # # # # # # # # # # #         },
# # # # # # # # # # # #         "7": {
# # # # # # # # # # # #             "0": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.1603870990835028,
# # # # # # # # # # # #                     1.0792297443023058,
# # # # # # # # # # # #                     1.2430900406074166,
# # # # # # # # # # # #                     0.986294706301861,
# # # # # # # # # # # #                     0.7707868336103318,
# # # # # # # # # # # #                     0.7707868336103318,
# # # # # # # # # # # #                     0.6381337723858419
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.01784919837344274,
# # # # # # # # # # # #                 "reduction": -27.811029012985024
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "1": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.1832995418097174,
# # # # # # # # # # # #                     1.0905482792026353,
# # # # # # # # # # # #                     1.3302953233764057,
# # # # # # # # # # # #                     0.9843368072021268,
# # # # # # # # # # # #                     0.7380420955546652,
# # # # # # # # # # # #                     0.7380420955546652,
# # # # # # # # # # # #                     0.6053890343301751
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.13256253891471256,
# # # # # # # # # # # #                 "reduction": 436.13350445572206
# # # # # # # # # # # #             }
# # # # # # # # # # # #         },
# # # # # # # # # # # #         "8": {
# # # # # # # # # # # #             "0": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0734044327847339,
# # # # # # # # # # # #                     0.68441838749763,
# # # # # # # # # # # #                     1.1349959529677995,
# # # # # # # # # # # #                     1.1226062798463852,
# # # # # # # # # # # #                     0.8950959113184866,
# # # # # # # # # # # #                     0.8950959113184869,
# # # # # # # # # # # #                     0.7624428500939969
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.018408325129518746,
# # # # # # # # # # # #                 "reduction": -25.54970700132038
# # # # # # # # # # # #             }
# # # # # # # # # # # #         },
# # # # # # # # # # # #         "9": {
# # # # # # # # # # # #             "0": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.2193592824972699,
# # # # # # # # # # # #                     0.9406217142369085,
# # # # # # # # # # # #                     1.047632519987185,
# # # # # # # # # # # #                     1.0702583040961495,
# # # # # # # # # # # #                     0.7391896092952116,
# # # # # # # # # # # #                     0.7391896092952116,
# # # # # # # # # # # #                     0.6065365480707217
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.029599499577130747,
# # # # # # # # # # # #                 "reduction": 19.71167396418586
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "1": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0210971227426333,
# # # # # # # # # # # #                     0.9763000135579325,
# # # # # # # # # # # #                     0.9934981787677983,
# # # # # # # # # # # #                     0.9971344654995962,
# # # # # # # # # # # #                     0.9439269966923025,
# # # # # # # # # # # #                     0.9439269966923025,
# # # # # # # # # # # #                     0.8112739354678127
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.026705543767133653,
# # # # # # # # # # # #                 "reduction": 8.007412090083053
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "2": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0858305279702722,
# # # # # # # # # # # #                     0.7408200098982979,
# # # # # # # # # # # #                     1.1504379654877448,
# # # # # # # # # # # #                     1.1031331979114531,
# # # # # # # # # # # #                     0.8773374716458932,
# # # # # # # # # # # #                     0.8773374716458933,
# # # # # # # # # # # #                     0.7446844104214033
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.05116422691203616,
# # # # # # # # # # # #                 "reduction": 106.92766223168104
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "3": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.3336503155513133,
# # # # # # # # # # # #                     0.9172250874422125,
# # # # # # # # # # # #                     0.9406337012000854,
# # # # # # # # # # # #                     0.903948653442149,
# # # # # # # # # # # #                     0.8731618144862163,
# # # # # # # # # # # #                     0.8731618144862163,
# # # # # # # # # # # #                     0.7405087532617264
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.053027194308125275,
# # # # # # # # # # # #                 "reduction": 114.46221344750083
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "4": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0229124427262146,
# # # # # # # # # # # #                     1.0113185349003293,
# # # # # # # # # # # #                     1.0347271486582024,
# # # # # # # # # # # #                     0.9980421009002659,
# # # # # # # # # # # #                     0.9672552619443331,
# # # # # # # # # # # #                     0.65238645727961,
# # # # # # # # # # # #                     0.8346022007198434
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.04495186515876774,
# # # # # # # # # # # #                 "reduction": 81.80250013841913
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "5": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0229124427262146,
# # # # # # # # # # # #                     1.0113185349003293,
# # # # # # # # # # # #                     1.0347271486582024,
# # # # # # # # # # # #                     0.9980421009002659,
# # # # # # # # # # # #                     0.9672552619443331,
# # # # # # # # # # # #                     0.9672552619443331,
# # # # # # # # # # # #                     0.5197333960551203
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.04576864858386692,
# # # # # # # # # # # #                 "reduction": 85.10588406320582
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "6": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0229124427262146,
# # # # # # # # # # # #                     1.0113185349003293,
# # # # # # # # # # # #                     1.0347271486582024,
# # # # # # # # # # # #                     0.9980421009002659,
# # # # # # # # # # # #                     0.65238645727961,
# # # # # # # # # # # #                     0.9672552619443331,
# # # # # # # # # # # #                     0.8346022007198434
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.04303818185496579,
# # # # # # # # # # # #                 "reduction": 74.06283443432574
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "7": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.119255760756178,
# # # # # # # # # # # #                     1.0792605256338257,
# # # # # # # # # # # #                     0.897040532263716,
# # # # # # # # # # # #                     1.158962843198647,
# # # # # # # # # # # #                     0.8295686455498469,
# # # # # # # # # # # #                     0.8295686455498468,
# # # # # # # # # # # #                     0.696915584325357
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.023082086565556676,
# # # # # # # # # # # #                 "reduction": -6.647231851043071
# # # # # # # # # # # #             },
# # # # # # # # # # # #             "8": {
# # # # # # # # # # # #                 "design_vector": [
# # # # # # # # # # # #                     1.0084541812875285,
# # # # # # # # # # # #                     0.9968602734616431,
# # # # # # # # # # # #                     1.020268887219516,
# # # # # # # # # # # #                     0.9835838394615796,
# # # # # # # # # # # #                     0.9527970005056469,
# # # # # # # # # # # #                     0.9527970005056469,
# # # # # # # # # # # #                     0.8201439392811571
# # # # # # # # # # # #                 ],
# # # # # # # # # # # #                 "objective_value": 0.02101097990975618,
# # # # # # # # # # # #                 "reduction": -15.023577676693623
# # # # # # # # # # # #             }
# # # # # # # # # # # #         }
# # # # # # # # # # # #     }
# # # # # # # # # # # # # Extract all design vectors and their corresponding objective values
# # # # # # # # # # # # # design_vectors = []
# # # # # # # # # # # # # objective_values = []

# # # # # # # # # # # # # for outer_key in intermediate_iteration_history:
# # # # # # # # # # # # #     for inner_key in intermediate_iteration_history[outer_key]:
# # # # # # # # # # # # #         entry = intermediate_iteration_history[outer_key][inner_key]
# # # # # # # # # # # # #         design_vectors.append(entry["design_vector"])
# # # # # # # # # # # # #         objective_values.append(entry["objective_value"])

# # # # # # # # # # # # # # Combine design vectors and objective values for sorting
# # # # # # # # # # # # # combined = list(zip(design_vectors, objective_values))
# # # # # # # # # # # # # combined_sorted = sorted(combined, key=lambda x: x[1])
# # # # # # # # # # # # # num_vectors_to_select = len(design_vectors[0]) + 1
# # # # # # # # # # # # # top_design_vectors = combined_sorted[:num_vectors_to_select]

# # # # # # # # # # # # # # Extract the design vectors from the sorted list
# # # # # # # # # # # # # best_design_vectors = [vector for vector, value in reversed(top_design_vectors)]
# # # # # # # # # # # # # best_objective_values = [value for vector, value in reversed(top_design_vectors)]

# # # # # # # # # # # # # print(best_design_vectors, best_objective_values)

# # # # # # # # # # # # # # Convert to 2D numpy array
# # # # # # # # # # # # # best_design_vectors_array = np.array(best_design_vectors)
# # # # # # # # # # # # # best_objective_values_array = np.array(best_objective_values)

# # # # # # # # # # # # # print(best_design_vectors_array)
# # # # # # # # # # # # # print(best_objective_values_array)


# # # # # # # # # # # # def get_best_simplex(intermediate_iteration_history):

# # # # # # # # # # # #     design_vectors = []
# # # # # # # # # # # #     objective_values = []

# # # # # # # # # # # #     for outer_key in intermediate_iteration_history:
# # # # # # # # # # # #         if outer_key == "0":
# # # # # # # # # # # #             for inner_key in intermediate_iteration_history[outer_key]:
# # # # # # # # # # # #                 entry = intermediate_iteration_history[outer_key][inner_key]
# # # # # # # # # # # #                 design_vectors.append(entry["design_vector"])
# # # # # # # # # # # #                 objective_values.append(entry["objective_value"])
# # # # # # # # # # # #                 n = len(entry["design_vector"])

# # # # # # # # # # # #                 # Combine design vectors and objective values for sorting
# # # # # # # # # # # #                 combined = list(zip(design_vectors, objective_values))
# # # # # # # # # # # #                 combined_sorted = sorted(combined, key=lambda x: x[1])
# # # # # # # # # # # #                 num_vectors_to_select = n + 1
# # # # # # # # # # # #                 top_design_vectors = combined_sorted[:num_vectors_to_select]

# # # # # # # # # # # #                 # Extract the design vectors from the sorted list
# # # # # # # # # # # #                 simplex = [vector for vector, value in reversed(top_design_vectors)]
# # # # # # # # # # # #                 objective_values = [value for vector, value in reversed(top_design_vectors)]

# # # # # # # # # # # #                 simplex = simplex[-n-1:]
# # # # # # # # # # # #                 objective_values = objective_values[-n-1:]



# # # # # # # # # # # #     for key, values in intermediate_iteration_history.items():
# # # # # # # # # # # #         sub_objective_values = []
# # # # # # # # # # # #         for subkey, subvalue in values.items():
# # # # # # # # # # # #             design_vector = subvalue["design_vector"]
# # # # # # # # # # # #             objective_value = subvalue["objective_value"]
# # # # # # # # # # # #             sub_objective_values.append(objective_value)
# # # # # # # # # # # #             n = len(design_vector)

# # # # # # # # # # # #             objective_value = max(sub_objective_values)

# # # # # # # # # # # #         if objective_value < objective_values[-1]:
# # # # # # # # # # # #             objective_values.append(objective_value)
# # # # # # # # # # # #             simplex.append(design_vector)
# # # # # # # # # # # #             if len(simplex) > n + 1:
# # # # # # # # # # # #                 simplex = simplex[1:]
# # # # # # # # # # # #                 objective_values = objective_values[1:]

# # # # # # # # # # # #     return simplex, objective_values

# # # # # # # # # # # # simplex, objective_values = get_best_simplex(intermediate_iteration_history)
# # # # # # # # # # # # print(get_best_simplex(intermediate_iteration_history))


# # # # # # # # # # # import numpy as np
# # # # # # # # # # # from scipy.optimize import minimize

# # # # # # # # # # # # Define the objective function
# # # # # # # # # # # def objective_function(x):
# # # # # # # # # # #     return x[0]**2 + x[1]**2

# # # # # # # # # # # # Create a list to store the simplex at each iteration
# # # # # # # # # # # simplex_history = []

# # # # # # # # # # # # Define the callback function
# # # # # # # # # # # def callback(xk, **kwargs):
# # # # # # # # # # #     # Retrieve the current simplex and save it to the history
# # # # # # # # # # #     simplex = kwargs['solver'].simps
# # # # # # # # # # #     simplex_history.append(simplex.copy())

# # # # # # # # # # # # Initial guess
# # # # # # # # # # # initial_guess = np.array([1.0, 1.0])

# # # # # # # # # # # # Perform the Nelder-Mead optimization
# # # # # # # # # # # result = minimize(objective_function, initial_guess, method='Nelder-Mead', callback=callback)

# # # # # # # # # # # # The final simplex is in result.x, and the history is in simplex_history
# # # # # # # # # # # print("Final optimized parameters:", result.x)
# # # # # # # # # # # print("Simplex history:", simplex_history)


# # # # # # # # # # import numpy as np
# # # # # # # # # # from scipy.optimize import minimize

# # # # # # # # # # # Define the objective function
# # # # # # # # # # def objective_function(x):
# # # # # # # # # #     return x[0]**2 + x[1]**2

# # # # # # # # # # # Create a list to store the simplex at each iteration
# # # # # # # # # # simplex_history = []

# # # # # # # # # # # Define the callback function
# # # # # # # # # # def callback(xk):
# # # # # # # # # #     # Access the current simplex and save it to the history
# # # # # # # # # #     simplex = np.vstack([result.allvecs[-1] + step for step in result.deltas])
# # # # # # # # # #     simplex_history.append(simplex)

# # # # # # # # # # # Initial guess
# # # # # # # # # # initial_guess = np.array([1.0, 1.0])

# # # # # # # # # # # Define an initial simplex (optional)
# # # # # # # # # # initial_simplex = initial_guess + np.array([[0, 0], [0.05, 0], [0, 0.05]])

# # # # # # # # # # # Perform the Nelder-Mead optimization
# # # # # # # # # # result = minimize(objective_function, initial_guess, method='Nelder-Mead',
# # # # # # # # # #                   callback=callback, options={'initial_simplex': initial_simplex, 'return_all': True})

# # # # # # # # # # # The final simplex is in result.x, and the history is in simplex_history
# # # # # # # # # # print("Final optimized parameters:", result.x)
# # # # # # # # # # print("Simplex history:", simplex_history)



# # # # # # # # # # list1 = [8, 35, 6, 555, 2]  # Example sorted list (largest to smallest)
# # # # # # # # # # list2 = ['A', 'B', 'C', 'D', 'E']  # Example corresponding list

# # # # # # # # # # # Create a mapping of indices from list1 to maintain order
# # # # # # # # # # order_mapping = sorted(range(len(list1)), key=lambda k: list1[k], reverse=True)
# # # # # # # # # # sorted_list2 = [list2[i] for i in order_mapping]

# # # # # # # # # # print(sorted_list2)  # Output: ['A', 'B', 'C', 'D', 'E']



# # # # # # # # # intermediate_iteration_history = {
# # # # # # # # #         "0": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     1.0,
# # # # # # # # #                     1.0
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.997710195859749,
# # # # # # # # #                 "reduction": 0.0
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.9,
# # # # # # # # #                     1.0
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.896449932243647,
# # # # # # # # #                 "reduction": -5.068816479285318
# # # # # # # # #             },
# # # # # # # # #             "2": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     1.0,
# # # # # # # # #                     0.9
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.8964756736455954,
# # # # # # # # #                 "reduction": -5.067527933929651
# # # # # # # # #             },
# # # # # # # # #             "3": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.8999999999999999,
# # # # # # # # #                     0.8999999999999999
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.7980625858681833,
# # # # # # # # #                 "reduction": -9.99382244758699
# # # # # # # # #             },
# # # # # # # # #             "4": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.8499999999999996,
# # # # # # # # #                     0.8499999999999996
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.701287776845883,
# # # # # # # # #                 "reduction": -14.838109132555912
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "1": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.7499999999999996,
# # # # # # # # #                     0.9499999999999996
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.6959467261380639,
# # # # # # # # #                 "reduction": -15.105467767401368
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.6249999999999991,
# # # # # # # # #                     0.9749999999999994
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.6046083771883861,
# # # # # # # # #                 "reduction": -19.677619881305393
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "2": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.5749999999999987,
# # # # # # # # #                     0.8249999999999991
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.3985462457480395,
# # # # # # # # #                 "reduction": -29.99253602216556
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.4124999999999981,
# # # # # # # # #                     0.7374999999999985
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.1509692099782718,
# # # # # # # # #                 "reduction": -42.38557662849929
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "3": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.18749999999999756,
# # # # # # # # #                     0.8624999999999983
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.0510927774826107,
# # # # # # # # #                 "reduction": -47.38512224340654
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.8687499999999977
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.9596931371890556,
# # # # # # # # #                 "reduction": -51.960342437155404
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "4": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.6312499999999968
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.7203439130622264,
# # # # # # # # #                 "reduction": -63.94152091954388
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.45937499999999565
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.5644730208855384,
# # # # # # # # #                 "reduction": -71.74399860122816
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "5": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.5906249999999948
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.694310271184568,
# # # # # # # # #                 "reduction": -65.24469502015234
# # # # # # # # #             }
# # # # # # # # #         }}



# # # # # # # # # intermediate_iteration_history =  {
# # # # # # # # #         "0": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     1.0,
# # # # # # # # #                     1.0
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.9985313995501006,
# # # # # # # # #                 "reduction": 0.0
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.9,
# # # # # # # # #                     1.0
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.896632998530773,
# # # # # # # # #                 "reduction": -5.098664001089321
# # # # # # # # #             },
# # # # # # # # #             "2": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     1.0,
# # # # # # # # #                     0.9
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.897285310297092,
# # # # # # # # #                 "reduction": -5.066024445540396
# # # # # # # # #             },
# # # # # # # # #             "3": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.8999999999999999,
# # # # # # # # #                     0.8999999999999999
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.804159406439324,
# # # # # # # # #                 "reduction": -9.725741269540855
# # # # # # # # #             },
# # # # # # # # #             "4": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.8499999999999996,
# # # # # # # # #                     0.8499999999999996
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.7054562969596347,
# # # # # # # # #                 "reduction": -14.664523292275591
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "1": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.7499999999999996,
# # # # # # # # #                     0.9499999999999996
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.7022102208125383,
# # # # # # # # #                 "reduction": -14.82694636693067
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.6249999999999991,
# # # # # # # # #                     0.9749999999999994
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.6072727554632333,
# # # # # # # # #                 "reduction": -19.577307825883818
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "2": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.5749999999999987,
# # # # # # # # #                     0.8249999999999991
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.4000752678131705,
# # # # # # # # #                 "reduction": -29.944795056592632
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.4124999999999981,
# # # # # # # # #                     0.7374999999999985
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.1531913872870787,
# # # # # # # # #                 "reduction": -42.29806008818879
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "3": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.18749999999999756,
# # # # # # # # #                     0.8624999999999983
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 1.0481969581559267,
# # # # # # # # #                 "reduction": -47.55163924910604
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.8687499999999977
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.9694997989400349,
# # # # # # # # #                 "reduction": -51.48938870020837
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "4": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.6312499999999968
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.7278493700426742,
# # # # # # # # #                 "reduction": -63.580788862935854
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.45937499999999565
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.5514024737286942,
# # # # # # # # #                 "reduction": -72.4096166888935
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "5": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.5906249999999948
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.6910682284623885,
# # # # # # # # #                 "reduction": -65.42119735431937
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "6": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.1812499999999928
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.28159445651601084,
# # # # # # # # #                 "reduction": -85.90993083324075
# # # # # # # # #             },
# # # # # # # # #             "1": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.09999999999999998
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.1958465404998021,
# # # # # # # # #                 "reduction": -90.20047718320112
# # # # # # # # #             }
# # # # # # # # #         },
# # # # # # # # #         "7": {
# # # # # # # # #             "0": {
# # # # # # # # #                 "design_vector": [
# # # # # # # # #                     0.09999999999999998,
# # # # # # # # #                     0.09999999999999998
# # # # # # # # #                 ],
# # # # # # # # #                 "objective_value": 0.19666542660911585,
# # # # # # # # #                 "reduction": -90.15950279022945
# # # # # # # # #             }
# # # # # # # # #         }}



# # # # # # # # # simplex = []
# # # # # # # # # objective_values = [9999]
# # # # # # # # # for iteration, steps in intermediate_iteration_history.items():
# # # # # # # # #     objective_values_step = []
# # # # # # # # #     design_vectors_step = []
# # # # # # # # #     for step, details in steps.items():
# # # # # # # # #         objective_value = details["reduction"]
# # # # # # # # #         design_vector = details["design_vector"]
# # # # # # # # #         n = len(design_vector)
# # # # # # # # #         if objective_value < objective_values[-1]:
# # # # # # # # #             objective_values_step.append(objective_value)
# # # # # # # # #             design_vectors_step.append(design_vector)
# # # # # # # # #     objective_values.extend(objective_values_step)
# # # # # # # # #     simplex.extend(design_vectors_step)

# # # # # # # # # if len(objective_values) > n + 1:
# # # # # # # # #     objective_values = objective_values[-(n+1):]
# # # # # # # # #     simplex = simplex[-(n+1):]


# # # # # # # # # print(objective_values, simplex)


# # # # # # # # # # list1 = [8, 35, 6, 555, 2]  # Example sorted list (largest to smallest)
# # # # # # # # # # list2 = ['A', 'B', 'C', 'D', 'E']  # Example corresponding list

# # # # # # # # # # # Create a mapping of indices from list1 to maintain order
# # # # # # # # # # order_mapping = sorted(range(len(list1)), key=lambda k: list1[k], reverse=True)
# # # # # # # # # # sorted_list2 = [list2[i] for i in order_mapping]

# # # # # # # # # # print(sorted_list2)  # Output: ['A', 'B', 'C', 'D', 'E']
# # # # # # # # # #



# # # # # # # # import numpy as np
# # # # # # # # from scipy.optimize import minimize
# # # # # # # # import pickle

# # # # # # # # # Define the objective function
# # # # # # # # def objective_function(x):
# # # # # # # #     return x[0]**2 + x[1]**2

# # # # # # # # # Create a list to store the simplex at each iteration
# # # # # # # # simplex_history = [[2.0, 2.0]]

# # # # # # # # # Define the callback function to save the simplex
# # # # # # # # def callback(xk, convergence=None):
# # # # # # # #     print("here")
# # # # # # # #     simplex_history.append(xk.copy())

# # # # # # # # print(simplex_history)

# # # # # # # # # Initial guess
# # # # # # # # initial_guess = np.array([2.0, 2.0])

# # # # # # # # # Perform the initial optimization
# # # # # # # # result = minimize(objective_function, initial_guess, method='Nelder-Mead', callback=callback)

# # # # # # # # # Save the result and simplex to a file
# # # # # # # # with open('optimization_state.pkl', 'wb') as f:
# # # # # # # #     pickle.dump({'result': result, 'simplex': simplex_history[-1]}, f)

# # # # # # # # # Load the result and simplex from the file
# # # # # # # # with open('optimization_state.pkl', 'rb') as f:
# # # # # # # #     saved_data = pickle.load(f)
# # # # # # # #     saved_result = saved_data['result']
# # # # # # # #     saved_simplex = saved_data['simplex']

# # # # # # # # # Use the result of the first optimization as the starting point for the second optimization
# # # # # # # # new_initial_guess = saved_result.x

# # # # # # # # # Continue the optimization with the saved simplex
# # # # # # # # result2 = minimize(objective_function, new_initial_guess, method='Nelder-Mead',
# # # # # # # #                    options={'initial_simplex': saved_simplex})

# # # # # # # # print("Final optimization result:", result2)




# # # # # # # import numpy as np

# # # # # # # def apply_magnitude_error(vector, magnitude_error):
# # # # # # #     """
# # # # # # #     Apply a magnitude error to the vector.
# # # # # # #     :param vector: numpy array, the original delta-v vector
# # # # # # #     :param magnitude_error: float, the relative error to apply to the magnitude (e.g., 0.05 for 5% error)
# # # # # # #     :return: numpy array, the vector with magnitude error applied
# # # # # # #     """
# # # # # # #     current_magnitude = np.linalg.norm(vector)
# # # # # # #     error = magnitude_error * current_magnitude
# # # # # # #     new_magnitude = current_magnitude + error
# # # # # # #     return vector * (new_magnitude / current_magnitude)

# # # # # # # def apply_direction_error(vector, direction_error_degrees):
# # # # # # #     """
# # # # # # #     Apply a direction error to the vector.
# # # # # # #     :param vector: numpy array, the original delta-v vector
# # # # # # #     :param direction_error_degrees: float, the error in degrees to apply to the direction
# # # # # # #     :return: numpy array, the vector with direction error applied
# # # # # # #     """
# # # # # # #     direction_error_radians = np.radians(direction_error_degrees)
# # # # # # #     rotation_axis = np.random.randn(3)
# # # # # # #     rotation_axis /= np.linalg.norm(rotation_axis)
# # # # # # #     rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, direction_error_radians)
# # # # # # #     return np.dot(rotation_matrix, vector)

# # # # # # # def rotation_matrix_from_axis_angle(axis, angle):
# # # # # # #     """
# # # # # # #     Compute the rotation matrix given an axis and an angle.
# # # # # # #     :param axis: numpy array, the axis to rotate around
# # # # # # #     :param angle: float, the angle in radians
# # # # # # #     :return: numpy array, the rotation matrix
# # # # # # #     """
# # # # # # #     axis = axis / np.linalg.norm(axis)
# # # # # # #     a = np.cos(angle / 2.0)
# # # # # # #     b, c, d = -axis * np.sin(angle / 2.0)
# # # # # # #     aa, bb, cc, dd = a*a, b*b, c*c, d*d
# # # # # # #     bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
# # # # # # #     return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
# # # # # # #                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
# # # # # # #                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# # # # # # # np.random.seed(0)
# # # # # # # # Example usage
# # # # # # # delta_v = np.array([1, 1, 1])
# # # # # # # magnitude_error = 0.05  # 5% error
# # # # # # # direction_error_degrees = 2.0  # 2 degrees

# # # # # # # # Apply magnitude error
# # # # # # # delta_v_with_magnitude_error = apply_magnitude_error(delta_v, magnitude_error)
# # # # # # # print("Delta-v with magnitude error:", delta_v_with_magnitude_error)

# # # # # # # # Apply direction error
# # # # # # # delta_v_with_direction_error = apply_direction_error(delta_v_with_magnitude_error, direction_error_degrees)
# # # # # # # print("Delta-v with direction error:", delta_v_with_direction_error)

# # # # # # # print(np.linalg.norm(delta_v_with_magnitude_error),
# # # # # # #       np.linalg.norm(delta_v_with_direction_error))



# # # # # # import numpy as np

# # # # # # def get_delta_v_noise(seed, delta_v0, magnitude_error, direction_error):

# # # # # #     rng = np.random.default_rng(seed=seed)
# # # # # #     new_delta_v = delta_v0.copy()
# # # # # #     old_magnitude = np.linalg.norm(delta_v0)

# # # # # #     # Apply magnitude errors relative to each component
# # # # # #     for i in range(3):
# # # # # #         magnitude_errors = rng.normal(loc=0, scale=(magnitude_error) * np.abs(delta_v0[i]))
# # # # # #         new_delta_v[i] += magnitude_errors

# # # # # #     new_magnitude = np.linalg.norm(new_delta_v)

# # # # # #     direction_vector = delta_v0.copy()
# # # # # #     for i in range(3):
# # # # # #         direction_errors = rng.normal(loc=0, scale=(direction_error) * np.abs(delta_v0[i]))
# # # # # #         direction_vector[i] += direction_errors

# # # # # #     direction_vector = direction_vector / np.linalg.norm(direction_vector)

# # # # # #     new_delta_v = new_magnitude * direction_vector

# # # # # #     print(delta_v0, new_delta_v, np.linalg.norm(new_delta_v))

# # # # # #     return new_delta_v

# # # # # # # Example usage
# # # # # # delta_v0 = np.array([-1.47356906e-03, -3.98291359e-04,  2.06277445e-07])
# # # # # # magnitude_error = 0.00
# # # # # # direction_error = 0.02

# # # # # # import matplotlib.pyplot as plt

# # # # # # a = []
# # # # # # for seed in range(1000):
# # # # # #     new_delta_v = get_delta_v_noise(seed, delta_v0, magnitude_error, direction_error)
# # # # # #     # print("New delta-v vector:", new_delta_v, np.linalg.norm(new_delta_v), new_delta_v/np.linalg.norm(new_delta_v))

# # # # # #     a.append(new_delta_v[0])

# # # # # # plt.hist(a)
# # # # # # plt.show()




# # # # # import numpy as np
# # # # # import matplotlib.pyplot as plt
# # # # # from scipy.interpolate import interp1d

# # # # # # Sample data points
# # # # # x = np.linspace(0, 10, 10)
# # # # # y = np.sin(x)

# # # # # # Points for interpolation
# # # # # x_interp = np.linspace(0.424, 2.452, 2)

# # # # # # Linear interpolation
# # # # # linear_interp = interp1d(x, y, kind='linear')
# # # # # y_linear_interp = linear_interp(x_interp)

# # # # # # Cubic interpolation
# # # # # cubic_interp = interp1d(x, y, kind='cubic')
# # # # # y_cubic_interp = cubic_interp(x_interp)

# # # # # # Plotting the results
# # # # # plt.figure(figsize=(10, 6))
# # # # # plt.plot(x, y, 'o', label='Data points')
# # # # # plt.plot(x_interp, y_linear_interp, '-', label='Linear interpolation')
# # # # # plt.plot(x_interp, y_cubic_interp, '--', label='Cubic interpolation')
# # # # # plt.legend()
# # # # # plt.xlabel('x')
# # # # # plt.ylabel('y')
# # # # # plt.title('Linear vs Cubic Interpolation')
# # # # # plt.show()


# # # # import matplotlib.pyplot as plt
# # # # import numpy as np

# # # # # Sample data
# # # # groups = ['Group 1', 'Group 2', 'Group 3']
# # # # subgroups = ['Subgroup 1', 'Subgroup 2', 'Subgroup 3']
# # # # values = np.random.rand(3, 3)

# # # # fig, ax = plt.subplots(figsize=(12, 8))

# # # # bar_width = 0.2
# # # # index = np.arange(len(groups))

# # # # for i in range(len(subgroups)):
# # # #     plt.bar(index + i * bar_width, values[:, i], bar_width, label=subgroups[i])

# # # # # Adding x labels
# # # # plt.xlabel('Groups')
# # # # plt.xticks(index + bar_width, groups)

# # # # # Rotate subgroup labels and position them lower
# # # # plt.xticks(rotation=45)
# # # # ax.set_xticks(index + bar_width / 2)
# # # # ax.set_xticklabels(groups)

# # # # # Add subgroup labels
# # # # ax.set_xticks(index + bar_width / 2, minor=True)
# # # # ax.set_xticklabels(subgroups, minor=True)

# # # # # Fine-tune the position of minor tick labels (subgroups)
# # # # for tick in ax.get_xticklabels(minor=True):
# # # #     tick.set_transform(tick.get_transform() + plt.gca().transData.inverted().transform((0, -20)))  # Adjust as needed

# # # # plt.legend()
# # # # plt.show()



# # # import numpy as np

# # # # Initialize RNG
# # # rng = np.random.default_rng()

# # # # Variables for arc_interval, arc_duration, and threshold
# # # arc_interval_vars = [5, 2]  # Example values: middle = 5, bounds = 2
# # # arc_duration_vars = [10, 3]  # Example values: middle = 10, bounds = 3
# # # threshold_vars = [20, 0]  # Example values: middle = 20, bounds = 4

# # # # Generating uniform distribution data using the initialized RNG
# # # arc_interval = rng.uniform(
# # #     low=arc_interval_vars[0] - arc_interval_vars[1],
# # #     high=arc_interval_vars[0] + arc_interval_vars[1],
# # #     size=100
# # # )

# # # arc_duration = rng.uniform(
# # #     low=arc_duration_vars[0] - arc_duration_vars[1],
# # #     high=arc_duration_vars[0] + arc_duration_vars[1],
# # #     size=100
# # # )

# # # threshold = rng.uniform(
# # #     low=threshold_vars[0] - threshold_vars[1],
# # #     high=threshold_vars[0] + threshold_vars[1],
# # #     size=100
# # # )

# # # # Output the generated data
# # # print("arc_interval:", arc_interval)
# # # print("arc_duration:", arc_duration)
# # # print("threshold:", threshold)




# # # import multiprocessing as mp

# # # # Example function placeholder for helper_functions.get_constant_arc_observation_windows
# # # def get_constant_arc_observation_windows(duration, arc_interval, arc_duration, mission_start_epoch):
# # #     # Placeholder implementation
# # #     return f"ObsWindows(duration={duration}, arc_interval={arc_interval}, arc_duration={arc_duration}, start_epoch={mission_start_epoch})"

# # # # Parameters
# # # params = [0.1, 0.2, 0.5, 1.0, 2.0]
# # # params2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# # # num_runs = 3
# # # duration = 100  # Example duration
# # # mission_start_epoch = 0  # Example mission start epoch

# # # # Generate the observation windows settings dynamically
# # # observation_windows_settings = {
# # #     f"{param2} day": [
# # #         (get_constant_arc_observation_windows(duration, arc_interval=param2, arc_duration=param, mission_start_epoch=mission_start_epoch), num_runs, str(param))
# # #         for param in params
# # #     ]
# # #     for param2 in params2
# # # }

# # # # Example method to run MC simulations with the given windows
# # # def run_mc_simulation(windows, num_runs):
# # #     # Placeholder implementation of the Monte Carlo simulation
# # #     results = []
# # #     for _ in range(num_runs):
# # #         # Simulate the MC run with the given windows
# # #         result = f"Simulated result for {windows}"
# # #         results.append(result)
# # #     return results

# # # # Running the MC simulations in parallel using multiprocessing.Pool
# # # def parallel_mc_runs(observation_windows_settings):
# # #     tasks = []
# # #     indices = []

# # #     for key, windows_list in observation_windows_settings.items():
# # #         for i, (windows, num_runs, param_str) in enumerate(windows_list):
# # #             tasks.append((windows, num_runs))
# # #             indices.append((key, i))  # Track the original position

# # #     with mp.Pool(processes=mp.cpu_count()) as pool:
# # #         results = pool.starmap(run_mc_simulation, tasks)

# # #     # Map results back to the original structure
# # #     ordered_results = {key: [None] * len(windows_list) for key, windows_list in observation_windows_settings.items()}
# # #     for (key, i), result in zip(indices, results):
# # #         ordered_results[key][i] = result

# # #     return ordered_results

# # # # Run the simulations
# # # mc_results = parallel_mc_runs(observation_windows_settings)

# # # # Print the results
# # # for key, result_list in mc_results.items():
# # #     print(f"{key}:")
# # #     for result in result_list:
# # #         print(f"  {result}")


# # import shelve

# # # Define the classes and create object instances
# # class Person:
# #     def __init__(self, name, age, city):
# #         self.name = name
# #         self.age = age
# #         self.city = city

# #     def __repr__(self):
# #         return f'Person({self.name}, {self.age}, {self.city})'

# # # Create object instances
# # obj_instance1 = Person('Alice', 30, 'New York')
# # obj_instance2 = Person('Bob', 25, 'Los Angeles')

# # # Create the nested dictionary
# # data = {"a": {"a1": obj_instance1, "a2": obj_instance2}}

# # # Save the dictionary to a shelve file
# # with shelve.open('data_shelve') as db:
# #     db['data'] = data

# # # Load the dictionary from the shelve file
# # with shelve.open('data_shelve') as db:
# #     loaded_data = db['data']

# # print(loaded_data)
# # print(loaded_data['a']['a1'].age)
# # print(loaded_data['a']['a2'])


# import numpy as np
# import os
# import sys
# import numpy as np
# import copy
# import matplotlib.pyplot as plt
# from memory_profiler import profile

# # Define path to import src files
# file_directory = os.path.realpath(__file__)
# for _ in range(4):
#     file_directory = os.path.dirname(file_directory)
#     sys.path.append(file_directory)

# from src import NavigationSimulator

# # Define the Rastrigin function for 6 dimensions
# def rastrigin(X):
#     # return 10 * len(X) + sum([(x ** 2 - 10 * np.cos(2 * np.pi * x)) for x in X])
#     return sum([x for x in X])

# # Parameters
# dim = 2 # Number of dimensions
# num_particles = 10
# num_iterations = 100
# lb = 0
# ub = 5.12
# w = 0.5  # Inertia weight
# c1 = 1.5  # Cognitive (particle) weight
# c2 = 1.5  # Social (swarm) weight

# # Initialize particle positions and velocities
# particles = np.random.uniform(low=lb, high=ub, size=(num_particles, dim))
# velocities = np.random.uniform(low=-1, high=1, size=(num_particles, dim))
# personal_best_positions = np.copy(particles)
# personal_best_scores = np.array([rastrigin(p) for p in particles])
# global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
# global_best_score = np.min(personal_best_scores)

# # History lists
# global_best_history = []
# particles_history = []

# # PSO algorithm
# for t in range(num_iterations):
#     for i in range(num_particles):
#         r1, r2 = np.random.rand(dim), np.random.rand(dim)
#         print(r1, r2)
#         velocities[i] = (w * velocities[i] +
#                          c1 * r1 * (personal_best_positions[i] - particles[i]) +
#                          c2 * r2 * (global_best_position - particles[i]))
#         particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
#         score = rastrigin(particles[i])

#         if score < personal_best_scores[i]:
#             personal_best_scores[i] = score
#             personal_best_positions[i] = particles[i]

#         if score < global_best_score:
#             global_best_score = score
#             global_best_position = particles[i]

#     global_best_history.append(global_best_score)
#     particles_history.append(np.copy(particles))

# # Plot the optimization history
# fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# # Plot global best fitness over iterations
# ax[0].plot(global_best_history, marker='o')
# ax[0].set_title('Global Best Fitness over Iterations')
# ax[0].set_xlabel('Iteration')
# ax[0].set_ylabel('Fitness')

# # Plot particle positions over iterations
# particles_history = np.array(particles_history)
# for dim in range(particles_history.shape[2]):
#     ax[1].plot(particles_history[:, :, dim], alpha=0.6)
# ax[1].set_title('Particle Positions over Iterations (One Dimension per Line)')
# ax[1].set_xlabel('Iteration')
# ax[1].set_ylabel('Position')

# plt.tight_layout()
# plt.show()

# print(f"Optimal solution: {global_best_position}")
# print(f"Optimal value: {global_best_score}")


# # Plot particle positions over iterations (2D convergence plot)
# for i in range(num_particles):
#     ax.plot(particles_history[:t, i, 0], particles_history[:t, i, 1], alpha=0.3)
# ax.scatter(global_best_position[0], global_best_position[1], color='red', marker='o', label='Global Best')
# ax.plot(global_best_position[0], global_best_position[1], color='red', marker='x', markersize=15)  # Mark final position
# ax.set_title('Particle Convergence towards Optimal Solution (2D)')
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.legend()

# plt.tight_layout()
# plt.show()



#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import random

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0):

        self.num_dimensions = len(x0)

        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,self.num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,self.num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,self.num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]

class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):

        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                print(swarm[j].position_i)
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # print final results
        # print 'FINAL:'
        print(pos_best_g)
        print(err_best_g)

if __name__ == "__PSO__":
    main()

#--- RUN ----------------------------------------------------------------------+

initial = [5,5]
bounds = [(4,10),(-10,10)]
PSO(func1,initial,bounds,num_particles=10,maxiter=40)

#--- END ----------------------------------------------------------------------+