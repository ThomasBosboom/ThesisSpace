# # # # # # # # # # # # # # # # # # # # # # import tracemalloc

# # # # # # # # # # # # # # # # # # # # # # class DataGenerator:
# # # # # # # # # # # # # # # # # # # # # #     def generate_large_list(self, size):
# # # # # # # # # # # # # # # # # # # # # #         return [i ** 2 for i in range(size)]

# # # # # # # # # # # # # # # # # # # # # #     def generate_large_dict(self, size):
# # # # # # # # # # # # # # # # # # # # # #         return {i: i ** 2 for i in range(size)}

# # # # # # # # # # # # # # # # # # # # # # class DataProcessor:
# # # # # # # # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # # # # # # # #         self.data = None

# # # # # # # # # # # # # # # # # # # # # #     def process_data(self, data):
# # # # # # # # # # # # # # # # # # # # # #         self.data = data
# # # # # # # # # # # # # # # # # # # # # #         # Simulate some processing
# # # # # # # # # # # # # # # # # # # # # #         processed_data = [x * 2 for x in self.data]
# # # # # # # # # # # # # # # # # # # # # #         return processed_data

# # # # # # # # # # # # # # # # # # # # # # def main():
# # # # # # # # # # # # # # # # # # # # # #     # Start tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # #     tracemalloc.start()

# # # # # # # # # # # # # # # # # # # # # #     generator = DataGenerator()
# # # # # # # # # # # # # # # # # # # # # #     processor = DataProcessor()

# # # # # # # # # # # # # # # # # # # # # #     # Generate large datasets
# # # # # # # # # # # # # # # # # # # # # #     large_list = generator.generate_large_list(10000)
# # # # # # # # # # # # # # # # # # # # # #     large_dict = generator.generate_large_dict(10000)

# # # # # # # # # # # # # # # # # # # # # #     # Process the large list
# # # # # # # # # # # # # # # # # # # # # #     processed_list = processor.process_data(large_list)

# # # # # # # # # # # # # # # # # # # # # #     # Process the large dict (just converting values to list for simplicity)
# # # # # # # # # # # # # # # # # # # # # #     processed_dict_values = processor.process_data(list(large_dict.values()))

# # # # # # # # # # # # # # # # # # # # # #     # Take a snapshot
# # # # # # # # # # # # # # # # # # # # # #     snapshot = tracemalloc.take_snapshot()

# # # # # # # # # # # # # # # # # # # # # #     # Display top memory allocations
# # # # # # # # # # # # # # # # # # # # # #     # top_stats = snapshot.statistics('lineno')
# # # # # # # # # # # # # # # # # # # # # #     top_stats = snapshot.statistics('traceback')

# # # # # # # # # # # # # # # # # # # # # #     print("[ Top 10 memory allocations ]")
# # # # # # # # # # # # # # # # # # # # # #     for stat in top_stats[:10]:
# # # # # # # # # # # # # # # # # # # # # #         print(stat)

# # # # # # # # # # # # # # # # # # # # # #     # Stop tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # #     tracemalloc.stop()

# # # # # # # # # # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # # # # # # # # # #     main()



# # # # # # # # # # # # # # # # # # # # # import tracemalloc
# # # # # # # # # # # # # # # # # # # # # import time

# # # # # # # # # # # # # # # # # # # # # # Start tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # tracemalloc.start()

# # # # # # # # # # # # # # # # # # # # # # Simulate a function that gradually increases memory usage
# # # # # # # # # # # # # # # # # # # # # def simulate_memory_usage():
# # # # # # # # # # # # # # # # # # # # #     data = []
# # # # # # # # # # # # # # # # # # # # #     for i in range(10):
# # # # # # # # # # # # # # # # # # # # #         data.append([i] * 1000000)  # Increase memory usage
# # # # # # # # # # # # # # # # # # # # #         time.sleep(1)
# # # # # # # # # # # # # # # # # # # # #         print(f"Snapshot {i + 1}")
# # # # # # # # # # # # # # # # # # # # #         snapshot = tracemalloc.take_snapshot()
# # # # # # # # # # # # # # # # # # # # #         top_stats = snapshot.statistics('lineno')
# # # # # # # # # # # # # # # # # # # # #         for stat in top_stats[:10]:
# # # # # # # # # # # # # # # # # # # # #             print(stat)
# # # # # # # # # # # # # # # # # # # # #         total_memory = sum(stat.size for stat in top_stats)
# # # # # # # # # # # # # # # # # # # # #         print(f"Total memory used after iteration {i + 1}: {total_memory / (1024 ** 2):.2f} MB")
# # # # # # # # # # # # # # # # # # # # #     return data

# # # # # # # # # # # # # # # # # # # # # simulate_memory_usage()

# # # # # # # # # # # # # # # # # # # # # # Optional: Stop tracing memory allocations
# # # # # # # # # # # # # # # # # # # # # tracemalloc.stop()

# # # # # # # # # # # # # # # # # # # # from memory_profiler import profile

# # # # # # # # # # # # # # # # # # # # class MyClass:
# # # # # # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # # # # # #         self.attribute1 = "initial_value1"
# # # # # # # # # # # # # # # # # # # #         self.attribute2 = "initial_value2"
# # # # # # # # # # # # # # # # # # # #         self.attribute3 = "initial_value3"
# # # # # # # # # # # # # # # # # # # #         self.initial_state = vars(self).copy()  # Store initial state

# # # # # # # # # # # # # # # # # # # #     @profile
# # # # # # # # # # # # # # # # # # # #     def generate_additional_attributes(self):
# # # # # # # # # # # # # # # # # # # #         # Method that generates additional attributes
# # # # # # # # # # # # # # # # # # # #         self.additional_attribute1 = "new_value1"
# # # # # # # # # # # # # # # # # # # #         self.additional_attribute2 = "new_value2"

# # # # # # # # # # # # # # # # # # # #     @profile
# # # # # # # # # # # # # # # # # # # #     def reset_to_init_state(self):

# # # # # # # # # # # # # # # # # # # #         current_state = vars(self)
# # # # # # # # # # # # # # # # # # # #         for attr_name in list(current_state.keys()):
# # # # # # # # # # # # # # # # # # # #             if attr_name is not str("initial_state"):
# # # # # # # # # # # # # # # # # # # #                 if attr_name not in self.initial_state.keys():
# # # # # # # # # # # # # # # # # # # #                     delattr(self, attr_name)


# # # # # # # # # # # # # # # # # # # # # Example usage:
# # # # # # # # # # # # # # # # # # # # obj = MyClass()

# # # # # # # # # # # # # # # # # # # # print(vars(obj))
# # # # # # # # # # # # # # # # # # # # # Call the method to generate additional attributes
# # # # # # # # # # # # # # # # # # # # obj.generate_additional_attributes()

# # # # # # # # # # # # # # # # # # # # # Print the attributes after generating additional attributes
# # # # # # # # # # # # # # # # # # # # print(vars(obj))

# # # # # # # # # # # # # # # # # # # # # Call the method to reset attributes to their initial state
# # # # # # # # # # # # # # # # # # # # obj.reset_to_init_state()

# # # # # # # # # # # # # # # # # # # # # Print the attributes after resetting
# # # # # # # # # # # # # # # # # # # # print(vars(obj))



# # # # # # # # # # # # # # # # # # # #             self.full_estimation_error_dict = dict()
# # # # # # # # # # # # # # # # # # # #             self.full_reference_state_deviation_dict = dict()
# # # # # # # # # # # # # # # # # # # #             self.full_propagated_covariance_dict = dict()
# # # # # # # # # # # # # # # # # # # #             self.full_propagated_formal_errors_dict = dict()
# # # # # # # # # # # # # # # # # # # #             self.full_state_history_reference_dict = dict()
# # # # # # # # # # # # # # # # # # # #             self.full_state_history_truth_dict = dict()
# # # # # # # # # # # # # # # # # # # #             self.full_state_history_estimated_dict = dict()
# # # # # # # # # # # # # # # # # # # #             self.full_state_history_final_dict = dict()
# # # # # # # # # # # # # # # # # # # #             self.delta_v_dict = dict()
# # # # # # # # # # # # # # # # # # # #             self.full_dependent_variables_history_estimated = dict()
# # # # # # # # # # # # # # # # # # # #             self.full_state_transition_matrix_history_estimated = dict()
# # # # # # # # # # # # # # # # # # # #             self.estimation_arc_results_dict = dict()
# # # # # # # # # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # # # # # # # # Example input dictionary
# # # # # # # # # # # # # # # # # # # data = {
# # # # # # # # # # # # # # # # # # #     '0.0': {
# # # # # # # # # # # # # # # # # # #         60391: {0: 0.0012819549366386333, 1: 0.001185896799597702},
# # # # # # # # # # # # # # # # # # #         60395: {0: 0.005411142569208645, 1: 0.006770971960928585},
# # # # # # # # # # # # # # # # # # #         60399: {0: 0.003109677105139794, 1: 0.002999612884133889},
# # # # # # # # # # # # # # # # # # #         60403: {0: 0.004734675217803918, 1: 0.004341626884436936}
# # # # # # # # # # # # # # # # # # #     },
# # # # # # # # # # # # # # # # # # #     '0.03': {
# # # # # # # # # # # # # # # # # # #         60399: {0: 0.0672568474862809, 1: 0.06727010238539159}
# # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # # }

# # # # # # # # # # # # # # # # # # # def calculate_stats(data, evaluation_threshold=14):
# # # # # # # # # # # # # # # # # # #     result_dict = {}

# # # # # # # # # # # # # # # # # # #     # Iterate through each test case
# # # # # # # # # # # # # # # # # # #     for case_type, epochs in data.items():


# # # # # # # # # # # # # # # # # # #         # Iterate through each epoch
# # # # # # # # # # # # # # # # # # #         epoch_stats = {}
# # # # # # # # # # # # # # # # # # #         combined_per_run = {}
# # # # # # # # # # # # # # # # # # #         combined_per_run_with_threshold = {}
# # # # # # # # # # # # # # # # # # #         for epoch, runs in epochs.items():
# # # # # # # # # # # # # # # # # # #             keys = list(runs.keys())
# # # # # # # # # # # # # # # # # # #             values = list(runs.values())
# # # # # # # # # # # # # # # # # # #             epoch_stats[epoch] = {'mean': np.mean(values), 'std': np.std(values)}

# # # # # # # # # # # # # # # # # # #             for key in keys:
# # # # # # # # # # # # # # # # # # #                 if key not in combined_per_run:
# # # # # # # # # # # # # # # # # # #                     combined_per_run[key] = []
# # # # # # # # # # # # # # # # # # #                     combined_per_run_with_threshold[key] = []

# # # # # # # # # # # # # # # # # # #                 combined_per_run[key].append(runs[key])
# # # # # # # # # # # # # # # # # # #                 if epoch >= 60390 + evaluation_threshold:
# # # # # # # # # # # # # # # # # # #                     combined_per_run_with_threshold[key].append(runs[key])


# # # # # # # # # # # # # # # # # # #         total = []
# # # # # # # # # # # # # # # # # # #         total_with_threshold = []
# # # # # # # # # # # # # # # # # # #         for run, combined in combined_per_run.items():
# # # # # # # # # # # # # # # # # # #             total.append(np.sum(combined))
# # # # # # # # # # # # # # # # # # #         for run, combined in combined_per_run_with_threshold.items():
# # # # # # # # # # # # # # # # # # #             total_with_threshold.append(np.sum(combined))

# # # # # # # # # # # # # # # # # # #         total_stats = {'mean': np.mean(total), 'std': np.std(total)}
# # # # # # # # # # # # # # # # # # #         total_stats_with_threshold = {'mean': np.mean(total_with_threshold), 'std': np.std(total_with_threshold)}

# # # # # # # # # # # # # # # # # # #         # Store statistics in the result dictionary
# # # # # # # # # # # # # # # # # # #         result_dict[case_type] = {'epoch_stats': epoch_stats, 'total_stats': total_stats, 'total_stats_with_threshold': total_stats_with_threshold}

# # # # # # # # # # # # # # # # # # #     return result_dict

# # # # # # # # # # # # # # # # # # import json

# # # # # # # # # # # # # # # # # # def create_design_vector_table(data, caption='', label='', file_name='design_vector_table.tex', decimals=4):
# # # # # # # # # # # # # # # # # #     table_str = r'% Please add the following required packages to your document preamble:' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'% \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'% Beamer presentation requires \usepackage{colortbl} instead of \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'\begin{table}[]' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'\centering' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'\begin{tabular}{lll}' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'\textbf{}      & \cellcolor[HTML]{EFEFEF}\textbf{Design vector} & \textbf{}          \\' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'\textbf{State} & \textbf{Initial}                               & \textbf{Final} \\' + '\n'

# # # # # # # # # # # # # # # # # #     states = ['x', 'y', 'z', 'v_{x}', 'v_{y}', 'v_{z}']
# # # # # # # # # # # # # # # # # #     initial_values = data.get('initial', [])
# # # # # # # # # # # # # # # # # #     final_values = data.get('final', [])

# # # # # # # # # # # # # # # # # #     for state, initial, final in zip(states, initial_values, final_values):
# # # # # # # # # # # # # # # # # #         table_str += f"${state}$ & {initial:.{decimals}f} & {final:.{decimals}f} \\" + '\n'

# # # # # # # # # # # # # # # # # #     table_str += r'\end{tabular}' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'\caption{' + caption + '}' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'\label{' + label + '}' + '\n'
# # # # # # # # # # # # # # # # # #     table_str += r'\end{table}'

# # # # # # # # # # # # # # # # # #     if file_name:
# # # # # # # # # # # # # # # # # #         with open(file_name, 'w') as f:
# # # # # # # # # # # # # # # # # #             f.write(table_str)

# # # # # # # # # # # # # # # # # #     return table_str

# # # # # # # # # # # # # # # # # # # Example usage
# # # # # # # # # # # # # # # # # # data = {
# # # # # # # # # # # # # # # # # #     "initial": [1, 1, 1, 1, 1, 1],
# # # # # # # # # # # # # # # # # #     "final": [2, 5, 1, 5, 2, 6]
# # # # # # # # # # # # # # # # # # }

# # # # # # # # # # # # # # # # # # # Generate the Overleaf table with custom caption, label, and decimals
# # # # # # # # # # # # # # # # # # file_name = "design_vector_table.tex"
# # # # # # # # # # # # # # # # # # overleaf_table = create_design_vector_table(data, caption="Design Vector comparison before and after optimization", label="tab:DesignVectorOptimization", file_name=file_name, decimals=4)

# # # # # # # # # # # # # # # # # # # Print the Overleaf table
# # # # # # # # # # # # # # # # # # print(overleaf_table)



# # # # # # # # # # # # # # # # # def create_design_vector_table(data, caption='', label='', file_name='design_vector_table.tex', decimals=4):
# # # # # # # # # # # # # # # # #     table_str = r'% Please add the following required packages to your document preamble:' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'% \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'% Beamer presentation requires \usepackage{colortbl} instead of \usepackage[table,xcdraw]{xcolor}' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'\begin{table}[]' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'\centering' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'\begin{tabular}{llll}' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'\textbf{}      & \cellcolor[HTML]{EFEFEF}\textbf{Design vector} & \textbf{}          & \textbf{}             \\' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'\textbf{State} & \textbf{Initial}                               & \textbf{Optimized} & \textbf{\% Difference} \\' + '\n'

# # # # # # # # # # # # # # # # #     states = ['x', 'y', 'z', 'v_{x}', 'v_{y}', 'v_{z}']
# # # # # # # # # # # # # # # # #     initial_values = data.get('initial', {}).get('values', [])
# # # # # # # # # # # # # # # # #     final_values = data.get('final', {}).get('values', [])

# # # # # # # # # # # # # # # # #     for state, initial, final in zip(states, initial_values, final_values):
# # # # # # # # # # # # # # # # #         if initial != 0:
# # # # # # # # # # # # # # # # #             percentage_diff = ((final - initial) / initial) * 100
# # # # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # # # #             percentage_diff = 0  # Handle division by zero case
# # # # # # # # # # # # # # # # #         table_str += f"${state}$ & {initial:.{decimals}f} & {final:.{decimals}f} & {round(percentage_diff, 2)}\%" + r' \\ ' + '\n'

# # # # # # # # # # # # # # # # #     initial_cost = data.get('initial', {}).get('cost', 0)
# # # # # # # # # # # # # # # # #     final_cost = data.get('final', {}).get('cost', 0)

# # # # # # # # # # # # # # # # #     if initial_cost != 0:
# # # # # # # # # # # # # # # # #         cost_percentage_diff = ((final_cost - initial_cost) / initial_cost) * 100
# # # # # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # # # # #         cost_percentage_diff = 0  # Handle division by zero case

# # # # # # # # # # # # # # # # #     table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
# # # # # # # # # # # # # # # # #     table_str += f"\\textbf{{Cost}}  & {initial_cost:.{decimals}f} & {final_cost:.{decimals}f} & {round(cost_percentage_diff, 2)}\%" + r' \\ ' + '\n'

# # # # # # # # # # # # # # # # #     table_str += r'\end{tabular}' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'\caption{' + caption + '}' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'\label{' + label + '}' + '\n'
# # # # # # # # # # # # # # # # #     table_str += r'\end{table}'

# # # # # # # # # # # # # # # # #     if file_name:
# # # # # # # # # # # # # # # # #         with open(file_name, 'w') as f:
# # # # # # # # # # # # # # # # #             f.write(table_str)

# # # # # # # # # # # # # # # # #     return table_str

# # # # # # # # # # # # # # # # # # Example usage
# # # # # # # # # # # # # # # # # data = {
# # # # # # # # # # # # # # # # #     "initial": {
# # # # # # # # # # # # # # # # #         "values": [1, 1, 1, 1, 1, 1],
# # # # # # # # # # # # # # # # #         "cost": 10
# # # # # # # # # # # # # # # # #     },
# # # # # # # # # # # # # # # # #     "final": {
# # # # # # # # # # # # # # # # #         "values": [2, 5, 1, 5, 2, 6],
# # # # # # # # # # # # # # # # #         "cost": 8
# # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # }

# # # # # # # # # # # # # # # # # # Generate the Overleaf table with custom caption, label, and decimals
# # # # # # # # # # # # # # # # # file_name = "design_vector_table.tex"
# # # # # # # # # # # # # # # # # overleaf_table = create_design_vector_table(data, caption="Design vector comparison before and after optimization", label="tab:DesignVectorOptimization", file_name=file_name, decimals=4)

# # # # # # # # # # # # # # # # # # Print the Overleaf table
# # # # # # # # # # # # # # # # # print(overleaf_table)

# # # # # # # # # # # # # # # # import copy

# # # # # # # # # # # # # # # # class NavigationSimulatorBase:
# # # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # # #         # Initialize some default attributes for the base class
# # # # # # # # # # # # # # # #         self.default_attr1 = "default_value1"
# # # # # # # # # # # # # # # #         self.default_attr2 = "default_value2"

# # # # # # # # # # # # # # # # class NavigationSimulator(NavigationSimulatorBase):

# # # # # # # # # # # # # # # #     def __init__(self, **kwargs):
# # # # # # # # # # # # # # # #         super().__init__()

# # # # # # # # # # # # # # # #         self.attr1 = 1
# # # # # # # # # # # # # # # #         self.attr2 = 2
# # # # # # # # # # # # # # # #         for key, value in kwargs.items():
# # # # # # # # # # # # # # # #             if hasattr(self, key):
# # # # # # # # # # # # # # # #                 setattr(self, key, value)


# # # # # # # # # # # # # # # #     def modify_attributes(self):

# # # # # # # # # # # # # # # #         # Example modifications
# # # # # # # # # # # # # # # #         if hasattr(self, 'attr1'):
# # # # # # # # # # # # # # # #             self.attr1 += 1
# # # # # # # # # # # # # # # #         if hasattr(self, 'attr2'):
# # # # # # # # # # # # # # # #             self.attr2 += 1

# # # # # # # # # # # # # # # #             self.new_attribute = self.attr1 + self.attr2

# # # # # # # # # # # # # # # #         return self.new_attribute


# # # # # # # # # # # # # # # #     def reset(self):
# # # # # # # # # # # # # # # #         # Reset attributes to their initial values
# # # # # # # # # # # # # # # #         self.__dict__ = copy.deepcopy(self.__dict__)

# # # # # # # # # # # # # # # # # Usage example
# # # # # # # # # # # # # # # # initial_instance = NavigationSimulator(attr1=1, attr2=2)

# # # # # # # # # # # # # # # # print(vars(initial_instance))

# # # # # # # # # # # # # # # # for _ in range(5):  # Example: call method 5 times
# # # # # # # # # # # # # # # #     new_attribute = initial_instance.modify_attributes()
# # # # # # # # # # # # # # # #     initial_instance.reset()
# # # # # # # # # # # # # # # #     print(new_attribute, vars(initial_instance))



# # # # # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # # # class RandomNumberGenerator:
# # # # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # # # #         self.seed = 0

# # # # # # # # # # # # # # #     def generate_normal(self, mean=0.0, std=1.0, size=1):
# # # # # # # # # # # # # # #         # Increment the seed for each call
# # # # # # # # # # # # # # #         self.seed = 1
# # # # # # # # # # # # # # #         # Create a new random generator with the updated seed
# # # # # # # # # # # # # # #         rng = np.random.default_rng(self.seed)
# # # # # # # # # # # # # # #         for _ in range(3):
# # # # # # # # # # # # # # #             print(rng.normal(mean, std, size))
# # # # # # # # # # # # # # #         # Generate the normal random numbers
# # # # # # # # # # # # # # #         # return rng.normal(mean, std, size)

# # # # # # # # # # # # # # # # Example usage
# # # # # # # # # # # # # # # rng_gen = RandomNumberGenerator()

# # # # # # # # # # # # # # # # Generate normal random numbers with different seeds
# # # # # # # # # # # # # # # print(rng_gen.generate_normal(mean=0, std=1, size=5))


# # # # # # # # # # # # # # class MyClass:
# # # # # # # # # # # # # #     def __init__(self, attr1, attr2):
# # # # # # # # # # # # # #         self.attr1 = attr1
# # # # # # # # # # # # # #         self.attr2 = attr2
# # # # # # # # # # # # # #         # Store the initial state of attributes
# # # # # # # # # # # # # #         self._initial_attrs = {attr: getattr(self, attr) for attr in vars(self)}
# # # # # # # # # # # # # #         print(self._initial_attrs)

# # # # # # # # # # # # # #     def modify_attributes(self):
# # # # # # # # # # # # # #         # Method that modifies attributes
# # # # # # # # # # # # # #         self.attr1 = "modified"
# # # # # # # # # # # # # #         self.attr2 = "modified"
# # # # # # # # # # # # # #         self.attr3 = "newattr"
# # # # # # # # # # # # # #         # Reset attributes to their initial state after modification
# # # # # # # # # # # # # #         self.reset_attributes()

# # # # # # # # # # # # # #     def reset_attributes(self):

# # # # # # # # # # # # # #         # Reset attributes to their initial state
# # # # # # # # # # # # # #         _initial_attrs = self._initial_attrs
# # # # # # # # # # # # # #         for attr, value in self._initial_attrs.items():
# # # # # # # # # # # # # #             setattr(self, attr, value)
# # # # # # # # # # # # # #         # Delete any newly created attributes
# # # # # # # # # # # # # #         for attr in list(vars(self)):
# # # # # # # # # # # # # #             if attr is not "_initial_attrs" and attr in self._initial_attrs:
# # # # # # # # # # # # # #                 delattr(self, attr)

# # # # # # # # # # # # # # # Example usage
# # # # # # # # # # # # # # obj = MyClass("initial_value1", "initial_value2")
# # # # # # # # # # # # # # print("Before modification:", obj.attr1, obj.attr2)  # Output: Before modification: initial_value1 initial_value2

# # # # # # # # # # # # # # obj.modify_attributes()
# # # # # # # # # # # # # # print("After modification:", obj.attr1, obj.attr2, getattr(obj, "attr3", None))  # Output: After modification: initial_value1 initial_value2 None


# # # # # # # # # # # # # import gc

# # # # # # # # # # # # # class A:
# # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # #         self.b = None

# # # # # # # # # # # # # class B:
# # # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # # #         self.a = None

# # # # # # # # # # # # # # Create instances of A and B
# # # # # # # # # # # # # a = A()
# # # # # # # # # # # # # b = B()

# # # # # # # # # # # # # # Create a cyclic reference
# # # # # # # # # # # # # a.b = b
# # # # # # # # # # # # # b.a = a

# # # # # # # # # # # # # # Manually break the reference cycle (uncomment to test cleanup)
# # # # # # # # # # # # # # a.b = None
# # # # # # # # # # # # # # b.a = None

# # # # # # # # # # # # # # Delete the references to a and b
# # # # # # # # # # # # # del a
# # # # # # # # # # # # # del b

# # # # # # # # # # # # # # Force garbage collection
# # # # # # # # # # # # # gc.collect()

# # # # # # # # # # # # # # Check for unreachable objects
# # # # # # # # # # # # # unreachable = gc.garbage
# # # # # # # # # # # # # print("Unreachable objects:", unreachable)


# # # # # # # # # # # # import tracemalloc

# # # # # # # # # # # # def large_function_call(data):
# # # # # # # # # # # #     # Simulate a large memory-consuming operation
# # # # # # # # # # # #     result = [x * 2 for x in data]
# # # # # # # # # # # #     return result

# # # # # # # # # # # # def process_data(dataset):
# # # # # # # # # # # #     results = []

# # # # # # # # # # # #     # Start tracing memory allocations
# # # # # # # # # # # #     tracemalloc.start()

# # # # # # # # # # # #     # Take an initial snapshot
# # # # # # # # # # # #     initial_snapshot = tracemalloc.take_snapshot()

# # # # # # # # # # # #     for index, data in enumerate(dataset):
# # # # # # # # # # # #         result = large_function_call(data)
# # # # # # # # # # # #         results.append(result)

# # # # # # # # # # # #         # Optionally take periodic snapshots to monitor memory usage
# # # # # # # # # # # #         if index % 100 == 0:  # Adjust the modulus value based on your use case
# # # # # # # # # # # #             intermediate_snapshot = tracemalloc.take_snapshot()
# # # # # # # # # # # #             top_stats = intermediate_snapshot.compare_to(initial_snapshot, 'lineno')
# # # # # # # # # # # #             print(f"Memory usage after {index + 1} iterations:")
# # # # # # # # # # # #             for stat in top_stats[:5]:
# # # # # # # # # # # #                 print(stat)

# # # # # # # # # # # #     # Take a final snapshot
# # # # # # # # # # # #     final_snapshot = tracemalloc.take_snapshot()
# # # # # # # # # # # #     top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')

# # # # # # # # # # # #     # Display the top memory usage differences
# # # # # # # # # # # #     print("[ Top 10 differences ]")
# # # # # # # # # # # #     for stat in top_stats[:10]:
# # # # # # # # # # # #         print(stat)

# # # # # # # # # # # #     # Stop tracing memory allocations
# # # # # # # # # # # #     tracemalloc.stop()

# # # # # # # # # # # #     return results

# # # # # # # # # # # # # Example usage
# # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # #     dataset = [list(range(1000)) for _ in range(1000)]  # Example dataset
# # # # # # # # # # # #     process_data(dataset)


# # # # # # # # # # # # import tracemalloc

# # # # # # # # # # # # class Base:

# # # # # # # # # # # #     def __init__(self):

# # # # # # # # # # # #         self.base_data = [list(range(1000)) for _ in range(1000)]


# # # # # # # # # # # # class DataProcessor(Base):
# # # # # # # # # # # #     def __init__(self):
# # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # #         self.data = None

# # # # # # # # # # # #     def process_data(self, data):
# # # # # # # # # # # #         self.data = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation
# # # # # # # # # # # #         self.data1 = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation
# # # # # # # # # # # #         self.data2 = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation
# # # # # # # # # # # #         self.data3 = {i: i * 2 for i in range(data)}  # Simulate a large memory-consuming operation

# # # # # # # # # # # #     def cleanup(self):
# # # # # # # # # # # #         self.data = None  # Release the reference to the dictionary

# # # # # # # # # # # # # Create an instance of DataProcessor
# # # # # # # # # # # # processor = DataProcessor()

# # # # # # # # # # # # # Start tracing memory allocations
# # # # # # # # # # # # tracemalloc.start()

# # # # # # # # # # # # # Take an initial snapshot before the loop
# # # # # # # # # # # # initial_snapshot = tracemalloc.take_snapshot()

# # # # # # # # # # # # # Simulate a loop that processes data multiple times
# # # # # # # # # # # # list = []
# # # # # # # # # # # # for i in range(10):
# # # # # # # # # # # #     processor.process_data(10000)
# # # # # # # # # # # #     # processor.cleanup()  # Clean up after each iteration

# # # # # # # # # # # #     list.append(processor.data[0])

# # # # # # # # # # # #     # Take a snapshot after each iteration
# # # # # # # # # # # #     snapshot = tracemalloc.take_snapshot()

# # # # # # # # # # # #     # Compare the snapshot to the initial snapshot
# # # # # # # # # # # #     top_stats = snapshot.compare_to(initial_snapshot, 'lineno')

# # # # # # # # # # # #     print(f"[Iteration {i + 1}] Top 1 memory usage differences:")
# # # # # # # # # # # #     for stat in top_stats[:1]:
# # # # # # # # # # # #         print(stat)

# # # # # # # # # # # # print(list)

# # # # # # # # # # # # # Stop tracing memory allocations
# # # # # # # # # # # # tracemalloc.stop()



# # # # # # # # # import numpy as np

# # # # # # # # # start = 60390
# # # # # # # # # end = 60390.91428571429
# # # # # # # # # num_points = 6

# # # # # # # # # array_with_endpoints = np.linspace(start, end, num_points)
# # # # # # # # # print(array_with_endpoints)


# # # # # # # import matplotlib.pyplot as plt
# # # # # # # import numpy as np
# # # # # # # import example

# # # # # # # # Generate some sample data
# # # # # # # x = np.linspace(0, 10, 100)  # Generate 100 points from 0 to 10
# # # # # # # y = np.sin(x)  # Compute sine of x for y values

# # # # # # # # Plot the data
# # # # # # # plt.figure(figsize=(8, 6))  # Create a new figure with size 8x6 inches
# # # # # # # plt.plot(x, y, label='sin(x)')  # Plot x vs y with label for the legend
# # # # # # # plt.title('Simple Plot Example')  # Set plot title
# # # # # # # plt.xlabel('x')  # Set x-axis label
# # # # # # # plt.ylabel('sin(x)')  # Set y-axis label
# # # # # # # plt.legend()  # Display legend based on labels

# # # # # # # # Save the plot as a PNG file
# # # # # # # plt.savefig('plot_example.png')

# # # # # # # # Display the plot
# # # # # # # plt.show()


# # # # # # import multiprocessing
# # # # # # import random
# # # # # # import math

# # # # # # def monte_carlo_pi_part(iterations):
# # # # # #     """
# # # # # #     Perform a Monte Carlo simulation to estimate the value of π.

# # # # # #     Args:
# # # # # #         iterations (int): Number of iterations for the simulation.

# # # # # #     Returns:
# # # # # #         int: Number of points inside the quarter circle.
# # # # # #     """
# # # # # #     inside_circle = 0
# # # # # #     for _ in range(iterations):
# # # # # #         x = random.uniform(0, 1)
# # # # # #         y = random.uniform(0, 1)
# # # # # #         if math.sqrt(x**2 + y**2) <= 1:
# # # # # #             inside_circle += 1
# # # # # #     return inside_circle

# # # # # # def parallel_monte_carlo_pi(total_iterations, num_processes):
# # # # # #     """
# # # # # #     Perform a parallelized Monte Carlo simulation to estimate the value of π.

# # # # # #     Args:
# # # # # #         total_iterations (int): Total number of iterations for the simulation.
# # # # # #         num_processes (int): Number of processes to use for parallelization.

# # # # # #     Returns:
# # # # # #         float: Estimated value of π.
# # # # # #     """
# # # # # #     iterations_per_process = total_iterations // num_processes

# # # # # #     # Create a pool of processes
# # # # # #     with multiprocessing.Pool(num_processes) as pool:
# # # # # #         results = pool.map(monte_carlo_pi_part, [iterations_per_process] * num_processes)

# # # # # #     total_inside_circle = sum(results)
# # # # # #     pi_estimate = (4.0 * total_inside_circle) / total_iterations
# # # # # #     return pi_estimate

# # # # # # if __name__ == "__main__":
# # # # # #     total_iterations = 1000000
# # # # # #     num_processes = multiprocessing.cpu_count()
# # # # # #     num_processes = 5
# # # # # #     print(num_processes)  # Use all available CPU cores

# # # # # #     pi_estimate = parallel_monte_carlo_pi(total_iterations, num_processes)
# # # # # #     print(f"Estimated value of π: {pi_estimate}")


# # # # # import datetime
# # # # # import time
# # # # # from itertools import product

# # # # # # Sensitivity analysis cases
# # # # # test_cases = {
# # # # #     "step_size": [0.00, 0.01, 0.02]
# # # # # }

# # # # # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# # # # # def generate_time_tag(parameters):
# # # # #     params_str = "_".join(f"{k}_{v:.2f}".replace('.', '_') for k, v in parameters.items())
# # # # #     return f"{current_time}_{params_str}"

# # # # # # Generate combinations of all test cases
# # # # # keys, values = zip(*test_cases.items())
# # # # # combinations = [dict(zip(keys, v)) for v in product(*values)]

# # # # # # Running the sensitivity analysis
# # # # # for case in combinations:
# # # # #     time_tag = generate_time_tag(case)
# # # # #     print(f"Run completed with time tag: {time_tag}")


# # # # # import pickle

# # # # # # class MyClass:
# # # # # #     def __init__(self, name, value):
# # # # # #         self.name = name
# # # # # #         self.value = value

# # # # # #     def display(self):
# # # # # #         print(f"Name: {self.name}, Value: {self.value}")

# # # # # # # Create an instance of the class
# # # # # # my_instance = MyClass(name="example", value=42)

# # # # # # # Save the class instance to a file
# # # # # # with open("my_instance.pkl", "wb") as f:
# # # # # #     pickle.dump(my_instance, f)

# # # # # # Load the class instance from the file
# # # # # with open("my_instance.pkl", "rb") as f:
# # # # #     loaded_instance = pickle.load(f)

# # # # # # Display the loaded instance
# # # # # loaded_instance.display()


# # # # import json

# # # # class MyClass:
# # # #     def __init__(self, name, value):
# # # #         self.name = name
# # # #         self.value = value

# # # #     def display(self):
# # # #         print(f"Name: {self.name}, Value: {self.value}")
# # # #         self.value = self.value*2

# # # #         return Output(self)



# # # # class Output(my_class):

# # # #     def __init__(self, my_class):
# # # #         self.my_class = my_class



# # # # # Custom encoder and decoder for MyClass
# # # # class MyClassEncoder(json.JSONEncoder):
# # # #     def default(self, obj):
# # # #         if isinstance(obj, MyClass):
# # # #             return obj.__dict__
# # # #         return super().default(obj)

# # # # def myclass_decoder(dct):
# # # #     if 'name' in dct and 'value' in dct:
# # # #         return MyClass(**dct)
# # # #     return dct

# # # # # Create an instance of the class
# # # # my_instance = MyClass(name="example", value=42)

# # # # # Save the class instance to a file
# # # # with open("my_instance.json", "w") as f:
# # # #     json.dump(my_instance, f, cls=MyClassEncoder)

# # # # # Load the class instance from the file
# # # # with open("my_instance.json", "r") as f:
# # # #     loaded_instance = json.load(f, object_hook=myclass_decoder)

# # # # # Display the loaded instance
# # # # loaded_instance.display()


# # # import itertools

# # # def generate_case_time_tag(case, custom_time=False, run=0):
# # #     params_str = "_".join(f"{run}_{k}_{v:.2f}".replace('.', '_') for k, v in case.items())
# # #     time = current_time
# # #     if custom_time is not False:
# # #         time = custom_time
# # #     return f"{time}_{params_str}"

# # # current_time =100000
# # # custom_input=False

# # # # Define the cases
# # # run_num = 1
# # # cases = {
# # #     "delta_v_min": [0.00, 0.01]
# # # }

# # # # Extract keys and values
# # # keys, values = zip(*cases.items())

# # # # Generate combinations of cases
# # # combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# # # # Initialize a list to store time tags
# # # time_tags = []
# # # for case in combinations:
# # #     for run in range(run_num):

# # #         time_tag = generate_case_time_tag(case, run=run)
# # #         if custom_input:
# # #             time_tag = generate_case_time_tag(case, custom_time=custom_time_tag, run=run)
# # #         time_tags.append(time_tag)

# # # # Output the time tags
# # # print(time_tags)
# # # a


# import multiprocessing
# import time

# def process_case(case):
#     # Simulate some work with a sleep
#     time.sleep(5)
#     return f"Processed case: {case}"

# def parallel_processing(cases, num_workers):
#     with multiprocessing.Pool(processes=num_workers) as pool:
#         results = pool.map(process_case, cases)
#     return results

# if __name__ == "__main__":
#     # Define the cases to process
#     cases = [f"Case {i}" for i in range(5)]

#     # Number of worker processes
#     num_workers = multiprocessing.cpu_count()

#     # Process cases in parallel
#     start_time = time.time()
#     results = parallel_processing(cases, num_workers)
#     end_time = time.time()

#     # Print results
#     for result in results:
#         print(result)

#     print(f"Processing took {end_time - start_time:.2f} seconds")


import concurrent.futures
import time

def task_function(x):
    # Your task implementation
    time.sleep(5)
    return x * x

def main():
    num_workers = 56  # Number of worker processes
    tasks = list(range(100))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(task_function, tasks))

    print("Results:", results)

if __name__ == "__main__":
    main()
