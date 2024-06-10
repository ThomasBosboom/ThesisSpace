# # # # import tracemalloc

# # # # class DataGenerator:
# # # #     def generate_large_list(self, size):
# # # #         return [i ** 2 for i in range(size)]

# # # #     def generate_large_dict(self, size):
# # # #         return {i: i ** 2 for i in range(size)}

# # # # class DataProcessor:
# # # #     def __init__(self):
# # # #         self.data = None

# # # #     def process_data(self, data):
# # # #         self.data = data
# # # #         # Simulate some processing
# # # #         processed_data = [x * 2 for x in self.data]
# # # #         return processed_data

# # # # def main():
# # # #     # Start tracing memory allocations
# # # #     tracemalloc.start()

# # # #     generator = DataGenerator()
# # # #     processor = DataProcessor()

# # # #     # Generate large datasets
# # # #     large_list = generator.generate_large_list(10000)
# # # #     large_dict = generator.generate_large_dict(10000)

# # # #     # Process the large list
# # # #     processed_list = processor.process_data(large_list)

# # # #     # Process the large dict (just converting values to list for simplicity)
# # # #     processed_dict_values = processor.process_data(list(large_dict.values()))

# # # #     # Take a snapshot
# # # #     snapshot = tracemalloc.take_snapshot()

# # # #     # Display top memory allocations
# # # #     # top_stats = snapshot.statistics('lineno')
# # # #     top_stats = snapshot.statistics('traceback')

# # # #     print("[ Top 10 memory allocations ]")
# # # #     for stat in top_stats[:10]:
# # # #         print(stat)

# # # #     # Stop tracing memory allocations
# # # #     tracemalloc.stop()

# # # # if __name__ == "__main__":
# # # #     main()



# # # import tracemalloc
# # # import time

# # # # Start tracing memory allocations
# # # tracemalloc.start()

# # # # Simulate a function that gradually increases memory usage
# # # def simulate_memory_usage():
# # #     data = []
# # #     for i in range(10):
# # #         data.append([i] * 1000000)  # Increase memory usage
# # #         time.sleep(1)
# # #         print(f"Snapshot {i + 1}")
# # #         snapshot = tracemalloc.take_snapshot()
# # #         top_stats = snapshot.statistics('lineno')
# # #         for stat in top_stats[:10]:
# # #             print(stat)
# # #         total_memory = sum(stat.size for stat in top_stats)
# # #         print(f"Total memory used after iteration {i + 1}: {total_memory / (1024 ** 2):.2f} MB")
# # #     return data

# # # simulate_memory_usage()

# # # # Optional: Stop tracing memory allocations
# # # tracemalloc.stop()

# # from memory_profiler import profile

# # class MyClass:
# #     def __init__(self):
# #         self.attribute1 = "initial_value1"
# #         self.attribute2 = "initial_value2"
# #         self.attribute3 = "initial_value3"
# #         self.initial_state = vars(self).copy()  # Store initial state

# #     @profile
# #     def generate_additional_attributes(self):
# #         # Method that generates additional attributes
# #         self.additional_attribute1 = "new_value1"
# #         self.additional_attribute2 = "new_value2"

# #     @profile
# #     def reset_to_init_state(self):

# #         current_state = vars(self)
# #         for attr_name in list(current_state.keys()):
# #             if attr_name is not str("initial_state"):
# #                 if attr_name not in self.initial_state.keys():
# #                     delattr(self, attr_name)


# # # Example usage:
# # obj = MyClass()

# # print(vars(obj))
# # # Call the method to generate additional attributes
# # obj.generate_additional_attributes()

# # # Print the attributes after generating additional attributes
# # print(vars(obj))

# # # Call the method to reset attributes to their initial state
# # obj.reset_to_init_state()

# # # Print the attributes after resetting
# # print(vars(obj))



# #             self.full_estimation_error_dict = dict()
# #             self.full_reference_state_deviation_dict = dict()
# #             self.full_propagated_covariance_dict = dict()
# #             self.full_propagated_formal_errors_dict = dict()
# #             self.full_state_history_reference_dict = dict()
# #             self.full_state_history_truth_dict = dict()
# #             self.full_state_history_estimated_dict = dict()
# #             self.full_state_history_final_dict = dict()
# #             self.delta_v_dict = dict()
# #             self.full_dependent_variables_history_estimated = dict()
# #             self.full_state_transition_matrix_history_estimated = dict()
# #             self.estimation_arc_results_dict = dict()
# import numpy as np

# # Example input dictionary
# data = {
#     '0.0': {
#         60391: {0: 0.0012819549366386333, 1: 0.001185896799597702},
#         60395: {0: 0.005411142569208645, 1: 0.006770971960928585},
#         60399: {0: 0.003109677105139794, 1: 0.002999612884133889},
#         60403: {0: 0.004734675217803918, 1: 0.004341626884436936}
#     },
#     '0.03': {
#         60399: {0: 0.0672568474862809, 1: 0.06727010238539159}
#     }
# }

# def calculate_stats(data, evaluation_threshold=14):
#     result_dict = {}

#     # Iterate through each test case
#     for case_type, epochs in data.items():


#         # Iterate through each epoch
#         epoch_stats = {}
#         combined_per_run = {}
#         combined_per_run_with_threshold = {}
#         for epoch, runs in epochs.items():
#             keys = list(runs.keys())
#             values = list(runs.values())
#             epoch_stats[epoch] = {'mean': np.mean(values), 'std': np.std(values)}

#             for key in keys:
#                 if key not in combined_per_run:
#                     combined_per_run[key] = []
#                     combined_per_run_with_threshold[key] = []

#                 combined_per_run[key].append(runs[key])
#                 if epoch >= 60390 + evaluation_threshold:
#                     combined_per_run_with_threshold[key].append(runs[key])


#         total = []
#         total_with_threshold = []
#         for run, combined in combined_per_run.items():
#             total.append(np.sum(combined))
#         for run, combined in combined_per_run_with_threshold.items():
#             total_with_threshold.append(np.sum(combined))

#         total_stats = {'mean': np.mean(total), 'std': np.std(total)}
#         total_stats_with_threshold = {'mean': np.mean(total_with_threshold), 'std': np.std(total_with_threshold)}

#         # Store statistics in the result dictionary
#         result_dict[case_type] = {'epoch_stats': epoch_stats, 'total_stats': total_stats, 'total_stats_with_threshold': total_stats_with_threshold}

#     return result_dict

# # Calculate the statistics
# result_dict = calculate_stats(data)

# # Print the resulting dictionary
# print(result_dict)
def create_overleaf_table(data, caption="Statistical results of Monte Carlo sensitivity analysis", label="tab:SensitivityAnalysis", file_name="sensitivity_analysis.tex", decimals=4):
    table_str = r'% Please add the following required packages to your document preamble:' + '\n'
    table_str += r'% \usepackage[table,xcdraw]{xcolor}' + '\n'
    table_str += r'% Beamer presentation requires \usepackage{colortbl} instead of \usepackage[table,xcdraw]{xcolor}' + '\n'
    table_str += r'\begin{table}[]' + '\n'
    table_str += r'\centering' + '\n'
    table_str += r'\begin{tabular}{llllll}' + '\n'
    table_str += r' &  & \cellcolor[HTML]{EFEFEF}\textbf{Total} &  & \cellcolor[HTML]{EFEFEF}\textbf{Last 14 days} &  \\' + '\n'
    table_str += r'\rowcolor[HTML]{EFEFEF} ' + '\n'
    table_str += r'\textbf{Case} & \textbf{Value} & \textbf{$\mu_{\Delta \boldsymbol{V}}$} & \textbf{$\sigma_{\Delta \boldsymbol{V}}$} & \textbf{$\mu_{\Delta \boldsymbol{V}}$} & \textbf{$\sigma_{\Delta \boldsymbol{V}}$} \\ ' + '\n'

    for case, case_data in data.items():
        is_first_row = True
        for subkey, stats in case_data.items():
            full_stats = stats.get('total_stats', {})
            threshold_stats = stats.get('total_stats_with_threshold', {})

            if is_first_row:
                table_str += f'{case} & '
                is_first_row = False
            else:
                table_str += ' & '

            table_str += f"{subkey} & " + \
                         f"{full_stats.get('mean', ''):.{decimals}f} & " + \
                         f"{full_stats.get('std', ''):.{decimals}f} & " + \
                         f"{threshold_stats.get('mean', ''):.{decimals}f} & " + \
                         f"{threshold_stats.get('std', ''):.{decimals}f}" + r' \\ ' + '\n'

    table_str += r'\end{tabular}' + '\n'
    table_str += r'\caption{' + caption + '}' + '\n'
    table_str += r'\label{' + label + '}' + '\n'
    table_str += r'\end{table}'

    if file_name:
        with open(file_name, 'w') as f:
            f.write(table_str)

    return table_str

# Example usage
sensitivity_statistics = {
    'Mission Start Epoch': {
        '60400': {
            'total_stats': {'mean': 0.035940300851514353, 'std': 0.0008319253572869126},
            'total_stats_with_threshold': {'mean': 0.010630449211852798, 'std': 2.7144310782730927e-05}
        },
        '60405': {
            'total_stats': {'mean': 0.03424555465277684, 'std': 0.0006336840370862809},
            'total_stats_with_threshold': {'mean': 0.016944068531082823, 'std': 0.00014095646779255233}
        }
    }
}

# Generate the Overleaf table with custom caption, title, label, and decimals
overleaf_table = create_overleaf_table(sensitivity_statistics)

# Print the Overleaf table
print(overleaf_table)
