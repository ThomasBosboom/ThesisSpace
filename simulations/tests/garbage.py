# # import tracemalloc

# # class DataGenerator:
# #     def generate_large_list(self, size):
# #         return [i ** 2 for i in range(size)]

# #     def generate_large_dict(self, size):
# #         return {i: i ** 2 for i in range(size)}

# # class DataProcessor:
# #     def __init__(self):
# #         self.data = None

# #     def process_data(self, data):
# #         self.data = data
# #         # Simulate some processing
# #         processed_data = [x * 2 for x in self.data]
# #         return processed_data

# # def main():
# #     # Start tracing memory allocations
# #     tracemalloc.start()

# #     generator = DataGenerator()
# #     processor = DataProcessor()

# #     # Generate large datasets
# #     large_list = generator.generate_large_list(10000)
# #     large_dict = generator.generate_large_dict(10000)

# #     # Process the large list
# #     processed_list = processor.process_data(large_list)

# #     # Process the large dict (just converting values to list for simplicity)
# #     processed_dict_values = processor.process_data(list(large_dict.values()))

# #     # Take a snapshot
# #     snapshot = tracemalloc.take_snapshot()

# #     # Display top memory allocations
# #     # top_stats = snapshot.statistics('lineno')
# #     top_stats = snapshot.statistics('traceback')

# #     print("[ Top 10 memory allocations ]")
# #     for stat in top_stats[:10]:
# #         print(stat)

# #     # Stop tracing memory allocations
# #     tracemalloc.stop()

# # if __name__ == "__main__":
# #     main()



# import tracemalloc
# import time

# # Start tracing memory allocations
# tracemalloc.start()

# # Simulate a function that gradually increases memory usage
# def simulate_memory_usage():
#     data = []
#     for i in range(10):
#         data.append([i] * 1000000)  # Increase memory usage
#         time.sleep(1)
#         print(f"Snapshot {i + 1}")
#         snapshot = tracemalloc.take_snapshot()
#         top_stats = snapshot.statistics('lineno')
#         for stat in top_stats[:10]:
#             print(stat)
#         total_memory = sum(stat.size for stat in top_stats)
#         print(f"Total memory used after iteration {i + 1}: {total_memory / (1024 ** 2):.2f} MB")
#     return data

# simulate_memory_usage()

# # Optional: Stop tracing memory allocations
# tracemalloc.stop()

from memory_profiler import profile

class MyClass:
    def __init__(self):
        self.attribute1 = "initial_value1"
        self.attribute2 = "initial_value2"
        self.attribute3 = "initial_value3"
        self.initial_state = vars(self).copy()  # Store initial state

    @profile
    def generate_additional_attributes(self):
        # Method that generates additional attributes
        self.additional_attribute1 = "new_value1"
        self.additional_attribute2 = "new_value2"

    @profile
    def reset_to_init_state(self):

        current_state = vars(self)
        for attr_name in list(current_state.keys()):
            if attr_name is not str("initial_state"):
                if attr_name not in self.initial_state.keys():
                    delattr(self, attr_name)


# Example usage:
obj = MyClass()

print(vars(obj))
# Call the method to generate additional attributes
obj.generate_additional_attributes()

# Print the attributes after generating additional attributes
print(vars(obj))

# Call the method to reset attributes to their initial state
obj.reset_to_init_state()

# Print the attributes after resetting
print(vars(obj))
