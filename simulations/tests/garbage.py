# # # # # # # # # import numpy as np


# # # # # # # # # matrix = np.random.rand(100, 12, 12)

# # # # # # # # # A1 = 500**2*np.eye(3,3)

# # # # # # # # # A2 = np.array([[2, 1],
# # # # # # # # #                [1, 2]])

# # # # # # # # # print(np.linalg.cond(A1.T), np.linalg.eigvals(A1))
# # # # # # # # # print(np.linalg.cond(A2), np.linalg.eigvals(A2))


# # # # # # # # import numpy as np
# # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # from matplotlib.patches import Ellipse

# # # # # # # # def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
# # # # # # # #     """
# # # # # # # #     Plots an error ellipse based on the covariance matrix `cov` and the mean `pos`.
# # # # # # # #     `nstd` is the number of standard deviations to determine the ellipse's radii.
# # # # # # # #     """
# # # # # # # #     eigvals, eigvecs = np.linalg.eigh(cov)
# # # # # # # #     order = eigvals.argsort()[::-1]
# # # # # # # #     eigvals, eigvecs = eigvals[order], eigvecs[:, order]

# # # # # # # #     angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

# # # # # # # #     width, height = 2 * nstd * np.sqrt(eigvals)
# # # # # # # #     ellip = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)

# # # # # # # #     return ellip

# # # # # # # # # Example: Assume we have the following covariance history
# # # # # # # # # Note: Replace this with your actual data
# # # # # # # # num_samples = 100
# # # # # # # # covariances = [np.random.rand(3, 3) for _ in range(num_samples)]
# # # # # # # # means = [np.random.rand(3) for _ in range(num_samples)]

# # # # # # # # # Plotting
# # # # # # # # fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # # # # # # # # Labels for the principal planes
# # # # # # # # plane_labels = [('State 1', 'State 2'), ('State 1', 'State 3'), ('State 2', 'State 3')]

# # # # # # # # for i, (x_idx, y_idx) in enumerate([(0, 1), (0, 2), (1, 2)]):
# # # # # # # #     ax = axes[i]
# # # # # # # #     ax.set_xlabel(plane_labels[i][0])
# # # # # # # #     ax.set_ylabel(plane_labels[i][1])

# # # # # # # #     for cov, mean in zip(covariances, means):
# # # # # # # #         sub_cov = cov[[x_idx, y_idx]][:, [x_idx, y_idx]]
# # # # # # # #         sub_mean = mean[[x_idx, y_idx]]

# # # # # # # #         ellip = plot_cov_ellipse(sub_cov, sub_mean, nstd=2, edgecolor='black', alpha=0.2)
# # # # # # # #         ax.add_patch(ellip)
# # # # # # # #         ax.scatter(*sub_mean, color='red')

# # # # # # # #     ax.grid(True)

# # # # # # # # plt.tight_layout()
# # # # # # # # plt.show()

# # # # # # # import numpy as np
# # # # # # # import matplotlib.pyplot as plt
# # # # # # # from matplotlib.patches import Ellipse

# # # # # # # def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
# # # # # # #     """
# # # # # # #     Plots an error ellipse based on the covariance matrix `cov` and the mean `pos`.
# # # # # # #     `nstd` is the number of standard deviations to determine the ellipse's radii.
# # # # # # #     """
# # # # # # #     eigvals, eigvecs = np.linalg.eigh(cov)
# # # # # # #     order = eigvals.argsort()[::-1]
# # # # # # #     eigvals, eigvecs = eigvals[order], eigvecs[:, order]

# # # # # # #     angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

# # # # # # #     width, height = 2 * nstd * np.sqrt(eigvals)
# # # # # # #     ellip = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)

# # # # # # #     return ellip

# # # # # # # # Example: Simulate smoother trajectory and increasing uncertainty
# # # # # # # num_samples = 50
# # # # # # # time = np.linspace(0, 2 * np.pi, num_samples)
# # # # # # # means = np.array([np.sin(time), np.cos(time), np.sin(2*time)]).T  # Smoothed trajectory

# # # # # # # # Increasing covariance over time
# # # # # # # covariances = [np.diag([0.1*(i+1), 0.005*(i+1), 0.01*(i+1)])/10 for i in range(num_samples)]

# # # # # # # # Plotting
# # # # # # # fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # # # # # # # Labels for the principal planes
# # # # # # # plane_labels = [('State 1', 'State 2'), ('State 1', 'State 3'), ('State 2', 'State 3')]

# # # # # # # for i, (x_idx, y_idx) in enumerate([(0, 1), (0, 2), (1, 2)]):
# # # # # # #     ax = axes[i]
# # # # # # #     ax.set_xlabel(plane_labels[i][0])
# # # # # # #     ax.set_ylabel(plane_labels[i][1])

# # # # # # #     # Extract mean positions for trajectory lines
# # # # # # #     trajectory = means[:, [x_idx, y_idx]]

# # # # # # #     # Plot trajectory
# # # # # # #     ax.plot(trajectory[:, 0], trajectory[:, 1], 'r--', alpha=0.5)

# # # # # # #     for j, (cov, mean) in enumerate(zip(covariances, means)):
# # # # # # #         sub_cov = cov[[x_idx, y_idx]][:, [x_idx, y_idx]]
# # # # # # #         sub_mean = mean[[x_idx, y_idx]]

# # # # # # #         ellip = plot_cov_ellipse(sub_cov, sub_mean, nstd=2, edgecolor='black', alpha=0.2)
# # # # # # #         ax.add_patch(ellip)
# # # # # # #         ax.scatter(*sub_mean, color='red')

# # # # # # #         # Optional: Add a label to indicate the time step
# # # # # # #         if j % 5 == 0:  # Show labels every 5 steps to avoid clutter
# # # # # # #             ax.text(*sub_mean, f'T{j}', fontsize=9, ha='center')

# # # # # # #     ax.grid(True)

# # # # # # # plt.tight_layout()
# # # # # # # plt.show()


# # # # # # import numpy as np
# # # # # # import matplotlib.pyplot as plt

# # # # # # def plot_error_bounds(ax, traj, error_bounds, x_idx, y_idx, label):
# # # # # #     """
# # # # # #     Plot the trajectory with error bounds in a specific plane.
# # # # # #     """
# # # # # #     upper_bound_x = traj[:, x_idx] + error_bounds[:, x_idx]
# # # # # #     lower_bound_x = traj[:, x_idx] - error_bounds[:, x_idx]

# # # # # #     upper_bound_y = traj[:, y_idx] + error_bounds[:, y_idx]
# # # # # #     lower_bound_y = traj[:, y_idx] - error_bounds[:, y_idx]

# # # # # #     ax.plot(traj[:, x_idx], traj[:, y_idx], 'r', label=label)
# # # # # #     ax.fill_between(traj[:, x_idx], lower_bound_y, upper_bound_y, color='gray', alpha=0.5, label='Error Bounds')
# # # # # #     ax.fill_betweenx(traj[:, y_idx], lower_bound_x, upper_bound_x, color='gray', alpha=0.5)
# # # # # #     ax.set_xlabel(f'State {x_idx + 1}')
# # # # # #     ax.set_ylabel(f'State {y_idx + 1}')
# # # # # #     ax.legend()
# # # # # #     ax.grid(True)

# # # # # # # Generate a smooth trajectory using sinusoidal functions
# # # # # # num_samples = 100
# # # # # # time = np.linspace(0, 0.3 * np.pi, num_samples)
# # # # # # trajectory = np.array([np.sin(time), np.cos(time), np.sin(2 * time)]).T  # 3D trajectory
# # # # # # print(np.shape(trajectory))

# # # # # # # Define the error bounds (e.g., ±0.2 around the trajectory)
# # # # # # error_bounds = 0.05 * (1 + np.sin(5 * time))[:, np.newaxis]
# # # # # # error_bounds = np.hstack([error_bounds] * 3)  # Apply the same bounds to all three states
# # # # # # print(np.shape(error_bounds))


# # # # # # # Plotting
# # # # # # fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # # # # # # Plot the trajectory and error bounds in the principal planes
# # # # # # plot_error_bounds(axes[0], trajectory, error_bounds, 0, 1, 'Trajectory (State 1 vs. State 2)')
# # # # # # plot_error_bounds(axes[1], trajectory, error_bounds, 0, 2, 'Trajectory (State 1 vs. State 3)')
# # # # # # plot_error_bounds(axes[2], trajectory, error_bounds, 1, 2, 'Trajectory (State 2 vs. State 3)')

# # # # # # plt.tight_layout()
# # # # # # plt.show()


# # # # # import matplotlib.pyplot as plt
# # # # # import numpy as np

# # # # # # Sample data
# # # # # x = np.linspace(0, 10, 10)
# # # # # y = np.sin(x)

# # # # # # Create a figure and axis
# # # # # fig, ax = plt.subplots()

# # # # # # Plot the data
# # # # # ax.plot(x, y)

# # # # # # Draw arrows
# # # # # start_positions = [3, 5, 7]  # X positions to start arrows
# # # # # arrow_lengths = [0.5, 0.7, 0.6]  # Lengths of arrows
# # # # # arrow_angles = [30, 45, 60]  # Angles of arrows in degrees

# # # # # for start, length, angle in zip(start_positions, arrow_lengths, arrow_angles):
# # # # #     ax.quiver(start, np.sin(start), length * np.cos(np.radians(angle)), length * np.sin(np.radians(angle)),
# # # # #               angles='xy', scale_units='xy', scale=0.01, color='red')

# # # # # # Set axis limits
# # # # # ax.set_xlim(0, 10)
# # # # # ax.set_ylim(-1, 1)

# # # # # # Show plot
# # # # # plt.show()

# # # # class Mission:
# # # #     def __init__(self, mission_start_epoch, duration, arc_length):
# # # #         self.mission_start_epoch = mission_start_epoch
# # # #         self.duration = duration
# # # #         self.arc_length = arc_length

# # # #     def generate_observation_windows(self, design_vector):
# # # #         observation_windows = []
# # # #         current_time = self.mission_start_epoch

# # # #         for arc_interval in design_vector:
# # # #             end_time = current_time + self.arc_length

# # # #             # Adjust the end_time if it exceeds the mission duration
# # # #             if end_time > self.mission_start_epoch + self.duration:
# # # #                 end_time = self.mission_start_epoch + self.duration

# # # #             observation_windows.append((current_time, end_time))
# # # #             current_time = end_time + arc_interval

# # # #             # Stop if the next start time exceeds the mission duration
# # # #             if current_time >= self.mission_start_epoch + self.duration:
# # # #                 break

# # # #         return observation_windows

# # # # # Example usage
# # # # mission_start_epoch = 0  # Start of the mission in some time units
# # # # duration = 100  # Total duration of the mission
# # # # arc_length = 10  # Length of each observation arc

# # # # # Each entry represents the interval between consecutive observation arcs
# # # # design_vector = [5, 3, 7, 2, 6]

# # # # mission = Mission(mission_start_epoch, duration, arc_length)
# # # # observation_windows = mission.generate_observation_windows(design_vector)

# # # # for window in observation_windows:
# # # #     print(f"Observation window: Start = {window[0]}, End = {window[1]}")

# # # # print(observation_windows)


# # # import numpy as np

# # # def generate_arc_sets(duration, arc_length, arc_interval):

# # #     arc_sets = []
# # #     current_time = 0
# # #     while current_time < duration:
# # #         arc_sets.append((current_time, current_time + arc_length))
# # #         current_time += arc_length + arc_interval

# # #     for arc_set in arc_sets:
# # #         if arc_set[1] >= duration:
# # #             arc_sets.remove(arc_set)
# # #             break
# # #     return arc_sets

# # # # Example usage:
# # # duration = 25
# # # arc_length = 5
# # # arc_interval = 2

# # # arc_sets = generate_arc_sets(duration, arc_length, arc_interval)
# # # print("Arc sets:", arc_sets)


# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Example data
# # epochs = np.linspace(0, 10, 100)
# # information_matrix_history_array = np.random.rand(100, 9, 9)

# # # Plotting
# # fig, ax = plt.subplots()

# # # Plot LPF
# # lpf_line = ax.plot(epochs, np.linalg.cond(information_matrix_history_array[:, 0:3, 0:3]), color="red", label="LPF")

# # # Plot LUMIO
# # lumio_line = ax.plot(epochs, np.linalg.cond(information_matrix_history_array[:, 6:9, 6:9]), color="blue", label="LUMIO")

# # # Create legend with handles
# # handles = [lpf_line, lumio_line]
# # labels = [handle.get_label() for handle in handles]
# # ax.legend(handles=handles, labels=labels)

# # plt.show()




# data = {
#         "0": {
#             "design_vector": [
#                 2.04,
#                 2.04,
#                 1.9,
#                 2.04,
#                 2.04
#             ],
#             "objective_value": 0.014174469877060197,
#             "reduction": -4.4227775712858275
#         },
#         "1": {
#             "design_vector": [
#                 2.04,
#                 2.04,
#                 1.9,
#                 2.04,
#                 2.04
#             ],
#             "objective_value": 0.014174469877060197,
#             "reduction": -4.4227775712858275
#         },
#         "2": {
#             "design_vector": [
#                 1.8576000000000006,
#                 1.9735999999999994,
#                 1.9160000000000004,
#                 2.1176000000000004,
#                 2.1176000000000004
#             ],
#             "objective_value": 0.01297308581041283,
#             "reduction": -12.523606255263637
#         },
#         "3": {
#             "design_vector": [
#                 1.8576000000000006,
#                 1.9735999999999994,
#                 1.9160000000000004,
#                 2.1176000000000004,
#                 2.1176000000000004
#             ],
#             "objective_value": 0.01297308581041283,
#             "reduction": -12.523606255263637
#         },
#         "4": {
#             "design_vector": [
#                 1.8576000000000006,
#                 1.9735999999999994,
#                 1.9160000000000004,
#                 2.1176000000000004,
#                 2.1176000000000004
#             ],
#             "objective_value": 0.01297308581041283,
#             "reduction": -12.523606255263637
#         },
#         "5": {
#             "design_vector": [
#                 1.8576000000000006,
#                 1.9735999999999994,
#                 1.9160000000000004,
#                 2.1176000000000004,
#                 2.1176000000000004
#             ],
#             "objective_value": 0.01297308581041283,
#             "reduction": -12.523606255263637
#         },
#         "6": {
#             "design_vector": [
#                 1.8706713600000011,
#                 2.040328959999999,
#                 1.8101376000000005,
#                 2.0776473600000003,
#                 2.13140736
#             ],
#             "objective_value": 0.01273035227551694,
#             "reduction": -14.160337452752639
#         },
#         "7": {
#             "design_vector": [
#                 1.8706713600000011,
#                 2.040328959999999,
#                 1.8101376000000005,
#                 2.0776473600000003,
#                 2.13140736
#             ],
#             "objective_value": 0.01273035227551694,
#             "reduction": -14.160337452752639
#         },
#         "8": {
#             "design_vector": [
#                 1.8706713600000011,
#                 2.040328959999999,
#                 1.8101376000000005,
#                 2.0776473600000003,
#                 2.13140736
#             ],
#             "objective_value": 0.01273035227551694,
#             "reduction": -14.160337452752639
#         },
#         "9": {
#             "design_vector": [
#                 1.8706713600000011,
#                 2.040328959999999,
#                 1.8101376000000005,
#                 2.0776473600000003,
#                 2.13140736
#             ],
#             "objective_value": 0.01273035227551694,
#             "reduction": -14.160337452752639
#         },
#         "10": {
#             "design_vector": [
#                 1.8641356800000008,
#                 2.0069644799999993,
#                 1.8630688000000004,
#                 2.0976236800000003,
#                 2.12450368
#             ],
#             "objective_value": 0.01260119535997549,
#             "reduction": -15.031231345221235
#         },
#         "11": {
#             "design_vector": [
#                 1.8835924057600009,
#                 2.0060491673599996,
#                 1.8541564416000003,
#                 2.0787864217600003,
#                 2.12757058176
#             ],
#             "objective_value": 0.012462843965700939,
#             "reduction": -15.964122811257775
#         },
#         "12": {
#             "design_vector": [
#                 1.8835924057600009,
#                 2.0060491673599996,
#                 1.8541564416000003,
#                 2.0787864217600003,
#                 2.12757058176
#             ],
#             "objective_value": 0.012462843965700939,
#             "reduction": -15.964122811257775
#         },
#         "13": {
#             "design_vector": [
#                 1.8720527751040006,
#                 2.0043557277439996,
#                 1.8473682246400003,
#                 2.0910998615040004,
#                 2.1297533255039998
#             ],
#             "objective_value": 0.012457978260848457,
#             "reduction": -15.996931837555875
#         },
#         "14": {
#             "design_vector": [
#                 1.8577502809856008,
#                 2.0185653090815996,
#                 1.8606205288960003,
#                 2.0812477459456007,
#                 2.1182490355456
#             ],
#             "objective_value": 0.012453536470413844,
#             "reduction": -16.02688244565853
#         },
#         "15": {
#             "design_vector": [
#                 1.8577502809856008,
#                 2.0185653090815996,
#                 1.8606205288960003,
#                 2.0812477459456007,
#                 2.1182490355456
#             ],
#             "objective_value": 0.012453536470413844,
#             "reduction": -16.02688244565853
#         },
#         "16": {
#             "design_vector": [
#                 1.8577502809856008,
#                 2.0185653090815996,
#                 1.8606205288960003,
#                 2.0812477459456007,
#                 2.1182490355456
#             ],
#             "objective_value": 0.012453536470413844,
#             "reduction": -16.02688244565853
#         },
#         "17": {
#             "design_vector": [
#                 1.8577502809856008,
#                 2.0185653090815996,
#                 1.8606205288960003,
#                 2.0812477459456007,
#                 2.1182490355456
#             ],
#             "objective_value": 0.012453536470413844,
#             "reduction": -16.02688244565853
#         },
#         "18": {
#             "design_vector": [
#                 1.8577502809856008,
#                 2.0185653090815996,
#                 1.8606205288960003,
#                 2.0812477459456007,
#                 2.1182490355456
#             ],
#             "objective_value": 0.012453536470413844,
#             "reduction": -16.02688244565853
#         },
#         "19": {
#             "design_vector": [
#                 1.8577502809856008,
#                 2.0185653090815996,
#                 1.8606205288960003,
#                 2.0812477459456007,
#                 2.1182490355456
#             ],
#             "objective_value": 0.012453536470413844,
#             "reduction": -16.02688244565853
#         },
#         "20": {
#             "design_vector": [
#                 1.8577502809856008,
#                 2.0185653090815996,
#                 1.8606205288960003,
#                 2.0812477459456007,
#                 2.1182490355456
#             ],
#             "objective_value": 0.012453536470413844,
#             "reduction": -16.02688244565853
#         },
#         "21": {
#             "design_vector": [
#                 1.8577502809856008,
#                 2.0185653090815996,
#                 1.8606205288960003,
#                 2.0812477459456007,
#                 2.1182490355456
#             ],
#             "objective_value": 0.012453536470413844,
#             "reduction": -16.02688244565853
#         },
#         "22": {
#             "design_vector": [
#                 1.8577502809856008,
#                 2.0185653090815996,
#                 1.8606205288960003,
#                 2.0812477459456007,
#                 2.1182490355456
#             ],
#             "objective_value": 0.012453536470413844,
#             "reduction": -16.02688244565853
#         }
#     }

# import matplotlib.pyplot as plt
# import numpy as np

# # Extract the iteration keys and the corresponding objective values
# iterations = list(map(int, data.keys()))
# design_vectors = np.array([data[str(key)]["design_vector"] for key in iterations])
# objective_values = np.array([data[str(key)]["objective_value"] for key in iterations])
# reduction = np.array([data[str(key)]["objective_value"] for key in iterations])

# # Plot the objective values over the iterations
# fig, axs = plt.subplots(2, 1, figsize=(10, 15), sharex=True)

# axs[0].plot(iterations, objective_values, marker='o', color='b')
# # axs[0].set_xlabel('Iteration')
# axs[0].set_ylabel(r"||$\Delta V$|| [m/s]")
# axs[0].set_title('Objective values')
# axs[0].grid(alpha=0.5, linestyle='--')

# for i in range(design_vectors.shape[1]):
#     axs[1].plot(iterations, design_vectors[:, i], marker='o', label=f'State {i+1}')
# axs[1].set_xlabel('Iteration')
# axs[1].set_ylabel("Arc length [days]")
# axs[1].set_title('Design vector history')
# axs[1].grid(alpha=0.5, linestyle='--')

# plt.legend(loc="upper right")
# plt.show()

import numpy as np
from scipy.optimize import minimize

# Example objective function
def objective_function(x):
    # Replace this with your actual objective function
    return np.sum((x - np.arange(1, len(x) + 1))**2)

# Initial guess (design vector of arbitrary length)
initial_guess = np.array([1.0] * 5)  # Example with a design vector of length 5

def generate_initial_simplex(initial_guess):
    # Perturbations based on parameter scale
    perturbation_scale = 0.2  # Adjust the scale as needed
    n = len(initial_guess)
    perturbations = np.eye(n) * perturbation_scale

    initial_simplex = [initial_guess]
    for i in range(n):
        vertex = initial_guess + perturbations[i]
        initial_simplex.append(vertex)
    initial_simplex = np.array(initial_simplex)

    return initial_simplex

initial_simplex = generate_initial_simplex(initial_guess)

# Run Nelder-Mead with custom initial simplex
result = minimize(objective_function, initial_guess, method='Nelder-Mead',
                  options={'initial_simplex': initial_simplex, 'xatol': 1e-8, 'fatol': 1e-8,
                           'maxiter': 2000, 'maxfev': 2000, 'adaptive': True})

print("Simplex at the start of the optimization process: \n", initial_simplex)
print("Simplex at the end of the optimization process: \n", result.final_simplex)
