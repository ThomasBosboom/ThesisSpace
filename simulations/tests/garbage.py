import numpy as np

# Initial vector
initial_vector = [1, 1, 1, 1, 1, 1, 1]


rng = np.random.default_rng(seed=1)

for _ in range(10):

    print(rng.integers(1000))