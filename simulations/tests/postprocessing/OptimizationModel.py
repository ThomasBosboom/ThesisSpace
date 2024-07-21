import numpy as np
import os
import sys
import copy
import scipy as sp
import json
import psutil

# Define path to import src files
file_directory = os.path.realpath(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
for _ in range(3):
    file_directory = os.path.dirname(file_directory)
    sys.path.append(file_directory)

from tests import utils


def func1(x):
    return np.sum([tup[-1]-tup[0] for tup in x]), [0, 0, 0, 0]

class Particle:
    def __init__(self, x0, seed=0):

        self.num_dimensions = len(x0)

        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        self.seed = seed

        self.rng = np.random.default_rng(seed=self.seed)

        for i in range(0,self.num_dimensions):
            self.velocity_i.append(self.rng.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc, window):
        self.err_i=costFunc(window)[0]
        self.individual_corrections = costFunc(window)[1]

    # check to see if the current position is an individual best
    def check_best(self):
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,self.num_dimensions):
            rng = np.random.default_rng(seed=self.rng.integers(0, 1000))
            r1 = rng.random()
            r2 = rng.random()
            # r1 = self.rng.integers(0, 1000)
            # r2 = self.rng.integers(0, 1000)
            # r1=random.random()
            # r2=random.random()
            # print(self, r1, r2)

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        bounds = [bounds for _ in range(len(self.position_i))]
        for i in range(0,self.num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]



class OptimizationModel:

    def __init__(self, json_settings={"save_dict": True, "current_time": float, "file_name": str}, custom_input={}, **kwargs):

        self.duration = 28
        self.arc_length = 1
        self.arc_interval = 3
        self.mission_start_epoch = 60390
        self.max_iterations = 10
        self.bounds = (0.1, 2)
        self.kwargs = {}

        self.optimization_method = "Nelder-Mead"

        self.design_vector_type = 'arc_lengths'
        self.initial_simplex_perturbation = -0.5

        self.num_particles = 10
        self.seed = 0

        self.iteration = 0
        self.total_iterations = 0
        self.iteration_history = {}
        self.intermediate_iteration_history = {}
        self.initial_objective_value = None
        self.best_objective_value = None
        self.latest_objective_value = None
        self.latest_individual_corrections = None
        self.run_counter = 0
        self.num_runs = 1
        self.evaluation_threshold = 14
        self.show_evaluations_in_terminal = False

        for key, value in json_settings.items():
            setattr(self, key, value)

        self.use_custom_input = False
        if custom_input:
            for key, value in custom_input.items():
                setattr(self, key, value)
            self.use_custom_input = True


        for key, value in kwargs.items():
            # if hasattr(self, key):
            setattr(self, key, value)
            self.kwargs[key] = value

        self.options = {'maxiter': self.max_iterations+1, 'disp': False, "adaptive": True}
        self.total_iterations += self.max_iterations

        if self.use_custom_input:
            self.iteration = 0
            self.run_counter = 0
            self.options = {'maxiter': self.total_iterations+1, 'disp': False, "adaptive": True}


    def load_from_json(self, time_tag, folder_name="optimization_analysis"):

        folder = os.path.join(os.path.dirname(__file__), "dicts")
        folder = os.path.join(folder, folder_name)
        filename=f'{time_tag}_optimization_analysis.json'

        file_path = os.path.join(folder, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)

        return data


    def convert_ndarray(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_ndarray(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarray(elem) for elem in obj]
        else:
            return obj


    def save_to_json(self):
        if self.save_dict:
            converted_vars = self.convert_ndarray(vars(self))
            utils.save_dict_to_folder(dicts=[converted_vars], labels=[f"{self.current_time}_optimization_analysis"], custom_sub_folder_name=self.file_name)


    def generate_observation_windows(self, design_vector):

        observation_windows = []
        current_time = self.mission_start_epoch

        if self.design_vector_type == 'arc_lengths':
            arc_interval = self.arc_interval
            for arc_length in design_vector:
                if current_time + arc_length > self.mission_start_epoch + self.duration:
                    arc_length = self.mission_start_epoch + self.duration - current_time
                observation_windows.append((current_time, current_time + arc_length))
                current_time += arc_length + arc_interval

                if current_time >= self.mission_start_epoch + self.duration:
                    break

        elif self.design_vector_type == 'arc_intervals':
            arc_length = self.arc_length
            for arc_interval in design_vector:
                end_time = current_time + arc_length
                if end_time > self.mission_start_epoch + self.duration:
                    end_time = self.mission_start_epoch + self.duration
                observation_windows.append((current_time, end_time))
                current_time = end_time + arc_interval

                if current_time >= self.mission_start_epoch + self.duration:
                    break

        else:
            raise ValueError("Invalid design_vector_type. Must be 'arc_lengths' or 'arc_intervals'.")

        return observation_windows


    def generate_initial_design_vector(self):

        initial_observation_windows = []
        current_time = 0
        while current_time < self.duration:
            initial_observation_windows.append((current_time, current_time + self.arc_length))
            current_time += self.arc_length + self.arc_interval

        for arc_set in initial_observation_windows:
            if arc_set[1]+self.bounds[1] >= self.duration:
                initial_observation_windows.remove(arc_set)
                break

        initial_design_vector = np.ones(len(initial_observation_windows))
        if self.design_vector_type == 'arc_lengths':
            initial_design_vector *= self.arc_length
        if self.design_vector_type == 'arc_intervals':
            initial_design_vector *= self.arc_interval

        return initial_design_vector


    def generate_initial_simplex(self, initial_design_vector):

        n = len(initial_design_vector)

        perturbations = np.eye(n)*self.initial_simplex_perturbation

        initial_simplex = [initial_design_vector]
        for i in range(n):
            vertex = initial_design_vector + perturbations[i]
            initial_simplex.append(vertex)
        initial_simplex = np.array(initial_simplex)

        return initial_simplex


    def generate_iteration_history_entry(self, design_vector, objective_value, initial_objective_value, individual_corrections=None):

        # print("Generate entry: ", design_vector, objective_value, initial_objective_value)

        iteration_history_entry = {
            'design_vector': design_vector,
            'objective_value': objective_value,
            'objective_value_annual': objective_value*365/(self.duration-self.evaluation_threshold),
            'reduction': (objective_value-initial_objective_value)/initial_objective_value*100
            }

        if individual_corrections is not None:
            iteration_history_entry.update({'individual_corrections': individual_corrections})

        return iteration_history_entry


    def has_intermediate_iteration_history(self, iteration, run_counter):

        if str(iteration) in self.intermediate_iteration_history.keys():
            if str(run_counter) in self.intermediate_iteration_history[str(iteration)].keys():
                return True
        return False


    def get_cached_objective_value(self, iteration, run_counter):
        history = self.intermediate_iteration_history[str(iteration)][str(run_counter)]
        return history["objective_value"]


    def particle_swarm(self, objective_function):

        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # Initialize the design vector with the maximum number of arcs
        self.initial_design_vector = self.generate_initial_design_vector()

        # Define the initial observation windows
        self.initial_observation_windows = self.generate_observation_windows(self.initial_design_vector)

        # establish the swarm
        swarm=[]
        rng = np.random.default_rng(seed=self.seed)
        for i in range(0,self.num_particles):
            seed = rng.integers(0, 1000)
            swarm.append(Particle(self.initial_design_vector, seed=seed))

        # begin optimization loop
        i=0
        while i < self.total_iterations:

            # cycle through particles in swarm and evaluate fitness
            for j in range(0,self.num_particles):

                observation_windows = self.generate_observation_windows(swarm[j].position_i)
                if self.has_intermediate_iteration_history(i, j):
                    objective_value = self.get_cached_objective_value(i, j)
                    swarm[j].err_i = objective_value
                    print(f"Retrieving iteration {i}, particle {j} from cache....")

                else:
                    if i == 0 and i in self.intermediate_iteration_history:
                        objective_value = self.intermediate_iteration_history[i][0]["objective_value"]
                        design_vector = self.intermediate_iteration_history[i][0]["design_vector"]
                        individual_corrections = self.intermediate_iteration_history[i][0]["individual_corrections"]

                    else:
                        swarm[j].evaluate(objective_function, observation_windows)

                        objective_value = swarm[j].err_i.copy()
                        design_vector = swarm[j].position_i.copy()
                        individual_corrections = swarm[j].individual_corrections.copy()

                    # Initialize initial objective value
                    if self.initial_objective_value is None:
                        self.initial_objective_value = objective_value

                    self.latest_objective_value = objective_value
                    self.latest_individual_corrections = individual_corrections

                    # Save all intermediate function evaluations between iterations
                    if i not in self.intermediate_iteration_history:
                        self.intermediate_iteration_history[i] = {}

                    self.intermediate_iteration_history[i][j] = self.generate_iteration_history_entry(design_vector, objective_value, self.initial_objective_value, individual_corrections)

                    # if i == 0 and j == 0:
                    #     self.iteration_history[i] = self.generate_iteration_history_entry(design_vector, objective_value, self.initial_objective_value, individual_corrections)

                    # Update the best objective value and arc lengths for the current iteration
                    if self.best_objective_value is None or objective_value < self.best_objective_value:
                        self.best_objective_value = objective_value
                        self.best_design_vector = design_vector
                        self.best_observation_windows = observation_windows
                        self.best_individual_corrections = individual_corrections

                    self.save_to_json()

                # determine if current particle is the best (globally)
                swarm[j].check_best()
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,self.num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(self.bounds)

            # Only save the final result of each iteration
            if str(i) not in self.iteration_history:
                self.iteration_history[i] = self.generate_iteration_history_entry(self.best_design_vector, self.best_objective_value, self.initial_objective_value, self.best_individual_corrections)

            if self.show_evaluations_in_terminal:
                print(f"Callback iteration {i} =================")
                print(f"Design vector: \n", self.best_design_vector)
                print(f"Objective value: \n", self.best_objective_value)
                print(f"Reduction: \n", (self.best_objective_value-self.initial_objective_value)/self.initial_objective_value*100)
                print("===========================")

            self.save_to_json()

            i+=1

        print(pos_best_g)
        print(err_best_g)


    def nelder_mead(self, objective_function):

        def objective(design_vector):

            observation_windows = self.generate_observation_windows(design_vector)

            # Retrieve latest simplex information from the cache of previous run
            if self.has_intermediate_iteration_history(self.iteration, self.run_counter):
                objective_value = self.get_cached_objective_value(self.iteration, self.run_counter)
                print(f"Retrieving iteration {self.iteration}, run counter {self.run_counter} from cache....")

            else:
                objective_value = objective_function(observation_windows)[0]
                individual_corrections = objective_function(observation_windows)[1]

                # Initialize initial objective value
                if self.initial_objective_value is None:
                    self.initial_objective_value = objective_value

                self.latest_objective_value = objective_value
                self.latest_individual_corrections = individual_corrections

                # Save all intermediate function evaluations between iterations
                if self.iteration not in self.intermediate_iteration_history:
                    self.intermediate_iteration_history[self.iteration] = {}

                self.intermediate_iteration_history[self.iteration][self.run_counter] = self.generate_iteration_history_entry(design_vector, objective_value, self.initial_objective_value, individual_corrections)

                if self.iteration == 0 and self.run_counter == 0:
                    self.iteration_history[self.iteration] = self.generate_iteration_history_entry(design_vector, objective_value, self.initial_objective_value, individual_corrections)

                # Update the best objective value and arc lengths for the current iteration
                if self.best_objective_value is None or objective_value < self.best_objective_value:
                    self.best_objective_value = objective_value
                    self.best_design_vector = design_vector
                    self.best_observation_windows = observation_windows
                    self.best_individual_corrections = individual_corrections

            if self.show_evaluations_in_terminal:
                print("==============")
                print(f"Function summary: \nDesign vector: {design_vector} \nObjective: {objective_value} \nObservation windows: \n {observation_windows}")
                # print(psutil.virtual_memory())
                print("==============")

            self.run_counter += 1

            self.save_to_json()

            return objective_value


        def callback_function(x):

            self.iteration += 1
            self.run_counter = 0

            # Only save the final result of each iteration
            if str(self.iteration) not in self.iteration_history:
                self.iteration_history[self.iteration] = self.generate_iteration_history_entry(self.best_design_vector, self.best_objective_value, self.initial_objective_value, self.best_individual_corrections)

            if self.show_evaluations_in_terminal:
                print(f"Callback iteration {self.iteration} =================")
                print(f"Design vector: \n", self.best_design_vector)
                print(f"Objective value: \n", self.best_objective_value)
                print(f"Reduction: \n", (self.best_objective_value-self.initial_objective_value)/self.initial_objective_value*100)
                print("===========================")


        # Initialize the design vector with the maximum number of arcs
        initial_design_vector = self.generate_initial_design_vector()
        self.initial_design_vector = initial_design_vector.copy()

        # Define the initial observation windows
        self.initial_observation_windows = self.generate_observation_windows(initial_design_vector)

        # Define bounds for the design vector entries
        self.bounds_vector = [self.bounds for state in self.generate_initial_design_vector()]

        # Adjust the initial simplex for better convergence
        self.initial_simplex = self.generate_initial_simplex(initial_design_vector)
        self.options.update({"initial_simplex": self.initial_simplex})


        # Plotting preliminary details
        print("===========")
        print("Current time: \n", self.current_time)
        print("Design vector type: \n", self.design_vector_type)
        print("Initial design vector: \n", initial_design_vector)
        print("Initial simplex: \n", self.initial_simplex)
        print("Initial observation windows: \n", self.initial_observation_windows)
        print("Bounds: \n", self.bounds)
        print("===========")

        # Performing the optimization itself
        result = sp.optimize.minimize(
            fun=objective,
            callback=callback_function,
            x0=initial_design_vector,
            method=self.optimization_method,
            bounds=self.bounds_vector,
            options=self.options,
        )

        self.final_solution = result.x.tolist()

        self.save_to_json()


    def optimize(self, objective_function):

        if self.optimization_method == "Nelder-Mead":
            self.nelder_mead(objective_function)
        elif self.optimization_method == "Particle-Swarm":
            self.particle_swarm(objective_function)


if __name__ == "__main__":


    time_tag = 1

    # mission_start_epoch = 60390
    # arc_interval = 3
    # arc_length = 1
    # duration = 28

    kwargs = {
        "num_particles": 20,
        "max_iterations": 20,
        'bounds': [0.5, 2],
        'seed': 1,
        "json_settings": {"save_dict": True, "current_time": time_tag, "file_name": file_name},
        "show_evaluations_in_terminal": True,
        # "optimization_method": "Nelder-Mead",
        "optimization_method": "Particle-Swarm"
    }

    optimization_model = OptimizationModel(**kwargs)
    # optimization_results = optimization_model.load_from_json(time_tag=time_tag, folder_name=file_name)
    # optimization_model = OptimizationModel(custom_input=optimization_results, **kwargs)
    optimization_model.optimize(func1)
