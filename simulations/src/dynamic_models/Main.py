# Standard
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Third party
import pytest
import pytest_html
from tudatpy.kernel import constants
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

# Define path to import src files
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Own
from tests import utils
from src.dynamic_models import validation_LUMIO
from src.dynamic_models import Interpolator, FrameConverter, TraditionalLowFidelity
from src.dynamic_models.full_fidelity import *
from src.dynamic_models.low_fidelity.three_body_problem import *
from src.dynamic_models.high_fidelity.point_mass import *
from src.dynamic_models.high_fidelity.point_mass_srp import *
from src.dynamic_models.high_fidelity.spherical_harmonics import *
from src.dynamic_models.high_fidelity.spherical_harmonics_srp import *
from src.estimation_models import estimation_model


mission_time = 10
mission_start_epoch = 60390
mission_end_epoch = 60390 + mission_time


# Argument settings for dynamic models to be used in estimation
simulation_start_epoch = mission_start_epoch
propagation_time = 1
package_dict = {"high_fidelity": ["point_mass"]}
# package_dict = {"low_fidelity": ["three_body_problem"], "high_fidelity": ["point_mass", "point_mass_srp"]}
# package_dict = None
get_only_first = True
custom_initial_state = None

apriori_covariance=np.diag([1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2])**2

# truth_model = full_fidelity.HighFidelityDynamicModel(*params[:2])


estimation_errors_list = list()
formal_errors_list = list()
# mission_epoch = mission_start_epoch
while simulation_start_epoch < mission_end_epoch:

    # Define estimation settings given batch
    dynamic_model_objects = utils.get_dynamic_model_objects(simulation_start_epoch,
                                                            propagation_time,
                                                            package_dict=package_dict,
                                                            get_only_first=get_only_first,
                                                            custom_initial_state=custom_initial_state)

    truth_model = high_fidelity_point_mass_01.HighFidelityDynamicModel(simulation_start_epoch,
                                                                       propagation_time,
                                                                       custom_initial_state=custom_initial_state)

    # Obtain estimation results for given batch
    estimation_model_objects_results = utils.get_estimation_model_objects_results(dynamic_model_objects,
                                                                                  estimation_model,
                                                                                  custom_truth_model=truth_model,
                                                                                  apriori_covariance=apriori_covariance)

    estimation_model_objects_result = estimation_model_objects_results["high_fidelity"]["point_mass"]

    parameter_history = estimation_model_objects_result[0][0].parameter_history
    final_covariance = np.linalg.inv(estimation_model_objects_result[0][0].inverse_covariance)
    formal_errors = estimation_model_objects_results["high_fidelity"]["point_mass"][0][0].formal_errors
    estimation_errors = parameter_history[:,0]-parameter_history[:,-1]

    dynamic_model_object = dynamic_model_objects["high_fidelity"]["point_mass"][0]

    # print("params", params)
    # print("time", estimation_model_objects_result[0][-1])
    # print("custom_initial_states", custom_initial_state)
    # print("final_parameters", parameter_history[:,-1])
    # print("final_covariance", final_covariance[0,0])
    print("formal_errors", formal_errors)
    print("estimation_errors", estimation_errors)

    print("=================")

    estimation_errors_list.append(estimation_errors)
    formal_errors_list.append(formal_errors)

    # Update settings for next batch
    simulation_start_epoch += 1
    custom_initial_state = parameter_history[:,-1]
    apriori_covariance = final_covariance


estimation_errors_history = np.array(estimation_errors_list)
formal_errors_history = np.array(formal_errors_list)

plt.plot(estimation_errors_history[:,:3], color="blue")
plt.plot(estimation_errors_history[:,:3]+formal_errors_history[:,:3], color="red", ls="--")
plt.plot(estimation_errors_history[:,:3]-formal_errors_history[:,:3], color="red", ls="--")
plt.show()
