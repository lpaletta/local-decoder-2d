import os
import sys
import time

import numpy as np
import pickle

# --- Add parent directory to path to allow imports from sibling folders ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Output path for simulation results ---
OUT_PATH = "visualization/out/"

# --- Import core modules for simulation ---
from sgn2d_to_visualize import SIGNAL2D
from init import *

#############################################
######### General view parameters ###########
#############################################

from param import PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS, update_existing

# --- Default simulation settings ---
update_existing(PHYSICS_PARAMS,{"meas_error": False})
update_existing(DECODER_PARAMS,{"online": False})
update_existing(SIMULATION_PARAMS,{"recorded_variable": "View"})

#############################################
################# Examples ##################
#############################################

# Choose which configuration to simulate
example = "online"
#example = "pair_of_defects"
#example = "single_measurement_error"
#example = "artificial_defect"
#example = "test"

if example == "pair_of_defects":

    # --- Initialize a 21x21 data array with a pair of defects ---
    data_array = init_data_array(13)
    data_array[0, 3:5] = 1
    data_array[1:5, 3] = 1
    data_array[1::2, ::2] = np.nan # Stabilizer sites
    data_array[::2, 1::2] = np.nan # Stabilizer sites
    data_array = np.roll(data_array, axis=(0, 1), shift=(10, 6))

    update_existing(PHYSICS_PARAMS,{"code_distance": 13})
    update_existing(DECODER_PARAMS,{}) #Keep defaults
    update_existing(SIMULATION_PARAMS,{"time_run":6,"initial_data_array": data_array})

elif example == "artificial_defect":

    # --- Artificial defects (meas error) at the center (for time <= 7) ---
    artificial_defect = {
        0: [(22, 23)],
        1: [(22, 23)],
        2: [(22, 23)],
        3: [(22, 23)]
    }

    update_existing(PHYSICS_PARAMS,{"code_distance": 23})
    update_existing(DECODER_PARAMS,{}) #Keep defaults
    update_existing(SIMULATION_PARAMS,{"time_run":14,"artificial_defect": artificial_defect})

elif example == "single_measurement_error":

    # --- Single measurement error defect ---
    artificial_defect = {0: [(2, 2)]}

    update_existing(PHYSICS_PARAMS,{"code_distance": 5})
    update_existing(DECODER_PARAMS,{}) #Keep defaults
    update_existing(SIMULATION_PARAMS,{"time_run":4,"artificial_defect": artificial_defect})

elif example == "test":

    # --- Artificial defects (meas error) at the center (for time <= 7) ---
    artificial_defect = {
        0: [(22, 23)],
        1: [(22, 23)],
        2: [(22, 23)],
        3: [(22, 23)],
        4: [(22, 23)],
        5: [(22, 23)]
    }

    update_existing(PHYSICS_PARAMS,{"data_error_rate":0.01,"code_distance": 23})
    update_existing(DECODER_PARAMS,{"online": True,"max_stack":3})
    update_existing(SIMULATION_PARAMS,{"time_run":100})

if example == "online":

    update_existing(PHYSICS_PARAMS,{"data_error_rate":0.02,"code_distance": 13})
    update_existing(DECODER_PARAMS,{"online": True,"max_stack":np.inf})
    update_existing(SIMULATION_PARAMS,{"time_run":100})


# --- Run the 2D signal simulation ---
configuration_history, step_history = SIGNAL2D(PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS)

filename_1 = os.path.join(OUT_PATH, "configuration_history.npy")
filename_2 = os.path.join(OUT_PATH, "step_history.npy")
with open(filename_1, 'wb') as f:
    pickle.dump(configuration_history, f)
with open(filename_2, 'wb') as f:
    pickle.dump(step_history, f)