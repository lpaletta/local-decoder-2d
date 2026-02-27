import os
import sys
import time

import cProfile
import pstats
import io

import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support

# --- Add parent directory to path to allow imports from sibling folders ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Import core modules for simulation ---
from sgn2d import SIGNAL2D
from mwpm import *
from view import *

# --- Import simulation parameters ---
from param import PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS, update_existing

# --- Core simulation parameters ---

PHYSICS_UPDATE = {"code_distance":50,
                  "data_error_rate":0.005,
                  "data_error_rate":0.005,
                  "meas_error_rate":True}

SIMULATION_UPDATE = {"time_run":1000,
                     "time_job":10**5,
                     "time_simulation_max":10**6,
                     "error_count_max":100,
                     "recorded_variable":"Data"}


update_existing(PHYSICS_PARAMS,PHYSICS_UPDATE)
update_existing(SIMULATION_PARAMS,SIMULATION_UPDATE)

pr = cProfile.Profile()
pr.enable()

# --- Run the code you want to profile ---
SIGNAL2D(PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS)

pr.disable()
s = io.StringIO()
sortby = 'cumulative'  # other options: 'tottime', 'ncalls'
ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
ps.print_stats(100)  # print top 100 lines
print(s.getvalue())