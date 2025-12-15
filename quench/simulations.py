import os
import sys
import time

import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

path_output = "quench/data/"

from sgn2d import SIGNAL2D
from mwpm import *
from view import *

import param

dict_trajectory = {}

#visualize_data(init_error_array(32,"diagonal"),np.zeros((64,64)),np.zeros((64,64)))

for k in range(len(param.d_list)):
    d = param.d_list[k]
    param.param_dict["d"] = d
    print("Processing d =",d)
    param.param_dict["init_data_array"] = init_error_array(d,"diagonal")

    #param.param_dict["error_bool"], param.param_dict["meas_error_bool"] = True, True
    defect_trajectory_list = [SIGNAL2D(param.param_dict, param.option_dict, param.view_dict) 
    for _ in range(5)]

    dict_trajectory[d] = defect_trajectory_list

    #param.param_dict["error_bool"], param.param_dict["meas_error_bool"] = False, False
    #defect_trajectory_list = [SIGNAL2D(param.param_dict, param.option_dict, param.view_dict) 
    #for _ in range(1)]

    #dict_trajectory[d] = defect_trajectory_list

np.save(path_output+'data_trajectory.npy', dict_trajectory)