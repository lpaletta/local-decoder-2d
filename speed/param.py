import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from init import *
from view import *

#d_list = [160]
d_list = [16,32,64,96,128]

param_dict = {"T" : 50,
              "d" : 64,
              "error_rate":0.001,
              "meas_error_rate":0,
              "anti_signal_velocity":2,
              "max_distance":5,
              "max_counter":3,
              "init_data_array":np.array(()),
              }

option_dict = {"error_bool":True,
               "meas_error_bool":False,
               "id_error_rate":False,
               "stabilisation_bool":False,
               "init_var":"Specified"
               }

view_dict = {"record_var" : "Speed"}
