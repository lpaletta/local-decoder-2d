import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from init import *
from view import *

#d = 256
d = 256

data_array = init_data_array(d)
data_array[2,1:d-1] = 1
data_array[2:d,d-1] = 1
data_array[1::2, ::2] = np.nan  # Odd rows, even columns
data_array[::2, 1::2] = np.nan  # Even rows, odd columns

coords = [(2,0),(2,2),(1,1),(1,3)]
rows, cols = zip(*coords)
values = [0,1,0,0]



#visualize_data(data_array,np.zeros_like(data_array),np.zeros_like(data_array))

param_dict = {"T" : 100,
              "d" : d,
              "error_rate":0.001,
              "meas_error_rate":0.001,
              "anti_signal_velocity":2,
              "max_self_stack":5,
              "max_counter":2,
              "init_data_array":data_array,
              "fixed_defect":(rows,cols,values),
              }

option_dict = {"error_bool":True,
               "meas_error_bool":True,
               "id_error_rate":True,
               "stabilisation_bool":True,
               "init_var":"Specified"
               }

view_dict = {"record_var" : "Field"}
