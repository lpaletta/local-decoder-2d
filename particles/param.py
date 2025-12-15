import numpy as np
from init import *

#data_array = init_data_array(14)
#data_array[0:8,1] = 1
#data_array[8,1:7] = 1
#data_array[1::2, ::2] = np.nan  # Odd rows, even columns
#data_array[::2, 1::2] = np.nan  # Even rows, odd columns
#data_array = np.roll(data_array,axis=(0,1),shift=(4,6))

param_dict = {"T" : 1000,
              "d" : 15,
              "error_rate":0.005,
              "meas_error_rate":0.005,
              "anti_signal_velocity":2,
              "max_self_stack":5,
              "max_counter":2,
              "init_data_array":np.array(())
              }

option_dict = {"error_bool":True,
               "meas_error_bool":True,
               "id_error_rate":True,
               "stabilisation_bool":True,
               "init_var":"Random Error"
               }

view_dict = {"record_var" : "Analysis"}
