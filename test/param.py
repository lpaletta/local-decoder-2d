import numpy as np
from init import *

data_array = init_data_array(29)
data_array[0,3:31] = 1
data_array[0:31,3] = 1
data_array[1::2, ::2] = np.nan  # Odd rows, even columns
data_array[::2, 1::2] = np.nan  # Even rows, odd columns
data_array = np.roll(data_array,axis=(0,1),shift=(6,6))

custom_error = init_data_array(29)
custom_error[30,4] = 1
#custom_error[30,6] = 1
custom_error[31,5] = 1
custom_error[33,5] = 1
#custom_error[33,3] = 1
#custom_error[15,3] = 1
custom_error[1::2, ::2] = np.nan  # Odd rows, even columns
custom_error[::2, 1::2] = np.nan  # Even rows, odd columns
custom_error = np.roll(custom_error,axis=(0,1),shift=(6,6))


param_dict = {"T" :200,
              "d" : 29,
              "error_rate":0.003,
              "meas_error_rate":0.003,
              "anti_signal_velocity":2,
              "max_distance":3,
              "max_counter":2,
              "init_data_array":data_array,#np.array(())
              "custom_error":custom_error,
              }

option_dict = {"error_bool":True,
               "meas_error_bool":True,
               "id_error_rate":True,
               "stabilisation_bool":True,
               "init_var":"Specified"
               }

view_dict = {"record_var" : "Stability"}
