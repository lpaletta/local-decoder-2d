import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

error_rate_list = [0.005]
d_list = [25]

#print(len(subroutines_list))

#error_rate_list = np.round(np.logspace(np.log10(0.004),np.log10(0.009),5,endpoint=False),4).tolist()

param_dict = {"T" : 1000,
              "d" : 15,
              "error_rate":0.005,
              "meas_error_rate":0.005,
              "anti_signal_velocity":2,
              "max_distance":3,
              "max_counter":2,
              "init_data_array":np.array(())
              }

option_dict = {"error_bool":True,
               "meas_error_bool":True,
               "id_error_rate":True,
               "stabilisation_bool":False,
               "init_var":"Zeros"
               }


view_dict = {"record_var" : "Data"}

mc_dict = {
    "T_inner_job": 10000,
    "T_cum_max": 10**7,
    "number_of_errors_max": 100
}
