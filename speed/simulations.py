import os
import sys
import time

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, freeze_support

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

path_output = "speed/data/"

from sgn2d import SIGNAL2D
from mwpm import *
from view import *

import param

rows = []

#visualize_data(init_error_array(32,"diagonal"),np.zeros((64,64)),np.zeros((64,64)))

for stab_bool in [True,False]:
    param.option_dict["stabilisation_bool"] = stab_bool
    for k in range(len(param.d_list)):
        d = param.d_list[k]
        param.param_dict["d"] = d
        print("Processing d =",d)
        param.param_dict["init_data_array"] = init_error_array(d,"diagonal")

        param.param_dict["matching"] = get_matching(d)
        param.param_dict["H"] = get_parity_check_matrix(d)

        t_list = [SIGNAL2D(param.param_dict, param.option_dict, param.view_dict) - (d//2) for _ in range(50)]

        rows.append({"stab_bool": stab_bool, "d": d, "t_list": t_list})

df = pd.DataFrame(rows).set_index(["stab_bool", "d"])

df.to_csv(path_output + "time_record.csv")
df.to_pickle(path_output + "time_record.pkl")
#np.save(path_output+'time_record.npy', time_record)