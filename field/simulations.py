import os
import sys
import time

import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

path_output = "field/data/"

from sgn2d import SIGNAL2D
from mwpm import *
from view import *

import param

field_array = [SIGNAL2D(param.param_dict, param.option_dict, param.view_dict) 
           for _ in range(50)]

np.save(path_output+'field_array.npy', field_array)