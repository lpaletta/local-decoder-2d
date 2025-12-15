import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from plot import *

path_in = "field/data/"
path_fig = "field/fig/"

# Pour recharger le tableau depuis le fichier
field_array_1 = np.load(path_in+'field_array_1.npy')
field_array_2 = np.load(path_in+'field_array_2.npy')
field_array = field_array_1 + field_array_2

np.save(path_in+'field_array.npy', field_array)