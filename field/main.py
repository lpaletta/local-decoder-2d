import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from plot import *

path_in = "field/data/"
path_fig = "field/fig/"

# Pour recharger le tableau depuis le fichier
field_array = np.load(path_in+'field_array.npy')

field_array_mean = np.mean(field_array,axis=(0,1))

plot_attraction_bassin(field_array_mean,path_fig)
plot_interaction_strength(field_array_mean,path_fig)
plot_interaction_strength_log(field_array_mean,path_fig)