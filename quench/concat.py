import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from plot import *

path_in = "quench/data/"
path_fig = "quench/fig/"

# Pour recharger le tableau depuis le fichier
A = np.load(path_in+'distance_trajectory_2.npy',allow_pickle=True).item()
B = np.load(path_in+'distance_trajectory_1.npy',allow_pickle=True).item()

for d in A.keys():
    A[d] = A[d] + B[d]

np.save(path_in+'distance_trajectory.npy', A)