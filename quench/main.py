import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from param import *
from plot import *
from mwpm import *

path_in = "quench/data/"
path_fig = "quench/fig/"


# Pour recharger le tableau depuis le fichier
dict_trajectory = np.load(path_in+'data_trajectory.npy',allow_pickle=True).item()
dict_distance = {}

for d in dict_trajectory.keys():
    print("Processing d =",d)
    trajectory_list = dict_trajectory[d]

    matching = get_matching(d)
    H = get_parity_check_matrix(d)

    distance_trajectory_list = []

    for i in range(len(trajectory_list)):
        if i%10==0:
            print(" * Processing trajectory",i,"/",len(trajectory_list))
        trajectory = trajectory_list[i]
        single_distance_trajectory = []
        for j in range(len(trajectory)):
            x = trajectory[j]
            single_distance_trajectory.append(int(np.sum(matching.decode(H@get_data_as_vector(x)%2))))

        distance_trajectory_list.append(single_distance_trajectory)

    dict_distance[d] = distance_trajectory_list

np.save(path_in+'distance_trajectory.npy', dict_distance)
dict_distance = np.load(path_in+'distance_trajectory.npy',allow_pickle=True).item()

plot_distance_to_code(dict_distance,path_fig)