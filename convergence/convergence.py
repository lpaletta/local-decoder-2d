import os
import sys
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgn2d import SIGNAL2D
from param import *
from view import *
from mwpm import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgb
import matplotlib.patches as patches

# use custom style
plt.style.use('rgplot')

path_fig = 'convergence/fig/'
number_of_runs = 1000
#d_list = [7,9,11]
d_list = [11,21,31,41,51,61,71,81,91]

fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))
plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = colors[0]
black = 'black'

mean_time_no_particles_array = np.zeros(len(d_list))
mean_time_no_syndrome_array = np.zeros(len(d_list))

for i in range(len(d_list)):

    time_no_syndrome_array = np.zeros(number_of_runs)
    time_no_particles_array = np.zeros(number_of_runs)

    d = d_list[i]
    param_dict["d"] = d

    print("Proceed to: \n d={}".format(d))
    time_i = time.time()

    for k in range(number_of_runs):
        time_no_syndrome_array[k], time_no_particles_array[k] = SIGNAL2D(param_dict,option_dict,view_dict)

    time_f = time.time()
    print("Completed in: \n {}s".format(np.around(time_f-time_i,2)))

    mean_time_no_syndrome_array[i] = np.mean(time_no_syndrome_array)
    mean_time_no_particles_array[i] = np.mean(time_no_particles_array)

# Save
np.savez("convergence/data/data_temp.npz", a=mean_time_no_syndrome_array, b=mean_time_no_particles_array)

# Load
#data = np.load("convergence/data/data.npz")
#mean_time_no_syndrome_array = data["a"]
#mean_time_no_particles_array = data["b"]



#mean_time_no_syndrome_array = np.concatenate((mean_time_no_syndrome_array_1, mean_time_no_syndrome_array_2))
#mean_time_no_particles_array = np.concatenate((mean_time_no_particles_array_1, mean_time_no_particles_array_2))

#np.savez("convergence/data/data.npz", a=mean_time_no_syndrome_array, b=mean_time_no_particles_array)

ax.scatter(d_list,mean_time_no_particles_array,facecolors=black,marker='D',s=18,linewidths=1,label="mean time to zero")
ax.scatter(d_list,mean_time_no_syndrome_array,facecolors=blue,marker='o',s=24,linewidths=1,label="mean time to code space")

ax.set_xlabel("distance ($d$)",fontsize=7.5,labelpad=0.5)
ax.set_ylabel("Time ($t$)",fontsize=7.5,labelpad=0.5)

ax.legend(loc="upper left",frameon=False,fontsize=7)

ax.set_xlim(0,160)
ax.set_ylim(0,100)

ax.tick_params(axis='both', which='major', labelsize=6)
ax.tick_params(axis='both', which='minor', labelsize=6)

ax.grid(False)

fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

plt.savefig(path_fig+"convergence.pdf")
plt.close()
