import os
import sys
import numpy as np
from scipy.optimize import curve_fit

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

d_list = [9,15,21,27,35]
steady_state_density_list = []

path_fig = 'particles/fig/'
number_of_runs = 100

for d in d_list:

    param_dict["d"] = d

    fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))
    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])

    for k in range(number_of_runs):
        data_outcome_array, number_of_particles_array, forward_signal_1_array, anti_signal_1_array, stack_1_array, forward_signal_2_array, anti_signal_2_array, stack_2_array = SIGNAL2D(param_dict,option_dict,view_dict)

        if k == 0:
            number_of_particles_tot_array = number_of_particles_array
        else:
            number_of_particles_tot_array += number_of_particles_array

        density_of_particles_array = number_of_particles_array/(param_dict["d"]**2)
        density_of_particles_tot_array = number_of_particles_tot_array/(param_dict["d"]**2)

        ax.plot(density_of_particles_array,color='black',linewidth=0.2)

    steady_state_density = np.mean(density_of_particles_tot_array[-200:]/number_of_runs)
    steady_state_density_list.append(steady_state_density)

    ax.plot(density_of_particles_tot_array/number_of_runs,color='red',linewidth=2)

    ax.set_xlabel("iteration ($t$)",fontsize=7.5,labelpad=0.5)
    ax.set_ylabel("density of particles ($\\rho$)",fontsize=7.5,labelpad=0.5)

    ax.set_ylim(0,1)

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)

    ax.grid(False)

    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    plt.savefig(path_fig+"particles_d={}.pdf".format(d))
    plt.close()

fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))
plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])

X = np.array(d_list)
Y = np.array(steady_state_density_list)

def model(x, A, C, alpha):
    return A + C * x**alpha

popt, pcov = curve_fit(model, X, Y, p0=[0,0.1,0.5], maxfev=20000)

A, C, alpha = popt

# generate smooth fit curve
x_fit = np.linspace(min(X), 40, 200)
y_fit = A + C*x_fit**alpha
#print(y_fit)
#print(alpha)

ax.scatter(X,Y,facecolors='black', edgecolors='black',marker='D',s=14,linewidths=1,zorder=-10,label="S2D")
ax.plot(x_fit, y_fit, 'k:', label=rf"$\propto d^{{{alpha:.2f}}}$")

ax.set_xlabel("distance ($d$)",fontsize=7.5,labelpad=0.5)
ax.set_ylabel("density of particles ($\\rho_\\infty$)",fontsize=7.5,labelpad=0.5)

ax.set_xlim(0,40)
ax.set_ylim(0,1)

ax.tick_params(axis='both', which='major', labelsize=6)
ax.tick_params(axis='both', which='minor', labelsize=6)

ax.grid(False)
ax.legend(fontsize=7.5)

fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

plt.savefig(path_fig+"density_of_particles.pdf")
plt.close()