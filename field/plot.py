import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# use custom style
plt.style.use('rgplot')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_attraction_bassin(field_array_mean,path_fig):

    # Suppose vecs is your (d,d,2) array
    l = field_array_mean.shape[0]

    # Grid coordinates (array indices)
    x = np.arange(l)-l//2
    y = np.arange(l)-l//2
    X, Y = np.meshgrid(x, y, indexing="ij")

    cond = ((np.absolute(X) + np.absolute(Y)) < l/2).astype(np.int8)

    U = field_array_mean[..., 0]*cond  # x-component
    V = field_array_mean[..., 1]*cond  # y-component

    fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))
    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])

    ax.quiver(Y, X, -U, -V, angles="xy", scale_units="xy", scale=0.8)

    ax.set_xlabel("x",fontsize=7.5,labelpad=0.5)
    ax.set_ylabel("y",fontsize=7.5,labelpad=0.5)

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)

    ax.grid(False)
    #ax.legend(fontsize=7.5)

    plt.savefig(path_fig+"field.pdf")
    plt.close()

def plot_interaction_strength(field_array_mean,path_fig):

    field_array_norm = np.absolute(field_array_mean[...,0])+np.absolute(field_array_mean[...,1])
    #field_array_norm = np.sqrt(field_array_mean[...,0]**2+field_array_mean[...,1]**2)
    l = field_array_norm.shape[0]

    x = np.arange(l)-l//2
    y = np.arange(l)-l//2
    X, Y = np.meshgrid(x, y, indexing="ij")

    cond = ((np.absolute(X) + np.absolute(Y)) < l/2).astype(np.int8)
    field_array_norm = field_array_norm*cond

    latt_interaction_strength = field_array_norm[l//2,:]
    vert_interaction_strength = field_array_norm[:,l//2]
    diag_interaction_strength = np.diag(field_array_norm)

    fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))
    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])

    ax.scatter(np.arange(l)-l//2,latt_interaction_strength,marker='o',edgecolors=colors[0],s=3,label="Latt")
    ax.scatter(np.arange(l)-l//2,vert_interaction_strength,marker='o',edgecolors=colors[1],s=3,label="Vert")
    ax.scatter(np.arange(2*l,step=2)-l,diag_interaction_strength,marker='o',edgecolors=colors[2],s=3,label="Diag")

    #ax.set_xlim((-65,63))
    ax.set_xlim((-l//2,l//2-1))
    ax.set_ylim((0,1))

    ax.set_xlabel("distance to defect ($\Delta$)",fontsize=7.5,labelpad=0.5)
    ax.set_ylabel("distance to defect ($\Gamma$)",fontsize=7.5,labelpad=0.5)

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)

    ax.grid(False)
    ax.legend(loc="upper right",frameon=False,fontsize=7.5)

    plt.savefig(path_fig+"interaction_strength.pdf")
    plt.close()

def plot_interaction_strength_log(field_array_mean,path_fig):

    field_array_norm = np.absolute(field_array_mean[...,0])+np.absolute(field_array_mean[...,1])
    #field_array_norm = np.sqrt(field_array_mean[...,0]**2+field_array_mean[...,1]**2)
    l = field_array_norm.shape[0]

    x = np.arange(l)-l//2
    y = np.arange(l)-l//2
    X, Y = np.meshgrid(x, y, indexing="ij")

    cond = ((np.absolute(X) + np.absolute(Y)) < l/2).astype(np.int8)
    field_array_norm = field_array_norm*cond

    latt_interaction_strength = field_array_norm[l//2,:]
    vert_interaction_strength = field_array_norm[:,l//2]
    diag_interaction_strength = np.diag(field_array_norm)

    fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))
    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])

    ax.scatter(np.arange(l)-l//2,latt_interaction_strength,marker='o',edgecolors=colors[0],s=3,label="Latt")
    ax.scatter(np.arange(l)-l//2,vert_interaction_strength,marker='o',edgecolors=colors[1],s=3,label="Vert")
    ax.scatter(np.arange(2*l,step=2)-l,diag_interaction_strength,marker='o',edgecolors=colors[2],s=3,label="Diag")

    #ax.set_xlim((-65,63))
    ax.set_xlim((-l//2,l//2-1))
    ax.set_ylim((0.1,1))

    ax.set_yscale("log")

    ax.set_xlabel("distance to defect ($\Delta$)",fontsize=7.5,labelpad=0.5)
    ax.set_ylabel("distance to defect ($\Gamma$)",fontsize=7.5,labelpad=0.5)

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)

    ax.grid(False)
    ax.legend(loc="upper right",frameon=False,fontsize=7.5)

    plt.savefig(path_fig+"interaction_strength_log.pdf")
    plt.close()