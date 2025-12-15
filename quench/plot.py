import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# use custom style
plt.style.use('rgplot')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_distance_to_code(dict_distance,path_fig):

    fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))
    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])

    for d in dict_distance.keys():

        #print(len(dict_distance[d]))
        Y = np.mean(dict_distance[d],axis=0)/d
        X = np.linspace(0, 2, num=Y.shape[0])
        #print(X.shape,Y.shape)
        
        line, = ax.plot(X,Y)
        ax.scatter(X,Y,color=line.get_color(),marker='o',s=8,label="d = "+str(d))
        

    #ax.set_xlim((-65,63))
    #ax.set_xlim((-l//2,l//2-1))
    ax.set_ylim((0,0.8))

    ax.set_xlabel("Time/distance ($t/d$)",fontsize=7.5,labelpad=0.5)
    ax.set_ylabel("distance to codespace ($\delta/d$)",fontsize=7.5,labelpad=0.5)

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)

    ax.grid(False)
    ax.legend(loc="upper right",frameon=False,fontsize=7.5)

    plt.savefig(path_fig+"distance_to_codespace.pdf")
    plt.close()