import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgb
import matplotlib.patches as patches

# use custom style
plt.style.use('rgplot')

grey = to_rgb("#D6D6D6")
light_grey = to_rgb("#F1F1F1")
magenta = to_rgb("#CE93D8")

def plot_f_E(df,fit_bool,key,plim_fit,path_fig):

    fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))

    df_plot = df.copy()
    df_plot = df_plot.reset_index()

    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors[5] = colors[4]
    colors[4] = magenta
    i=0

    for d, group in df_plot.groupby("d"):
        X = group["error_rate"].to_numpy()
        Y = group["pL"].to_numpy()
        S = group["sigma"].to_numpy()
        
        if d <= 6:
            ax.errorbar(X,Y,yerr=S,fmt="o",color=grey,capsize=2.5,markersize=5,label="$d=%i$"%d)
        elif d > 6:
            ax.errorbar(X,Y,yerr=S,fmt="o",color=colors[i],capsize=2.5,markersize=5,label="$d=%i$"%d)
            #ax.errorbar(X,Y,yerr=S,fmt="o",capsize=2.5,markersize=5,label="$d=%i$"%d)
            i+=1

    if fit_bool:
        i=0
        for name, group in df_plot.groupby("d"):
            group = group[group["error_rate"]<plim_fit]
            if name <= 6:
                ax.plot(group["error_rate"].to_numpy(), group[key].to_numpy(),color=grey, linestyle='dotted')
            elif name > 6:
                ax.plot(group["error_rate"].to_numpy(), group[key].to_numpy(),color=colors[i], linestyle='dotted')
                #ax.plot(group["error_rate"].to_numpy(), group[key].to_numpy(), color = 'black', linestyle='dotted')
                i+=1

    ax.set_xlabel("physical error ($\\varepsilon = \\varepsilon_d = \\varepsilon_m$)",fontsize=7.5,labelpad=0.5)
    ax.set_ylabel("logical error ($\\varepsilon_L$)",fontsize=7.5,labelpad=0.5)

    ax.set_xscale("log")
    #ax.set_xlim(10**(-2),0.1)
    #ax.set_xlim(10**(-3),0.01)
    ax.set_xlim(10**(-3),10**(-2))

    ax.set_yscale("log")
    ax.set_ylim(10**(-8),10**(-3))
    #ax.set_ylim(10**(-4),10**(-1))

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)

    ax.grid(False)

    ax.legend(loc="upper left",frameon=False,handletextpad=0.1,labelspacing=0.25,borderpad=0.25,fontsize=7)

    if not fit_bool:
        plt.savefig(path_fig+"/logical_f_E_wo_fit.pdf")
    elif fit_bool:
        plt.savefig(path_fig+"/logical_f_E_{}.pdf".format(key))
    plt.close()

def plot_gamma_d(df,fit_bool,key,path_fig):

    fig, ax = plt.subplots(1,1,figsize=(1.9,2.4))

    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    blue = colors[0]
    black = 'black'
        
    d_reduced_list = list(set(df["d"].to_list()))
    df_plot = df[df["d"].isin(d_reduced_list)] 

    X = df_plot["d"].to_numpy()
    Y = df_plot["gamma_d"].to_numpy()
    
    ax.scatter(X,Y,facecolors=blue, edgecolors=blue,marker='o',s=14,linewidths=1,zorder=-10,label="S2D")
    ax.plot(X,Y,color=blue,linewidth=1,zorder=-10)

    X = np.linspace(0,100,50)
    Y = (X+1)/2
    ax.plot(X,Y,color=black,linestyle="dashdot",zorder=-40,label=" $\\frac{d+1}{2}$")

    df  = df[df["d"]>15]

    if fit_bool:
        alpha = df["alpha"].iloc[0]
        beta = df["beta"].iloc[0]
        label="$\\alpha=%.2f, \\beta=%.2f$"%(alpha,beta)
        ax.plot(df["d"].to_numpy(),df[key].to_numpy(),color=black,linestyle='dotted',label=label)
        ax.set_title("$\\varepsilon_L = Ad(\\varepsilon/\\varepsilon_{th})^{\\gamma_d}$")

    ax.set_xlabel("distance ($d$)",labelpad=0.5,fontsize=7.5)
    ax.set_xlim(0,100)

    ax.set_ylabel("effective distance ($\\gamma_d$)",labelpad=0.5,fontsize=7.5)
    ax.set_ylim(0,40)

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)

    ax.grid(False)
    ax.legend(loc="lower right",frameon=False,fontsize=7)

    fig.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.1)

    if not fit_bool:
        plt.savefig(path_fig+"/gamma_f_d_wo_fit.pdf")
    elif fit_bool:
        plt.savefig(path_fig+"/gamma_f_d.pdf")
    plt.close()

def plot_estimate_f_n(df,df_proof,path_fig):
    fig, ax = plt.subplots(1,1,figsize=(2.55,2.4))

    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    blue = colors[0]
    yellow = colors[3]
    grey = colors[4]
    black = 'black'

    rect = patches.Rectangle((10**(-9), 0), 1, 100, linewidth=0, edgecolor=None, facecolor=light_grey,zorder=-40)
    ax.add_patch(rect)

    df_plot = df.copy()
    df_plot = df_plot.reset_index()

    error_rate=0.001
    group_0001=df_plot[df_plot["error_rate"]==error_rate]
    X_proof = df_proof[df_proof["error_rate"]==error_rate]["pL"].to_numpy()
    Y_proof = df_proof[df_proof["error_rate"]==error_rate]["d"].to_numpy()
    S_proof = df_proof[df_proof["error_rate"]==error_rate]["sigma"].to_numpy()
    ax.scatter(X_proof,Y_proof,facecolors='white', edgecolors=blue,marker='o',s=50,linewidths=1,zorder=-8)
    ax.errorbar(X_proof,Y_proof,xerr=S_proof,fmt="o",color=blue,capsize=3,markersize=6,zorder=-9)
    Y = group_0001["d"].to_numpy()
    X = group_0001["pL_fit"].to_numpy()
    Y_inf = group_0001[group_0001["pL_fit"]<10**(-9)]["d"].to_numpy()
    X_inf = group_0001[group_0001["pL_fit"]<10**(-9)]["pL_fit"].to_numpy()
    Y_sup = group_0001[group_0001["pL_fit"]>10**(-9)]["d"].to_numpy()
    X_sup = group_0001[group_0001["pL_fit"]>10**(-9)]["pL_fit"].to_numpy()
    ax.scatter(X_inf,Y_inf,facecolors='white', edgecolors=blue,marker='o',s=16,linewidths=1,zorder=-9)
    ax.scatter(X_sup,Y_sup,facecolors=light_grey, edgecolors=blue,marker='o',s=16,linewidths=1,zorder=-0)
    ax.plot(X,Y,color=blue,linewidth=1,zorder=-10)
    error_rate=0.01
    group_001=df_plot[df_plot["error_rate"]==error_rate]
    X_proof = df_proof[df_proof["error_rate"]==error_rate]["pL"].to_numpy()
    Y_proof = df_proof[df_proof["error_rate"]==error_rate]["d"].to_numpy()
    S_proof = df_proof[df_proof["error_rate"]==error_rate]["sigma"].to_numpy()
    ax.errorbar(X_proof,Y_proof,xerr=S_proof,fmt="o",color=blue,capsize=3,markersize=6,zorder=-9)
    Y = group_001["d"].to_numpy()
    X = group_001["pL_fit"].to_numpy()
    ax.scatter(X,Y,facecolors=blue, edgecolors=blue,marker='o',s=16,linewidths=1,zorder=-10)
    ax.plot(X,Y,color=blue,linewidth=1,zorder=-10)

    ax.scatter([],[],color=light_grey,linewidth=1,label="$\\varepsilon = 10^{-2}$")
    ax.scatter([],[],color=light_grey,linewidth=1,label="$\\varepsilon = 10^{-3}$")

    ax.set_xlabel("logical error ($\\varepsilon_L$)",fontsize=7.5,labelpad=0.5)
    ax.set_ylabel("distance ($d$)",fontsize=7.5,labelpad=5,rotation=-90)

    ax.set_xscale("log")
    ax.set_xlim(10**(-14),10**(-6))
    ax.set_ylim(0,100)

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.grid(False)

    ax.legend(loc="upper right",fontsize=7,frameon=False)

    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    plt.savefig(path_fig+"/logical_estimate_f_d.pdf")
    
    plt.close()