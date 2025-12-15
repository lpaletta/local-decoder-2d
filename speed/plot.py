import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# use custom style
plt.style.use('rgplot')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def time_to_half_correction(df,path_fig):

    fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))
    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])

    for stab_bool in [True, False]:
        df_stab = df[df["stab_bool"]==stab_bool]
        time_record_as_array = {int(d):np.vstack(df_stab[df_stab["d"] == d]["t_list"].to_numpy()) for d in df_stab["d"].unique()}

        for d in time_record_as_array.keys():
            print("{} unfinished runs".format(np.sum(time_record_as_array[d] == np.inf)))
            time_record_as_array[d] = np.where(time_record_as_array[d] == np.inf, 10*d, time_record_as_array[d])
        mean_time_record = {d:np.mean(time_record_as_array[d][np.isfinite(time_record_as_array[d])]) for d in time_record_as_array.keys()}

        X = np.array(list(mean_time_record.keys()))
        Y = np.array(list(mean_time_record.values()))

        if stab_bool:
            ax.plot(X,3*X/4,color='black',linestyle='dotted',label='code capacity',zorder=-1)

        #line, = ax.plot(X,Y)
        ax.scatter(X,Y,marker='o',s=8,label='with stabilisation' if stab_bool else 'without stabilisation')
            
        ax.set_xlim((0,130))
        ax.set_ylim((0,400))

        ax.set_xlabel("distance ($d$)",fontsize=7.5,labelpad=0.5)
        ax.set_ylabel("time to half correction ($t_{1/2}$)",fontsize=7.5,labelpad=0.5)

        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        ax.grid(False)
        ax.legend(loc="upper left",frameon=False,fontsize=7)

    plt.savefig(path_fig+"time_to_half_correction.pdf")
    plt.close()

"""def distribution_time_to_half_correction(df,path_fig):

    for d in df["d"].unique():

        fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))
        plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])

        df_d = df[df["d"]==d]
        df_d = df_d.drop(['d'], axis=1)
        df_exploded = df_d.explode("t_list")
        df_exploded["t_list"] = df_exploded["t_list"].astype(float)  # make numeric
        
        Y1 = df_exploded[df_exploded["stab_bool"]==True]["t_list"].to_numpy()
        ax.hist(Y1, bins=20, alpha=0.5, density=True, label="with stabilisation")
        Y2 = df_exploded[df_exploded["stab_bool"]==False]["t_list"].to_numpy()
        ax.hist(Y2, bins=20, alpha=0.5, density=True, label="without stabilisation")
        print(Y1)

        ax.set_ylim((0,1))
        ax.set_xlabel("time to half correction ($t_{1/2}$)")
        ax.set_ylabel("Frequency")

        ax.legend(loc="upper left",frameon=False,fontsize=7)

        #ax.histplot(data=df_exploded, x="t_list", hue="stab_bool", kde=True, element="step")
    
        plt.savefig(path_fig+"distribution_time_to_half_d={}.pdf".format(d))"""


def distribution_time_to_half_correction(df, path_fig):
    df["t_list"] = df.apply(
    lambda row: [(x - 3*row["d"]//4)/ row["d"] for x in row["t_list"]],
    axis=1
)

    for d in df["d"].unique():
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
        plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
        #print(df)
        df_d = df[df["d"] == d].drop(['d'], axis=1)
        df_exploded = df_d.explode("t_list")
        df_exploded["t_list"] = df_exploded["t_list"].astype(float)

        bin_edges = np.arange(0, 13, 1)

        for i, stab in enumerate([True, False]):
            ax = axes[i]
            Y = df_exploded[df_exploded["stab_bool"] == stab]["t_list"].to_numpy()
            counts, bins = np.histogram(Y, bins=bin_edges)
            ax.bar((bins[:-1] + bins[1:]) / 2, counts / counts.sum(),
                   width=(bins[1] - bins[0]) * 0.9, alpha=0.7,
                   color='C0' if stab else 'C1')
            ax.set_xlim((0, 12))
            ax.set_ylim((0, 1))
            ax.set_xlabel("scaled time to half correction ($t_{1/2}/d$)",fontsize=7.5,labelpad=0.5)
            ax.set_ylabel("Fraction of entries",fontsize=7.5,labelpad=0.5)
            title = "with stabilisation" if stab else "without stabilisation"
            ax.set_title(title, fontsize=7.5)

            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.tick_params(axis='both', which='minor', labelsize=6)

        plt.tight_layout()
        plt.savefig(path_fig + f"distribution_time_to_half_d={d}.pdf")
