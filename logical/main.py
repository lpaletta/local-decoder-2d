import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from import_data import *
from fit import *
from plot import *

path, data_name, proof_name = "logical/data/", "data.csv", "proof.csv"
path_fig = "logical/fig/"

pth_guess = 0.008

df = import_data(path, data_name, "Logical")

########## Test ##########

var_name = "stabilisation_bool"

config_dict = {
    "stabilisation_bool": {True: "stab", False: "no_stab"},
}



for config in config_dict.keys():

    for key in config_dict[config].keys():
    
        df_config = df[df[config]==key]
        name = config_dict[config][key]
        path_fig_key = path_fig + name

        d_reduced_list = list(set(df["d"].to_list()))
        plot_f_E(df_config[df_config["d"].isin(d_reduced_list)],False,"",pth_guess,path_fig_key)

        ######### fit pL #########

        fit_pL_single_alg(df_config,pth_guess,pth_guess)
        add_pL_fit(df_config,"pL_fit")

        ######### fit gamma n #########


        fit_gamma_d_single_alg(df_config)
        add_gamma_d_fit(df_config,"gamma_d_fit")

        ######### plot pL #########

        d_reduced_list = [5,9,15,25,50,100]
        plot_f_E(df_config[df_config["d"].isin(d_reduced_list)],True,"pL_fit",pth_guess,path_fig_key)

        ######### plot gamma d #########

        plot_gamma_d(df_config,False,"gamma_d_fit",path_fig_key)
        plot_gamma_d(df_config,True,"gamma_d_fit",path_fig_key)

        ######### plot logical estimated #########



"""

dict_pth_guess = {"Signal":0.08,
                "Toom":0.08,
                "Shearing":0.08,
                "Harrington":0.03}

dict_plim_fit = dict_pth_guess
for key in dict_plim_fit:
    dict_plim_fit[key] = dict_plim_fit[key]/2

df = import_data(path, data_name, "Logical")

########## Test ##########

for alg_name in list(set(df["alg_name"].to_list())):
    df_single_alg = df[df["alg_name"]==alg_name]
    alg_name_lower = alg_name.lower()
    path_fig_single_alg = path_fig + alg_name_lower
    try:
        os.mkdir(path_fig_single_alg)
    except:
        print("Directory {} already exists.".format(alg_name_lower))

    if alg_name == "Signal":
        n_reduced_list = [5,9,15,25,50,100]
    elif alg_name == "Shearing":
        n_reduced_list = [6,10,16,24,50,100]
    else:
        n_reduced_list = list(set(df_single_alg["n"].to_list()))

    plot_f_E(df_single_alg[df_single_alg["n"].isin(n_reduced_list)],False,"",dict_plim_fit[alg_name],path_fig_single_alg)


######### fit pL #########

df_algs_fit = pd.DataFrame()

for alg_name in list(set(df["alg_name"].to_list())):
    print("Fit {}".format(alg_name))
    df_single_alg = df[df["alg_name"]==alg_name]
    df_single_alg_fit = fit_pL_single_alg(df_single_alg,dict_pth_guess[alg_name],dict_plim_fit[alg_name])
    df_algs_fit = pd.concat([df_algs_fit,df_single_alg_fit],ignore_index=True)

df = df.merge(df_algs_fit[["alg_name","n","error_rate","A","pth","gamma_n"]],left_on=["alg_name","n","error_rate"],right_on=["alg_name","n","error_rate"],how="outer")
df = add_pL_fit(df,"pL_fit")

######### fit gamma n #########


df_algs_fit = pd.DataFrame()

for alg_name in list(set(df["alg_name"].to_list())):
    df_single_alg = df[df["alg_name"]==alg_name]
    df_single_alg_fit = fit_gamma_n_single_alg(df_single_alg)
    df_algs_fit = pd.concat([df_algs_fit,df_single_alg_fit],ignore_index=True)

df = df.merge(df_algs_fit[["alg_name","n","error_rate","alpha","beta"]],left_on=["alg_name","n","error_rate"],right_on=["alg_name","n","error_rate"],how="outer")
df = add_gamma_n_fit(df,"gamma_n_fit")

######### plot pL #########

for alg_name in list(set(df["alg_name"].to_list())):
    df_single_alg = df[df["alg_name"]==alg_name]
    alg_name_lower = alg_name.lower()
    path_fig_single_alg = path_fig + alg_name_lower
    try:
        os.mkdir(path_fig_single_alg)
    except:
        print("Directory {} already exists.".format(alg_name_lower))

    error_rate_reduced_list = [0.0193,0.01,0.0052,0.0019,0.001]
    if alg_name == "Signal":
        n_reduced_list = [5,9,15,25,50,100]
    elif alg_name == "Shearing":
        n_reduced_list = [6,10,16,24,50,100]
    elif alg_name == "Toom":
        n_reduced_list = [9,16,25,49,100]
    else:
        n_reduced_list = list(set(df_single_alg["n"].to_list()))

    plot_f_E(df_single_alg[df_single_alg["n"].isin(n_reduced_list)],True,"pL_fit",dict_plim_fit[alg_name],path_fig_single_alg)

######### plot gamma n #########

plot_gamma_n(df,False,"gamma_n_fit",path_fig)
#plot_gamma_n(df,True,"gamma_n_fit",path_fig)

######### plot logical estimated #########

df_proof = import_data(path, proof_name, "Proof")
plot_estimate_f_n(df,df_proof,path_fig)








"""