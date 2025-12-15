import numpy as np

from scipy.optimize import curve_fit


def fit_pL_single_alg(df,pth_guess,plim_fit):

    df_fit = df.copy()
    df_fit = df_fit.dropna(subset="pL")

    df_fit = df_fit[df_fit["error_rate"]<plim_fit]

    d_list = np.sort(list(set(df_fit["d"].to_list())))
    dict_d_index = dict(zip(list(d_list),[int(i) for i in range(len(d_list))]))
    df_fit["i_d"] = df_fit["d"].apply(lambda x: map_id_to_d(x,dict_d_index))
    df["i_d"] = df["d"].apply(lambda x: map_id_to_d(x,dict_d_index))

    gamma_d_guess_list = [d/10 for d in d_list]
    gamma_d_bound_list = len(d_list)*[50]

    guess = np.array(gamma_d_guess_list+[0.1,pth_guess])
    upper_bound = np.array(gamma_d_bound_list+[100,2*pth_guess])

    fit = curve_fit(f=func_logical,xdata=(df_fit["d"].to_numpy(),df_fit["error_rate"].to_numpy(),df_fit["i_d"].to_numpy()),ydata=np.log(df_fit["pL"].to_numpy()),p0=guess,bounds=(0,list(upper_bound)))

    param_opt_value = fit[0]
    A, pth = param_opt_value[-1], param_opt_value[-2]

    df.loc[:,"A"] = A
    df.loc[:,"pth"] = pth
    print(A,pth)
    for i in range(len(d_list)):
        df.loc[df["i_d"]==i,"gamma_d"] = param_opt_value[i]

    df = df.drop(columns=["i_d"])

    return(df)

def fit_gamma_d_single_alg(df):

    df_fit = df.copy()
    df_fit = df_fit.dropna(subset="gamma_d")

    df_fit = df_fit[df_fit["d"]>=15]
    df_fit = df_fit[df_fit["d"]<=100]

    fit = curve_fit(f=get_exp,xdata=df_fit["d"].to_numpy(),ydata=df_fit["gamma_d"].to_numpy(),p0=[1,1],bounds=(0,[2,1]))

    param_opt_value = fit[0]
    alpha, beta = param_opt_value[0], param_opt_value[1]

    df.loc[:,"alpha"] = alpha
    df.loc[:,"beta"] = beta
    print(alpha,beta)

    return(df)


def add_pL_fit(df,key):
    df.loc[:,key] = np.exp(ansatz(df.loc[:,"A"],df.loc[:,"d"],df.loc[:,"error_rate"],df.loc[:,"pth"],df.loc[:,"gamma_d"]))
    return(df)

def add_gamma_d_fit(df,key):
    df.loc[:,key] = get_exp(df.loc[:,"d"],df.loc[:,"alpha"],df.loc[:,"beta"])
    return(df)

def func_logical(X,*param):
    d, p, i_d = X
    gamma_d_list = param[:-2]
    A, pth = param[-1], param[-2]
    gamma_d = [gamma_d_list[int(i)] for i in i_d]
    return(ansatz(A,d,p,pth,gamma_d))

def ansatz(A,d,p,pth,gamma_d):
    return(np.log((d*A)*(p/pth)**(gamma_d)))

def get_exp(X,alpha,beta):
    d = X
    gamma_d = alpha*d**beta
    return(gamma_d)

def map_id_to_d(d,dict_d_index):
    if d in dict_d_index:
        return(dict_d_index[d])
    else:
        return(float("NaN"))