import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def import_data(path,data_name,analysis_var):
    dtypes = pd.read_csv(path+data_name, nrows=1, na_filter=False).iloc[0].to_dict()
    df = pd.read_csv(path+data_name,dtype=dtypes,skiprows=[1], na_filter=True)

    df["T"] = df["T"] + 1
    df['error_rate'] = df.apply(lambda x: np.around(x['error_rate'],4),axis=1)

    if analysis_var in ["Logical"]:

        df = df[df["number_of_errors"]>=2]
        df = df[df["number_of_runs"]>=2.5*df["number_of_errors"]]

        df['pL'] = df.apply(lambda x: compute_pL(number_of_errors=x['number_of_errors'],T=x['T'],number_of_runs=x['number_of_runs'],error_bool=x['error_bool']),axis=1)
        df["sigma"] = df.apply(lambda x: compute_sigma(number_of_errors=x['number_of_errors'],T=x['T'],number_of_runs=x['number_of_runs'],error_bool=x['error_bool']),axis=1)
        df['pU'] = df.apply(lambda x: compute_pL(number_of_errors=x['number_of_unconverged'],T=x['T'],number_of_runs=x['number_of_runs'],error_bool=x['error_bool']),axis=1)
        df["sigmaU"] = df.apply(lambda x: compute_sigma(number_of_errors=x['number_of_unconverged'],T=x['T'],number_of_runs=x['number_of_runs'],error_bool=x['error_bool']),axis=1)

    df_sorted = df.sort_values(by=['d', 'error_rate'])
    return(df_sorted)

def compute_pL(number_of_errors,T,number_of_runs,error_bool):
    if error_bool == False:
        T=1
    r = number_of_errors/number_of_runs
    if r<=0.5:
        return((1-(1-2*r)**(1/T))/2)
    else:
        return(0.5)
    
def compute_ratio(number_of_errors,number_of_runs):
    r = number_of_errors/number_of_runs
    if r<0.5:
        return(r)
    elif r>0.5:
        return(0.5)
    
def compute_sigma(number_of_errors,T,number_of_runs,error_bool):
    if error_bool == False:
        T=1
    r = number_of_errors/number_of_runs
    return(1.96*np.sqrt(r*(1-r))/(T*np.sqrt(number_of_runs)))

def compute_sigma_ratio(number_of_errors,number_of_runs):
    r = number_of_errors/number_of_runs
    return(1.96*np.sqrt(r*(1-r))/(np.sqrt(number_of_runs)))