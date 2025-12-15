import pandas as pd

from param import *
from simulations import *

data = []

path_data = "logical/data/"
path_out = "logical/out/"

#suffix = "_temoin"
suffix = ""

    
for j in range(len(error_rate_list)):

    param_dict_copy = param_dict.copy()
    option_dict_copy = option_dict.copy()
    input_dict = param_dict_copy | option_dict_copy

    input_dict["error_rate"] = error_rate_list[j]
    if input_dict["meas_error_bool"] and input_dict["id_error_rate"]:
            input_dict["meas_error_rate"] = error_rate_list[j]

    for k in range(len(d_list)):
        input_dict["d"] = d_list[k]       
        input_dict_merged = dict(zip(list(input_dict.keys()),list(input_dict.values())))
        try:
            file = open(path_out+'data_'+str(j)+'_'+str(k)+suffix+'.txt')
            output_list_of_string = [s.split(" ") for s in file.readlines()]
            input_dict_merged["number_of_errors"] = np.sum([int(output_list_of_string[i][0]) for i in range(len(output_list_of_string))],dtype=int)
            input_dict_merged["number_of_unconverged"] = np.sum([int(output_list_of_string[i][1]) for i in range(len(output_list_of_string))],dtype=int)
            input_dict_merged["number_of_runs"] = np.sum([int(output_list_of_string[i][2]) for i in range(len(output_list_of_string))],dtype=int)
            input_dict_merged["T"] = int(output_list_of_string[0][3])
            data.append(input_dict_merged)
            file.close()
        except:
            pass

df = pd.DataFrame.from_records(data)
df.loc[-1] = df.dtypes
df.sort_index(inplace=True)

df = df[["d","error_bool","meas_error_bool","error_rate","meas_error_rate","number_of_errors","number_of_unconverged","number_of_runs","T"]]

df.to_csv(path_data+"data"+suffix+".csv",index=False)