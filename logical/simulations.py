import os
import sys
import time

import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

path_output = "logical/out/"

from sgn2d import SIGNAL2D
from mwpm import *
from view import *

import param

number_of_cores = 8

def main():
    T_inner_job = param.mc_dict["T_inner_job"]
    T_cum_max = param.mc_dict["T_cum_max"]
    number_of_errors_max = param.mc_dict["number_of_errors_max"]

    for j in range(len(param.error_rate_list)):
        param.param_dict["error_rate"] = param.error_rate_list[j]
        if param.option_dict["meas_error_bool"] and param.option_dict["id_error_rate"]:
            param.param_dict["meas_error_rate"] = param.error_rate_list[j]

        for k in range(len(param.d_list)):
            time_i = time.time()

            param.param_dict["d"] = param.d_list[k]

            if not param.option_dict["error_bool"]:
                param.param_dict["T"] = 10*param.param_dict["d"]
            else:
                matching = get_matching(param.param_dict["d"])
            H = get_parity_check_matrix(param.param_dict["d"])

            print("Proceed to: \n d={}, e={}".format(param.param_dict["d"],param.param_dict["error_rate"]))

            """A, pth, alpha, beta = X, X, X, X
            d, p = param.param_dict["d"], param.param_dict["error_rate"]
            estimate_logical = get_estimate_logical_signal(d,p,A,pth,alpha*d**beta)
            print("Estimated logical rate: \n {}".format(estimate_logical))
            if estimate_logical < 10**(-7):
                continue
            else:
                pass"""

            #param_dict,option_dict,view_dict,mc_dict = calibrate_param(param.param_dict,param.option_dict,param.view_dict,param.mc_dict)
            param_dict,option_dict,view_dict,mc_dict = param.param_dict,param.option_dict,param.view_dict,param.mc_dict
            args = number_of_cores*[(param_dict,option_dict,view_dict,mc_dict)]

            T_cum=0
            number_of_unconverged=0
            number_of_errors=0
            number_of_errors_cum=0
            number_of_runs=0
            number_of_jobs=0

            while number_of_errors_cum<number_of_errors_max and T_cum<T_cum_max:
                with Pool(number_of_cores) as pool:
                    results = pool.starmap(sMC, args)

                result_data_array = [d for r in results for d in r[0]]
                result_number_of_runs = [r[1] for r in results]

                if param.option_dict["error_bool"]:
                    result_not_converged = [0]*len(result_data_array)
                    result_error = [int(get_logical_from_array((x+vector_to_array(matching.decode(H@get_data_as_vector(x)%2)))%2) != (0,0)) for x in result_data_array]
                else:
                    result_not_converged = [int(~np.all((H@get_data_as_vector(x))%2 == 0)) for x in result_data_array]
                    result_error = [int(get_logical_from_array(x) != (0,0))*int(np.all((H@get_data_as_vector(x)%2) == 0)) for x in result_data_array]

                """for bit, x in zip(result_not_converged, result_data_array):
                    if bit == 0:
                        #visualize_data(x)
                        visualize_data(x+vector_to_array(matching.decode(H@get_data_as_vector(x)%2)))
                        #print(get_logical_from_array(x+vector_to_array(matching.decode(H@get_data_as_vector(x))%2)))
                        print("")
                        time.sleep(0.5)"""

                number_of_unconverged+=np.sum(result_not_converged)
                number_of_errors+=np.sum(result_error)
                number_of_errors_cum+=np.sum(result_error)
                number_of_runs+=np.sum(result_number_of_runs)
                number_of_jobs+=len(results)

                T_cum+=len(result_not_converged)*T_inner_job

                if (np.sum(number_of_unconverged)>0) or (np.sum(number_of_errors)>=2) or ((T_inner_job*number_of_jobs)>=(T_cum_max/100)):
                    file = open(path_output+"data_{}_{}.txt".format(j,k), "a")
                    file.write(str(number_of_errors)+" "+str(number_of_unconverged)+" "+str(number_of_runs)+" "+str(param_dict["T"])+"\n")
                    file.close()
                    
                    number_of_errors=0
                    number_of_unconverged=0
                    number_of_runs=0
                    number_of_jobs=0
            
            time_f = time.time()
            print("Completed in: \n {}s".format(np.around(time_f-time_i,2)))

def calibrate_param(param,option,view,mc):
    for T_candidate in [10,20,50,100,200,500,1000]:
        positive=0
        number_of_param_jobs=0

        param["T"] = T_candidate
        args = number_of_cores*[(param,option,view,mc)]

        while number_of_param_jobs<100:
            with Pool(number_of_cores) as pool:
                #results = [sMC(param,option,view,mc) for i in range(10)]
                results = pool.starmap(sMC, args)

            result_positive = [r[0] for r in results]
            result_number_of_param_jobs = [r[1] for r in results]

            positive+=np.sum(result_positive)
            number_of_param_jobs+=np.sum(result_number_of_param_jobs)
        
        if (positive/number_of_param_jobs)>0.1:
            param["T"] = T_candidate
            return(param,option,view,mc)
    param["T"] = 1000
    return(param,option,view,mc)

def sMC(param,option,view,mc):
    T_inner_job = mc["T_inner_job"]
    T = param["T"]
    result_data_array = []
    for i in range(T_inner_job//T):
        result_data_array.append(SIGNAL2D(param, option, view))
    return(result_data_array,T_inner_job//T)

def get_estimate_logical_signal(n,p,A,pth,gamma):
    pL = A*n*(p/pth)**gamma
    return(pL)

if __name__=="__main__":
    freeze_support()
    main()