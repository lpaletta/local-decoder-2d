import os
import sys
import time

import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support

# --- Add parent directory to path to allow imports from sibling folders ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Output path for simulation results ---
OUT_PATH = "threshold/out/"

# --- Import core modules for simulation ---
from sgn2d import SIGNAL2D
from mwpm import *
from view import *

# --- Import simulation parameters ---
from param import PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS, update_existing

# --- Lists of simulation parameters to sweep over ---
code_distance_list = [9,15,25,35]
error_rate_list = np.round(np.logspace(np.log10(0.007),np.log10(0.01),12,endpoint=True),5).tolist()

# --- Core simulation parameters ---

SIMULATION_UPDATE = {"time_simulation_max":10**6,
                     "error_count_max":1000,
                     "recorded_variable":"Data"}

update_existing(SIMULATION_PARAMS,SIMULATION_UPDATE)

DISTANCE_TO_TIME_RUN_RATIO = 10

# --- Multiprocessing setup ---
number_of_cores = cpu_count()  # Automatically detect number of CPU cores
CORE_MULTIPLIER = 5

def main():
    T_simulation_max = SIMULATION_PARAMS["time_simulation_max"]
    error_count_max = SIMULATION_PARAMS["error_count_max"]

    for i_e, error_rate in enumerate(error_rate_list):
        update_existing(PHYSICS_PARAMS,{"data_error_rate":error_rate})
        update_existing(PHYSICS_PARAMS,{"meas_error_rate":error_rate}) if PHYSICS_PARAMS["meas_error"] else update_existing(PHYSICS_PARAMS,{"meas_error_rate":0})
    
        for i_d, distance in enumerate(code_distance_list):
            update_existing(SIMULATION_PARAMS,{"time_run":DISTANCE_TO_TIME_RUN_RATIO*distance})
            update_existing(PHYSICS_PARAMS,{"code_distance":distance})
            

            print(
                "Proceed to:\n d={}, e_d={}, e_m={}".format(
                    distance,
                    PHYSICS_PARAMS["data_error_rate"],
                    PHYSICS_PARAMS["meas_error_rate"],
                )
            )

            time_start = time.time()

            matching = get_matching(distance)
            H = get_parity_check_matrix(distance)

            physics, decoder, simulation = PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS
            
            args = CORE_MULTIPLIER * number_of_cores * [(physics, decoder, simulation)]

            T_run = simulation["time_run"]
            T_simulation = 0
            error_count = 0

            # --- Batch-level accumulators (flushed periodically)
            error_count_i = 0
            unconverged_count_i = 0
            run_count_i = 0

            while error_count < error_count_max and T_simulation < T_simulation_max:
                with Pool(number_of_cores) as pool:
                    batch_results = pool.starmap(SIGNAL2D_REPEATED, args)

                batch_results_flattened = [item for sublist in batch_results for item in sublist]
                batch_data = [r for r in batch_results_flattened]
                batch_run_count = [1 for r in batch_results_flattened]

                if decoder["online"]:
                    batch_unconverged = [0] * len(batch_data)
                    batch_error = [
                        int(
                            get_logical_from_array(
                                (d + vector_to_array(matching.decode((H @ get_data_as_vector(d))%2))) % 2
                            ) != (0, 0)
                        )
                        for d in batch_data
                    ]
                else:
                    batch_unconverged = [
                        int(~np.all((H @ get_data_as_vector(d)) % 2 ==0))
                        for d in batch_data
                    ]
                    batch_error = [
                        int(get_logical_from_array(d) != (0, 0))
                        * int(np.all((H @ get_data_as_vector(d)) %2 == 0))
                        for d in batch_data
                    ]

                T_simulation += sum(batch_run_count) * T_run
                error_count += sum(batch_error)

                error_count_i += sum(batch_error)
                unconverged_count_i += sum(batch_unconverged)
                run_count_i += sum(batch_run_count)

                # --- Periodic flush to disk
                if (
                    unconverged_count_i > 0
                    or error_count_i > 10
                    or (run_count_i * T_run) >= (T_simulation_max / 100)
                ):
                    with open(OUT_PATH + f"data_{i_e}_{i_d}.txt", "a") as file:
                        file.write(
                            f"{error_count_i} "
                            f"{unconverged_count_i} "
                            f"{run_count_i} "
                            f"{simulation['time_run']}\n"
                        )

                    error_count_i = 0
                    unconverged_count_i = 0
                    run_count_i = 0

            print(
                "Completed in:\n {}s".format(
                    np.around(time.time() - time_start, 2)
                )
            )

def SIGNAL2D_REPEATED(*args):
    return [SIGNAL2D(*args) for _ in range(10)]

def get_estimate_logical_signal(n,p,A,pth,gamma):
    pL = A*n*(p/pth)**gamma
    return(pL)

if __name__=="__main__":
    freeze_support()
    main()