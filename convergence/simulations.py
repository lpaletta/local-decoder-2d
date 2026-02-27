import os
import sys
import time

import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support

# --- Add parent directory to path to allow imports from sibling folders ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Output path for simulation results ---
OUT_PATH = "convergence/out/"

# --- Import core modules for simulation ---
from sgn2d import SIGNAL2D
from mwpm import *
from view import *

# --- Parameters ---
from param import PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS, update_existing

# --- Worker-local globals ---
W_PHYSICS = None
W_DECODER = None
W_SIMULATION = None
W_MATCHING = None
W_H = None

# --- Lists of simulation parameters to sweep over ---
code_distance_list = [9,15,25,35,50]

# --- Core simulation parameters ---

PHYSICS_UPDATE = {"data_error_rate":0.006,
                  "meas_error_rate":0.006,
                  "meas_error":True}

SIMULATION_UPDATE = {"time_run":1000,
                     "time_step":2,
                     "time_simulation_max":10**8,
                     "recorded_variable":"Poisson"}

DECODER_UPDATE = {"max_stack":np.inf}

update_existing(PHYSICS_PARAMS,PHYSICS_UPDATE)
update_existing(SIMULATION_PARAMS,SIMULATION_UPDATE)
update_existing(DECODER_PARAMS,DECODER_UPDATE)

# --- Multiprocessing setup ---
number_of_cores = cpu_count()
CORE_MULTIPLIER = 100

def init_worker(physics, decoder, simulation, distance):
    """Initialize worker-local read-only objects."""
    global W_PHYSICS, W_DECODER, W_SIMULATION, W_MATCHING, W_H
    W_PHYSICS = physics
    W_DECODER = decoder
    W_SIMULATION = simulation
    W_MATCHING = get_matching(distance)
    W_H = get_parity_check_matrix(distance)


def SIGNAL2D_DECODED(_):
    """Run one simulation + decoding entirely inside the worker."""
    data_history = SIGNAL2D(W_PHYSICS, W_DECODER, W_SIMULATION)
    logical_history = np.array([
                        int(
                            get_logical_from_array(
                                (data_history[i] + vector_to_array(
                                    W_MATCHING.decode(W_H @ get_data_as_vector(data_history[i]) % 2)
                                )) % 2
                            ) != (0, 0)
                        )
                        for i in range(data_history.shape[0])
                    ])
    return logical_history

def main():
    T_run = SIMULATION_PARAMS["time_run"]
    dt = SIMULATION_PARAMS["time_step"]
    time_grid = np.array([t for t in range(0,T_run,dt)])
    T_simulation_max = SIMULATION_PARAMS["time_simulation_max"]

    for i_d, distance in enumerate(code_distance_list):

        update_existing(PHYSICS_PARAMS, {"code_distance": distance})

        print(
            f"Proceed to:\n d={distance}, "
            f"e_d={PHYSICS_PARAMS['data_error_rate']}, "
            f"e_m={PHYSICS_PARAMS['meas_error_rate']}"
        )

        time_start = time.time()

        physics, decoder, simulation = PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS

        T_simulation = 0
        run_count_i = 0
        logical_history_i = np.zeros(len(time_grid)).astype(np.int8)

        with Pool(
            processes=number_of_cores,
            initializer=init_worker,
            initargs=(physics, decoder, simulation, distance),
        ) as pool:

            while (
                T_simulation < T_simulation_max
            ):

                batch_logical_history = pool.map(
                    SIGNAL2D_DECODED,
                    range(CORE_MULTIPLIER * number_of_cores),
                )
                
                T_simulation += len(batch_logical_history) * T_run
                
                logical_history_i += sum([h for h in batch_logical_history])
                run_count_i += len(batch_logical_history)

                # --- Periodic flush to disk
                if ((T_run * run_count_i) >= (T_simulation_max / 100)):
                    for t in range(len(logical_history_i)):
                        with open(OUT_PATH + f"data_{i_d}.txt", "a") as file:
                            file.write(
                                f"{logical_history_i[t]} "
                                f"{run_count_i} "
                                f"{time_grid[t]}\n"
                            )

                    logical_history_i = np.zeros(len(time_grid)).astype(np.int8)
                    run_count_i = 0

            print(
            "Completed in:\n {}s".format(
                np.around(time.time() - time_start, 2)
            )
            )


if __name__ == "__main__":
    freeze_support()
    main()