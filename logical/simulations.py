import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support

# --- Add parent directory to path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Output path ---
OUT_PATH = "logical/out/"

# --- Core simulation imports ---
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

# --- Parameter sweeps ---
code_distance_list = [5, 9, 15, 25, 35, 50, 75, 100]
error_rate_list = np.round(np.logspace(np.log10(0.001), np.log10(0.01), 14, endpoint=False)[1:], 5).tolist()
max_stack_list = [3, 7, np.inf]

# --- Run constants ---
MIN_LOGICAL_ERROR = 1e-8
DISTANCE_TO_TIME_RUN_RATIO = 20

# --- Simulation control ---
SIMULATION_UPDATE = {
    "time_run": 10_000,
    "time_simulation_max": 10**7,
    "error_count_max": 100,
    "recorded_variable": "Data",
}

DECODER_UPDATE = {"max_stack":np.inf}

update_existing(SIMULATION_PARAMS, SIMULATION_UPDATE)
update_existing(DECODER_PARAMS,DECODER_UPDATE)

number_of_cores = cpu_count()
CORE_MULTIPLIER = 1

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
    d = SIGNAL2D(W_PHYSICS, W_DECODER, W_SIMULATION)
    syndrome = (W_H @ get_data_as_vector(d)) % 2

    if W_DECODER["online"]:
        logical_error = int(
            get_logical_from_array(
                (d + vector_to_array(W_MATCHING.decode(syndrome))) % 2
            ) != (0, 0)
        )
        unconverged = 0
    else:
        unconverged = int(~np.all(syndrome == 0))
        logical_error = int(
            get_logical_from_array(d) != (0, 0)
        ) * int(np.all(syndrome == 0))

    return logical_error, unconverged


def main():
    T_simulation_max = SIMULATION_PARAMS["time_simulation_max"]
    error_count_max = SIMULATION_PARAMS["error_count_max"]

    for i_e, error_rate in enumerate(error_rate_list):
        update_existing(PHYSICS_PARAMS, {"data_error_rate": error_rate})
        update_existing(
            PHYSICS_PARAMS,
            {"meas_error_rate": error_rate if PHYSICS_PARAMS["meas_error"] else 0},
        )

        for i_d, distance in enumerate(code_distance_list):

            update_existing(PHYSICS_PARAMS, {"code_distance": distance})
            update_existing(SIMULATION_PARAMS, {"time_run": DISTANCE_TO_TIME_RUN_RATIO*distance})

            for i_m, max_stack in enumerate(max_stack_list):

                update_existing(DECODER_PARAMS, {"max_stack": max_stack})

                print(
                    f"Proceed to:\n d={distance}, "
                    f"e_d={PHYSICS_PARAMS['data_error_rate']}, "
                    f"e_m={PHYSICS_PARAMS['meas_error_rate']}, "
                    f"max_stack={max_stack}"
                )

                time_start = time.time()

                physics, decoder, simulation = PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS
                T_run = simulation["time_run"]

                T_simulation = error_count = run_count = 0
                error_count_i = unconverged_count_i = run_count_i = 0

                with Pool(
                    processes=number_of_cores,
                    initializer=init_worker,
                    initargs=(physics, decoder, simulation, distance),
                ) as pool:

                    while (
                        error_count < error_count_max
                        and T_simulation < T_simulation_max
                    ):

                        batch = pool.map(
                            SIGNAL2D_DECODED,
                            range(CORE_MULTIPLIER * number_of_cores),
                        )

                        batch_error = [r[0] for r in batch]
                        batch_unconv = [r[1] for r in batch]

                        n = len(batch)
                        T_simulation += n * T_run
                        error_count += sum(batch_error)
                        run_count += n

                        error_count_i += sum(batch_error)
                        unconverged_count_i += sum(batch_unconv)
                        run_count_i += n

                        if (
                            unconverged_count_i > 0
                            or error_count_i > 10
                            or (T_run * run_count_i) >= T_simulation_max / 100
                        ):
                            with open(f"{OUT_PATH}data_{i_e}_{i_d}_{i_m}.txt", "a") as f:
                                f.write(
                                    f"{error_count_i} "
                                    f"{unconverged_count_i} "
                                    f"{run_count_i} "
                                    f"{simulation['time_run']}\n"
                                )
                            error_count_i = unconverged_count_i = run_count_i = 0

                        # --- Real time estimate of the logical error rate to skip runs with too low rate
                        if (T_simulation > T_simulation_max / 20):
                            if error_count == 0:
                                conservative_logical_error = 3.0 / T_simulation
                            else:
                                r = error_count / run_count
                                sigma = (
                                    np.sqrt(r * (1 - r))
                                    / (T_run * np.sqrt(run_count))
                                )
                                conservative_logical_error = error_count / T_simulation + sigma

                            if conservative_logical_error < MIN_LOGICAL_ERROR :
                                return

                            print("Completed in:", round(time.time() - time_start, 2), "s")

if __name__ == "__main__":
    freeze_support()
    main()
