import numpy as np

rules = [
    {
        "pattern": np.array([
            [-1, -1, -1],
            [ 1,  1, -1],
            [-1, -1, -1]
        ]),
            "offset": (0, 1)
    },
    {
        "pattern": np.array([
            [-1, -1, -1],
            [-1,  1,  1],
            [-1, -1, -1]
        ]),
             "offset": (0,-1)
    },
    {
        "pattern": np.array([
            [-1, -1, -1],
            [-1,  1, -1],
            [-1,  1, -1]
        ]),
            "offset": (-1, 0)
    },
    {
        "pattern": np.array([
            [-1,  1, -1],
            [-1,  1, -1],
            [-1, -1, -1]
        ]),
            "offset": (1, 0)
    }
]

PHYSICS_PARAMS = {
    "code_distance": 0,
    "data_error_rate": 0,
    "meas_error_rate": 0,
    "meas_error": True 
}

DECODER_PARAMS = {
    "online": True,
    "anti_signal_velocity": 3,
    "max_stack": np.inf,
    "matching_rules": rules,
}

SIMULATION_PARAMS = {
    # Time control
    "time_run": 1000,
    "time_step": None,

    # Job / runtime control
    "time_job": 10**4,
    "time_simulation_max": 10**6,
    "error_count_max": 100,

    # Initialization / state
    "initial_data_array": None,
    "artificial_defect": None,

    # Observables / output
    "recorded_variable": "Data", #["Poisson","Logical","View"]
}

def update_existing(dict_to_update, updates):
    for key, value in updates.items():
        if key not in dict_to_update:
            raise KeyError(f"Key '{key}' not found in dictionary")
        dict_to_update[key] = value