import pandas as pd
import numpy as np

from simulations import PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS, error_rate_list, code_distance_list

data = []

DATA_PATH = "threshold/data/"
OUT_PATH = "threshold/out/"

#suffix = "_temoin"
suffix = ""

for i_e, error_rate in enumerate(error_rate_list):

    # Make copies of the current parameter dicts for this iteration
    physics_copy = PHYSICS_PARAMS.copy()
    decoder_copy = DECODER_PARAMS.copy()
    simulation_copy = SIMULATION_PARAMS.copy()

    # Merge into a single dict for convenience
    input_dict = {**physics_copy, **decoder_copy, **simulation_copy}

    # Set the current error rate
    input_dict["physical_error_rate"] = error_rate

    for i_d, distance in enumerate(code_distance_list):
        input_dict["code_distance"] = distance

        # Create a fresh merged dict for this iteration
        input_dict_merged = dict(input_dict)

        try:
            file_path = OUT_PATH + f"data_{i_e}_{i_d}{suffix}.txt"
            with open(file_path, "r") as file:
                output_lines = [line.strip().split() for line in file.readlines()]

            # Aggregate numeric values from the file
            input_dict_merged["error_count"] = np.sum([int(line[0]) for line in output_lines], dtype=int)
            input_dict_merged["unconverged_count"] = np.sum([int(line[1]) for line in output_lines], dtype=int)
            input_dict_merged["run_count"] = np.sum([int(line[2]) for line in output_lines], dtype=int)
            input_dict_merged["time_run"] = int(output_lines[0][3])  # assume T is the same for all lines

            data.append(input_dict_merged)

        except FileNotFoundError:
            # skip missing files silently
            pass

# Convert to DataFrame
df = pd.DataFrame.from_records(data)

# Optional: move dtypes into first row (like your old code)
df.loc[-1] = df.dtypes
df.sort_index(inplace=True)

# Keep only relevant columns
df = df[
    [   "online",
        "code_distance",
        "meas_error",
        "physical_error_rate",
        "error_count",
        "unconverged_count",
        "run_count",
        "time_run",
    ]
]

# Save CSV
df.to_csv(DATA_PATH + f"data{suffix}.csv", index=False)
