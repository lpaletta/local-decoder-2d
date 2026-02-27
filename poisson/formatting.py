import pandas as pd
import numpy as np

from simulations import PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS, code_distance_list

data = []

DATA_PATH = "poisson/data/"
OUT_PATH = "poisson/out/"

#suffix = "_temoin"
suffix = ""

# Make copies of the current parameter dicts for this iteration
physics_copy = PHYSICS_PARAMS.copy()
decoder_copy = DECODER_PARAMS.copy()
simulation_copy = SIMULATION_PARAMS.copy()

# Merge into a single dict for convenience
input_dict = {**physics_copy, **decoder_copy, **simulation_copy}
input_dict["physical_error_rate"] = input_dict["data_error_rate"] if input_dict["data_error_rate"] == input_dict["meas_error_rate"] else print("Warning: different data and measurement error rates")

for i_d, distance in enumerate(code_distance_list):
    input_dict["code_distance"] = distance

    try:
        file_path = OUT_PATH + f"data_{i_d}{suffix}.txt"
        with open(file_path, "r") as file:
            output_lines = [line.strip().split() for line in file.readlines()]

        for line in output_lines:
            # Aggregate numeric values from the file
            # Create a fresh merged dict for this iteration
            input_dict_merged = dict(input_dict)
            input_dict_merged["error_count"] = int(line[0])
            input_dict_merged["run_count"] = int(line[1])
            input_dict_merged["time_run"] = int(line[2])
            data.append(input_dict_merged)

    except FileNotFoundError:
        # skip missing files silently
        pass

# Convert to DataFrame
df = pd.DataFrame.from_records(data)

cols_to_sum = ["error_count", "run_count"]
df = (
    df
    .groupby(
        ["online", "code_distance", "meas_error", "physical_error_rate", "time_run"],
        as_index=False
    )[cols_to_sum]
    .sum()
)

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
        "run_count",
        "time_run",
    ]
]

# Save CSV
df.to_csv(DATA_PATH + f"data{suffix}.csv", index=False)
