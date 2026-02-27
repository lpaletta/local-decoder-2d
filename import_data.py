import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

VIRTUAL_INF = 0.001


# -----------------------
# I/O
# -----------------------

def import_data(path, data_name, record_var):
    """
    Import CSV data and compute logical error statistics if requested.
    """
    dtypes = pd.read_csv(path + data_name, nrows=1, na_filter=False).iloc[0].to_dict()

    df = pd.read_csv(path + data_name, dtype=dtypes, skiprows=[1], na_filter=True)

    if record_var == "Logical":
        df = process_data_for_logical(df)
    elif record_var == "Poisson":
        df = process_data_for_poisson(df)
    elif record_var == "Convergence":
        df = process_data_for_convergence(df)
    elif record_var == "Threshold":
        df = process_data_for_threshold(df)

    return df.sort_values(by=["code_distance", "physical_error_rate"])

# -----------------------
# Logical analysis
# -----------------------

def process_data_for_threshold(df):
    df["time_run"] = df["time_run"] + 1
    df["physical_error_rate"] = df["physical_error_rate"].round(5)
    """
    Remove entries with unsifficent error count and compute pL and uncertainties.
    """
    df = df[(df["error_count"]>=2) & (df["run_count"]>=(4*df["error_count"]/3))]

    df["logical_error_rate"] = df.apply(lambda x: compute_pL(x["error_count"], x["time_run"], x["run_count"], x["online"]), axis=1)
    df["sigma"] = df.apply(lambda x: compute_sigma(x["error_count"], x["time_run"], x["run_count"], x["online"]), axis=1)

    return df

def process_data_for_logical(df):
    df["time_run"] = df["time_run"] + 1
    df["physical_error_rate"] = df["physical_error_rate"].round(5)
    """
    Remove entries with unsifficent error count and compute pL and uncertainties.
    """
    df = df[(df["error_count"]>=2) & (df["run_count"]>=(4*df["error_count"]/3))]

    df["logical_error_rate"] = df.apply(lambda x: compute_pL(x["error_count"], x["time_run"], x["run_count"], x["online"]), axis=1)
    df["sigma"] = df.apply(lambda x: compute_sigma(x["error_count"], x["time_run"], x["run_count"], x["online"]), axis=1)

    all_p = list(df["physical_error_rate"].unique()) + [VIRTUAL_INF]
    all_d = list(df["code_distance"].unique())
    all_ms = list(df["max_stack"].unique())

    df = (
        df.set_index(
            ["physical_error_rate", "code_distance", "max_stack"]
        )
        .reindex(
            pd.MultiIndex.from_product(
                [all_p, all_d, all_ms],
                names=[
                    "physical_error_rate",
                    "code_distance",
                    "max_stack",
                ],
            )
        )
        .reset_index()
    )

    # Keep logical_error_rate and sigma as NaN for new rows
    # (this is correct — they were never simulated)

    return df

def process_data_for_poisson(df):
    df["time_run"] = df["time_run"] + 1
    """
    Compute cummulative error probability and uncertainty for Poisson-type data and regularize the grid.
    """
    df["cummulative_error"] = df.apply(lambda x: compute_ratio(x["error_count"], x["run_count"]), axis=1)
    df["sigma"] = df.apply(lambda x: compute_sigma_ratio(x["error_count"], x["run_count"]), axis=1)

    df = df.set_index(["code_distance", "physical_error_rate", "time_run"]).unstack(fill_value=0).stack().reset_index()

    df[["cummulative_error", "sigma"]] = df[["cummulative_error", "sigma"]].replace(0, np.nan)
    df["cummulative_error"] = df["cummulative_error"].fillna(1)
    df["sigma"] = df["sigma"].fillna(0)

    return df

def process_data_for_convergence(df):
    df["time_run"] = df["time_run"] + 1
    """
    Compute logical error probability and uncertainty.
    """
    df["cummulative_error"] = df.apply(lambda x: compute_ratio(x["error_count"], x["run_count"]), axis=1)
    df["sigma"] = df.apply(lambda x: compute_sigma_ratio(x["error_count"], x["run_count"]), axis=1)

    df = df.set_index(["code_distance", "physical_error_rate", "time_run"]).unstack(fill_value=0).stack().reset_index()

    df[["cummulative_error", "sigma"]] = df[["cummulative_error", "sigma"]].replace(0, np.nan)
    df["cummulative_error"] = df["cummulative_error"].fillna(1)
    df["sigma"] = df["sigma"].fillna(0)

    return df

# -----------------------
# Statistics helpers
# -----------------------

def compute_pL(error_count, T, run_count, online):
    """
    Estimate logical error probability.
    """
    if not online:
        T = 1

    r = error_count / run_count

    if r < 0.75:
        return (1 - (1 - 4/3 * r) ** (1 / T)) / (3/4)

    return np.nan


def compute_sigma(error_count, T, run_count, online):
    """
    95% confidence interval for pL.
    """
    if not online:
        T = 1

    r = error_count / run_count

    dr = np.sqrt((r*(1-r))/run_count)
    if r < 0.75:
        return 1.96 * 1/T * (1 - 4/3 * r)**(1/T - 1)*dr
    
    return np.nan


def compute_ratio(error_count, run_count):
    """
    Raw error ratio capped at 0.5.
    """
    r = error_count / run_count
    return min(r, 0.75)


def compute_sigma_ratio(error_count, run_count):
    """
    95% confidence interval for raw error ratio.
    """
    r = error_count / run_count
    return 1.96 * np.sqrt(r * (1 - r)) / np.sqrt(run_count)