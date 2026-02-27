import numpy as np
from scipy.stats import linregress

def get_cutoff(df):
    TIME_MIN_FIT = 900
    for distance, group in df.groupby("code_distance"):
        logical_error_mean = (group.loc[group["time_run"] >= TIME_MIN_FIT, "cummulative_error"]/group.loc[group["time_run"] >= TIME_MIN_FIT, "time_run"]).mean()
        cond = (group["cummulative_error"] / group["time_run"]) >= (9 * logical_error_mean / 10)

        # Ensure ordered by time
        group_sorted = group.sort_values("time_run")

        # True only if condition holds for this row AND all later rows
        steady_from_here = cond.iloc[::-1].cummin().iloc[::-1]

        time_cutoff = group_sorted.loc[steady_from_here, "time_run"].min()

        df.loc[df["code_distance"] == distance, "logical_error_mean"] = logical_error_mean
        df.loc[df["code_distance"] == distance, "time_cutoff"] = time_cutoff

def fit_cutoff(df):
    df_fit = df[df["time_run"] == df["time_run"].max()]
    X = df_fit["code_distance"].to_numpy()
    Y = df_fit["time_cutoff"].to_numpy()
    slope, intercept, *_ = linregress(X, Y)
    df["time_cutoff_fit"] = slope * df["code_distance"] + intercept