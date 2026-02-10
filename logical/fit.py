import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def fit_logical(df, pth_guess, plim_fit):
    """
    Fit logical error rate pL using:
        log pL = log(A / d * (p / p_th)^{gamma_d})
    with one gamma_d per distance d.
    """
    for max_stack in df["max_stack"].unique():
        print(f"Fit logical max stack = {int(max_stack) if max_stack < np.inf else 'inf'}")
        df_subset = df[df["max_stack"] == max_stack]

        df_fit = (
            df_subset.copy()
            .dropna(subset=["logical_error_rate"])
            .loc[lambda x: x["physical_error_rate"] < plim_fit]
        )

        d_values = np.sort(df_fit["code_distance"].unique())
        d_to_index = {d: i for i, d in enumerate(d_values)}
        n_d = len(d_values)

        gamma_guess = [d / 10 for d in d_values]
        gamma_bounds = [50 for _ in range(n_d)]

        initial_guess = np.array(gamma_guess + [0.1, pth_guess])
        upper_bounds = np.array(gamma_bounds + [100, 2 * pth_guess])

        xdata = (df_fit["code_distance"].map(d_to_index).to_numpy(),df_fit["code_distance"].to_numpy(), df_fit["physical_error_rate"].to_numpy())

        ydata = np.log(df_fit["logical_error_rate"].to_numpy())
        popt, _ = curve_fit(
            f=func_logical,
            xdata=xdata,
            ydata=ydata,
            p0=initial_guess,
            bounds=(0, upper_bounds),
        )

        gamma_d_values = popt[:-2]
        pth = popt[-2]
        A = popt[-1]

        d_to_gamma_d = dict(zip(d_values, gamma_d_values))

        df.loc[df["max_stack"] == max_stack, "A"] = A
        df.loc[df["max_stack"] == max_stack, "error_threshold"] = pth
        df.loc[df["max_stack"] == max_stack, "effective_distance"] = df["code_distance"].map(d_to_gamma_d)

        print(f"A = {A}, p_th = {pth}")
    return df

def fit_effective_distance(df):
    """Fit gamma_d(d) = alpha * d^beta."""
    for max_stack in df["max_stack"].unique():
        print(f"Fit effective distance max stack = {int(max_stack) if max_stack < np.inf else 'inf'}")
        df_subset = df[df["max_stack"] == max_stack]
        df_fit = (
            df_subset.copy()
            .dropna(subset=["effective_distance"])
            .loc[lambda x: (x["code_distance"] >= 9) & (x["code_distance"] <= 100)]
        )

        popt, _ = curve_fit(
            f=func_exp,
            xdata=df_fit["code_distance"].to_numpy(),
            ydata=df_fit["effective_distance"].to_numpy(),
            p0=[1, 1],
            bounds=(0, [5, 1]),
        )

        alpha, beta = popt
        df.loc[df["max_stack"] == max_stack, "alpha"] = alpha
        df.loc[df["max_stack"] == max_stack, "beta"] = beta

        print(f"alpha = {alpha}, beta = {beta}")
    return df

def func_logical(X, *params):
    d_idx, d, p = X
    gamma_d_list = params[:-2]
    pth = params[-2]
    A = params[-1]

    gamma_d = [gamma_d_list[int(i)] for i in d_idx]

    return ansatz(A, d, p, pth, gamma_d)


def add_logical_fit(df, key):
    df.loc[:, key] = np.exp(ansatz(df["A"], df["code_distance"], df["physical_error_rate"], df["error_threshold"], df["effective_distance"]))
    return df


def add_effective_distance_fit(df, key):
    df.loc[:, key] = func_exp(df["code_distance"], df["alpha"], df["beta"])
    return df


def ansatz(A, d, p, pth, gamma_d):
    return np.log(A / d * (p / pth) ** gamma_d)


def func_exp(d, alpha, beta):
    return alpha * d ** beta
