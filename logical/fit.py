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
        yerr = np.log(df_fit["sigma"].to_numpy())

        popt, pcov = curve_fit(
                    f=func_logical,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=yerr,
                    p0=initial_guess,
                    bounds=(0, upper_bounds),
                )

        perr = np.sqrt(np.diag(pcov))

        gamma_d_values = popt[:-2]
        gamma_d_errs = perr[:-2]
        pth = popt[-2]
        pth_err = perr[-2]
        A = popt[-1]
        A_err = perr[-1]

        d_to_gamma_d = dict(zip(d_values, gamma_d_values))
        d_to_gamma_d_err = dict(zip(d_values, gamma_d_errs))

        mask = df["max_stack"] == max_stack
        df.loc[mask, "A"] = A
        df.loc[mask, "sigma_A"] = A_err
        df.loc[mask, "error_threshold"] = pth
        df.loc[mask, "sigma_error_threshold"] = pth_err
        df.loc[mask, "effective_distance"] = df["code_distance"].map(d_to_gamma_d)
        df.loc[mask, "sigma_effective_distance"] = df["code_distance"].map(d_to_gamma_d_err)

        print(f"A     = {A:.3g} ± {2*A_err:.2g}")
        print(f"p_th  = {pth:.3g} ± {2*pth_err:.2g}")
        print("")
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

        popt, pcov = curve_fit(
            f=func_exp,
            xdata=df_fit["code_distance"].to_numpy(),
            ydata=df_fit["effective_distance"].to_numpy(),
            sigma=df_fit["sigma_effective_distance"].to_numpy(),
            p0=[1, 1],
            bounds=(0, [5, 1]),
            absolute_sigma=True,
        )

        perr = np.sqrt(np.diag(pcov))
        alpha, beta = popt
        alpha_err, beta_err = perr

        mask = df["max_stack"] == max_stack
        df.loc[mask, "alpha"] = alpha
        df.loc[mask, "sigma_alpha"] = alpha_err
        df.loc[mask, "beta"] = beta
        df.loc[mask, "sigma_beta"] = beta_err

        print(f"alpha = {alpha:.3g} ± {2*alpha_err:.2g}")
        print(f"beta  = {beta:.3g} ± {2*beta_err:.2g}")
        print("")

        # popt, _ = curve_fit(
        #     f=func_exp,
        #     xdata=df_fit["code_distance"].to_numpy(),
        #     ydata=df_fit["effective_distance"].to_numpy(),
        #     p0=[1, 1],
        #     bounds=(0, [5, 1]),
        # )

        # alpha, beta = popt
        # df.loc[df["max_stack"] == max_stack, "alpha"] = alpha
        # df.loc[df["max_stack"] == max_stack, "beta"] = beta

        # print(f"alpha = {alpha}, beta = {beta}")
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

def func_lin(d, a, b):
    return a * d + b
