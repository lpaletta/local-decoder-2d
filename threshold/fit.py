import numpy as np
from scipy.optimize import curve_fit


def fit_logical(df, pth_guess, plim_fit):
    """
    Fit logical error rate pL using:
        log pL = log(d * A * (p / p_th)^{gamma_d})
    with one gamma_d per distance d.
    """
    df_fit = (
        df.copy()
        .dropna(subset=["logical_error_rate"])
        .loc[lambda x: x["physical_error_rate"] <= plim_fit]
    )

    d_values = np.sort(df_fit["code_distance"].unique())
    d_to_index = {d: i for i, d in enumerate(d_values)}
    n_d = len(d_values)

    gamma_guess = [d / 2 for d in d_values]
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

    df.loc[:, "A"] = A
    df.loc[:, "error_threshold"] = pth
    df.loc[:, "effective_distance"] = df["code_distance"].map(d_to_gamma_d)

    print(f"A = {A}, p_th = {pth}")
    return df

def fit_effective_distance(df):
    """Fit gamma_d(d) = alpha * d^beta."""
    df_fit = (
        df.copy()
        .dropna(subset=["effective_distance"])
        .loc[lambda x: (x["code_distance"] >= 9) & (x["code_distance"] <= 100)]
    )

    popt, _ = curve_fit(
        f=func_exp,
        xdata=df_fit["code_distance"].to_numpy(),
        ydata=df_fit["effective_distance"].to_numpy(),
        p0=[1, 1],
        bounds=(0, [2, 1]),
    )

    alpha, beta = popt
    df.loc[:, "alpha"] = alpha
    df.loc[:, "beta"] = beta

    print(f"alpha = {alpha}, beta = {beta}")
    return df

def fit_threshold_d(df, pth_guess, plim_fit):
    """
    Fit logical error rate pL using:
        log pL = log(d * A * (p / p_th)^{gamma_d})
    with one gamma_d per distance d.
    """
    df_fit = (
        df.copy()
        .dropna(subset=["logical_error_rate"])
        .loc[lambda x: x["physical_error_rate"] <= plim_fit]
    )

    d_values = np.sort(df_fit["code_distance"].unique())
    n_d = len(d_values)

    for i_d, distance in enumerate(d_values[:-1]):
        short_distance_list = [d_values[i_d],d_values[i_d+1]]
        df_short = df_fit[df_fit["code_distance"].isin(short_distance_list)]
        d_to_index = {d: i for i, d in enumerate(short_distance_list)}

        gamma_guess = [d / 2 for d in short_distance_list]
        gamma_bounds = [3 * d / 4 for d in short_distance_list]

        initial_guess = np.array(gamma_guess + [pth_guess, 0.1])
        upper_bounds = np.array(gamma_bounds + [2 * pth_guess, 100])

        xdata = (df_short["code_distance"].map(d_to_index).to_numpy(),df_short["code_distance"].to_numpy(), df_short["physical_error_rate"].to_numpy())
        ydata = np.log(df_short["logical_error_rate"].to_numpy())

        popt, _ = curve_fit(
            f=func_logical,
            xdata=xdata,
            ydata=ydata,
            p0=initial_guess,
            bounds=(0, upper_bounds),
        )

        short_distance_gamma_d_values = popt[:-2]
        pth = popt[-2]
        A = popt[-1]

        d_to_gamma_d = dict(zip(short_distance_list, short_distance_gamma_d_values))

        df.loc[df["code_distance"]==d_values[i_d], "A"] = A
        df.loc[df["code_distance"]==d_values[i_d], "error_threshold"] = pth
        df.loc[df["code_distance"]==d_values[i_d], "effective_distance"] = df["code_distance"].map(d_to_gamma_d)

        df.loc[df["code_distance"]==d_values[i_d+1], "A"] = A
        df.loc[df["code_distance"]==d_values[i_d+1], "error_threshold"] = pth
        df.loc[df["code_distance"]==d_values[i_d+1], "effective_distance"] = df["code_distance"].map(d_to_gamma_d)

        print(f"d = {d_values[i_d]}, error_threshold(d) = {pth}")

    return(df)



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
    return np.log(A * (p / pth) ** gamma_d)


def func_exp(d, alpha, beta):
    return alpha * d ** beta
