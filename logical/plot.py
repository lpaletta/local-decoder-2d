import numpy as np
import pandas as pd

import logging
logging.getLogger("fontTools").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter

# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

plt.style.use("rgplot")

# ---------------------------------------------------------------------
# Figure / font constants
# ---------------------------------------------------------------------

FIGSIZE_ERROR = (3.2, 2.0)
FIGSIZE_GAMMA = (1.85, 2.1)
FIGSIZE_GAMMA_COMP = (2.8,1.7)

LABEL_SMALL_FONTSIZE = 6
LABEL_FONTSIZE = 7
TICK_FONTSIZE_SMALL = 5.5

THIN_AXWIDTH = 0.5
THIN_LINEWIDTH = 0.8
AXWIDTH = 0.6
LINEWIDTH = 1.2
MARKER_SIZE = 2.5
SCATTER_SIZE = 10

# ---------------------------------------------------------------------
# Physical / numerical cutoffs
# ---------------------------------------------------------------------

D_PLOT_CUTOFF = 6
D_FIT_CUTOFF = 15
PL_FIT_CUTOFF = 1e-9

# ---------------------------------------------------------------------
# Axis limits
# ---------------------------------------------------------------------

X_ERROR_LIM = (1e-3, 1e-2)
Y_ERROR_LIM = (1e-8, 1e-2)

X_GAMMA_LIM = (0, 110)
Y_GAMMA_LIM = (0, 30)

X_GAMMA_COMP_LIM = (0, 27)
Y_GAMMA_COMP_LIM = (2, 12)

# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

GREY = "#D6D6D6"
MAGENTA = "#CE93D8"
SKYBLUE = "#AEC6CF"
NAVY = "#1F3A5F"
APPLE = "#A8D5A2"
BLACK = "black"
PASTELRED = "#FF6961"

BLUE, RED, GREEN, YELLOW = colors[:4]

DISTANCE_TO_COLOR = {
    5: GREY,
    7: GREY,
    9: BLUE,
    11: BLUE,
    13: APPLE,
    15: RED,
    19: RED,
    25: GREEN,
    35: NAVY,
    50: YELLOW,
    75: APPLE,
    100: MAGENTA,
}

STACK_TO_COLOR = {
    3: YELLOW,
    4: YELLOW,
    7: GREEN,
    8: GREEN,
    np.inf: BLUE,
}

STACK_TO_MARKER = {
    3: "^",
    4: "^",
    7: "d",
    8: "d",
    np.inf: "o",
}


# ---------------------------------------------------------------------
# Digitalized data
# ---------------------------------------------------------------------


d_lake = [5,7,9,13,19]
d_eff_lake = [3.3, 4.3, 4.9, 6.3, 8.5]

d_winter = [5,9,15,21]
d_eff_winter = [3.096, 4.831, 6.819, 8.554]
d_eff_winter_uncertainty = [0.066, 0.038, 0.058, 0.217]

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_logical(df, fit, key, plim, suffix="", path=""):
    max_stack = df["max_stack"].unique()[0]  # Assuming only one max_stack value in the subset
    fig, ax = plt.subplots(figsize=FIGSIZE_ERROR)

    for distance, group in df.groupby("code_distance"):
        color = DISTANCE_TO_COLOR[distance]

        ax.scatter(
            group["physical_error_rate"],
            group["logical_error_rate"],
            s=SCATTER_SIZE*0.8 if max_stack != np.inf else SCATTER_SIZE*0.5,
            marker=STACK_TO_MARKER[max_stack],
            color=color,
            zorder=10,
            label=rf"$d={distance}$",   
        )

        ax.errorbar(
            group["physical_error_rate"],
            group["logical_error_rate"],
            yerr=2*group["sigma"],
            fmt="none",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            color=color,
        )

        ax.plot(
            group["physical_error_rate"],
            group["logical_error_rate"],
            linewidth=LINEWIDTH,
            color=color,
        )

    if fit:
        for distance, group in df.groupby("code_distance"):
            group = group[group["physical_error_rate"] < plim]
            ax.plot(
                group["physical_error_rate"],
                group[key],
                linestyle=":",
                linewidth=LINEWIDTH,
                color=DISTANCE_TO_COLOR[distance],
            )

    ax.set_xlabel(
        r"physical error ($\varepsilon = \varepsilon_d = \varepsilon_m$)",
        fontsize=LABEL_FONTSIZE,
        labelpad=-5,
    )
    ax.set_ylabel(
        r"logical error ($\varepsilon_L$)",
        fontsize=LABEL_FONTSIZE,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*X_ERROR_LIM)
    ax.set_ylim(*Y_ERROR_LIM)

    ax.xaxis.set_minor_formatter(NullFormatter())
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position("top")

    ax.tick_params(labelsize=TICK_FONTSIZE_SMALL, width=AXWIDTH)

    # Grid (as in original)
    ax.grid(which='major', color=GREY, linestyle='-', linewidth=AXWIDTH/2, alpha=0.2)
    ax.grid(which='minor', color=GREY, linestyle='-', linewidth=AXWIDTH/3, alpha=0.2)
    #ax.grid(False)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=7,
        ncol=2,
        columnspacing=0.5,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(AXWIDTH)

    fig.tight_layout()
    if fit:
        suffix += "_fit"
    plt.savefig(f"{path}/logical{suffix}.pdf")
    plt.close()

# def plot_logical_stack_comp(df, fit, key, plim, suffix="", path=""):
#     fig, ax = plt.subplots(figsize=FIGSIZE_ERROR)

#     for max_stack, group in df.groupby("max_stack"):
#         df_subset = df[df["max_stack"] == max_stack]

#         if max_stack == 7: #skip 7 to avoid overloading the plot
#             continue

#         for distance, group in df_subset.groupby("code_distance"):
#             color = DISTANCE_TO_COLOR[distance]

#             if max_stack == np.inf:

#                 ax.errorbar(
#                     group["physical_error_rate"],
#                     group["logical_error_rate"],
#                     yerr=2*group["sigma"],
#                     fmt=STACK_TO_MARKER[max_stack],
#                     markersize=MARKER_SIZE,
#                     linewidth=LINEWIDTH,
#                     color=color,
#                     label=rf"$d={distance}$",
#                     zorder = 5,
#                 )
    
#                 ax.plot(
#                     group["physical_error_rate"],
#                     group["logical_error_rate"],
#                     linewidth=LINEWIDTH,
#                     color=color
#                 )

#             if max_stack == 3:

#                 ax.scatter(
#                     group["physical_error_rate"],
#                     group["logical_error_rate"],
#                     s=SCATTER_SIZE*0.5,
#                     marker=STACK_TO_MARKER[max_stack],
#                     color=color,
#                     alpha=0.5,
#                     zorder=10,
#                     label=None,
#                 )

#                 ax.errorbar(
#                     group["physical_error_rate"],
#                     group["logical_error_rate"],
#                     yerr=2*group["sigma"],
#                     fmt="none",
#                     markersize=MARKER_SIZE,
#                     linewidth=LINEWIDTH,
#                     color=color,
#                     alpha=0.5,
#                     label=None,
#                     zorder=10
#                 )

#         if fit:
#             if max_stack == np.inf:
#                 for distance, group in df_subset.groupby("code_distance"):
#                     group = group[group["physical_error_rate"] < plim]
#                     ax.plot(
#                         group["physical_error_rate"],
#                         group[key],
#                         linestyle=":",
#                         linewidth=LINEWIDTH,
#                         color=DISTANCE_TO_COLOR[distance],
#                     )

#     ax.set_xlabel(
#         r"physical error ($\varepsilon = \varepsilon_d = \varepsilon_m$)",
#         fontsize=LABEL_FONTSIZE,
#         labelpad=-5,
#     )
#     ax.set_ylabel(
#         r"logical error ($\varepsilon_L$)",
#         fontsize=LABEL_FONTSIZE,
#     )

#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlim(*X_ERROR_LIM)
#     ax.set_ylim(*Y_ERROR_LIM)

#     ax.xaxis.set_minor_formatter(NullFormatter())
#     ax.xaxis.tick_top()
#     ax.xaxis.set_label_position("top")

#     ax.tick_params(labelsize=TICK_FONTSIZE_SMALL, width=AXWIDTH)

#     # Grid (as in original)
#     ax.grid(which='major', color=GREY, linestyle='-', linewidth=AXWIDTH/2, alpha=0.2)
#     ax.grid(which='minor', color=GREY, linestyle='-', linewidth=AXWIDTH/3, alpha=0.2)
#     #ax.grid(False)

#     ax.legend(
#         loc="upper left",
#         frameon=False,
#         fontsize=7,
#         ncol=2,
#         columnspacing=0.5,
#     )

#     for spine in ax.spines.values():
#         spine.set_linewidth(AXWIDTH)

#     fig.tight_layout()
#     suffix += "_fit" if fit else suffix
#     plt.savefig(f"{path}/logical{suffix}.pdf")
#     plt.close()


def plot_effective_distance(df, fit, key, suffix="", path=""):
    fig, ax = plt.subplots(figsize=FIGSIZE_GAMMA)

    for max_stack in df["max_stack"].unique():
        df_subset = df[df["max_stack"] == max_stack]
        df_plot = df_subset.drop_duplicates(subset="code_distance")

        ax.errorbar(
            df_plot["code_distance"],
            df_plot["effective_distance"],
            yerr=2*df_plot["sigma_effective_distance"],
            fmt="none",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            color=STACK_TO_COLOR[max_stack]
        )

        ax.scatter(
            df_plot["code_distance"],
            df_plot["effective_distance"],
            s=SCATTER_SIZE,
            marker=STACK_TO_MARKER[max_stack],
            color=STACK_TO_COLOR[max_stack],
            zorder=10,
            label=rf"$m={int(max_stack)}$" if max_stack != np.inf else r"$m=d$",
        )

        ax.plot(
            df_plot["code_distance"],
            df_plot["effective_distance"],
            linewidth=LINEWIDTH,
            color=STACK_TO_COLOR[max_stack],
            zorder=10,
        )

        if fit:
            df_fit = df_subset[df_subset["code_distance"] > D_FIT_CUTOFF]
            alpha = df_fit["alpha"].iloc[0]
            beta = df_fit["beta"].iloc[0]

            ax.plot(
                df_fit["code_distance"],
                df_fit[key],
                linestyle="dotted",
                color=STACK_TO_COLOR[max_stack],
                label=rf"${alpha:.2f}d^{{{beta:.2f}}}$",
            )

    X_ref = np.linspace(*X_GAMMA_LIM, 50)
    ax.plot(
        X_ref,
        (X_ref + 1) / 2,
        linewidth=LINEWIDTH,
        linestyle="dashdot",
        color=BLACK,
        label=r"$\frac{d+1}{2}$",
        zorder=5,
    )

    ax.set_xlabel(r"distance ($d$)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"effective distance ($\gamma_d$)", fontsize=LABEL_FONTSIZE,rotation=270,labelpad=10,)

    ax.set_xlim(*X_GAMMA_LIM)
    ax.set_ylim(*Y_GAMMA_LIM)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.tick_params(which="both", labelsize=TICK_FONTSIZE_SMALL, width=AXWIDTH)

    # Grid (as in original)
    ax.grid(which='major', color=GREY, linestyle='-', linewidth=AXWIDTH/2, alpha=0.2)
    ax.grid(which='minor', color=GREY, linestyle='-', linewidth=AXWIDTH/3, alpha=0.2)
    #ax.grid(False)

    ax.legend(loc="lower right", frameon=False, fontsize=LABEL_FONTSIZE)

    for spine in ax.spines.values():
        spine.set_linewidth(AXWIDTH)

    fig.tight_layout()
    suffix += "_fit" if fit else ""
    plt.savefig(f"{path}/effective_distance{suffix}.pdf")
    plt.close()


def plot_effective_distance_comp(df, fit, key, suffix="", path=""):
    fig, ax = plt.subplots(figsize=FIGSIZE_GAMMA_COMP)

    for max_stack in [3]:
        df_subset = df[df["max_stack"] == max_stack]
        df_plot = df_subset.drop_duplicates(subset="code_distance")

        ax.errorbar(
            df_plot["code_distance"],
            df_plot["effective_distance"],
            yerr=2*df_plot["sigma_effective_distance"],
            fmt="none",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            color=STACK_TO_COLOR[max_stack]
        )

        ax.scatter(
            df_plot["code_distance"],
            df_plot["effective_distance"],
            s=SCATTER_SIZE*2,
            marker=STACK_TO_MARKER[max_stack],
            color=STACK_TO_COLOR[max_stack],
            zorder=10,
            label=r"2DSR*"
        )

        if fit:
            df_fit = df_subset[df_subset["code_distance"] > D_FIT_CUTOFF]
            alpha = df_fit["alpha"].iloc[0]
            beta = df_fit["beta"].iloc[0]

            ax.plot(
                df_fit["code_distance"],
                df_fit[key],
                linestyle="dotted",
                color=STACK_TO_COLOR[max_stack],
                label=rf"${alpha:.2f}d^{{{beta:.2f}}}$",
            )


    ax.scatter(
                    d_lake,
                    d_eff_lake,
                    s=SCATTER_SIZE*2,
                    edgecolor=BLUE,
                    facecolors='none',
                    marker='o',
                    zorder=8,
                    label=r"Lake",
                )
    
    ax.scatter(
                    d_winter,
                    d_eff_winter,
                    s=SCATTER_SIZE*2,
                    marker="x",
                    color=GREEN,
                    linewidths=1,
                    zorder=8,
                    label=r"Winter",
                )

    ax.errorbar(
                    d_winter,
                    d_eff_winter,
                    yerr=[2*s for s in d_eff_winter_uncertainty],
                    fmt="none",
                    markersize=MARKER_SIZE,
                    linewidth=1,
                    color=GREEN,
                    zorder=8,
                )

    X_ref = np.linspace(*X_GAMMA_LIM, 50)
    ax.plot(
        X_ref,
        (X_ref + 1) / 2,
        linewidth=THIN_LINEWIDTH,
        linestyle="dashdot",
        color=BLACK,
        label=r"$\frac{d+1}{2}$",
        zorder=5,
    )

    ax.set_xlabel(r"distance ($d$)", fontsize=LABEL_SMALL_FONTSIZE)
    ax.set_ylabel(r"effective distance ($\gamma_d$)", fontsize=LABEL_SMALL_FONTSIZE)

    ax.set_xlim(*X_GAMMA_COMP_LIM)
    ax.set_ylim(*Y_GAMMA_COMP_LIM)

    ax.tick_params(which="both",labelsize=TICK_FONTSIZE_SMALL, width=THIN_AXWIDTH)

    # Grid (as in original)
    ax.grid(which='major', color=GREY, linestyle='-', linewidth=THIN_AXWIDTH/2, alpha=0.2)
    ax.grid(which='minor', color=GREY, linestyle='-', linewidth=THIN_AXWIDTH/3, alpha=0.2)
    #ax.grid(False)

    ax.legend(loc="lower right", frameon=False, fontsize=LABEL_SMALL_FONTSIZE)

    for spine in ax.spines.values():
        spine.set_linewidth(THIN_AXWIDTH)

    fig.tight_layout()
    suffix += "_fit" if fit else ""
    plt.savefig(f"{path}/effective_distance_comp{suffix}.pdf")
    plt.close()