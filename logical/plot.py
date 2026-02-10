import numpy as np
import pandas as pd

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

LABEL_FONTSIZE = 7
TICK_FONTSIZE_SMALL = 5.5

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

# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

GREY = "#D6D6D6"
MAGENTA = "#CE93D8"
NAVY = "#1F3A5F"
APPLE = "#A8D5A2"
BLACK = "black"

BLUE, RED, GREEN, YELLOW = colors[:4]

DISTANCE_TO_COLOR = {
    5: GREY,
    9: BLUE,
    15: RED,
    25: GREEN,
    35: NAVY,
    50: YELLOW,
    75: APPLE,
    100: MAGENTA,
}

STACK_TO_COLOR = {
    3: GREEN,
    4: GREEN,
    7: YELLOW,
    8: YELLOW,
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
# Plots
# ---------------------------------------------------------------------

def plot_logical(df, fit, key, plim, suffix="", path=""):
    fig, ax = plt.subplots(figsize=FIGSIZE_ERROR)

    for distance, group in df.groupby("code_distance"):
        color = DISTANCE_TO_COLOR[distance]

        ax.errorbar(
            group["physical_error_rate"],
            group["logical_error_rate"],
            yerr=group["sigma"],
            fmt="o",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            color=color,
            label=rf"$d={distance}$",
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
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

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
    suffix += "_fit" if fit else suffix
    plt.savefig(f"{path}/logical{suffix}.pdf")
    plt.close()


def plot_effective_distance(df, fit, key, suffix="", path=""):
    fig, ax = plt.subplots(figsize=FIGSIZE_GAMMA)

    for max_stack in df["max_stack"].unique():
        df_subset = df[df["max_stack"] == max_stack]
        df_plot = df_subset.drop_duplicates(subset="code_distance")

        ax.scatter(
            df_plot["code_distance"],
            df_plot["effective_distance"],
            s=SCATTER_SIZE,
            marker=STACK_TO_MARKER[max_stack],
            color=STACK_TO_COLOR[max_stack],
            zorder=10,
            label=rf"$m={int(max_stack) if max_stack < np.inf else 'd'}$",
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
    ax.set_ylabel(r"effective distance ($\gamma_d$)", fontsize=LABEL_FONTSIZE)

    ax.set_xlim(*X_GAMMA_LIM)
    ax.set_ylim(*Y_GAMMA_LIM)

    ax.tick_params(labelsize=TICK_FONTSIZE_SMALL, width=AXWIDTH)

    # Grid (as in original)
    ax.grid(which='major', color=GREY, linestyle='-', linewidth=AXWIDTH/2, alpha=0.2)
    ax.grid(which='minor', color=GREY, linestyle='-', linewidth=AXWIDTH/3, alpha=0.2)
    #ax.grid(False)

    ax.legend(loc="lower right", frameon=False, fontsize=7)

    for spine in ax.spines.values():
        spine.set_linewidth(AXWIDTH)

    fig.tight_layout()
    suffix += "_fit" if fit else ""
    plt.savefig(f"{path}/effective_distance{suffix}.pdf")
    plt.close()



# import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt
# from matplotlib.colors import to_rgb
# from matplotlib.ticker import NullLocator, NullFormatter
# import matplotlib.ticker as ticker


# # ---------------------------------------------------------------------
# # Plot style
# # ---------------------------------------------------------------------

# plt.style.use('rgplot')

# FIGSIZE_ERROR = (3.2, 2)
# FIGSIZE_GAMMA = (1.85, 2.1)
# FIGSIZE_ESTIMATE = (2.55, 2.4)

# LABEL_FONTSIZE = 7
# TICK_FONTSIZE_SMALL = 5.5
# TICK_FONTSIZE = 6.5
# AXWIDTH  = 0.8

# LINEWIDTH = 1.2
# MARKER_SIZE = 2.5
# SCATTER_SIZE = 10

# # ---------------------------------------------------------------------
# # Physical / numerical cutoffs
# # ---------------------------------------------------------------------

# D_PLOT_CUTOFF = 6
# D_FIT_CUTOFF = 15

# PL_FIT_CUTOFF = 1e-9

# # ---------------------------------------------------------------------
# # Axis limits
# # ---------------------------------------------------------------------

# X_ERROR_LIM = (1e-3, 1e-2)
# Y_ERROR_LIM = (1e-8, 1e-2)

# X_GAMMA_LIM = (0, 100)
# Y_GAMMA_LIM = (0, 30)

# # ---------------------------------------------------------------------
# # Colors
# # ---------------------------------------------------------------------

# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# GREY = "#D6D6D6"
# LIGHT_GREY = "#F1F1F1"
# MAGENTA = "#CE93D8"
# ORANGE = "#FBB03B"
# TEAL = "#1B9E77"
# NAVY = "#1F3A5F"
# SLATE = "#4C72B0"
# COPPER = "#8C564B"
# BLACK = "black"
# LAVENDER = "#C3B1E1"
# APPLE = "#A8D5A2"


# BLUE = colors[0]
# RED = colors[1]
# GREEN = colors[2]
# YELLOW = colors[3]

# DISTANCE_TO_COLOR = {5:GREY,
#                     9:BLUE,
#                     15:RED,
#                     25:GREEN,
#                     35:NAVY,
#                     50:YELLOW,
#                     75:APPLE,
#                     100:MAGENTA}

# STACK_TO_COLOR = {3:GREEN,
#                   4:GREEN,
#                   7:YELLOW,
#                   8:YELLOW,
#                   np.inf:BLUE}

# STACK_TO_MARKER = {3:'^',
#                   4:'^',
#                   7:'d',
#                   8:'d',
#                   np.inf:'o'}


# def plot_logical(df, fit, key, plim, path, suffix=""):
#     fig, ax = plt.subplots(figsize=FIGSIZE_ERROR)

#     df_plot = df.copy().reset_index(drop=True)

#     for distance, group in df_plot.groupby("code_distance"):

#         # ax.errorbar(
#         #     group["physical_error_rate"], group["logical_error_rate"], yerr=group["sigma"],
#         #     fmt="o", color=DISTANCE_TO_COLOR[distance], capsize=2.5,
#         #     markersize=MARKER_SIZE, label=rf"$d={distance}$"
#         # )

#         # ax.errorbar(
#         #     group["physical_error_rate"], group["logical_error_rate"], yerr=group["sigma"],
#         #     fmt="o",                    # square markers instead of circles
#         #     color=DISTANCE_TO_COLOR[distance], 
#         #     capsize=3,                  # slightly larger caps
#         #     elinewidth=1,             # thicker error bars
#         #     markersize=3,               # slightly bigger markers
#         #     #markerfacecolor='white',    # hollow markers for contrast
#         #     markeredgewidth=1,          # thicker edge for clarity
#         #     label=rf"$d={distance}$"                # slight transparency for overlapping points
#         # )

#         ax.errorbar(
#             group["physical_error_rate"], group["logical_error_rate"], yerr=group["sigma"],
#             fmt='o',                     # line only, no markers
#             color=DISTANCE_TO_COLOR[distance],
#             markersize=MARKER_SIZE,
#             linewidth=LINEWIDTH,
#             label=rf"$d={distance}$"
#         )
#         ax.plot(group["physical_error_rate"], group["logical_error_rate"], color=DISTANCE_TO_COLOR[distance], linewidth=LINEWIDTH)

#     if fit:
#         for distance, group in df_plot.groupby("code_distance"):
#             group = group[group["physical_error_rate"] < plim]
#             ax.plot(group["physical_error_rate"], group[key],
#                     linestyle=":", linewidth=LINEWIDTH, color=DISTANCE_TO_COLOR[distance])

#     ax.set_xlabel(r"physical error ($\varepsilon = \varepsilon_d = \varepsilon_m$)",
#                   fontsize=LABEL_FONTSIZE, labelpad=-5)
#     ax.set_ylabel(r"logical error ($\varepsilon_L$)",
#                   fontsize=LABEL_FONTSIZE, labelpad=0.5)

#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlim(*X_ERROR_LIM)
#     ax.set_ylim(*Y_ERROR_LIM)

#     # Disable minor ticks on the x-axis
#     ax.xaxis.set_minor_formatter(NullFormatter())

#     ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE_SMALL)
#     ax.xaxis.tick_top()
#     ax.xaxis.set_label_position("top")


#     # Major grid
#     ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

#     # Minor grid
#     ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.2)

#     #ax.grid(False)

#     ax.legend(loc="upper left", frameon=False, fontsize=7,
#               handletextpad=0.1, labelspacing=0.25, borderpad=0.25, ncol=2)
    
#     # Thinner axis lines (spines)
#     for spine in ax.spines.values():
#         spine.set_linewidth(AXWIDTH)  # default is usually ~1.5

#     # Thinner ticks
#     ax.tick_params(width=AXWIDTH, length=4)  # width of ticks, length in points

#     fig.tight_layout()
#     suffix += "_fit" if fit else ""
#     plt.savefig(f"{path}/logical{suffix}.pdf")
#     plt.close()


# def plot_effective_distance(df, fit, key, path):
#     fig, ax = plt.subplots(figsize=FIGSIZE_GAMMA)

#     for max_stack in df["max_stack"].unique():
#         df_subset = df[df["max_stack"] == max_stack]

#         df_plot = df_subset.copy().drop_duplicates(subset="code_distance")

#         ax.scatter(df_plot["code_distance"], df_plot["effective_distance"],
#                 s=SCATTER_SIZE, color=STACK_TO_COLOR[max_stack], marker=STACK_TO_MARKER[max_stack], zorder=-10, label=f"$m={int(max_stack) if max_stack < np.inf else 'd'}$")
#         ax.plot(df_plot["code_distance"], df_plot["effective_distance"],
#                 color=STACK_TO_COLOR[max_stack], linewidth=LINEWIDTH, zorder=-10)

#         if fit:
#             df_fit = df_subset[df_subset["code_distance"] > D_FIT_CUTOFF]
#             alpha = df_fit["alpha"].iloc[0]
#             beta = df_fit["beta"].iloc[0]

#             ax.plot(df_fit["code_distance"], df_fit[key],
#                     linestyle="dotted", color=STACK_TO_COLOR[max_stack],
#                     label=rf"${alpha:.2f}d^{{{beta:.2f}}}$")

#             ax.set_title(r"$\varepsilon_L = A d (\varepsilon/\varepsilon_{th})^{\gamma_d}$")

#     X_ref = np.linspace(*X_GAMMA_LIM, 50)
#     ax.plot(X_ref, (X_ref + 1) / 2,
#                 linestyle="dashdot", color=BLACK,
#                 label=r"$\frac{d+1}{2}$", zorder=-40)

#     ax.set_xlabel(r"distance ($d$)", fontsize=LABEL_FONTSIZE, labelpad=0.5)
#     ax.set_ylabel(r"effective distance ($\gamma_d$)", fontsize=LABEL_FONTSIZE, labelpad=0.5)

#     ax.set_xlim(*X_GAMMA_LIM)
#     ax.set_ylim(*Y_GAMMA_LIM)

#     ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE_SMALL)
#     ax.grid(False)

#     ax.legend(loc="lower right", frameon=False, fontsize=7)

#     for spine in ax.spines.values():
#         spine.set_linewidth(AXWIDTH)  # default is usually ~1.5

#     # Thinner ticks
#     ax.tick_params(width=AXWIDTH, length=4)  # width of ticks, length in points

#     fig.tight_layout()

#     suffix = "" if not fit else "_fit"
#     plt.savefig(f"{path}/effective_distance{suffix}.pdf")
#     plt.close()