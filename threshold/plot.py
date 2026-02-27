import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import PercentFormatter

# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

plt.style.use("rgplot")

# ---------------------------------------------------------------------
# Figure / font constants
# ---------------------------------------------------------------------

FIGSIZE_ERROR = (2.0, 2.0)
FIGSIZE_GAMMA = (2.0, 2.1)
FIGSIZE_ESTIMATE = (2.55, 2.4)

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

ERROR_RATES_ESTIMATE = [1e-3, 1e-2]
PL_FIT_CUTOFF = 1e-9

# ---------------------------------------------------------------------
# Axis limits
# ---------------------------------------------------------------------

X_ERROR_LIM = (6e-3, 8e-3)
Y_ERROR_LIM = (1e-6, 1e-2)

X_GAMMA_LIM = (0, 110)
Y_GAMMA_LIM = (0, 40)

X_ESTIMATE_LIM = (1e-14, 1e-6)
Y_ESTIMATE_LIM = (0, 100)

X_THRESHOLD_LIM = (0, 110)
Y_THRESHOLD_LIM = (0.006, 0.009)

# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

BLUE, RED, GREEN, YELLOW = colors[:4]

GREY = "#D6D6D6"
LIGHT_GREY = "#F1F1F1"
MAGENTA = "#CE93D8"
ORANGE = "#FBB03B"
TEAL = "#1B9E77"
NAVY = "#1F3A5F"
SLATE = "#4C72B0"
COPPER = "#8C564B"
BLACK = "black"

DISTANCE_TO_COLOR = {
     5: GREY,
    10: BLUE,
    15: RED,
    20: RED,
    25: GREEN,
    30: GREEN,
    35: YELLOW,
    40: YELLOW,
    45: MAGENTA,
    50: MAGENTA,
    55: COPPER,
    60: COPPER,
    65: TEAL,
    70: TEAL,
    75: BLACK,
    80: BLACK,
    90: SLATE,
    95: SLATE,
   100: BLACK,
   120: NAVY,
   140: ORANGE,
}

# ---------------------------------------------------------------------
# Logical error rate vs physical error
# ---------------------------------------------------------------------

def plot_logical(df, fit, key, plim, path):
    fig, ax = plt.subplots(figsize=FIGSIZE_ERROR)

    df_plot = df.copy().reset_index(drop=True)

    for distance, group in df_plot.groupby("code_distance"):
        color = DISTANCE_TO_COLOR[distance]

        ax.errorbar(
            group["physical_error_rate"],
            group["logical_error_rate"],
            yerr=group["sigma"],
            fmt="o",
            markersize=MARKER_SIZE,
            capsize=2.5,
            color=color,
            label=rf"$d={distance}$",
        )

    if fit:
        for distance, group in df_plot.groupby("code_distance"):
            group = group[group["physical_error_rate"] <= plim]
            color = DISTANCE_TO_COLOR[distance]

            ax.plot(
                group["physical_error_rate"],
                group[key],
                linewidth=LINEWIDTH,
                color=color,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*X_ERROR_LIM)
    ax.set_ylim(*Y_ERROR_LIM)

    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE_SMALL)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    # Grid (as in original)
    ax.grid(which='major', color=GREY, linestyle='-', linewidth=AXWIDTH/2, alpha=0.2)
    ax.grid(which='minor', color=GREY, linestyle='-', linewidth=AXWIDTH/3, alpha=0.2)
    #ax.grid(False)

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=7,
        handletextpad=0.1,
        labelspacing=0.25,
        borderpad=0.25,
    )

    suffix = "" if not fit else "_fit"
    fig.tight_layout()
    plt.savefig(f"{path}/logical{suffix}.pdf")
    plt.close()

# ---------------------------------------------------------------------
# Effective distance vs code distance
# ---------------------------------------------------------------------

def plot_effective_distance(df, fit, key, path):
    fig, ax = plt.subplots(figsize=FIGSIZE_GAMMA)

    df_plot = df.drop_duplicates(subset="code_distance")

    ax.scatter(
        df_plot["code_distance"],
        df_plot["effective_distance"],
        marker='o',
        s=SCATTER_SIZE,
        color=BLUE,
        zorder=12,
        label="signal-rule",
    )

    ax.plot(
        df_plot["code_distance"],
        df_plot["effective_distance"],
        color=BLUE,
        linewidth=LINEWIDTH,
        zorder=10,
    )

    X_ref = np.linspace(*X_GAMMA_LIM, 50)
    ax.plot(
        X_ref,
        (X_ref + 1) / 2,
        linestyle="dashdot",
        color=BLACK,
        label=r"$\frac{d+1}{2}$",
        zorder=10,
    )

    if fit:
        df_fit = df[df["code_distance"] > D_FIT_CUTOFF]
        alpha = df_fit["alpha"].iloc[0]
        beta = df_fit["beta"].iloc[0]

        ax.plot(
            df_fit["code_distance"],
            df_fit[key],
            linestyle="dotted",
            color=BLACK,
            label=rf"${alpha:.2f}d^{{{beta:.2f}}}$",
            zorder=10,
        )

        ax.set_title(
            r"$\varepsilon_L = A d (\varepsilon/\varepsilon_{th})^{\gamma_d}$"
        )

    ax.set_xlabel(r"distance ($d$)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"effective distance ($\gamma_d$)", fontsize=LABEL_FONTSIZE)

    ax.set_xlim(*X_GAMMA_LIM)
    ax.set_ylim(*Y_GAMMA_LIM)

    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE_SMALL)
    # Grid (as in original)
    ax.grid(which='major', color=GREY, linestyle='-', linewidth=AXWIDTH/2, alpha=0.2)
    ax.grid(which='minor', color=GREY, linestyle='-', linewidth=AXWIDTH/3, alpha=0.2)
    #ax.grid(False)

    ax.legend(loc="lower right", frameon=False, fontsize=7)

    suffix = "" if not fit else "_fit"
    fig.tight_layout(pad=0.4)
    plt.savefig(f"{path}/effective_distance{suffix}.pdf")
    plt.close()

# ---------------------------------------------------------------------
# Threshold vs code distance
# ---------------------------------------------------------------------

def plot_threshold_d(df, pth, path):
    fig, ax = plt.subplots(figsize=FIGSIZE_GAMMA)

    df_plot = df.drop_duplicates(subset="code_distance")
    max_distance = df_plot["code_distance"].max()
    df_plot = df_plot[df_plot["code_distance"] != max_distance]

    ax.scatter(
        df_plot["code_distance"],
        df_plot["error_threshold"],
        marker='o',
        s=SCATTER_SIZE,
        color=BLUE,
        zorder=10,
    )

    ax.plot(
        df_plot["code_distance"],
        df_plot["error_threshold"],
        color=BLUE,
        linewidth=LINEWIDTH,
        zorder=10,
    )

    ax.axhline(
        y=pth,
        linestyle="--",
        linewidth=AXWIDTH,
        color=BLACK,
        label=r"$\varepsilon_c$",
    )

    ax.set_xlabel(r"distance ($d$)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(
        r"crossing-point",
        fontsize=LABEL_FONTSIZE,
        rotation=270,
        labelpad=10,
    )

    ax.set_xlim(*X_THRESHOLD_LIM)
    ax.set_ylim(*Y_THRESHOLD_LIM)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))

    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE_SMALL)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # Grid (as in original)
    ax.grid(which='major', color=GREY, linestyle='-', linewidth=AXWIDTH/2, alpha=0.2)
    ax.grid(which='minor', color=GREY, linestyle='-', linewidth=AXWIDTH/3, alpha=0.2)
    #ax.grid(False)

    ax.legend(loc="upper right", frameon=False, fontsize=LABEL_FONTSIZE)

    for spine in ax.spines.values():
        spine.set_linewidth(AXWIDTH)

    ax.tick_params(width=AXWIDTH, length=4)

    fig.tight_layout()
    plt.savefig(f"{path}/threshold.pdf")
    plt.close()


# import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# from matplotlib.colors import to_rgb
# import matplotlib.patches as patches
# from matplotlib.ticker import PercentFormatter


# # ---------------------------------------------------------------------
# # Plot style
# # ---------------------------------------------------------------------

# plt.style.use('rgplot')

# FIGSIZE_ERROR = (2, 2)
# FIGSIZE_GAMMA = (2, 2.1)
# FIGSIZE_ESTIMATE = (2.55, 2.4)

# LABEL_FONTSIZE = 7
# TICK_FONTSIZE_SMALL = 5.5
# TICK_FONTSIZE_SMALL = 6.5
# AXWIDTH  = 0.8

# LINEWIDTH = 1.2
# MARKER_SIZE = 2.5
# SCATTER_SIZE = 10

# # ---------------------------------------------------------------------
# # Physical / numerical cutoffs
# # ---------------------------------------------------------------------

# D_PLOT_CUTOFF = 6
# D_FIT_CUTOFF = 15

# ERROR_RATES_ESTIMATE = [1e-3, 1e-2]

# PL_FIT_CUTOFF = 1e-9

# # ---------------------------------------------------------------------
# # Axis limits
# # ---------------------------------------------------------------------

# #X_ERROR_LIM = (5e-3, 9e-3)
# #Y_ERROR_LIM = (1e-8, 1e-2)
# X_ERROR_LIM = (6e-3, 8e-3)
# Y_ERROR_LIM = (1e-6, 1e-2)


# X_GAMMA_LIM = (0, 100)
# Y_GAMMA_LIM = (0, 40)

# X_ESTIMATE_LIM = (1e-14, 1e-6)
# Y_ESTIMATE_LIM = (0, 100)

# X_THRESHOLD_LIM = (0,100)
# Y_THRESHOLD_LIM = (0.006,0.009)

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

# BLUE = colors[0]
# RED = colors[1]
# GREEN = colors[2]
# YELLOW = colors[3]

# DISTANCE_TO_COLOR = {5:GREY,
#                     10:BLUE,
#                     15:RED,
#                     20:RED,
#                     25:GREEN,
#                     30:GREEN,
#                     35:YELLOW,
#                     40:YELLOW,
#                     45:MAGENTA,
#                     50:MAGENTA,
#                     55:COPPER,
#                     60:COPPER,
#                     65:TEAL,
#                     70:TEAL,
#                     75:BLACK,
#                     80:BLACK,
#                     80:COPPER,
#                     90:SLATE,
#                     95:SLATE,
#                     100:BLACK,
#                     120:NAVY,
#                     140:ORANGE}


# def plot_logical(df, fit, key, plim, path):
#     fig, ax = plt.subplots(figsize=FIGSIZE_ERROR)

#     df_plot = df.copy().reset_index(drop=True)

#     for distance, group in df_plot.groupby("code_distance"):
#         color = DISTANCE_TO_COLOR[distance]

#         ax.errorbar(
#             group["physical_error_rate"], group["logical_error_rate"], yerr=group["sigma"],
#             fmt="o", color=color, capsize=2.5,
#             markersize=MARKER_SIZE, label=rf"$d={distance}$"
#         )

#     if fit:
#         for distance, group in df_plot.groupby("code_distance"):
#             group = group[group["physical_error_rate"] <= plim]
#             color = DISTANCE_TO_COLOR[distance]
#             ax.plot(group["physical_error_rate"], group[key],
#                     linewidth=LINEWIDTH, color=color)

#     # ax.set_xlabel(r"physical error ($\varepsilon = \varepsilon_d = \varepsilon_m$)",
#     #               fontsize=LABEL_FONTSIZE, labelpad=0.5)
#     # ax.set_ylabel(r"logical error ($\varepsilon_L$)",
#     #               fontsize=LABEL_FONTSIZE, labelpad=0.5)

#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlim(*X_ERROR_LIM)
#     ax.set_ylim(*Y_ERROR_LIM)

#     ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE_SMALL)
#     ax.yaxis.tick_right()
#     ax.yaxis.set_label_position("right")
#     ax.grid(False)

#     ax.legend(loc="lower right", frameon=False, fontsize=7,
#               handletextpad=0.1, labelspacing=0.25, borderpad=0.25)

#     suffix = "" if not fit else "_fit"
#     fig.tight_layout()
#     plt.savefig(f"{path}/logical{suffix}.pdf")
#     plt.close()


# def plot_effective_distance(df, fit, key, path):
#     fig, ax = plt.subplots(figsize=FIGSIZE_GAMMA)

#     df_plot = df.drop_duplicates(subset="code_distance")

#     ax.scatter(df_plot["code_distance"], df_plot["effective_distance"],
#                s=SCATTER_SIZE, color=BLUE, marker = 'o', zorder=-10, label="signal-rule")
#     ax.plot(df_plot["code_distance"], df_plot["effective_distance"],
#             color=BLUE, linewidth=LINEWIDTH, zorder=-10)

#     X_ref = np.linspace(*X_GAMMA_LIM, 50)
#     ax.plot(X_ref, (X_ref + 1) / 2,
#             linestyle="dashdot", color=BLACK,
#             label=r"$\frac{d+1}{2}$", zorder=-40)

#     if fit:
#         df_fit = df[df["code_distance"] > D_FIT_CUTOFF]
#         alpha = df_fit["alpha"].iloc[0]
#         beta = df_fit["beta"].iloc[0]

#         ax.plot(df_fit["code_distance"], df_fit[key],
#                 linestyle="dotted", color=BLACK,
#                 label=rf"${alpha:.2f}d^{{{beta:.2f}}}$")

#         ax.set_title(r"$\varepsilon_L = A d (\varepsilon/\varepsilon_{th})^{\gamma_d}$")

#     ax.set_xlabel(r"distance ($d$)", fontsize=LABEL_FONTSIZE, labelpad=0.5)
#     ax.set_ylabel(r"effective distance ($\gamma_d$)", fontsize=LABEL_FONTSIZE, labelpad=0.5)

#     ax.set_xlim(*X_GAMMA_LIM)
#     ax.set_ylim(*Y_GAMMA_LIM)

#     ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE_SMALL)
#     ax.grid(False)

#     ax.legend(loc="lower right", frameon=False, fontsize=7)
#     fig.tight_layout(pad=0.4)

#     suffix = "" if not fit else "_fit"
#     fig.tight_layout()
#     plt.savefig(f"{path}/effective_distance{suffix}.pdf")
#     plt.close()


# def plot_threshold_d(df, pth, path):
#     fig, ax = plt.subplots(figsize=FIGSIZE_GAMMA)

#     df_plot = df.drop_duplicates(subset="code_distance")

#     # Find the maximum code_distance
#     max_distance = df_plot["code_distance"].max()

#     # Filter out the row(s) with the maximum code_distance
#     df_filtered = df_plot[df_plot["code_distance"] != max_distance]

#     # Plot using the filtered DataFrame
#     ax.scatter(df_filtered["code_distance"], df_filtered["error_threshold"],
#             s=SCATTER_SIZE, color=BLUE, marker="o",zorder=-10)
#     ax.plot(df_filtered["code_distance"], df_filtered["error_threshold"],
#         color=BLUE, linewidth=LINEWIDTH, zorder=-10)
    
#     ax.axhline(
#         y=pth,
#         linestyle="--",
#         linewidth=AXWIDTH,
#         color="black",
#         label=r"$\varepsilon_c$"
#     )

#     ax.set_xlabel(r"distance ($d$)", fontsize=LABEL_FONTSIZE, labelpad=0.5)
#     ax.set_ylabel(r"crossing-point", fontsize=LABEL_FONTSIZE, rotation=270, labelpad=10)

#     ax.set_xlim(*X_THRESHOLD_LIM)
#     ax.set_ylim(*Y_THRESHOLD_LIM)

#     plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))

#     ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE_SMALL)
#     ax.yaxis.tick_right()
#     ax.yaxis.set_label_position("right")
#     ax.grid(False)

#     ax.legend(loc="upper right", frameon=False, fontsize=LABEL_FONTSIZE)

#     suffix = ""

#     # Thinner axis lines (spines)
#     for spine in ax.spines.values():
#         spine.set_linewidth(AXWIDTH)  # default is usually ~1.5

#     # Thinner ticks
#     ax.tick_params(width=AXWIDTH, length=4)  # width of ticks, length in points

#     fig.tight_layout()
#     plt.savefig(f"{path}/threshold{suffix}.pdf")
#     plt.close()