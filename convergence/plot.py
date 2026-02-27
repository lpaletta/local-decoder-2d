import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

plt.style.use("rgplot")

# ---------------------------------------------------------------------
# Figure / font constants
# ---------------------------------------------------------------------

FIGSIZE_CONVERGENCE = (2.6, 2.0)
FIGSIZE_CUTOFF = (2.0, 2.0)

LABEL_FONTSIZE = 7
TICK_FONTSIZE = 5.5

AXWIDTH = 0.6
LINEWIDTH = 1.2
SCATTER_SIZE = 10

# ---------------------------------------------------------------------
# Axis limits
# ---------------------------------------------------------------------

X_CONVERGENCE_LIM = (0, 1000)
Y_CONVERGENCE_LIM = (1e-7, 1e-3)

X_CUTOFF_LIM = (0, 55)
Y_CUTOFF_LIM = (0, 800)

# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

GREY = "#D6D6D6"
MAGENTA = "#CE93D8"
APPLE = "#A8D5A2"
PINK = "#FADADD"

BLUE, RED, GREEN, YELLOW = colors[:4]

DISTANCE_TO_COLOR = {
    5: GREY,
    9: BLUE,
    15: RED,
    25: GREEN,
    35: PINK,
    50: YELLOW,
    75: APPLE,
    100: MAGENTA
}

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_convergence(df, path):
    fig, ax = plt.subplots(figsize=FIGSIZE_CONVERGENCE)
    ax.errorbar([],[],[], label="")

    for distance, group in df.groupby("code_distance"):
        group = group[group["error_count"] < 3 * group["run_count"] / 4]

        # ax.plot(group["time_run"],
        #     group["cummulative_error"] / group["time_run"],
        #     color=DISTANCE_TO_COLOR[distance],
        #     label=rf"$d={distance}$")
        # ax.fill_between(
        #     group["time_run"],
        #     group["cummulative_error"] / group["time_run"] - 1.96 * group["sigma"] / group["time_run"],
        #     group["cummulative_error"] / group["time_run"] + 1.96 * group["sigma"] / group["time_run"],
        #     color=DISTANCE_TO_COLOR[distance],
        #     alpha=0.3,
        # )

        ax.errorbar(
            group["time_run"],
            group["cummulative_error"] / group["time_run"],
            yerr=2 * group["sigma"] / group["time_run"],
            fmt="o",
            markersize=2,
            color=DISTANCE_TO_COLOR[distance],
            label=rf"$d={distance}$",
        )

    ax.set_xlabel(r"simulation time ($\tau$)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"$P(\tau)/\tau$", fontsize=LABEL_FONTSIZE)

    ax.set_yscale("log")
    ax.set_xlim(*X_CONVERGENCE_LIM)
    ax.set_ylim(*Y_CONVERGENCE_LIM)

    ax.tick_params(labelsize=TICK_FONTSIZE, width=AXWIDTH)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Grid (as in original)
    ax.grid(which='major', color=GREY, linestyle='-', linewidth=AXWIDTH/2, alpha=0.2)
    ax.grid(which='minor', color=GREY, linestyle='-', linewidth=AXWIDTH/3, alpha=0.2)
    #ax.grid(False)

    # ax.legend(
    #     loc="lower right",
    #     frameon=False,
    #     fontsize=LABEL_FONTSIZE,
    #     ncol=2,
    #     columnspacing=0.5,
    # )

    for spine in ax.spines.values():
        spine.set_linewidth(AXWIDTH)

    fig.tight_layout()
    plt.savefig(f"{path}/convergence.pdf")
    plt.close()


def plot_cutoff(df, path):
    fig, ax = plt.subplots(figsize=FIGSIZE_CUTOFF)

    df_plot = df[["code_distance", "time_cutoff", "time_cutoff_fit"]].drop_duplicates()

    ax.scatter(
        df_plot["code_distance"],
        df_plot["time_cutoff"],
        marker="o",
        s=SCATTER_SIZE,
        color=BLUE,
        zorder=6
    )

    mask = df_plot["code_distance"] >= 10
    # ax.plot(
    #     df_plot.loc[mask, "code_distance"],
    #     df_plot.loc[mask, "time_cutoff_fit"],
    #     linestyle="dotted",
    #     color="black",
    #     zorder=5,
    # )

    ax.set_xlabel(r"distance ($d$)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"cut-off time ($\tau_d$)", fontsize=LABEL_FONTSIZE,rotation=270,labelpad=10,)

    ax.set_xlim(*X_CUTOFF_LIM)
    ax.set_ylim(*Y_CUTOFF_LIM)

    ax.tick_params(labelsize=TICK_FONTSIZE, width=AXWIDTH)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # Grid (as in original)
    ax.grid(which='major', color=GREY, linestyle='-', linewidth=AXWIDTH/2, alpha=0.2)
    ax.grid(which='minor', color=GREY, linestyle='-', linewidth=AXWIDTH/3, alpha=0.2)
    #ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(AXWIDTH)

    fig.tight_layout()
    plt.savefig(f"{path}/cutoff.pdf")
    plt.close()

# import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# from matplotlib.colors import to_rgb
# import matplotlib.patches as patches


# # ---------------------------------------------------------------------
# # Plot style
# # ---------------------------------------------------------------------

# plt.style.use('rgplot')

# FIGSIZE = (3.2, 2)
# CUTOFF_FIGSIZE = (2, 2)

# LABEL_FONTSIZE = 7
# TICK_FONTSIZE = 5.5
# TICK_FONTSIZE = 6.5
# AXWIDTH = 0.6

# LINEWIDTH = 1.2
# MARKER_SIZE = 3.5
# SCATTER_SIZE = 14

# FIGSIZE = (2.4, 1.8)

# # ---------------------------------------------------------------------
# # Physical / numerical cutoffs
# # ---------------------------------------------------------------------

# # ---------------------------------------------------------------------
# # Axis limits
# # ---------------------------------------------------------------------

# CONVERGENCE_X_LIM = (0, 1000)
# CONVERGENCE_Y_LIM = (10**-7, 10**-3)

# # DENSITY_X_LIM = (0, 100)
# # DENSITY_Y_LIM = (0, 0.2)

# CUTOFF_X_LIM = (0,50)
# CUTOFF_Y_LIM = (0,800)

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
#                     35:GREY,
#                     50:YELLOW,
#                     75:APPLE,
#                     100:MAGENTA}


# def plot_convergence(df, path):
#     fig, ax = plt.subplots(figsize=FIGSIZE)

#     df_plot = df.copy().reset_index(drop=True)

#     # Precompute sorted unique time grid
#     #time_grid = np.sort(df_plot["time_run"].unique())

#     for distance, group in df_plot.groupby("code_distance"):

#         # Subsample time grid depending on n
#         # if distance == 9:
#         #     step = 2
#         # elif distance == 15:
#         #     step = 10
#         # elif distance > 15:
#         #     step = 20
#         # else:
#         #     step = None

#         # if step is not None:
#         #     time_reduced = time_grid[::step]
#         #     group = group[group["time_run"].isin(time_reduced)]

#         # Error count filter
#         group = group[
#             group["error_count"] < (3 * group["run_count"] / 4)
#         ]

#         ax.errorbar(
#             group["time_run"],
#             group["cummulative_error"]/group["time_run"],
#             yerr=2 * group["sigma"]/group["time_run"],
#             fmt="o",
#             color=DISTANCE_TO_COLOR[distance],
#             capsize=1,
#             markersize=2,
#             label=rf"$d={distance}$",
#         )

#     # Axes formatting
#     ax.set_xlabel(r"simulation time ($\tau$)",
#                   fontsize=LABEL_FONTSIZE, labelpad=0.5)
#     ax.set_ylabel(r"$P(\tau)/\tau$",
#                   fontsize=LABEL_FONTSIZE, labelpad=0.5)

#     #ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlim(*CONVERGENCE_X_LIM)
#     ax.set_ylim(*CONVERGENCE_Y_LIM)

#     #ax.set_xticks([1, 1e5, 5e5, 1e6])
#     #ax.set_xticklabels(["1", "10⁵", "5×10⁵", "10⁶"])
#     #ax.set_xticks([i * 1e5 for i in range(1, 10)], minor=True)

#     ax.tick_params(axis="both", which="both",
#                    labelsize=TICK_FONTSIZE)
#     ax.yaxis.set_minor_formatter(ticker.NullFormatter())

#     ax.grid(False)

#     ax.legend(
#         loc="lower right",
#         frameon=False,
#         fontsize=LABEL_FONTSIZE,
#         handletextpad=0.1,
#         labelspacing=0.25,
#         borderpad=0.25,
#         ncol=2,
#         columnspacing=0.5,
#     )

#     for spine in ax.spines.values():
#         spine.set_linewidth(AXWIDTH)  # default is usually ~1.5

#     # Thinner ticks
#     ax.tick_params(width=AXWIDTH, length=4)  # width of ticks, length in points

#     plt.savefig(f"{path}/convergence.pdf")
#     plt.close()


# def plot_cutoff(df, path):
#     fig, ax = plt.subplots(figsize=CUTOFF_FIGSIZE)

#     df_plot = (
#         df[["code_distance", "time_cutoff", "time_cutoff_fit"]]
#         .drop_duplicates()
#         .copy()
#     )

#     ax.scatter(
#         df_plot["code_distance"],
#         df_plot["time_cutoff"],
#         s=SCATTER_SIZE,
#         color=BLUE,
#         marker="o",
#     )

#     mask = df_plot["code_distance"] >= 10
#     ax.plot(
#         df_plot.loc[mask, "code_distance"],
#         df_plot.loc[mask, "time_cutoff_fit"],
#         color=BLACK,
#         linestyle="dotted",
#         zorder=-1,
#     )

#     # Axes
#     ax.set_xlim(*CUTOFF_X_LIM)
#     ax.set_ylim(*CUTOFF_Y_LIM)

#     ax.set_xlabel(
#         r"distance ($d$)",
#         fontsize=LABEL_FONTSIZE,
#         labelpad=0.5,
#     )
#     ax.set_ylabel(
#         r"cut-off time ($\tau_d$)",
#         fontsize=LABEL_FONTSIZE,
#         labelpad=0.5,
#     )

#     ax.tick_params(
#         axis="both",
#         which="both",
#         labelsize=TICK_FONTSIZE,
#     )

#     ax.grid(False)

#     for spine in ax.spines.values():
#         spine.set_linewidth(AXWIDTH)  # default is usually ~1.5

#     # Thinner ticks
#     ax.tick_params(width=AXWIDTH, length=4)  # width of ticks, length in points

#     plt.savefig(f"{path}/cutoff.pdf")
#     plt.close()