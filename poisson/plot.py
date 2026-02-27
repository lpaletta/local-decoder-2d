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

FIGSIZE_POISSON = (3.2, 2.0)

LABEL_FONTSIZE = 7
TICK_FONTSIZE = 5.5

AXWIDTH = 0.8
LINEWIDTH = 1.2
MARKER_SIZE = 3.5

# ---------------------------------------------------------------------
# Axis limits
# ---------------------------------------------------------------------

X_POISSON_LIM = (0, 1e5)
Y_POISSON_LIM = (0.1, 1.0)

# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

GREY = "#D6D6D6"
MAGENTA = "#CE93D8"
ORANGE = "#FBB03B"
APPLE = "#A8D5A2"
COPPER = "#8C564B"
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
# Plot
# ---------------------------------------------------------------------

def plot_poisson(df, path):
    fig, ax = plt.subplots(figsize=FIGSIZE_POISSON)

    for distance, group in df.groupby("code_distance"):
        group = group[group["error_count"] < 3 * group["run_count"] / 4]

        # ax.plot(group["time_run"],
        #     1 - 4 / 3 * group["cummulative_error"],
        #     color=DISTANCE_TO_COLOR[distance],
        #     label=rf"$d={distance}$")
        # ax.fill_between(
        #     group["time_run"],
        #     1 - 4 / 3 * (group["cummulative_error"] - group["sigma"]),
        #     1 - 4 / 3 * (group["cummulative_error"] + group["sigma"]),
        #     color=DISTANCE_TO_COLOR[distance],
        #     alpha=0.3,
        # )

        ax.errorbar(
            group["time_run"],
            (1 - 4/3*group["cummulative_error"]),
            yerr= 4/3*group["sigma"],
            fmt="o",
            markersize=2,
            color=DISTANCE_TO_COLOR[distance],
            label=rf"$d={distance}$",
        )

    ax.set_xlabel(r"simulation time ($\tau$)", fontsize=LABEL_FONTSIZE, labelpad=1)
    ax.set_ylabel(r"$1 - \frac{4}{3} \times P(\tau)$", fontsize=LABEL_FONTSIZE, labelpad=-12)

    ax.set_yscale("log")
    ax.set_xlim(*X_POISSON_LIM)
    ax.set_ylim(*Y_POISSON_LIM)

    ax.tick_params(labelsize=TICK_FONTSIZE, width=AXWIDTH)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Grid (as in original)
    ax.grid(which='major', color=GREY, linestyle='-', linewidth=AXWIDTH/2, alpha=0.2)
    ax.grid(which='minor', color=GREY, linestyle='-', linewidth=AXWIDTH/3, alpha=0.2)
    #ax.grid(False)

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=LABEL_FONTSIZE,
        ncol=2,
        columnspacing=0.5,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(AXWIDTH)

    fig.tight_layout()
    plt.savefig(f"{path}/poisson.pdf")
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

# LABEL_FONTSIZE = 7
# TICK_FONTSIZE = 5.5
# TICK_FONTSIZE = 6.5

# LINEWIDTH = 1.2
# MARKER_SIZE = 3.5
# SCATTER_SIZE = 14

# # ---------------------------------------------------------------------
# # Physical / numerical cutoffs
# # ---------------------------------------------------------------------

# # ---------------------------------------------------------------------
# # Axis limits
# # ---------------------------------------------------------------------

# X_LIM = (0, 10**5)
# Y_LIM = (0.1, 1)

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
#                     35:ORANGE,
#                     50:YELLOW,
#                     75:APPLE,
#                     100:MAGENTA,
#                     125:COPPER}


# def plot_poisson(df, path):
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
#             1 - 4/3 * group["cummulative_error"],
#             yerr=4/3 * group["sigma"],
#             fmt="o",
#             color=DISTANCE_TO_COLOR[distance],
#             capsize=1,
#             markersize=2,
#             label=rf"$d={distance}$",
#         )

#     # Axes formatting
#     ax.set_xlabel(r"simulation time ($\tau$)",
#                   fontsize=LABEL_FONTSIZE, labelpad=1)
#     ax.set_ylabel(r"$1 - 2 \times P(\tau)$",
#                   fontsize=LABEL_FONTSIZE, labelpad=-12)

#     #ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlim(*X_LIM)
#     ax.set_ylim(*Y_LIM)

#     #ax.set_xticks([1, 1e5, 5e5, 1e6])
#     #ax.set_xticklabels(["1", "10⁵", "5×10⁵", "10⁶"])
#     #ax.set_xticks([i * 1e5 for i in range(1, 10)], minor=True)

#     ax.tick_params(axis="both", which="both",
#                    labelsize=TICK_FONTSIZE)
#     ax.yaxis.set_minor_formatter(ticker.NullFormatter())

#     ax.grid(False)

#     # ax.legend(
#     #     loc="lower right",
#     #     frameon=False,
#     #     fontsize=LABEL_FONTSIZE,
#     #     handletextpad=0.1,
#     #     labelspacing=0.25,
#     #     borderpad=0.25,
#     #     ncol=2,
#     #     columnspacing=0.5,
#     # )

#     plt.savefig(f"{path}/poisson.pdf")
#     plt.close()