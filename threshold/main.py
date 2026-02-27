import os
import sys

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from import_data import *
from fit import *
from plot import *

# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

DATA_PATH = "threshold/data/"
FIG_PATH = "threshold/fig/"

DATA_FILE = "data.csv"

PTH_GUESS = 0.008
PTH_LIM_FIT = 0.007
PTH_PLOT = 0.0068

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = import_data(DATA_PATH, DATA_FILE, "Threshold")

# Distances used for plotting
D_PLOT_ALL = df["code_distance"].unique()
D_PLOT_REDUCED = [5, 9, 15, 25, 50, 100]

# -------------------------------------------------------------
# Plot raw logical error (all distances)
# -------------------------------------------------------------

plot_logical(
    df[df["code_distance"].isin(D_PLOT_ALL)],
    fit=False,
    key="",
    plim=PTH_GUESS,
    path=FIG_PATH,
)

# -------------------------------------------------------------
# Fit threshold(d)
# -------------------------------------------------------------

fit_threshold_d(df, PTH_GUESS, PTH_LIM_FIT)

# -------------------------------------------------------------
# Fit logical error rate p_L
# -------------------------------------------------------------

#fit_logical(df, PTH_GUESS, PTH_GUESS)
add_logical_fit(df, "logical_error_rate_fit")

# # -------------------------------------------------------------
# # Fit effective distance
# # -------------------------------------------------------------

fit_effective_distance(df)
add_effective_distance_fit(df, "effective_distance_fit")

# # -------------------------------------------------------------
# # Plot fitted logical error (reduced distances)
# # -------------------------------------------------------------


plot_logical(
    df[df["code_distance"].isin(D_PLOT_ALL)],
    fit=True,
    key="logical_error_rate_fit",
    plim=PTH_GUESS,
    path=FIG_PATH,
)

# # -------------------------------------------------------------
# # Plot effective distance scaling
# # -------------------------------------------------------------

plot_effective_distance(df, fit=False, key="effective_distance_fit", path=FIG_PATH)
plot_effective_distance(df, fit=True, key="effective_distance_fit", path=FIG_PATH)

# # -------------------------------------------------------------
# # Plot threshold(d)
# # -------------------------------------------------------------

plot_threshold_d(df, pth=PTH_PLOT, path=FIG_PATH)
