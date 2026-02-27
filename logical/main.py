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

DATA_PATH = "logical/data/"
FIG_PATH = "logical/fig/"

DATA_FILE = "data_2.csv"
PROOF_FILE = "proof.csv"

PTH_GUESS = 0.007

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = import_data(DATA_PATH, DATA_FILE, "Logical")

# Distances used for plotting
D_PLOT_ALL = df["code_distance"].unique()
D_PLOT_REDUCED = [5, 9, 15, 25, 50, 100]

# Max stacks used for plotting
MAX_STACK_ALL = df["max_stack"].unique()

# -------------------------------------------------------------
# Plot raw logical error (all distances)
# -------------------------------------------------------------

for max_stack in MAX_STACK_ALL:
    df_subset = df[df["max_stack"] == max_stack]
    plot_logical(
        df_subset[df_subset["code_distance"].isin(D_PLOT_ALL)],
        fit=False,
        key="",
        plim=PTH_GUESS,
        suffix=f"_maxstack_{int(max_stack) if max_stack < np.inf else 'inf'}",
        path=FIG_PATH,
    )

# -------------------------------------------------------------
# Fit logical error rate p_L
# -------------------------------------------------------------

fit_logical(df, PTH_GUESS, PTH_GUESS)
df = add_logical_fit(df, "logical_error_rate_fit")

# -------------------------------------------------------------
# Fit effective distance
# -------------------------------------------------------------

fit_effective_distance(df)
add_effective_distance_fit(df, "effective_distance_fit")

# -------------------------------------------------------------
# Plot fitted logical error (reduced distances)
# -------------------------------------------------------------

for max_stack in MAX_STACK_ALL:
    df_subset = df[df["max_stack"] == max_stack]
    plot_logical(
        df_subset[df_subset["code_distance"].isin(D_PLOT_REDUCED)],
        fit=True,
        key="logical_error_rate_fit",
        plim=PTH_GUESS,
        suffix=f"_maxstack_{int(max_stack) if max_stack < np.inf else 'inf'}",
        path=FIG_PATH,
    )

# -------------------------------------------------------------
# Plot effective distance scaling
# -------------------------------------------------------------

plot_effective_distance(df, fit=False, key="effective_distance_fit", path=FIG_PATH)
plot_effective_distance(df, fit=True, key="effective_distance_fit", path=FIG_PATH)
