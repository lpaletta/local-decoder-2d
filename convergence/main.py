import os
import sys

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from import_data import *
from poisson.fit import *
from plot import *
from fit import *


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

DATA_PATH = "convergence/data/"
FIG_PATH = "convergence/fig/"

DATA_FILE = "data.csv"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = import_data(DATA_PATH, DATA_FILE, "Convergence")

# Distances used for plotting
D_PLOT_ALL = df["code_distance"].unique()

# -------------------------------------------------------------
# Plot jump probability (selected distances)
# -------------------------------------------------------------

plot_convergence(df, path=FIG_PATH)
get_cutoff(df)
fit_cutoff(df)
plot_cutoff(df, path=FIG_PATH)