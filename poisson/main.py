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


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

DATA_PATH = "poisson/data/"
FIG_PATH = "poisson/fig/"

DATA_FILE = "data.csv"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = import_data(DATA_PATH, DATA_FILE, "Poisson")

# Distances used for plotting
D_PLOT_ALL = df["code_distance"].unique()

# -------------------------------------------------------------
# Plot jump probability (selected distances)
# -------------------------------------------------------------

plot_poisson(df, path=FIG_PATH)

# plot_logical_convergence(
#     df[df["d"].isin(D_PLOT_REDUCED)],
#     path=FIG_PATH
# )

# # -------------------------------------------------------------
# # Plot particles density (selected distances)
# # -------------------------------------------------------------

# plot_density_convergence(
#     df[df["d"].isin(D_PLOT_REDUCED)],
#     path=FIG_PATH
# )

# # -------------------------------------------------------------
# # Fit density convergence
# # -------------------------------------------------------------

# fit_density_cutoff(df[df["d"].isin(D_PLOT_ALL)],key="tau_density")

# # -------------------------------------------------------------
# # Fit logical convergence
# # -------------------------------------------------------------

# fit_logical_cutoff(df[df["d"].isin(D_PLOT_ALL)],key="tau_logical")

# # -------------------------------------------------------------
# # Plot density qnd logical convergences
# # -------------------------------------------------------------

# plot_cutoff(df[df["d"].isin(D_PLOT_ALL)],
#                 keys=["tau_density","tau_logical"],
#                 path=FIG_PATH
# )






"""import os
import sys
import numpy as np

sys.path.append("/scratch/lpaletta/toric")

from sgn2d import SIGNAL2D

from param import *
from mwpm import *
from multiprocessing import Pool, cpu_count, freeze_support

number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

i_e = int(sys.argv[1])
i_d = int(sys.argv[2])

param_dict["error_rate"] = error_rate_list[i_e]
param_dict["meas_error_rate"] = param_dict["error_rate"] if option_dict["id_error_rate"] else param_dict["meas_error_rate"]
param_dict["d"] = d_list[i_d]

T_cum_max = mc_dict["T_cum_max"]

T = param_dict["T"]
dt = view_dict["dt"]

args = number_of_cores*[(param_dict,option_dict,view_dict)]

def main():

    T_cum=0
    number_of_runs = 0

    t_array = np.array([t for t in range(0,T,dt)])
    logical_array = np.zeros(len(t_array))

    while T_cum<T_cum_max:
        with Pool(number_of_cores) as pool:
            results_as_list_of_hist_data_array = pool.starmap(SIGNAL2D, args)

    results_as_list_of_hist_logical_error = [
    np.array([
        int(
            get_logical_from_array(
                (A[i] + vector_to_array(
                    matching.decode(H @ get_data_as_vector(A[i]) % 2)
                )) % 2
            ) != (0, 0)
        )
        for i in range(A.shape[0])
    ])
    for A in results_as_list_of_hist_data_array
    ]

    hist_of_logical_error += sum([r for r in results_as_list_of_hist_logical_error])
    number_of_runs += len(results_as_list_of_hist_logical_error)
    T_cum += len(results_as_list_of_hist_logical_error)*T

    if (number_of_runs*T)>=(T_cum_max/1000):
        for t in range(len(logical_array)):
            print(int(logical_array[t]),number_of_runs,int(t_array[t]))
        logical_array = np.zeros(len(t_array))
        number_of_runs = 0


matching = get_matching(param_dict["d"])
H = get_parity_check_matrix(param_dict["d"])

def get_estimate_logical_signal(d,p,A,pth,gamma):
    pL = A*d*(p/pth)**gamma
    return(pL)

if __name__=="__main__":
    freeze_support()
    main()"""