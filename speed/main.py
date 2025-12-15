import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from param import *
from plot import *
from mwpm import *

path_in = "speed/data/"
path_fig = "speed/fig/"


# Pour recharger le tableau depuis le fichier
df_loaded = pd.read_pickle(path_in + "time_record.pkl")
#time_record = np.load(path_in+'time_record.npy',allow_pickle=True).item()
df_loaded = df_loaded.reset_index()

time_to_half_correction(df_loaded,path_fig)
distribution_time_to_half_correction(df_loaded,path_fig)