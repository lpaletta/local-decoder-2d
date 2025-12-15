import numpy as np

def recombine_signals(signal_1_array,signal_2_array):
    inter_array = ((signal_1_array*signal_2_array)>0).astype(np.int8)
    signal_1_array = signal_1_array - inter_array
    signal_2_array = signal_2_array - inter_array
    return(signal_1_array,signal_2_array)