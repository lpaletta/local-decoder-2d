import numpy as np

from init import *
from error import *

def get_data_to_visualize(defect_array,forward_signal_1_array,forward_signal_2_array,anti_signal_1_array,anti_signal_2_array,stack_1_array,stack_2_array):
    return(defect_array,np.sum(forward_signal_1_array,axis=0).astype(int),np.sum(forward_signal_2_array,axis=0).astype(int),np.sum(anti_signal_1_array,axis=0).astype(int),np.sum(anti_signal_2_array,axis=0).astype(int),np.max(stack_1_array,axis=0).astype(int),np.max(stack_2_array,axis=0).astype(int))

def get_number_of_particles(forward_signal_1_array,anti_signal_1_array,stack_1_array,forward_signal_2_array,anti_signal_2_array,stack_2_array):
    return(np.sum(forward_signal_1_array+anti_signal_1_array+stack_1_array+forward_signal_2_array+anti_signal_2_array+stack_2_array))

def verify_charge_1_conservation(forward_signal_1_array,anti_signal_1_array,stack_1_array):

    charge_conservation_rows = np.sum(forward_signal_1_array - anti_signal_1_array - stack_1_array, axis=2)
    charge_conservation_columns = np.sum(forward_signal_1_array - anti_signal_1_array - stack_1_array, axis=1)

    return(charge_conservation_rows[::2].astype(np.int8),charge_conservation_columns[1::2].astype(np.int8))

def verify_charge_2_conservation(forward_signal_2_array,anti_signal_2_array,stack_2_array):
    charge_conservation_rows = np.sum(forward_signal_2_array - anti_signal_2_array - stack_2_array, axis=2)
    charge_conservation_columns = np.sum(forward_signal_2_array - anti_signal_2_array - stack_2_array, axis=1)

    return(charge_conservation_rows[::2].astype(np.int8),charge_conservation_columns[1::2].astype(np.int8))