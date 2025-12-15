import numpy as np

from propagation import *
from init import *

def error_channel(h,L,error_rate,rng):
    new_errors_array = (rng.random((h,L)) < error_rate)
    return(new_errors_array.astype(np.int8))

def get_defect(data_array,mask_array,meas_error_bool,meas_error_rate,rng):

    dim = data_array.shape
    defect_array = np.zeros(dim)

    if meas_error_bool:
        # Compute sum of all four translations mod 2
        defect_array += (mask_array==1)*(
            shift(data_array,'v',1) +
            shift(data_array,'v',-1) +
            shift(data_array,'h',1) +
            shift(data_array,'h',-1) +
            error_channel(dim[0],dim[1],meas_error_rate,rng)
        ) % 2
    else:
        # Compute sum of all four translations mod 2
        defect_array += (mask_array==1)*(
            shift(data_array,'v',1) +
            shift(data_array,'v',-1) +
            shift(data_array,'h',1) +
            shift(data_array,'h',-1)
        ) % 2

    defect_array = np.nan_to_num(defect_array,nan=0).astype(np.int8)

    return(defect_array)

def get_defect_determistic(data_array,mask_array):
    defect_array = get_defect(data_array,mask_array,False,0,0).astype(np.int8)
    return(defect_array)

def get_instantaneous_correction(defect_array):
    instantaneous_correction_array = np.zeros_like(defect_array)

    # Correction of (x,x,x) and (x,1,x) and (x,0,x) defect substrings
    #               (1,1,0)     (x,1,0)     (0,1,x)
    #               (x,0,x)     (x,0,x)     (x,1,x)

    corr_1 = shift(defect_array,'h',+2)*defect_array*(shift(defect_array,'h',-2)==0)*(shift(defect_array,'v',-2)==0)
    corr_2 = shift(defect_array,'v',+2)*defect_array*(shift(defect_array,'v',-2)==0)*(shift(defect_array,'h',-2)==0)#*(corr_1==0)
    corr_3 = shift(defect_array,'v',-2)*defect_array*(shift(defect_array,'v',+2)==0)*(shift(defect_array,'h',+2)==0)#*(corr_1==0)

    instantaneous_correction_array[:,:] = ((shift(corr_1,'h',-1) + shift(corr_2,'v',-1) + shift(corr_3,'v',+1))>0).astype(np.int8)

    return(instantaneous_correction_array.astype(np.int8))

def get_desactivated_defect(defect_array):
    # desactivated defects do not emit signals
    desactivated_defect_array = np.zeros_like(defect_array)

    desactivated_defect_array = (
            defect_array*shift(defect_array,'v',2) +
            defect_array*shift(defect_array,'v',-2) +
            defect_array*shift(defect_array,'h',2) +
            defect_array*shift(defect_array,'h',-2)
        ) > 0

    return(desactivated_defect_array.astype(np.int8))

def get_correction(defect_array,forward_signal_1_array,forward_signal_2_array):

    # Correction

    inter_11_array = defect_array*forward_signal_1_array[1]
    inter_12_array = defect_array*forward_signal_1_array[2]*(inter_11_array==0)
    inter_13_array = defect_array*forward_signal_1_array[3]*(inter_11_array==0)*(inter_12_array==0)

    not_inter_1x_array = (inter_11_array==0)*(inter_12_array==0)*(inter_13_array==0)

    inter_20_array = defect_array*forward_signal_2_array[0]*not_inter_1x_array
    inter_21_array = defect_array*forward_signal_2_array[1]*not_inter_1x_array*(inter_20_array==0)
    inter_23_array = defect_array*forward_signal_2_array[3]*not_inter_1x_array*(inter_20_array==0)*(inter_21_array==0)

    corr_11 = shift(inter_11_array,'v',-1)
    corr_12 = shift(inter_12_array,'h',+1)
    corr_13 = shift(inter_13_array,'v',+1)

    corr_20 = shift(inter_20_array,'h',-1)
    corr_21 = shift(inter_21_array,'v',-1)
    corr_23 = shift(inter_23_array,'v',+1)

    final_correction_array = ((corr_11 + corr_12 + corr_13 + corr_20 + corr_21 + corr_23) > 0).astype(np.int8)

    return(final_correction_array)

def get_desactivated_forward(defect_array,forward_signal_1_array,forward_signal_2_array,self_stack_array):

    full_defect_array = np.stack([defect_array]*4)
    full_self_stack_array = np.stack([self_stack_array]*4)

    inter_1_array = forward_signal_1_array*(full_self_stack_array>0)*full_defect_array
    inter_2_array = forward_signal_2_array*(full_self_stack_array>0)*full_defect_array

    desactivated_forward_signal_1_array = inter_1_array.astype(np.int8)
    desactivated_forward_signal_2_array = inter_2_array.astype(np.int8)

    self_stack_array = self_stack_array - (1 + (self_stack_array>1).astype(np.int8))*np.any(inter_1_array | inter_2_array,axis=0).astype(np.int8)
    #self_stack_array = self_stack_array - (1 + (self_stack_array>1).astype(np.int8) + (self_stack_array>2).astype(np.int8))*np.any(inter_1_array | inter_2_array,axis=0).astype(np.int8)

    return(desactivated_forward_signal_1_array,desactivated_forward_signal_2_array,self_stack_array)

    