from pymatching import Matching

from init import *
from error import *

def get_matching(d):
    H = get_parity_check_matrix(d)
    matching=Matching(H)
    return(matching)

def MWPM(matching,H,error_vector):
    syndrome = H@error_vector % 2
    correction = matching.decode(syndrome)

    error_array = vector_to_array(error_vector)
    correction_array = vector_to_array(correction)

    logicals = get_logical_from_array((error_array+correction_array)%2)
    return(logicals)

def get_defect(data_array,meas_error_bool,meas_error_rate,rng,mask_defect_array):
    L1, L2 = data_array.shape
    defect_array = np.zeros((4,L1,L2)).astype(np.int8)
    if meas_error_bool:
        defect_array[0,:,:] = mask_defect_array*(data_array + np.roll(data_array,shift=(-1,0),axis=(0,1))+np.roll(data_array,shift=(0,-1),axis=(0,1))+np.roll(data_array,shift=(-1,-1),axis=(0,1))+error_channel(L1,L2,meas_error_rate,rng))%2
        defect_array[1,:,:] = np.roll(defect_array[0,:,:],shift=(1,1),axis=(0,1))
        defect_array[2,:,:] = np.roll(defect_array[0,:,:],shift=(1,0),axis=(0,1))
        defect_array[3,:,:] = np.roll(defect_array[0,:,:],shift=(0,1),axis=(0,1))
    else:
        defect_array[0,:,:] = mask_defect_array*(data_array + np.roll(data_array,shift=(-1,0),axis=(0,1))+np.roll(data_array,shift=(0,-1),axis=(0,1))+np.roll(data_array,shift=(-1,-1),axis=(0,1)))%2
        defect_array[1,:,:] = np.roll(defect_array[0,:,:],shift=(1,1),axis=(0,1))
        defect_array[2,:,:] = np.roll(defect_array[0,:,:],shift=(1,0),axis=(0,1))
        defect_array[3,:,:] = np.roll(defect_array[0,:,:],shift=(0,1),axis=(0,1))
    return(defect_array.astype(np.int8))