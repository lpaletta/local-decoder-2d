import numpy as np

from propagation import *
from init import *

def error_channel(h,L,error_rate,rng):
    new_errors_array = (rng.random((2*h,2*L)) < error_rate)
    return(new_errors_array.astype(np.int8))

def get_defect(data_array,mask_array,meas_error_bool,meas_error_rate,rng):

    dim = data_array.shape
    full_defect_array = np.zeros(dim)

    if meas_error_bool:
        # Compute sum of all four translations mod 2
        full_defect_array += (mask_array==1)*(
            shift(data_array,'v',1) +
            shift(data_array,'v',-1) +
            shift(data_array,'h',1) +
            shift(data_array,'h',-1) +
            error_channel(dim[0]//2,dim[1]//2,meas_error_rate,rng)
        ) % 2
    else:
        # Compute sum of all four translations mod 2
        full_defect_array += (mask_array==1)*(
            shift(data_array,'v',1) +
            shift(data_array,'v',-1) +
            shift(data_array,'h',1) +
            shift(data_array,'h',-1)
        ) % 2

    #full_defect_array = np.nan_to_num(full_defect_array,nan=0).astype(np.int8)
    defect_array = full_defect_array[::2,1::2]

    return(defect_array.astype(np.int8))


def get_defect_determistic(data_array,mask_array):
    defect_array = get_defect(data_array,mask_array,False,0,0)
    return(defect_array.astype(np.int8))



# def match_pattern(full_grid, pattern, spacing=2):
#     """
#     Match a 3x3 pattern on a full_grid with given spacing.
#     spacing=2 means neighbors are at ±2 in the full_grid.
#     """
#     matches = np.ones_like(full_grid, dtype=bool)

#     for i in range(3):
#         for j in range(3):
#             val = pattern[i, j]
#             if val == -1:
#                 continue

#             dy = (i - 1) * spacing
#             dx = (j - 1) * spacing

#             shifted = np.roll(full_grid, shift=(dy, dx), axis=(0, 1))
#             matches &= (shifted == val)

#     return matches

# def get_instantaneous_correction(full_defect_array,rules):
#     correction = np.zeros_like(full_defect_array, dtype=np.int8)

#     for rule in rules:
#         matches = match_pattern(full_defect_array, rule["pattern"], spacing=2)

#         # correction offsets are still ±1 in real grid
#         dy, dx = rule["offset"]
#         correction |= np.roll(matches, shift=(dy, dx), axis=(0, 1))

#     return correction.astype(np.int8)


def precompute_shifts_2d(array, distances=(-1,0,1)):

    """
    Precompute all 2D shifts for a reduced lattice.
    Returns a dict: shifts[(dy, dx)] = array rolled by (dy, dx)
    """

    shifts = {}
    for dy in distances:
        for dx in distances:
            if (abs(dx)+abs(dy)>1):
                continue # Skip diagonal shifts that are not needed in the set of rules
            shifts[(dy, dx)] = np.roll(array, shift=(dy, dx), axis=(0,1))

    return shifts

def match_pattern(grid, pattern, shifts):
    """Match a 3x3 pattern on reduced lattice using precomputed shifts"""
    matches = np.ones_like(grid, dtype=bool)
    for i in range(3):
        for j in range(3):
            val = pattern[i,j]
            if val == -1:
                 continue
            dy = i - 1
            dx = j - 1
            matches &= (shifts[(dy, dx)] == val)
            if not matches.any():
                break
    return matches

def get_instantaneous_correction(defect_array, rules):
    """
    Compute instantaneous correction exactly like original code,
    but all pattern matching is done on reduced lattice.
    """
    correction = np.zeros_like(map_to_full_defect(defect_array), dtype=np.int8)
    
    shifts = precompute_shifts_2d(defect_array)
    
    for rule in rules:
        matches = match_pattern(defect_array, rule["pattern"], shifts)
        dy, dx = rule["offset"]
        
        # expand to full lattice
        full_matches = np.zeros_like(correction, dtype=bool)
        full_matches[::2,1::2] = matches
        correction |= np.roll(full_matches, shift=(dy, dx), axis=(0,1))
    
    return correction.astype(np.int8)

# def get_instantaneous_correction(defect_array, rules):
#     """
#     Compute instantaneous correction on reduced lattice (d x d),
#     avoiding any full-lattice rolls.
#     Returns same full-lattice output as original.
#     """
#     d = defect_array.shape[0]
#     correction_compact = np.zeros_like(defect_array, dtype=np.int8)
    
#     shifts = precompute_shifts_2d(defect_array)
    
#     for rule in rules:
#         # match on compact lattice
#         matches = match_pattern(defect_array, rule["pattern"], shifts)
        
#         # map full-lattice offset to compact-lattice offset
#         dy, dx = rule["offset"]
#         dy_c = dy // 2
#         dx_c = dx // 2
        
#         # roll on compact lattice only
#         correction_compact |= np.roll(matches, shift=(dy_c, dx_c), axis=(0,1))
    
#     # expand compact correction to full lattice
#     correction_full = np.zeros((2*d, 2*d), dtype=np.int8)
#     correction_full[::2, 1::2] = correction_compact  # active sites
#     return correction_full


def get_correction(defect_array,forward_signal_1_array,forward_signal_2_array):

    # Correction

    inter_12_array = defect_array*forward_signal_1_array[2]
    inter_11_array = defect_array*forward_signal_1_array[1]*(forward_signal_1_array[2]==0)*(forward_signal_1_array[3]==0)
    inter_13_array = defect_array*forward_signal_1_array[3]*(forward_signal_1_array[1]==0)*(forward_signal_1_array[2]==0)

    not_inter_1x_array = (inter_11_array==0)*(inter_12_array==0)*(inter_13_array==0)

    inter_20_array = defect_array*forward_signal_2_array[0]*not_inter_1x_array
    inter_21_array = defect_array*forward_signal_2_array[1]*(forward_signal_2_array[0]==0)*(forward_signal_2_array[3]==0)*not_inter_1x_array
    inter_23_array = defect_array*forward_signal_2_array[3]*(forward_signal_2_array[0]==0)*(forward_signal_2_array[1]==0)*not_inter_1x_array

    corr_11 = shift(map_to_full_defect(inter_11_array),'v',-1)
    corr_12 = shift(map_to_full_defect(inter_12_array),'h',+1)
    corr_13 = shift(map_to_full_defect(inter_13_array),'v',+1)

    corr_20 = shift(map_to_full_defect(inter_20_array),'h',-1)
    corr_21 = shift(map_to_full_defect(inter_21_array),'v',-1)
    corr_23 = shift(map_to_full_defect(inter_23_array),'v',+1)

    final_correction_array = ((corr_11 + corr_12 + corr_13 + corr_20 + corr_21 + corr_23) > 0).astype(np.int8)
    

    return(final_correction_array)

    