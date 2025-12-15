import numpy as np

from init import *
from error import *

def get_condensed_defects(defect_array):
    condensed_defect_array = defect_array[::2, 1::2]
    return(condensed_defect_array)

def get_condensed_signals(signal_array):
    condensed_signal_array = np.stack([signal_array[0,::2,1::2],signal_array[1,::2,1::2]])
    return(condensed_signal_array)

def get_number_of_particles(forward_signal_1_array,anti_signal_1_array,stack_1_array,forward_signal_2_array,anti_signal_2_array,stack_2_array):
    c_fs1_array, c_as1_array, c_s1_array = get_condensed_signals(forward_signal_1_array), get_condensed_signals(anti_signal_1_array), get_condensed_signals(stack_1_array)
    c_fs2_array, c_as2_array, c_s2_array = get_condensed_signals(forward_signal_2_array), get_condensed_signals(anti_signal_2_array), get_condensed_signals(stack_2_array)
    return(np.sum(c_fs1_array+c_as1_array+c_s1_array+c_fs2_array+c_as2_array+c_s2_array))

def verify_charge_1_conservation(forward_signal_1_array,backward_signal_1_array,anti_signal_1_array,stack_1_array,switch_1_array):
    c_forward_signal_array, c_backward_signal_array, c_anti_signal_array, c_stack_array = get_condensed_signals(forward_signal_1_array), get_condensed_signals(backward_signal_1_array), get_condensed_signals(anti_signal_1_array), get_condensed_signals(stack_1_array)

    charge_conservation_rows = np.sum(c_forward_signal_array + c_backward_signal_array - c_anti_signal_array - c_stack_array, axis=2)
    charge_conservation_columns = np.sum(c_forward_signal_array + c_backward_signal_array - c_anti_signal_array - c_stack_array, axis=1)

    return(charge_conservation_rows[::2,:],charge_conservation_columns[1::2,:])

def verify_charge_2_conservation(forward_signal_2_array,anti_signal_2_array,stack_2_array):
    c_forward_signal_array, c_anti_signal_array, c_stack_array = get_condensed_signals(forward_signal_2_array), get_condensed_signals(anti_signal_2_array), get_condensed_signals(stack_2_array)

    charge_conservation_rows = np.sum(c_forward_signal_array - c_anti_signal_array - c_stack_array, axis=2)
    charge_conservation_columns = np.sum(c_forward_signal_array - c_anti_signal_array - c_stack_array, axis=1)

    return(charge_conservation_rows[::2,:],charge_conservation_columns[1::2,:])

# ANSI escape sequences
BLACK = "\033[30m"
RED     = "\033[31m"
BLUE    = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"
PINK = "\033[38;5;211m"
LIGHT_PINK = "\033[38;5;217m"  # soft light pink
WHITE = "\033[97m"
GREY = "\033[90m"

def visualize_data(data_array,signal_1_array,signal_2_array):

    dim = data_array.shape
    d = dim[0]//2
    mask_array = init_mask(d)
    
    data_array_to_visualize = np.nan_to_num(data_array, nan=8).astype(np.int8)
    defect_array = get_defect_determistic(data_array,mask_array)

    for i in range(dim[0]):
        for j in range(dim[1]):
            val = data_array_to_visualize[i, j]
            if val == 8:
                if mask_array[i, j] == 1:
                    if defect_array[i,j] == 1:
                        print(BLUE + '●' + RESET, end=" ")
                    elif signal_1_array[i,j] == 1:
                        print(RED + '●' + RESET, end=" ")
                    elif signal_2_array[i,j] == 1:
                        print(WHITE + '●' + RESET, end=" ")
                    else:
                        print(BLACK + '*' + RESET, end=" ")
                elif mask_array[i, j] == -1:
                    print(BLACK + '-' + RESET, end=" ")
            else:
                print(str(val), end=" ")
        print()

def visualize_ancilla(defect_array,signal_1_array,signal_2_array,anti_signal_1_array,anti_signal_2_array,stack_1_array,stack_2_array,alt_signal_1_array,alt_signal_2_array,alt_stack_array):

    reduced_defect_array = defect_array[::2, 1::2]
    reduced_signal_1_array = signal_1_array[::2, 1::2]
    reduced_signal_2_array = signal_2_array[::2, 1::2]

    reduced_anti_signal_1_array = anti_signal_1_array[::2, 1::2]
    reduced_anti_signal_2_array = anti_signal_2_array[::2, 1::2]
    reduced_stack_1_array = stack_1_array[::2, 1::2]
    reduced_stack_2_array = stack_2_array[::2, 1::2]

    reduced_alt_signal_1_array = alt_signal_1_array[::2, 1::2]
    reduced_alt_signal_2_array = alt_signal_2_array[::2, 1::2]
    reduced_alt_stack_array = alt_stack_array[::2, 1::2]

    dim = reduced_defect_array.shape
    d = dim[0]

    for i in range(dim[0]):
        for j in range(dim[1]):
            if reduced_defect_array[i,j] == 1:
                print(BLUE + '●' + RESET, end=" ")
            elif reduced_signal_1_array[i,j] > 0:
                print(RED + '●' + RESET, end=" ")
            elif reduced_signal_2_array[i,j] > 1:
                print(WHITE + '●' + RESET, end=" ")
            elif reduced_signal_2_array[i,j] == 1:
                print(GREY + '●' + RESET, end=" ")
            else:
                print(BLACK + '*' + RESET, end=" ")
        print(' ', end=" ")

        for j in range(dim[1]):
            if reduced_stack_1_array[i,j] > 0:
                if reduced_stack_1_array[i,j] < 10:
                    print(RED + str(reduced_stack_1_array[i,j]) + RESET, end=" ")
                else:
                    print(RED + '■' + RESET, end=" ")
            elif reduced_stack_2_array[i,j] > 0:
                if reduced_stack_2_array[i,j] < 10:
                    print(GREY + str(reduced_stack_2_array[i,j]) + RESET, end=" ")
                else:
                    print(GREY + '■' + RESET, end=" ")
            elif reduced_anti_signal_1_array[i,j] > 0:
                print(RED + '●' + RESET, end=" ")
            elif reduced_anti_signal_2_array[i,j] > 1:
                print(WHITE + '●' + RESET, end=" ")
            elif reduced_anti_signal_2_array[i,j] == 1:
                print(GREY + '●' + RESET, end=" ")
            else:
                print(BLACK + '*' + RESET, end=" ")
                
        print(' ', end=" ")
        for j in range(dim[1]):
            if reduced_alt_stack_array[i,j] > 0:
                print(WHITE + str(reduced_alt_stack_array[i,j]) + RESET, end=" ")
            elif reduced_alt_signal_1_array[i,j] == 1:
                print(RED + '●' + RESET, end=" ")
            elif reduced_alt_signal_2_array[i,j] == 1:
                print(WHITE + '●' + RESET, end=" ")
            else:
                print(BLACK + '*' + RESET, end=" ")
        print()