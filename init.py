import numpy as np

def init_data_array(d):
    data_array = np.zeros((2*d,2*d))
    data_array[1::2, ::2] = np.nan  # Odd rows, even columns
    data_array[::2, 1::2] = np.nan  # Even rows, odd columns
    return(data_array)

def init_mask(d):
    stab_array = np.zeros((2*d,2*d), dtype = np.int8)
    stab_array[::2] = 1
    stab_array[1::2] = -1
    stab_array[::2, ::2] = 0  # even rows, even columns
    stab_array[1::2, 1::2] = 0  # odd rows, odd columns
    return(stab_array)

def get_data_as_vector(data_array):
    return(data_array[~np.isnan(data_array)].ravel())

def get_parity_check_matrix(d):
    data_array = init_data_array(d)
    single_stab_vector_list = []
    stab_index_list = [(r, c) for r, c in zip(*np.where(np.isnan(data_array))) if r % 2 == 0]
    for idx in stab_index_list:
        single_stab_array = set_neighbors_to_one_periodic(data_array.copy(), idx[0], idx[1])
        single_stab_vector_list.append(single_stab_array[~np.isnan(data_array)])
    return(np.vstack(single_stab_vector_list[:-1]).astype(np.int8))


def set_neighbors_to_one_periodic(array, i, j):
    rows, cols = array.shape
    # Top (with periodicity)
    if array[(i-1) % rows, j] == 0:
        array[(i-1) % rows, j] = 1
    # Bottom (with periodicity)
    if array[(i+1) % rows, j] == 0:
        array[(i+1) % rows, j] = 1
    # Left (with periodicity)
    if array[i, (j-1) % cols] == 0:
        array[i, (j-1) % cols] = 1
    # Right (with periodicity)
    if array[i, (j+1) % cols] == 0:
        array[i, (j+1) % cols] = 1
    return(array)

def vector_to_array(data_vector):
    d = int(np.sqrt(len(data_vector)//2))
    data_array = init_data_array(d)
    data_array[np.where(data_array == 0)] = data_vector
    return(data_array)

def get_logical_from_array(data_array):
    data_array_without_nan = np.nan_to_num(data_array, nan=0)
    X1 = data_array_without_nan[1, :].sum() % 2
    X2 = data_array_without_nan[:, 0].sum() % 2
    return(int(X1),int(X2))

def init_signal_array(d):
    signal_array = np.zeros((4,2*d,2*d))
    return(signal_array.astype(np.int8))

def init_error_array(d,orientation):
    data_array = init_data_array(d)

    if orientation == "horizontal":
        data_array[0,1:d//2+1] = 1
    elif orientation == "vertical":
        data_array[:d//2,1] = 1
    elif orientation == "diagonal":
        data_array[0,1:d//2+1] = 1
        data_array[:d//2,1] = 1

    data_array[1::2, ::2] = np.nan  # Odd rows, even columns
    data_array[::2, 1::2] = np.nan  # Even rows, odd columns

    s = d//2+d//4 + ((d//2+d//4 % 2) == 1)

    data_array = np.roll(data_array,axis=(0,1),shift=(s,s))

    return(data_array)

def fix_defect(defect_array,data_array,rows,cols,values,defect_to_fix):
    data_array[rows, cols] = values
    defect_array[defect_to_fix[0], defect_to_fix[1]] = 0
    return(defect_array,data_array)

def add_artificial_defect(defect_array,dict_artificial_defect,t):
    if t in dict_artificial_defect:
        for coord in dict_artificial_defect[t]:
            defect_array[coord[0], coord[1]] = 1
    return(defect_array)