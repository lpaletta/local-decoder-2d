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
    rows, cols = np.indices(data_array.shape)
    vector = data_array[(rows % 2) == (cols % 2)]
    return(vector.astype(np.int8))

def get_parity_check_matrix(d):
    stab_array = init_mask(d)
    single_stab_vector_list = []
    stab_index_list = [(r, c) for r, c in zip(*np.where(stab_array != 0)) if r % 2 == 0]
    for idx in stab_index_list:
        single_stab_array = set_neighbors_to_one_periodic(stab_array.copy(), idx[0], idx[1])
        single_stab_vector_list.append(single_stab_array[~(stab_array != 0)])
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
    data_array = np.zeros((2*d,2*d))
    data_array[1::2, ::2] = np.nan  # Odd rows, even columns
    data_array[::2, 1::2] = np.nan  # Even rows, odd columns
    data_array[np.where(data_array == 0)] = data_vector
    return(data_array)

def get_logical_from_array(data_array):
    data_array_without_nan = np.nan_to_num(data_array, nan=0)
    X1 = data_array_without_nan[1, :].sum() % 2
    X2 = data_array_without_nan[:, 0].sum() % 2
    return(int(X1),int(X2))

def init_signal_array(d):
    signal_array = np.zeros((4,d,d))
    return(signal_array.astype(np.int8))

def add_artificial_defect(defect_array,dict_artificial_defect,t):
    if t in dict_artificial_defect:
        for coord in dict_artificial_defect[t]:
            defect_array[coord[0], coord[1]] = 1
    return(defect_array)

def map_to_full_defect(defect_array):
    d = defect_array.shape[0]
    full_defect_array = np.zeros((2*d, 2*d), dtype=np.int8)
    full_defect_array[::2, 1::2] = defect_array
    return(full_defect_array)

def map_to_full_signal(signal_array):
    d = signal_array.shape[1]
    full_signal_array = np.zeros((4,2*d, 2*d), dtype=np.int8)
    full_signal_array[:,::2, 1::2] = signal_array
    return(full_signal_array)