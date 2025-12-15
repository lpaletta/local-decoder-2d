import numpy as np

def shift(array , direction, distance):
    if direction == 'v':
        shifted_array = np.roll(array, shift=distance, axis=0)
    if direction == 'h':
        shifted_array = np.roll(array, shift=distance, axis=1)
    return(shifted_array)

"""def propagate_signals_1(signal_array, step):
    # Each roll is along one axis only
    #signal_array[0] = np.roll(signal_array[0], shift=2*step, axis=1)   # right
    signal_array[1] = np.roll(signal_array[1], shift=2*step, axis=0)   # down
    signal_array[2] = np.roll(signal_array[2], shift=-2*step, axis=1)  # left
    signal_array[3] = np.roll(signal_array[3], shift=-2*step, axis=0)  # up
    return(signal_array)

def propagate_signals_2(signal_array, step):
    # Each roll is along one axis only
    signal_array[0] = np.roll(signal_array[0], shift=2*step, axis=1)   # right
    signal_array[1] = np.roll(signal_array[1], shift=2*step, axis=0)   # down
    #signal_array[2] = np.roll(signal_array[2], shift=-2*step, axis=1)  # left
    signal_array[3] = np.roll(signal_array[3], shift=-2*step, axis=0)  # up
    return(signal_array)"""

def propagate_signals_1(signal_array,step,wrap=True):

    if wrap:
        # With wraparound (regular np.roll)
        signal_array[1] = np.roll(signal_array[1], shift=2*step, axis=0)   # down
        signal_array[2] = np.roll(signal_array[2], shift=-2*step, axis=1)  # left
        signal_array[3] = np.roll(signal_array[3], shift=-2*step, axis=0)  # up
    else:
        signal_array[1] = np.roll(signal_array[1], shift=2*step, axis=0)   # down
        signal_array[1][:2*step, :] = 0  

        signal_array[2] = np.roll(signal_array[2], shift=-2*step, axis=1)  # left
        signal_array[2][:, -2*step:] = 0  

        signal_array[3] = np.roll(signal_array[3], shift=-2*step, axis=0)  # up
        signal_array[3][-2*step:, :] = 0  

    return signal_array


def propagate_signals_2(signal_array,step,wrap=True):

    if wrap:
        signal_array[0] = np.roll(signal_array[0], shift=2*step, axis=1)   # right
        signal_array[1] = np.roll(signal_array[1], shift=2*step, axis=0)   # down
        signal_array[3] = np.roll(signal_array[3], shift=-2*step, axis=0)  # up
    else:
        signal_array[0] = np.roll(signal_array[0], shift=2*step, axis=1)   # right
        signal_array[0][:, :2*step] = 0  
        
        signal_array[1] = np.roll(signal_array[1], shift=2*step, axis=0)   # down
        signal_array[1][:2*step, :] = 0  
        
        signal_array[3] = np.roll(signal_array[3], shift=-2*step, axis=0)  # up
        signal_array[3][-2*step:, :] = 0  

    return signal_array