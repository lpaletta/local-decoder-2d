import numpy as np

def create_forward_signals_1(defect_array,forward_signal_1_array,stack_1_array):
    inter_array = np.stack([np.zeros_like(defect_array)]+[defect_array]*3)*(forward_signal_1_array==0)
    forward_signal_1_array, stack_1_array = forward_signal_1_array + inter_array, stack_1_array + inter_array
    return(forward_signal_1_array.astype(np.int8),stack_1_array.astype(np.int8))

def create_forward_signals_2(defect_array,forward_signal_1_array,forward_signal_2_array,stack_2_array):
    inter_array_0 = (defect_array | forward_signal_1_array[1] | forward_signal_1_array[3])
    inter_array_1 = forward_signal_1_array[2]
    inter_array_3 = forward_signal_1_array[2]
    inter_array = np.stack([inter_array_0,inter_array_1,np.zeros_like(defect_array),inter_array_3])*(forward_signal_2_array==0)
    forward_signal_2_array, stack_2_array = forward_signal_2_array + inter_array, stack_2_array + inter_array
    return(forward_signal_2_array.astype(np.int8),stack_2_array.astype(np.int32))

def create_anti_signals_1(defect_array,anti_signal_1_array,stack_1_array,cond):
    inter_array = (np.stack([defect_array]*4)==0)*(anti_signal_1_array==0)*(stack_1_array>0)*cond
    anti_signal_1_array, stack_1_array = anti_signal_1_array + inter_array, stack_1_array - inter_array
    return(anti_signal_1_array.astype(np.int8),stack_1_array.astype(np.int32))

def create_anti_signals_2(defect_array,forward_signal_1_array,anti_signal_2_array,stack_2_array,cond):
    inter_array_0 = (defect_array == 0) & (forward_signal_1_array[1] == 0) & (forward_signal_1_array[3] == 0)
    inter_array_1 = (forward_signal_1_array[2] == 0)
    inter_array_3 = (forward_signal_1_array[2] == 0)
    inter_array = np.stack([inter_array_0,inter_array_1,np.zeros_like(defect_array),inter_array_3]) & (anti_signal_2_array == 0) & (stack_2_array > 0) & cond
    anti_signal_2_array, stack_2_array = anti_signal_2_array + inter_array, stack_2_array - inter_array
    return(anti_signal_2_array.astype(np.int8),stack_2_array.astype(np.int32))

def decrement_self_stack(defect_array,self_stack_array):
    self_stack_array = self_stack_array - (defect_array==0)*(self_stack_array>0)
    return(self_stack_array)

def decrement_self(self_1_array,self_2_array):
    self_1_array = self_1_array - (self_1_array>0)
    self_2_array = self_2_array - (self_2_array>0)
    return(self_1_array,self_2_array)

def create_self_1(defect_array,self_1_array,self_2_array,self_stack_array):
    inter_array_1 = self_stack_array*(defect_array==0)
    inter_array_2 = self_stack_array*(defect_array==0)
    inter_array_3 = self_stack_array*(defect_array==0)
    inter_array = np.stack([np.zeros_like(defect_array),inter_array_1,inter_array_2,inter_array_3])
    self_1_array = self_1_array + inter_array
    return(self_1_array)

def create_self_2(defect_array,self_1_array,self_2_array,self_stack_array):
    inter_array_0 = self_1_array[1] + self_1_array[3] + self_stack_array*(defect_array==0)
    inter_array_1 = self_1_array[2]
    inter_array_3 = self_1_array[2]
    inter_array = np.stack([inter_array_0,inter_array_1,np.zeros_like(defect_array),inter_array_3])
    self_2_array = self_2_array + inter_array
    return(self_2_array)

def modify_counter(defect_array,signal_array,counter_array,self_stack_array,max_counter,max_self_stack):
    counter_array = counter_array + (defect_array==1)*(signal_array==0) - (defect_array==0)*(counter_array>0)
    self_stack_array = self_stack_array + 2*(counter_array==max_counter)*(self_stack_array<max_self_stack)
    counter_array = counter_array - counter_array*(counter_array==max_counter)
    return(counter_array.astype(np.int32),self_stack_array.astype(np.int32))