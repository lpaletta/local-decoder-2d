import numpy as np
import time
np.set_printoptions(threshold=np.inf)

from init import *
from error import *
from creation import *
from recombination import *
from view import *

def SIGNAL2D(param, option, view={"record_var" : "Logical"}):

    rng = np.random.default_rng() # Random number generator for reproducibility

    # Parameters for time steps and grid dimensions
    T = param["T"]
    d = param["d"]
    K = 0 # Step counter for visualization

    # error parameters
    error_bool = option["error_bool"]
    meas_error_bool = option["meas_error_bool"]
    error_rate = param["error_rate"]
    meas_error_rate = param["meas_error_rate"] if meas_error_bool else 0

    # Signal velocity parameters
    anti_signal_velocity = param["anti_signal_velocity"]
    
    # Initial configuration and options
    init_var = option["init_var"]
    artificial_defect_bool = option["artificial_defect_bool"]
    dict_artificial_defect = param["dict_artificial_defect"]

    # Recorder variable
    record_var = view["record_var"]
    if record_var == "Analysis":
        number_of_particles_array = np.zeros((T)).astype(np.int32)
    if record_var == "Convergence":
        first_time_zero_defect = -1
        first_time_zero_particles = -1
    if record_var == "Field":
        defect_on_focus, defect_to_fix = [d,d-1], [2,1]
        radius, T = d - (d+1)%2, 3*d
        t_to_record_array = list(range(radius//2,T))
        field_array = np.zeros((len(t_to_record_array),radius,radius,2))
        rows, cols, values = param["fixed_defect"][0],param["fixed_defect"][1],param["fixed_defect"][2]
    if record_var == "Stability":
        custom_error = param["custom_error"]
    if record_var == "Quench":
        T, nb_step = 2*d, 32
        defect_trajectory = np.zeros((nb_step,2*d,2*d))
    if record_var == "Speed":
        T, nb_step = 20*d, 16
        H = param["H"]
        matching = param["matching"]
    if record_var == "View":
        subgrid = view["subgrid_var"]
        single_layer_view = view["single_layer_view"]


    #Initialize Mask: {0 : Data, 1 : Sx, -1 : Sz}
    mask_array = init_mask(d)

    # Initialize data array based on initial condition
    if init_var == "Zeros":
        data_array = init_data_array(d)
    elif init_var == "Random":
        data_array = init_data_array(d) + (mask_array==0)*error_channel(2*d,2*d,0.5,rng).astype(np.int8)
    elif init_var == "Random Error":
        data_array = init_data_array(d) + (mask_array==0)*error_channel(2*d,2*d,error_rate,rng).astype(np.int8)
    elif init_var == "Specified":
        data_array = param["init_data_array"]

    # Initialize signal arrays
    forward_signal_1_array = init_signal_array(d)
    anti_signal_1_array = init_signal_array(d)
    stack_1_array = init_signal_array(d)

    forward_signal_2_array = init_signal_array(d)
    anti_signal_2_array = init_signal_array(d)
    stack_2_array = init_signal_array(d)

    # Initialize defect array
    defect_array = np.zeros_like(data_array)

    # Main simulation loop
    for t in range(T):

################################### Errors #####################################

        if error_bool: # Introduce random errors if enabled
            new_errors_array = (mask_array==0)*error_channel(2*d,2*d,error_rate,rng).astype(np.int8)
            data_array = (data_array + new_errors_array)%2

        if (record_var == "Stability") and (t == 9):
            data_array = (data_array + custom_error)%2

        if record_var == "Field":
            data_array[rows, cols] = values

############################### Measure Parities ###############################

        #visualize_ancilla_particles(K,defect_array,forward_signal_1_array,forward_signal_2_array,anti_signal_1_array,anti_signal_2_array,stack_1_array,stack_2_array)
        #K+=1

        defect_array = get_defect(data_array,mask_array,meas_error_bool,meas_error_rate,rng)

        #defect_array,data_array = fix_defect(defect_array,data_array,rows,cols,values,defect_to_fix) if record_var == "Field" else (defect_array,data_array)
        defect_array = add_artificial_defect(defect_array,dict_artificial_defect,t) if artificial_defect_bool else defect_array

########################## Instantaneous correction ############################
        
        instantaneous_correction_array = get_instantaneous_correction(defect_array)
        
        data_array, defect_array = (data_array + instantaneous_correction_array)%2, (defect_array + get_defect_determistic(instantaneous_correction_array,mask_array))%2
        defect_array,data_array = fix_defect(defect_array,data_array,rows,cols,values,defect_to_fix) if record_var == "Field" else (defect_array,data_array)

################################### Self rules ################################
        
        
################################## Update rules ################################

        if record_var == "View":
            view_particles(K,defect_array,forward_signal_1_array,forward_signal_2_array,anti_signal_1_array,anti_signal_2_array,stack_1_array,stack_2_array,subgrid,single_layer_view)
            view_field(K,forward_signal_1_array,forward_signal_2_array,subgrid)
            K+=1

        # Propagate and recombine anti signals

        emptying_stack_1_array = (stack_1_array > 0)
        emptying_stack_2_array = (stack_2_array > 0)
        
        anti_signal_1_array,stack_1_array = create_anti_signals_1(defect_array,anti_signal_1_array,stack_1_array,emptying_stack_1_array)
        anti_signal_2_array,stack_2_array = create_anti_signals_2(defect_array,forward_signal_1_array,anti_signal_2_array,stack_2_array,emptying_stack_2_array)

        for k in range(anti_signal_velocity):
            anti_signal_1_array = propagate_signals_1(anti_signal_1_array,1,wrap=False if record_var in ["Quench","Speed"] else True)
            forward_signal_1_array,anti_signal_1_array = recombine_signals(forward_signal_1_array,anti_signal_1_array)
            anti_signal_2_array = propagate_signals_2(anti_signal_2_array,1,wrap=False if record_var in ["Quench","Speed"] else True)
            forward_signal_2_array,anti_signal_2_array = recombine_signals(forward_signal_2_array,anti_signal_2_array)
            

        # Emission & propagation of type 1 forward signals // type 1 stack increment
        forward_signal_1_array,stack_1_array = create_forward_signals_1(defect_array,forward_signal_1_array,stack_1_array)
        forward_signal_1_array = propagate_signals_1(forward_signal_1_array,1,wrap=False if record_var in ["Quench","Speed"] else True)

        if record_var == "View":
            view_particles(K,defect_array,forward_signal_1_array,forward_signal_2_array,anti_signal_1_array,anti_signal_2_array,stack_1_array,stack_2_array,subgrid,single_layer_view)
            view_field(K,forward_signal_1_array,forward_signal_2_array,subgrid)
            K+=1

        # Emission & propagation of type 2 forward signals // type 2 stack increment
        forward_signal_2_array,stack_2_array = create_forward_signals_2(defect_array,forward_signal_1_array,forward_signal_2_array,stack_2_array)
        forward_signal_2_array = propagate_signals_2(forward_signal_2_array,1,wrap=False if record_var in ["Quench","Speed"] else True)

        if record_var == "Field":
            if t in t_to_record_array:
                vector_array = get_vector(forward_signal_1_array,forward_signal_2_array)
                field_array[t-t_to_record_array[0]] = slice_on_focus(defect_on_focus,vector_array,radius)
        if record_var == "Quench":
            if t%(T//nb_step) == 0:
                defect_trajectory[t//(T//nb_step)] = data_array
        if record_var == "Speed":
            if (t%(T//nb_step) == 0) and (t >= d//2+d//4):
                error_weight = int(np.sum(matching.decode(H@get_data_as_vector(data_array)%2)))
                if error_weight < d//4:
                    return(t)
        
        #desactivated_forward_1_array, desactivated_forward_2_array = np.zeros_like(forward_signal_1_array), np.zeros_like(forward_signal_2_array)

        # Correction from forward signals meeting defects
        final_correction_array = get_correction(defect_array,forward_signal_1_array,forward_signal_2_array)
        data_array, defect_array = (data_array + final_correction_array)%2, (defect_array + get_defect_determistic(final_correction_array,mask_array))%2
        defect_array,data_array = fix_defect(defect_array,data_array,rows,cols,values,defect_to_fix) if record_var == "Field" else (defect_array,data_array)

        if record_var == "Analysis":
            number_of_particles = get_number_of_particles(forward_signal_1_array,anti_signal_1_array,stack_1_array,forward_signal_2_array,anti_signal_2_array,stack_2_array)
            number_of_particles_array[t] = number_of_particles
        if record_var == "Convergence":
            number_of_particles = get_number_of_particles(forward_signal_1_array,anti_signal_1_array,stack_1_array,forward_signal_2_array,anti_signal_2_array,stack_2_array)
            number_of_defects = np.sum(defect_array)
            if (first_time_zero_defect == -1) and (number_of_defects == 0):
                first_time_zero_defect = t
            if (first_time_zero_defect != -1) and (number_of_particles == 0):
                first_time_zero_particles = t
            if (first_time_zero_defect != -1) and (first_time_zero_particles != -1):
                return(first_time_zero_defect,first_time_zero_particles)


################################# Output ######################################

    if record_var == "Data":
        return(data_array)
    elif record_var == "Analysis":
        return(data_array,number_of_particles_array,forward_signal_1_array,anti_signal_1_array,stack_1_array,forward_signal_2_array,anti_signal_2_array,stack_2_array)
    elif record_var == "Convergence":
        print("Simulation too short")
        return(T,T)
    elif record_var == "Field":
        return(field_array)
    elif record_var == "Speed":
        return(float('inf'))
    elif record_var == "Quench":
        return(defect_trajectory)