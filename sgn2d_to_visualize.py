import numpy as np

from init import *
from error import *
from creation import *
from recombination import *
from view import *

def SIGNAL2D(physics, decoder, simulation):

    rng = np.random.default_rng()

    # ===============================
    # Physics parameters
    # ===============================
    d = physics["code_distance"]
    data_error_rate = physics["data_error_rate"]
    meas_error_rate = physics["meas_error_rate"]
    meas_error = physics["meas_error"]

    # ===============================
    # Decoder parameters
    # ===============================
    online = decoder["online"]
    max_stack = decoder["max_stack"]
    anti_signal_velocity = decoder["anti_signal_velocity"]
    rules = decoder["matching_rules"]

    # ===============================
    # Simulation parameters
    # ===============================
    T = simulation["time_run"]
    dt = simulation["time_step"]

    initial_data_array = simulation["initial_data_array"]
    artificial_defect_bool = simulation["artificial_defect"] is not None
    dict_artificial_defect = simulation["artificial_defect"]

    record_var = simulation["recorded_variable"]

    if record_var == "Poisson":
        t_array = np.arange(0, T, dt, dtype=np.int32)
        data_history = np.zeros((len(t_array), 2*d, 2*d), dtype=float)
    if record_var == "View":
        configuration_history = []
        step_history = []

    # ===============================
    # Initialization
    # ===============================

    # Mask: {0 : Data, 1 : Sx, -1 : Sz}
    mask_array = init_mask(d)

    if initial_data_array is None:
        data_array = np.zeros((2*d,2*d))
        data_array[1::2, ::2] = np.nan  # Odd rows, even columns
        data_array[::2, 1::2] = np.nan  # Even rows, odd columns
    else:
        data_array = initial_data_array.copy()

    # Defect array
    defect_array = np.zeros((d,d)).astype(np.int8)

    # Signal arrays
    forward_signal_1_array = init_signal_array(d)
    anti_signal_1_array = init_signal_array(d)
    stack_1_array = init_signal_array(d)

    forward_signal_2_array = init_signal_array(d)
    anti_signal_2_array = init_signal_array(d)
    stack_2_array = init_signal_array(d)

    # ===============================
    # Main simulation loop
    # ===============================
    
    for t in range(T):

        if online:
            # Inject new physical errors
            new_errors_array = (mask_array == 0) * error_channel(d, d, data_error_rate, rng).astype(np.int8)
            data_array = (data_array + new_errors_array) % 2

            # Noisy syndrome extraction
            defect_array = get_defect(data_array, mask_array, meas_error, meas_error_rate, rng)
        else:
            # Deterministic (noise-free) syndrome extraction
            defect_array = get_defect_determistic(data_array, mask_array)

        # Optional manual defect injection (for testing / probing dynamics)
        defect_array = add_artificial_defect(defect_array, dict_artificial_defect, t) if artificial_defect_bool else defect_array

        # Matching of neighboring defects
        instantaneous_correction_array = get_instantaneous_correction(defect_array, rules)
        data_array = (data_array + instantaneous_correction_array) % 2

        # Update defects after instantaneous correction
        defect_array = (defect_array + get_defect_determistic(instantaneous_correction_array, mask_array)) % 2

        if record_var == "View": #Creation and propagation of 1-forward-signals - 1
            # Append configuration
            configuration = defect_array, forward_signal_1_array, forward_signal_2_array, anti_signal_1_array, anti_signal_2_array,stack_1_array, stack_2_array
            configuration_history.append(configuration)
            step_history.append((t+1,1))

        # Emit and propagate forward signals
        forward_signal_1_array, stack_1_array = create_forward_signals_1(
            defect_array, forward_signal_1_array, stack_1_array, max_stack
        )
        forward_signal_1_array = propagate_signals_1(forward_signal_1_array, 1, wrap=True)

        if record_var == "View": #Creation and propagation of 2-forward-signals - 2
            # Append configuration
            configuration = defect_array, forward_signal_1_array, forward_signal_2_array, anti_signal_1_array, anti_signal_2_array,stack_1_array, stack_2_array
            configuration_history.append(configuration)
            step_history.append((t+1,2))

        forward_signal_2_array, stack_2_array = create_forward_signals_2(
            defect_array, forward_signal_1_array, forward_signal_2_array, stack_2_array, max_stack
        )
        forward_signal_2_array = propagate_signals_2(forward_signal_2_array, 1, wrap=True)

        if record_var == "View": #Creation and propagation of 1-forward-signals - 3
            # Append configuration
            configuration = defect_array, forward_signal_1_array, forward_signal_2_array, anti_signal_1_array, anti_signal_2_array,stack_1_array, stack_2_array
            configuration_history.append(configuration)
            step_history.append((t+1,3))

        # Final correction inferred from signal interference
        final_correction_array = get_correction(
            defect_array, forward_signal_1_array, forward_signal_2_array
        )
        data_array = (data_array + final_correction_array) % 2

        # Update defects after final correction
        defect_array = (defect_array + get_defect_determistic(final_correction_array, mask_array)) % 2

        if record_var == "View": #Emission of anti-signals - 4
            # Append configuration
            configuration = defect_array, forward_signal_1_array, forward_signal_2_array, anti_signal_1_array, anti_signal_2_array,stack_1_array, stack_2_array
            configuration_history.append(configuration)
            step_history.append((t+1,4))

        # Identify stacks that are currently non-empty
        non_empty_stack_1_array = stack_1_array > 0
        non_empty_stack_2_array = stack_2_array > 0

        # Create anti-signals from defects and stacks
        anti_signal_1_array, stack_1_array = create_anti_signals_1(
            defect_array, anti_signal_1_array, stack_1_array, non_empty_stack_1_array
        )
        anti_signal_2_array, stack_2_array = create_anti_signals_2(
            defect_array, forward_signal_1_array, anti_signal_2_array,
            stack_2_array, non_empty_stack_2_array
        )

        if record_var == "View": #Emission of anti-signals - 4
            # Append configuration
            configuration = defect_array, forward_signal_1_array, forward_signal_2_array, anti_signal_1_array, anti_signal_2_array,stack_1_array, stack_2_array
            configuration_history.append(configuration)
            step_history.append((t+1,5))

        # Propagate anti-signals and recombine with forward signals
        for _ in range(anti_signal_velocity):

            anti_signal_1_array = propagate_signals_1(anti_signal_1_array, 1, wrap=True)
            forward_signal_1_array, anti_signal_1_array = recombine_signals(
                forward_signal_1_array, anti_signal_1_array
            )

            if record_var == "View": #Propagation of anti-signals and recombination - 5
                # Append configuration
                configuration = defect_array, forward_signal_1_array, forward_signal_2_array, anti_signal_1_array, anti_signal_2_array,stack_1_array, stack_2_array
                configuration_history.append(configuration)
                step_history.append((t+1,6))

            anti_signal_2_array = propagate_signals_2(anti_signal_2_array, 1, wrap=True)
            forward_signal_2_array, anti_signal_2_array = recombine_signals(
                forward_signal_2_array, anti_signal_2_array
            )

            if record_var == "View": #Propagation of anti-signals and recombination - 5
                # Append configuration
                configuration = defect_array, forward_signal_1_array, forward_signal_2_array, anti_signal_1_array, anti_signal_2_array,stack_1_array, stack_2_array
                configuration_history.append(configuration)
                step_history.append((t+1,6))

        # Store data for Poisson-time snapshots
        if record_var == "Poisson" and t in t_array:
            idx = np.where(t_array == t)[0]
            data_history[idx] = data_array

    # ===============================
    # Output
    # ===============================

    if record_var == "Data":
        return(data_array)
    elif record_var == "Poisson":
        return(data_history)
    elif record_var == "View":
        return(configuration_history,step_history)