import os
import sys
#import cProfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgn2d import SIGNAL2D
import param
from view import *
from mwpm import *

#def main():
#data_outcome_array, number_of_particles_array, forward_signal_1_array, backward_signal_1_array, anti_signal_1_array, stack_1_array, switch_1_array, #forward_signal_2_array, anti_signal_2_array, stack_2_array = SIGNAL2D(param_dict,option_dict,view_dict)


data_outcome_array = SIGNAL2D(param.param_dict,param.option_dict,param.view_dict)

#matching = get_matching(param.param_dict["d"])
#H = get_parity_check_matrix(param.param_dict["d"])

#print("MWPM from perfect syndrome measurement: \n")

#visualize_data((data_outcome_array+vector_to_array(matching.decode(H@get_data_as_vector(data_outcome_array)%2)))%2)

#if __name__ == "__main__":
#    cProfile.run('main()')