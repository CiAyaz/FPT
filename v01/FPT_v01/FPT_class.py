import numpy as np
from tools import compute_passage_times

class FPT():
    """Main class for computing passage times distributions. 
    Computes passage times: first first passage times, all first passage times and transition path times.
    trajectories: 1d numpy array trajectory or a list of strings with paths to 1d trajectories,
    dt: time step,
    xstart_vector and xfinal_vector: 1d arrays with positions to compute the times it takes to go from xstart to xfinal,
    array_size: the size of the passage times arrays. As an estimate, use expected maximum number of passage events."""
    
    def __init__(self, dt, array):
        self.dt = dt


    def compute_passage_time_arrays(trajectories , dt , xstart_vector , xfinal_vector , array_size):
        fpt_wr_dict = dict()
        fpt_dict = dict()
        tpt_dict = dict()
        number_xstarts = len(xstart_vector)
        number_xfinals = len(xfinal_vector)
        if isinstance(trajectories ,(list, np.ndarray)):
            fpt_array_with_recrossings = np.zeros((array_size, ), dtype=np.float64)
            if isinstance(trajectories, list):
                number_of_trajs = len(trajectories)
                for index_trajs, path_to_traj in enumerate(trajectories):
                    x = np.load(path_to_traj)
                    for index_xstart, xstart in enumerate(xstart_vector):
                        for index_xfinal, xfinal in enumerate (xfinal_vector):
                            if xstart == xfinal:
                                continue
                            else:
                                sign_x_minus_xstart = np.sign(x - xstart)
                                sign_x_minus_xfinal = np.sign(x - xfinal)
                                if index_trajs == 0:
                                    float_stats, integer_stats, fpt_array, tpt_array, fpt_array_with_recrossings = compute_passage_times(x, dt, sign_x_minus_xstart, sign_x_minus_xfinal, fpt_array_with_recrossings, xstart, xfinal)
                                    previous_x = float_stats[0]
                                    previous_sign_xstart = float_stats[1]
                                    previous_sign_xfinal = float_stats[2]
                                    time_step = integer_stats[0]
                                    current_number_recrossings = integer_stats[1]
                                else:
                                    float_stats, integer_stats, fpt_array, tpt_array, fpt_array_with_recrossings = compute_passage_times(x, dt, sign_x_minus_xstart, sign_x_minus_xfinal, fpt_array_with_recrossings, xstart, xfinal, previous_x, previous_sign_xstart, previous_sign_xfinal, time_step, current_number_recrossings)
                                    previous_x = float_stats[0]
                                    previous_sign_xstart = float_stats[1]
                                    previous_sign_xfinal = float_stats[2]
                                    time_step = integer_stats[0]
                                    current_number_recrossings = integer_stats[1]
                                fpt_array_with_recrossings = fpt_array_with_recrossings[fpt_array_with_recrossings != 0.]
                                if current_number_recrossings != 0:
                                    nonzero_length = len(fpt_array_with_recrossings)
                                    fpt_dummy = fpt_array_with_recrossings[ - current_number_recrossings: ]
                                    fpt_array_with_recrossings = fpt_array_with_recrossings[ :nonzero_length - current_number_recrossings ]
                                else:
                                    fpt_dummy = 0
                                if index_trajs == 0:
                                    fpt_wr_dict[index_xstart, index_xfinal] = fpt_array_with_recrossings
                                    fpt_dict[index_xstart, index_xfinal] = fpt_array
                                    tpt_dict[index_xstart, index_xfinal] = tpt_array
                                else:
                                    fpt_wr_dict[index_xstart, index_xfinal] = np.append(fpt_wr_dict[index_xstart, index_xfinal] , fpt_array_with_recrossings)
                                    fpt_dict[ index_xstart, index_xfinal] = np.append(fpt_dict[index_xstart, index_xfinal ], fpt_array)
                                    tpt_dict[ index_xstart, index_xfinal] = np.append(tpt_dict[index_xstart, index_xfinal ], tpt_array)
                                fpt_array_with_recrossings = np.zeros((array_size, ), dtype=np.float64)
                                fpt_array_with_recrossings[ :current_number_recrossings ] = fpt_dummy
            else:
                x = trajectories
                for index_xstart, xstart in enumerate(xstart_vector):
                    for index_xfinal, xfinal in enumerate(xfinal_vector):
                        if xstart==xfinal:
                            continue
                        else:
                            sign_x_minus_xstart = np.sign( x - xstart )
                            sign_x_minus_xfinal = np.sign( x - xfinal )
                            float_stats, integer_stats, fpt_array, tpt_array, fpt_array_with_recrossings = compute_passage_times( x, dt, sign_x_minus_xstart, sign_x_minus_xfinal, fpt_array_with_recrossings, xstart, xfinal)
                            previous_x = float_stats[0]
                            previous_sign_xstart = float_stats[1]
                            previous_sign_xfinal = float_stats[2]
                            time_step = integer_stats[0]
                            current_number_recrossings = integer_stats[1]
                            fpt_array_with_recrossings = fpt_array_with_recrossings[ fpt_array_with_recrossings != 0. ]
                            if current_number_recrossings != 0:
                                    nonzero_length = len( fpt_array_with_recrossings )
                                    fpt_array_with_recrossings = fpt_array_with_recrossings[ :nonzero_length - current_number_recrossings ]
                            fpt_wr_dict[ index_xstart, index_xfinal ] = fpt_array_with_recrossings
                            fpt_dict[ index_xstart, index_xfinal ] = fpt_array
                            tpt_dict[ index_xstart, index_xfinal ] = tpt_array
            return fpt_wr_dict, fpt_dict, tpt_dict
        else:
            assert('trajectory must be a list of paths to numpy arrays or a numpy array!')