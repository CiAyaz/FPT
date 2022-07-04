#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit

@njit()
def compute_passage_times(x, dt, sign_x_minus_xstart, sign_x_minus_xfinal, fpt_array_with_recrossings, xstart, xfinal, previous_x = 0., previous_sign_xstart = 0., previous_sign_xfinal = 0., time_step = 0, current_number_recrossings = 0):
    """
    Compute passage times (first passage times without recrossings, first passage times with recrossings and transition path times) between configurations xstart and xfinal in time series data x with time step dt.
    """
    array_size = len(fpt_array_with_recrossings)
    fpt_array = np.zeros((array_size, ), dtype=np.float64)
    tpt_array = np.zeros((array_size, ), dtype=np.float64)
    total_number_recrossings = current_number_recrossings
    index = 0
    if previous_sign_xstart == 0 and previous_sign_xfinal == 0:
        previous_sign_xstart = sign_x_minus_xstart[0]
        previous_sign_xfinal = sign_x_minus_xfinal[0]
    for i in range(0 ,len(x)):
        if sign_x_minus_xstart[i] != previous_sign_xstart:
            v = (x[i] - previous_x) / dt
            delta_t = (xstart - previous_x) / v
            fpt_array_with_recrossings[total_number_recrossings] = - (time_step * dt - delta_t)
            total_number_recrossings += 1
            current_number_recrossings += 1
        if sign_x_minus_xfinal[i] != previous_sign_xfinal and current_number_recrossings != 0:
            v = (x[i] - previous_x) / dt
            delta_t = (xfinal - previous_x) / v
            for recross in range(current_number_recrossings):
                fpt_array_with_recrossings[total_number_recrossings - 1 - recross] += time_step * dt - delta_t
            tpt_array[index] = fpt_array_with_recrossings[total_number_recrossings - 1] 
            fpt_array[index] = fpt_array_with_recrossings[total_number_recrossings - current_number_recrossings]
            index += 1
            current_number_recrossings = 0
            time_step = 0
        previous_sign_xstart = sign_x_minus_xstart[i]
        previous_sign_xfinal = sign_x_minus_xfinal[i]
        previous_x = x[i]
        if current_number_recrossings != 0:
            time_step += 1
    integer_stats_for_continuation = np.array([time_step, current_number_recrossings], dtype=np.int64)
    float_stats_for_continuation = np.array([previous_x, previous_sign_xstart, previous_sign_xfinal])
    return float_stats_for_continuation, integer_stats_for_continuation, fpt_array, tpt_array, fpt_array_with_recrossings

def compute_passage_time_arrays(trajectories , dt , xstart_vector , xfinal_vector , array_size):
    """
    Computes passage times: first first passage times, all first passage times and transition path times.
    trajectories: 1d numpy array trajectory or a list of strings with paths to 1d trajectories,
    dt: time step,
    xstart_vector and xfinal_vector: 1d arrays with positions to compute the times it takes to go from xstart to xfinal,
    array_size: the size of the passage times arrays. As an estimate, use expected maximum number of passage events.
    """
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