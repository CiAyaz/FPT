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