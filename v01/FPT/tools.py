#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit

@njit()
def compute_passage_times(
    x, 
    dt,
    sign_x_minus_xstart,
    sign_x_minus_xfinal,
    fpt_array_with_recrossings,
    xstart,
    xfinal,
    float_values_for_continuation,
    integer_values_for_continuation):
    """
    Compute passage times (first passage times without recrossings, 
    first passage times with recrossings and transition path times) 
    between configurations xstart and xfinal in time series data x with time step dt.
    Arguments
    Returns
    """
    previous_x = float_values_for_continuation[0]
    previous_sign_xstart = float_values_for_continuation[1]
    previous_sign_xfinal = float_values_for_continuation[2]
    time_step = integer_values_for_continuation[0]
    current_number_recrossings = integer_values_for_continuation[1]
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
            if current_number_recrossings > array_size:
                raise IndexError('number of passage events with recrossings is larger than array_size!')
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
    integer_values_for_continuation = np.array([time_step, current_number_recrossings], dtype=np.int64)
    float_values_for_continuation = np.array([previous_x, previous_sign_xstart, previous_sign_xfinal])
    return float_values_for_continuation, integer_values_for_continuation, fpt_array, tpt_array, fpt_array_with_recrossings

@njit()
def find_transition_paths(
    x, 
    sign_x_minus_xstart,
    sign_x_minus_xfinal,
    array_size,
    float_values_for_continuation,
    integer_values_for_continuation):
    """
    Compute passage times (first passage times without recrossings, 
    first passage times with recrossings and transition path times) 
    between configurations xstart and xfinal in time series data x with time step dt.
    Arguments
    Returns
    """
    transition_path_indices = np.zeros((array_size, 2), dtype=np.int64)
    total_number_recrossings = integer_values_for_continuation[0]
    total_number_transitions = integer_values_for_continuation[1]
    current_number_recrossings = integer_values_for_continuation[2]
    previous_sign_xstart = float_values_for_continuation[0]
    previous_sign_xfinal = float_values_for_continuation[1]
    starting_index = 0
    index = 0
    if previous_sign_xstart == 0 and previous_sign_xfinal == 0:
        previous_sign_xstart = sign_x_minus_xstart[0]
        previous_sign_xfinal = sign_x_minus_xfinal[0]
    for i in range(0 ,len(x)):
        if sign_x_minus_xstart[i] != previous_sign_xstart:
            starting_index = i
            current_number_recrossings += 1
            total_number_recrossings +=1
        if sign_x_minus_xfinal[i] != previous_sign_xfinal and current_number_recrossings != 0:
            total_number_transitions +=1
            index += 1
            current_number_recrossings = 0
            ending_index = i
            transition_path_indices[index] = np.array([starting_index, ending_index])
        previous_sign_xstart = sign_x_minus_xstart[i]
        previous_sign_xfinal = sign_x_minus_xfinal[i]
    if current_number_recrossings != 0:
        index +=1
        transition_path_indices[index] = np.array([starting_index, 0])
    integer_values_for_continuation = np.array([total_number_recrossings, total_number_transitions, current_number_recrossings], dtype=np.int64)
    float_values_for_continuation = np.array([previous_sign_xstart, previous_sign_xfinal])
    return float_values_for_continuation, integer_values_for_continuation, transition_path_indices

@njit()
def kde_epanechnikov(x, edges, bw, norm=1e6):
    nrbins=len(edges)
    p=np.zeros(nrbins)
    for i in range(len(x)):
        for j in range(nrbins):    
            if abs(edges[j] - x[i]) <= bw:
                p[j] += (1 - ((edges[j] - x[i]) / bw) **2)
    p /= (norm * 4 * bw / 3)
    return p