#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import math
import numba
from numba import njit

@njit()
def calc_fpt( x , dt , sign_x_minus_xstart , sign_x_minus_xfinal , fpt_array_with_recrossings , xstart , xfinal , previous_x = 0 , previous_sign_xstart = 0 , previous_sign_xfinal = 0 , time_step = 0 , current_number_recrossings = 0 ):
    """
    Compute first passage times between configurations xstart and xfinal in time series data  x with time step dt.
    """
    array_size = len( fpt_array_with_recrossings )
    fpt_array = np.zeros(( array_size ,), dtype=np.float64)
    tpt_array = np.zeros(( array_size ,), dtype=np.float64)
    total_number_recrossings = current_number_recrossings
    index = 0
    if previous_sign_xstart == 0 and previous_sign_xfinal == 0:
        previous_sign_xstart = sign_x_minus_xstart [0]
        previous_sign_xfinal = sign_x_minus_xfinal [0]
    for i in range( 0 , traj_length ):
        if sign_x_minus_xstart [i] != previous_sign_xstart :
            v = ( x [i] - previous_x )/dt
            delta_t = ( xstart - previous_x )/ v
            fpt_array_with_recrossings [ total_number_recrossings ] = -( time_step * dt - delta_t )
            total_number_recrossings += 1
            current_number_recrossings += 1
        if sign_x_minus_xfinal [i] != previous_sign_xfinal :
            if current_number_recrossings != 0:
                v = ( x [i] - previous_x )/ dt
                delta_t = ( xfinal - previous_x )/ v
                for recross in range( current_number_recrossings ):
                    fpt_array_with_recrossings [ total_number_recrossings - 1 - recross ] += time_step * dt - delta_t
                tpt_array [ index ] = fpt_array_with_recrossings [ total_number_recrossings - 1 ] 
                fpt_array [ index ] = fpt_array_with_recrossings [ total_number_recrossings - current_number_recrossings ]
                index += 1
                current_number_recrossings = 0
                time_step = 0
        previous_sign_xstart = sign_x_minus_xstart [i]
        previous_sign_xfinal = sign_x_minus_xfinal [i]
        previous_x = x [i]
        if current_number_recrossings != 0:
            time_step += 1
    integer_stats_for_continuation = np.array([ time_step , current_number_recrossings , previous_sign_xstart , previous_sign_xfinal ], dtype=np.int64)
    float_stats_for_continuation = np.array([ previous_x ])
    return float_stats_for_continuation , interger_stats_for_continuation , fpt_array , tpt_array , fpt_array_with_recrossings

def comp_tarray(x, dt, st_pos_vec, fin_pos_vec, integer_stats, float_stats):
    fp_first_dict=dict()
    npos_st=len(st_pos_vec)
    npos_fin=len(fin_pos_vec)
    for i in range(npos_st):
        for j in range(npos_fin):
            if st_pos_vec[i]==fin_pos_vec[j]:
                continue
            else:
                previous_x,=times_arr[i]
                stats=stats_arr[i,j]
                stat,time,fp_first_array = calc_fp(x, dt=dt, xs=st_pos_vec[i], xf=fin_pos_vec[j], llauf=stats[0], tlauf=stats[1], nrevents=stats[2], mfpT=times[0], mfp=times[1], mfpT_first=times[2], mfp_first=times[3], corr=stats[3], nr_fp=stats[4])
                fp_first_dict[i,j]=fp_first_array
                times_arr[i,j]=time
                stats_arr[i,j]=stat
    return stats_arr, times_arr, fp_first_dict

@njit()
def calc_tr(x, dt, xs, xf, T, llauf, t1, t2, nrevents):
    x0=x[0]
    if x0!=xs:
        signs=int((x0-xs)/abs(x0-xs))
    else:
        signs=int(0)
    if x0!=xf:
        signf=int((x0-xf)/abs(x0-xf))
    else:
        signf=int(0)
    length=len(x)
    trarr1=np.zeros((length,),dtype=np.float64)
    trarr2=np.zeros((length,),dtype=np.float64)
    xxx=x0
    fplauf1=0
    fplauf2=0
    for i in range(length):
        xx=x[i]
        if xx!=xs:
            signs_n=int((xx-xs)/abs(xx-xs))
        else:
            signs_n=int(0)
        if xx!=xf:
            signf_n=int((xx-xf)/abs(xx-xf))
        else:
            signf_n=int(0)
        if signs_n!=signs or signs_n==0:
            v=(xx-xxx)/dt
            if v==0:
                ddt=0
            else:
                ddt=(xx-xs)/v
            if t2!=0:
                trarr1[fplauf1]=llauf*dt-ddt-T
                fplauf1+=1
                t2=0
                nrevents+=1
            T=-ddt
            t1+=1
            llauf=0
            signs=signs_n
        if signf_n!=signf:
            v=(xx-xxx)/dt
            if v==0:
                ddt=0
            else:
                ddt=(xx-xf)/v
            if t1!=0:
                trarr2[fplauf2]=llauf*dt-ddt-T
                fplauf2+=1
                t1=0
                nrevents+=1
            T=-ddt
            t2+=1
            llauf=0
            signf=signf_n
        xxx=xx
        llauf+=1
    times=np.array([T, llauf, t1, t2, nrevents], dtype=np.float64)
    return trarr1, trarr2, times
