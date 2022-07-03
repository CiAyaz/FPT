#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import math
import numba
from numba import njit

@njit()
def calc_fpt( x , xstart_signs , xfinal_signs , x_start , x_final , dt , time_step = 0 , recrossings = 0 , fpt = 0. , sign_x_start_before = 0 , sign_x_final_before = 0 , x_past = 0. ):
    """
    Compute first passage times between configurations x_start and x_final in time series data  x with time step dt.
    """
    traj_length =len( x )
    fpt_array = np.zeros(( traj_length ,), dtype=np.float64)
    tpt_array = np.zeros(( traj_length ,), dtype=np.float64)
    fpt_array_with_recrossings = np.zeros(( traj_length ,), dtype=np.float64)
    nr_events_with_recrossings = 0
    index = 0
    for i in range( 0 , traj_length ):
        if xstart_signs [i] != xstart_signs [i+1] :
            if recrossings == 0:
                time_step = 0
            if xstart_signs [i] == 0:
                aa
            elif xstart_signs [i+1] == 0:
                aa
            else:
                v = ( x [i+1] - x [i] )/dt
                delta_t = ( x [i+1] - x_start )/ v
           if recrossings == 0:
               time_step = 0
               fpt -= delta_t
           fpt_array_with_recrossings [ nr_events_with_recrossings ] = -( time_step * dt - delta_t )
           nr_events_with_recrossings += 1
           recrossings += 1
           sign_x_start_before = sign_x_start_after
        if sign_x_final_after != sign_x_final_before :
            if recrossings != 0:
                v = ( x_present - x_past )/ dt
                delta_t = ( x_present - x_final )/ v
                for recross in range( recrossings ):
                    fpt_array_with_recrossings [ nr_events_with_recrossings - 1 - recross ] += time_step * dt - delta_t
                tpt_array [ index ] = fpt_array_with_recrossings [ nr_events_with_recrossings - 1 ] 
                fpt_array [ index ] = time_step * dt - delta_t - fpt
                index += 1
                recrossings = 0
            sign_x_final_before = sign_x_final_after
            time_step = 0
        x_past = x_present
        time_step += 1
    fp_first_array = fp_first_array [np.nonzero( fp_first_array )[0]]
    stats = np.array([ time_step , recrossings , nr_events_no_recrossings , single_prec_corrections , nr_events_with_recrossings ], dtype=np.int64)
    times = np.array([ mfpT , mfp , mfpT_first , mfp_first ], dtype=np.float64)
    return stats , times , fp_first_array

@njit()
def comp_tarray(x, dt, st_pos_vec, fin_pos_vec, stats_arr, times_arr):
    fp_first_dict=dict()
    npos_st=len(st_pos_vec)
    npos_fin=len(fin_pos_vec)
    for i in range(npos_st):
        for j in range(npos_fin):
            if st_pos_vec[i]==fin_pos_vec[j]:
                continue
            else:
                times=times_arr[i,j]
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
