#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import math
import numba
from numba import njit

@njit()
def calc_fp(x, dt, xs, xf, llauf, tlauf, nrevents, mfpT, mfp, mfpT_first, mfp_first, corr, nr_fp):
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
    fp_first_array=np.zeros((length,), dtype=np.float64)
    index=0
    xxx=x0
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
           if tlauf==0:
               llauf=0
               mfpT_first=llauf*dt-ddt
           mfpT+=llauf*dt-ddt
           if mfpT>1e9:
               mfpT-=1e9
               corr+=1
           nr_fp+=1
           tlauf+=1
           signs=signs_n
        if signf_n!=signf:
            if tlauf!=0:
                v=(xx-xxx)/dt
                ddt=(xx-xf)/v
                fp=tlauf*(llauf*dt-ddt)
                fp_corr=fp-corr*1e9
                mfp+=fp_corr-mfpT
                mfp_first+=llauf*dt-ddt-mfpT_first
                fp_first_array[index]=llauf*dt-ddt-mfpT_first
                index+=1

                nrevents+=1
                mfpT=0
                tlauf=0
                corr=0
            signf=signf_n
            llauf=0
        xxx=xx
        llauf+=1
    fp_first_array=fp_first_array[np.nonzero(fp_first_array)[0]]
    stats=np.array([llauf, tlauf, nrevents, corr, nr_fp], dtype=np.int64)
    times=np.array([mfpT, mfp, mfpT_first, mfp_first], dtype=np.float64)
    return stats, times, fp_first_array

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
