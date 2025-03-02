# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:27:36 2015

@author: jonas
"""

import numpy as N
import scipy.signal as si


def filterData(data,low=600,high=6000,Fs=30000):
    fto= float(high)*2/float(Fs)
    ffrom= float(low)*2/float(Fs)
    b,a=si.butter(2,[ffrom,fto],btype="bandpass", analog=False, output="ba")
    filtered=si.filtfilt(b,a,data)
    return filtered

def lowpass(data,high=6000,Fs=30000):
    fto= float(high)*2/float(Fs)
    
    b,a=si.butter(2,fto,btype="lowpass", analog=False, output="ba")
    filtered=si.filtfilt(b,a,data)
    return filtered

def envelope(data):
    if len(data)%2 == 1:                       # Cut the lenghts of the vectors to an even number.
        data=data[:(len(data)-1)]    
    hil=si.hilbert(data)
    env=N.abs(hil)
    return env
  
def mrl(data):
    alpha=N.asarray(data[~N.isnan(data)],dtype='f8')
    t=N.exp(1j*alpha)
    r=N.sum(t)
    r=N.abs(r)/alpha.shape
    return r[0]

