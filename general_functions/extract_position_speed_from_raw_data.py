#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:20:09 2024

@author: Mei
"""

import numpy as N
import matplotlib.pyplot as pl
import open_helper as op
import os
import matplotlib
import scipy.signal as si


# Directories of the data and target.
target_dir="path_to_target_dir"


### function for extracting position and estimating speed
def estimate_speed_from_position(target_dir,track_length=400,thr=1,rect_thres=20,lowpass=0.5,Fs=30000,plotData=True):
    ''' Inputs:
        adc: raw recording of pulse-width modulated track position.
        Keyword arguments:
        Track length: Real length in cm
        thr: Threshold to detect position pulses
        rect_thres: Threshold of track reset for rectification
        Fs: sampling frequency (in Hz)
        Outputs:
        pos: Track position (in au)
        rect: rectified (cumulative) track position (in cm)
        speed: instant speed in cm/s
    
    '''
    # Obtain starting value of result = pulse width by measuring the first pulse 
    # and selecting this value as the starting value of result.
    os.chdir(target_dir)
    adc=op.open_helper('100_ADC1.continuous')
    for n in range(1,3000):
        if adc[-n]<0.5:
            adc=adc[:-n]
            print 'index end', -n
            break
    
    pos=create_position_signal_from_pulses(adc,thr=thr,Fs=Fs)
    rect=rectify_position_signal(pos,track_length=track_length)    
    speed=create_speed_from_rectified_signal(rect)
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if plotData==True:
        f,(ax1,ax2,ax3)=pl.subplots(3,1,sharex=True)
        
        ax1.plot(pos)
        ax1.set_ylabel("Track position (au)")
        ax2.plot(rect)
        ax2.set_ylabel("Cumulative position (cm)")
        ax3.plot(speed)
        ax3.set_ylabel("Speed (cm/s)")
        f.savefig('pos_speed_rect.png' )
        f.savefig('pos_speed_rect.eps' ,transparent=True)
    
    os.chdir(target_dir)
    N.save('estimate speed from position_corrected posnew.npy', pos) 
    N.save('estimate speed from position_rect.npy', rect) 
    N.save('estimate speed from position_speed.npy', speed) 
    return pos,rect,speed

def correct_speed_measurement(pos,start,end,value,target_dir,lowpass=0.5,track_length=400,plotData=True):
### Manual correction for speed recording based on VR output. 
    '''
    input: 
    pos: output pos from 'estimate_speed_from_position'
    start, end: starting and finishing indice of misdetected track position.
    value: correct track position
    output:
    corrected track position and speed
    '''

    posnew=[]
    for n in range(len(pos)):
        if start<n<end:
            posnew.append(value)
        else:
            posnew.append(pos[n])
    
    rect=rectify_position_signal(posnew,track_length=track_length)    
    speed=create_speed_from_rectified_signal(rect)
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if plotData==True:
        f,(ax1,ax2,ax3)=pl.subplots(3,1,sharex=True)
        ax1.plot(posnew)
        ax1.set_ylabel("Track position (au)")
        ax2.plot(rect)
        ax2.set_ylabel("Cumulative position (cm)")
        ax3.plot(speed)
        ax3.set_ylabel("Speed (cm/s)")
    
    os.chdir(target_dir)
    N.save('estimate speed from position_corrected posnew.npy', posnew) 
    N.save('estimate speed from position_rect.npy', rect) 
    N.save('estimate speed from position_speed.npy', speed) 
    return posnew,rect,speed

#### Helper functions for estimate_speed_from_position and correct_speed_measurement.     
def create_position_signal_from_pulses(adc,thr=1,Fs=30000):
    n=0
    result=0
    while n<(len(adc)-1):
        if adc[n]<thr and adc[n+1]>thr:
            l=0
            while l<100:
                if adc[n+l]>thr and adc[n+l+1]<thr:                    
                    result=l
                    break
                else:
                    l+=1
            break
        else:
            n+=1
        
    pos=[]
    n=0
    while n<(len(adc)-1):
        if adc[n]<thr and adc[n+1]<thr:
            pos.append(result)
            n+=1
        elif adc[n]<thr and adc[n+1]>thr:
            l=0
            while l<100:
                if adc[n+l]>thr and adc[n+l+1]<thr:                    
                    result=l
                    break
                else:
                    l+=1
            pos.append(result)
            n+=1
        else:
             pos.append(result)  
             n+=1
    
    pos.append(result)
    return pos
        
def rectify_position_signal(pos,lowpass=0.5,rect_thres=20,track_length=150):  
    # rectify the signal.
    total_span=N.max(pos)-N.min(pos)
    scale=track_length/total_span
    rect=[]
    n=1
    maxi=N.max(pos)
    counter=0
    rect.append(pos[0]*scale)
    while n<len(pos):
        if pos[n]<rect_thres and pos[n-1]>rect_thres:            
            counter+=1
        value=(pos[n]+(maxi*counter))*scale
        rect.append(value)
        n+=1
    rect.append(value)                            
    rect=rect-rect[0]
    # smooth position by filtering.   
    rect=datatool_lowpass(rect,high=lowpass)
    return rect
    
def create_speed_from_rectified_signal(rect):
    
    speed=[]
    for n in range(1,len(rect),1):
        speed.append((rect[n]-rect[n-1])*30000)
    # smooth speed by filtering.
    speed=datatool_lowpass(speed,high=0.5)
    return speed


def datatool_lowpass(data,high=6000,Fs=30000):
    fto= float(high)*2/float(Fs)
    
    b,a=si.butter(2,fto,btype="lowpass", analog=False, output="ba")
    filtered=si.filtfilt(b,a,data)
    return filtered


#%%
### Executed part of the code.
pos,rect,speed=estimate_speed_from_position(target_dir)
start=16048000
end=16058000
value=48
posnew,rect,speed=correct_speed_measurement(pos,start,end,value,target_dir)