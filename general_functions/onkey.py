#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:11:13 2019

@author: physiol1
functions:
    onkey needs to be run each time using onkey function;
    DS detection manual can be independently used; to check which channel has the largest DS amplitude;
    then start using DS_detect_find_del (not as a function) run single lines, to find false positive DS;
    run function DS_artefact_del to delete false positive DS;
    run spike_DS_coupling_all_batch for all units on one shank.    
"""

import numpy as N
import js_data_tools as data_tools
import tkFileDialog
import open_helper as op
import os
import matplotlib.pyplot as pl
from scipy import stats
import time
#from matplotlib.widgets import Slider
#from matplotlib.widgets import Cursor
#from matplotlib.widgets import Button
import matplotlib.widgets as wdg
import math as ma
import scipy.signal as sig
import matplotlib

#Helper function to find false positive DS artefacts
def onkey(event):
    ev.append(int(event.xdata))
    _=pl.plot(event.xdata,event.ydata,'r.') #shift+click with mouse
    
#plot all raw channel data to decide which channel be used to detect DS    
def DS_detection_manual(low=1,high=1000,tscale=100,scale_std=3,Fs=30000,shank_No=1,plotData=True):
    '''
    low, high for filtering DS
    tscale: ms, scale for representing 8 channels
    scale_std: for setting threshold of detecting DS
    '''
    
    dir1=tkFileDialog.askopenfilename(title='position data directory')          
    directory, fname = os.path.split(dir1)
    dir1=op.open_helper(dir1)
    
    os.chdir(directory)
    posnew_immo = N.load('estimate speed from position_corrected posnew_immo.npy')
    speed_immo = N.load('estimate speed from position_speed_immo.npy')
    ind_immo = N.load('estimate speed from position_ind_immo.npy')
    
    if shank_No==1:
        raw_batch=op.open_helper('shank1.batch')
    elif shank_No==2:
        raw_batch=op.open_helper('shank2.batch')
    elif shank_No==3:
        raw_batch=op.open_helper('shank3.batch')
    elif shank_No==4:
        raw_batch=op.open_helper('shank4.batch')
    
    #dir2=tkFileDialog.askopenfilename(title='raw list of one shank_shankx.batch') 
    #raw_batch=op.open_helper(dir2)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if plotData==True:
        
        fig,axs=pl.subplots(int(ma.ceil(float(len(raw_batch))/2)),1,sharex=True,sharey=True,figsize=[16,12], subplot_kw=dict(frameon=False))
        #fig,axs=pl.subplots(sharex=True,sharey=True,figsize=[16,12])
        fig.subplots_adjust(hspace=0)
        axs=axs.ravel()
        for n in range(0,len(raw_batch),2): 
            os.chdir(directory)
            raw = op.open_helper(raw_batch[n])
            raw_immo = N.take(raw,ind_immo)
            raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high)
            N.save('%s_raw_immo.npy' %raw_batch[n],raw_immo)
            mean_immo=N.mean(raw_immo)
            std_immo=N.std(raw_immo)
            x=range(len(raw_immo))
            x=N.divide(x,float(Fs)/1000)
            #ax = fig.add_subplot(8,1,n/2+1)
            #ax.plot(x,raw_immo)
            #ax.hlines(mean_immo+scale_std*std_immo,0,max(x),colors='orange',linestyles='dashed')
            axs[n/2].plot(x,raw_immo)
            axs[n/2].hlines(mean_immo+scale_std*std_immo,0,max(x),colors='orange',linestyles='dashed')
            #plot a scrolling bar
            '''
            axcolor = 'lightgoldenrodyellow'
            axpos = pl.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
            spos = wdg.Slider(axpos, 'Pos', 0.1, max(x)-150000)
    
            def update(val):
                pos = spos.val
                axs[n/2].axis([pos,pos+tscale*Fs/1000,-1000,1000])
                fig.canvas.draw_idle()
        
            spos.on_changed(update)
        '''
       
    #ev=[]
    #cid=fig.canvas.mpl_connect('key_press_event',onkey)
    
    #file_to_save=tkFileDialog.asksaveasfilename(title="Choose target filename")
    #N.savetxt('%s_all_false_positive_DS_ind.txt' %file_to_save,ev)
    
    #ind=input('channel index in raw list? ')
    #return ind,posnew_immo,speed_immo,ind_immo,directory,raw_batch
    return posnew_immo,speed_immo,ind_immo,directory,raw_batch

def DS_detection_manual_scroll(low=1,high=1000,tscale=3000,scale_std=3,Fs=30000,shank_No=1,plotData=True):
    '''
    low, high for filtering DS
    tscale: ms, scale for representing 8 channels
    scale_std: for setting threshold of detecting DS
    '''
    
    dir1=tkFileDialog.askopenfilename(title='position data directory')          
    directory, fname = os.path.split(dir1)
    dir1=op.open_helper(dir1)
    
    os.chdir(directory)
    posnew_immo = N.load('estimate speed from position_corrected posnew_immo.npy')
    speed_immo = N.load('estimate speed from position_speed_immo.npy')
    ind_immo = N.load('estimate speed from position_ind_immo.npy')
    
    if shank_No==1:
        raw_batch=op.open_helper('shank1.batch')
    elif shank_No==2:
        raw_batch=op.open_helper('shank2.batch')
    elif shank_No==3:
        raw_batch=op.open_helper('shank3.batch')
    elif shank_No==4:
        raw_batch=op.open_helper('shank4.batch')
    else:
        print 'please type a proper shank No. 1-4, present shank No. %s' %shank_No
    
    #dir2=tkFileDialog.askopenfilename(title='raw list of one shank_shankx.batch') 
    #raw_batch=op.open_helper(dir2)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if plotData==True:
        decimation_factor=10
        fig,axs=pl.subplots(figsize=[16,12], subplot_kw=dict(frameon=False))
        #fig,axs=pl.subplots(sharex=True,sharey=True,figsize=[16,12])
        #fig.subplots_adjust(hspace=0)
        #axs=axs.ravel()
        for n in range(0,len(raw_batch),2): 
            os.chdir(directory)
            raw = op.open_helper(raw_batch[n])
            raw_immo = N.take(raw,ind_immo)
            raw_immo_down=sig.decimate(raw_immo,decimation_factor)
            raw_immo_filt = data_tools.filterData(raw_immo_down,low=low,high=high)
            #raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high)
            N.save('%s_raw_immo.npy' %raw_batch[n],raw_immo)
            mean_immo=N.mean(raw_immo)
            std_immo=N.std(raw_immo)
            x=N.multiply(range(len(raw_immo_filt)),decimation_factor) #keep the same timestamp after downsampling
            x=N.divide(x,float(Fs)/1000)    # ms
            #ax = fig.add_subplot(8,1,n/2+1)
            #ax.plot(x,raw_immo)
            #ax.hlines(mean_immo+scale_std*std_immo,0,max(x),colors='orange',linestyles='dashed')
            axs.plot(x,raw_immo_filt-2000*n/2)
            axs.hlines(mean_immo+scale_std*std_immo-2000*n/2,0,max(x),colors='orange',linestyles='dashed')
        axs.set_yticks(N.arange(-1000-2000*n/2,1000,step=1000))
        axs.set_yticklabels(N.concatenate((N.tile([-1000,0],int(ma.ceil(float(len(raw_batch))/2))),[1000])))
        #plot a scrolling bar
        axcolor = 'lightgoldenrodyellow'
        axpos = pl.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
        spos = wdg.Slider(axpos, 'Time (ms)', 0.1, max(x)-tscale, valstep=tscale*Fs/1000)

        def update(val):
            pos = spos.val
            axs.axis([pos,pos+tscale,-1000-2000*n/2,1000])
            fig.canvas.draw_idle()
        spos.on_changed(update)
    
    return posnew_immo,speed_immo,ind_immo,directory,raw_batch

#algorhithm to detect dentate spike    
def dentate_spike_detection(ind,posnew_immo,speed_immo,ind_immo,raw_batch,directory,scale_std=3,exceed_std=1,low=1,high=1000,Fs=30000):
    
    '''
    immo_thres: threshold of speed for immobility
    thres: for detecting dentate spike event, baseline+thres*SD
    low: low frequency for bandpass filterung
    high: high frequency for bandpass filterung
    mid: middle of the time window to detect the maximum exceeding thres
    start, end: start and end of the time window to detect the maximum exceeding thres    
    '''    
    #posnew_immo,speed_immo,ind_immo,directory,raw_batch=DS_detection_manual(low=low,high=high,scale_std=scale_std,Fs=Fs,plotData=False)
    #ind=input('channel index in raw list? ')
    
    os.chdir(directory)
    raw = op.open_helper(raw_batch[ind])
    raw_immo = N.take(raw,ind_immo)
    raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high)
    thr=N.mean(raw_immo)+scale_std*N.std(raw_immo)
    
    #detect dentate spike according to Nokia et al. 2017 & threshold according to Szabo et al. 2017
    ind_above_thr,ind_DS,ind_peak_DS,raw_DS,ind_peak_above_thr,res,ind_DS_range=[],[],[],[],[],[],[]
    for i in range(len(raw_immo))[1:]:        
        #find the first index exceeding threshold positive deflection
        if raw_immo_filt[i-1]<thr and raw_immo_filt[i]>thr:
            ind_above_thr.append(i)
            #use 20 ms time window to find the maximum of later half exceeding threshold 
            #and difference between the first half max and latter half max exceeding threshold
            #new threshold 100 ms before DS starts
            '''
            thr_new= N.mean(raw_immo[i-110*Fs/1000:i-10*Fs/1000])+thres*N.std(raw_immo[i-110*Fs/1000:i-10*Fs/1000])
            res.append(thr_new)
            print thr_new
            '''
            if N.amax(raw_immo_filt[i:i+10*Fs/1000])-N.amax(raw_immo_filt[i-10*Fs/1000+1:i])>=exceed_std*N.std(raw_immo):
                ind_DS.append(i)
                result=N.ravel(N.argmax(raw_immo_filt[i:i+20*Fs/1000]))
                result = i+result[0]
                ind_peak_DS.append(result)
                raw_DS.append(raw_immo_filt[result-20*Fs/1000:result+20*Fs/1000])
                ind_DS_range.append(range(result-20*Fs/1000,result+20*Fs/1000))
    if len(ind_peak_DS)==0:
        print 'no dentate spike detected by threshold of %s X SD' %scale_std
    else:
        os.chdir(directory)
        N.save('temp_ind_peak_DS.npy',ind_peak_DS)
        N.save('temp_raw_DS.npy',raw_DS)
        N.save('temp_ind_DS_range.npy',ind_DS_range)
        return ind_peak_DS,raw_DS,ind_DS_range
    
#manually find false positive DS artefacts
def DS_detect_find_del(low=10,high=1000,tscale=100,shank_No=1,scale_std=3,exceed_std=1,Fs=30000):
    '''
    low=10
    high=1000
    tscale=3000
    scale_std=3
    Fs=30000
    shank_No=2
    '''
    #import onkey as on
    posnew_immo,speed_immo,ind_immo,directory,raw_batch=DS_detection_manual(low=low,high=high,tscale=tscale,scale_std=scale_std,Fs=Fs,shank_No=shank_No,plotData=True)
    #posnew_immo,speed_immo,ind_immo,directory,raw_batch=on.DS_detection_manual_scroll(low=low,high=high,tscale=tscale,scale_std=scale_std,Fs=Fs,shank_No=shank_No,plotData=True)
    ind=input('channel index in raw list? ')
    print 'raw channel %s for DS detection' %raw_batch[ind]
    ind_peak_DS,raw_DS,ind_DS_range=on.dentate_spike_detection(ind,posnew_immo,speed_immo,ind_immo,raw_batch,directory,scale_std=scale_std,exceed_std=exceed_std,low=low,high=high,Fs=Fs)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    #plot 8 channels of raw immo and use onkey to select false positive out
    fig,axs=pl.subplots(figsize=[16,12], subplot_kw=dict(frameon=False))
    # create a memory map for the continuous files
    os.chdir(directory)
    raw = op.open_helper(raw_batch[0])
    klusta_file=N.memmap('data.dat',dtype='int16',mode='w+',shape=(int(ma.ceil(float(len(raw_batch))/2)),len(raw)))
    klusta_file[0]=raw
    raw_immo = N.take(klusta_file[0],ind_immo)
    raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high)
    N.save('%s_raw_immo.npy' %raw_batch[0],raw_immo)
    mean_immo=N.mean(raw_immo)
    std_immo=N.std(raw_immo)
    x=range(len(raw_immo))
    x=N.divide(x,float(Fs)/1000)
    plots=axs.plot(x,raw_immo)
    axs.hlines(mean_immo+scale_std*std_immo,0,max(x),colors='orange',linestyles='dashed')
    for n in range(2,len(raw_batch),2): 
        os.chdir(directory)
        raw = op.open_helper(raw_batch[n])
        # create a memory map for the continuous files
        klusta_file[n/2]=raw 
        raw_immo = N.take(klusta_file[n/2],ind_immo)
        raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high)
        N.save('%s_raw_immo.npy' %raw_batch[n],raw_immo)
        mean_immo=N.mean(raw_immo)
        std_immo=N.std(raw_immo)
        x=range(len(raw_immo))
        x=N.divide(x,float(Fs)/1000)
        plots=axs.plot(x,raw_immo-2000*n/2)
        axs.hlines(mean_immo+scale_std*std_immo-2000*n/2,0,max(x),colors='orange',linestyles='dashed')
    axs.vlines(N.divide(ind_peak_DS,float(Fs)/1000),-1000-2000*n/2,1000,'g',linestyles='dotted',alpha=0.5)
    multi = wdg.Cursor(axs, color='r', lw=1, horizOn=False, vertOn=True)
    
    axs.set_yticks(N.concatenate((N.arange(-1000-2000*n/2,1000,step=1000),[1000])))
    axs.set_yticklabels(N.concatenate((N.tile([-1000,0],int(ma.ceil(float(len(raw_batch))/2))),[1000])))
    axs.set_xlabel('time (ms)',fontsize=16)
    
    #plot a scrolling bar
    axcolor = 'lightgoldenrodyellow'
    axpos = pl.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
    #rax = pl.axes([0.05, 0.4, 0.03, 0.15])
    spos = wdg.Slider(axpos, 'Time (ms)', 0.1, max(x)-tscale, valstep=tscale*Fs/1000)

    def update(val):
        pos = spos.val
        axs.axis([pos,pos+tscale*Fs/1000,-1000-2000*n/2,1000])
        fig.canvas.draw_idle()
    spos.on_changed(update)
    
    
    starts=N.arange(0,len(x),tscale)
    curr_pos=0
    def key_event(e):
        global curr_pos
        if e.key == "right":
            if curr_pos + tscale*Fs/1000 <= max(x):
                curr_pos = curr_pos + tscale*Fs/1000
            else:
                curr_pos = max(x) - tscale*Fs/1000
        elif e.key == "left":
            if curr_pos - tscale*Fs/1000 >= 0:
                curr_pos = curr_pos - tscale*Fs/1000
            else:
                curr_pos=0

        axs.axis([curr_pos,curr_pos+tscale,-1000-2000*n/2,1000])
        fig.canvas.draw_idle()
        #print curr_pos 
    
    
    fig.canvas.mpl_connect('key_press_event', key_event)
    #klusta_file._mmap.close()
    #os.remove('data.dat')
    '''
    fig,axs=pl.subplots(len(raw_batch)/2,1,sharex=True,sharey=True,figsize=[16,12])
    fig.subplots_adjust(hspace=0)
    axs=axs.ravel()
    for n in range(0,len(raw_batch),2): 
        os.chdir(directory)
        raw = op.open_helper(raw_batch[n])
        raw_immo = N.take(raw,ind_immo)
        raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high)
        mean_immo=N.mean(raw_immo)
        std_immo=N.std(raw_immo)
        x=range(len(raw_immo))
        x=N.divide(x,float(Fs)/1000)
        axs[n/2].plot(raw_immo)
        #axs[n/2].plot(x,raw_immo)
        #axs[n/2].vlines(N.divide(ind_peak_DS,float(Fs)/1000),min(raw_immo),max(raw_immo),'g',linestyles='dotted',alpha=0.5)
        axs[n/2].vlines(ind_peak_DS,min(raw_immo),max(raw_immo),'g',linestyles='dotted',alpha=0.5)
        axs[n/2].hlines(mean_immo+scale_std*std_immo,0,len(raw_immo),colors='orange',linestyles='dashed')
        axcolor = 'lightgoldenrodyellow'
        axpos = pl.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
        spos = wdg.Slider(axpos, 'Pos', 0.1, max(x)-150000)

        def update(val):
            pos = spos.val
            axs[n/2].axis([pos,pos+tscale*Fs/1000,-1000,1000])
            fig.canvas.draw_idle()
    
        spos.on_changed(update)
        '''
    #axs[-1].set_xlabel('time (ms)',fontsize=16)
    
    def onkey(event):
        if event.key=='d':
            ev.append(int(event.xdata))
            axs.plot(event.xdata,event.ydata,'r.')
        if event.key=='m':
            ev_miss.append(int(event.xdata))
            axs.plot(event.xdata,event.ydata,'g.')
        #ev.append(int(event.xdata))
        #axs.plot(event.xdata,event.ydata,'r.') #shift+click with mouse
    
    ev,ev_miss=[],[]
    cid=fig.canvas.mpl_connect('key_press_event',onkey)
    
    file_to_save=tkFileDialog.asksaveasfilename(title="Choose target filename")
    N.savetxt('%s_all_false_positive_DS_ind.txt' %file_to_save,ev)
    N.savetxt('%s_all_missed_DS_ind.txt' %file_to_save,ev_miss)
    N.savetxt('%s_all_peak_DS_ind.txt' %file_to_save,ind_peak_DS)
    N.savetxt('%s_all_raw_DS_range.txt' %file_to_save,raw_DS)
    N.savetxt('%s_all_ind_DS_range.txt' %file_to_save,ind_DS_range)

    return ind_DS_range,raw_DS,ind_peak_DS,raw_batch,ind_immo,directory,fig,axs

def DS_detect_find_del_memory_efficient(low=10,high=1000,tscale=100,shank_No=1,scale_std=3,exceed_std=1,Fs=30000):
    '''
    low=10
    high=1000
    tscale=3000
    scale_std=3
    exceed_std=1
    Fs=30000
    shank_No=input('shank No.? ')
    exceed_std=input('exceed_std? ')
    '''
    #import onkey as on
    #posnew_immo,speed_immo,ind_immo,directory,raw_batch=DS_detection_manual(low=low,high=high,tscale=tscale,scale_std=scale_std,Fs=Fs,shank_No=shank_No,plotData=True)
    posnew_immo,speed_immo,ind_immo,directory,raw_batch=on.DS_detection_manual_scroll(low=low,high=high,tscale=tscale,scale_std=scale_std,Fs=Fs,shank_No=shank_No,plotData=True)
    ind=input('channel index in raw list? ')
    print 'raw channel %s for DS detection' %raw_batch[ind]
    ind_peak_DS,raw_DS,ind_DS_range=on.dentate_spike_detection(ind,posnew_immo,speed_immo,ind_immo,raw_batch,directory,scale_std=scale_std,exceed_std=exceed_std,low=low,high=high,Fs=Fs)
    
    
    os.chdir(directory)
    decimation_factor=10
    klusta_file=N.memmap('data.dat',dtype='int16',mode='w+',shape=(int(ma.ceil(float(len(raw_batch))/2)),int(ma.ceil(float(len(ind_immo))/decimation_factor))))
    mean_immo_all,std_immo_all=[],[]
    for n in range(0,len(raw_batch),2): 
        os.chdir(directory)
        raw = op.open_helper(raw_batch[n])
        raw_immo = N.take(raw,ind_immo)
        raw_immo_down=sig.decimate(raw_immo,decimation_factor)
        raw_immo_filt = data_tools.filterData(raw_immo_down,low=low,high=high)
        N.save('%s_raw_immo.npy' %raw_batch[n],raw_immo)
        # create a memory map for the continuous files
        klusta_file[n/2]=raw_immo_filt 
        mean_immo_all.append(N.mean(raw_immo_filt))
        std_immo_all.append(N.std(raw_immo_filt))
    Fs_ds=Fs/decimation_factor
    N.save('Shank_%s_raw_immo.npy'%shank_No,raw_immo)
    N.save('Shank_%s_mean_immo_all.npy'%shank_No,mean_immo_all)
    N.save('Shank_%s_std_immo_all.npy'%shank_No,std_immo_all)
    #klusta_file._mmap.close()
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    #plot 8 channels of raw immo and use onkey to select false positive out
    fig,axs=pl.subplots(figsize=[16,12], subplot_kw=dict(frameon=False))
    
    for n in range(len(klusta_file)):
        x=N.multiply(range(len(klusta_file[n])),decimation_factor) #keep the same timestamp after downsampling
        x=N.divide(x,float(Fs)/1000)    # ms
        plots=axs.plot(x,klusta_file[n]-2000*n)
        axs.hlines(mean_immo_all[n]+scale_std*std_immo_all[n]-2000*n,0,max(x),colors='orange',linestyles='dashed')
    #axs.vlines(N.divide(ind_peak_DS,float(Fs)/1000),-1000-2000*n,1000,'g',linestyles='dotted',alpha=0.5)
    axs.vlines(N.divide(ind_peak_DS,float(Fs)/1000),-1000-2000*n,1000,'g',linestyles='dotted')
    multi = wdg.Cursor(axs, color='r', lw=1, horizOn=False, vertOn=True)
    
    axs.set_yticks(N.arange(-1000-2000*n,2000,step=1000))
    axs.set_yticklabels(N.concatenate((N.tile([-1000,0],int(ma.ceil(float(len(raw_batch))/2))),[1000])))
    axs.set_xlabel('time (ms)',fontsize=16)    
    
    #plot a scrolling bar
    axcolor = 'lightgoldenrodyellow'
    axpos = pl.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
    #rax = pl.axes([0.05, 0.4, 0.03, 0.15])
    spos = wdg.Slider(axpos, 'Time (ms)', 0.1, max(x)-tscale, valstep=tscale*Fs/1000)

    def update(val):
        pos = spos.val
        #axs.axis([pos,pos+tscale*Fs/(1000*decimation_factor),-1000-2000*n/2,1000])
        axs.axis([pos,pos+tscale,-1000-2000*n,1000])
        fig.canvas.draw_idle()
    spos.on_changed(update)
    
    
    #check every 3s (time_scale) window once by key pressing right/left
    curr_pos=0
    def key_event(e):
        global curr_pos
        if e.key == "right":
            if curr_pos + tscale <= max(x):
                curr_pos = curr_pos + tscale
            else:
                curr_pos = max(x) - tscale
        elif e.key == "left":
            if curr_pos - tscale>= 0:
                curr_pos = curr_pos - tscale
            else:
                curr_pos=0
        #axs.axis([curr_pos,curr_pos+tscale*Fs/(1000*decimation_factor),-1000-2000*n/2,1000])
        axs.axis([curr_pos,curr_pos+tscale,-1000-2000*n,1000])
        fig.canvas.draw_idle()
        #print curr_pos 
    fig.canvas.mpl_connect('key_press_event', key_event)
    
    #create a zoom window to check zoomed LFPs
    figzoom, axzoom = pl.subplots(figsize=[4,3])  ##https://matplotlib.org/stable/gallery/event_handling/zoom_window.html
    axzoom.plot(x,klusta_file[ind/2])
    axzoom.set(title='Zoom window')
    axzoom.hlines(mean_immo_all[ind/2]+scale_std*std_immo_all[ind/2],0,max(x),colors='orange',linestyles='dashed')
    axzoom.vlines(N.divide(ind_peak_DS,float(Fs)/1000),-1000,1000,'g',linestyles='dotted',alpha=0.5)
    def onclick(event):
        if event.dblclick != 1:
            return
        axzoom.set_xlim([event.xdata - 50, event.xdata + 50])
        figzoom.canvas.draw()
            
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    #manually find false positive events with 'd' and missed events with 'm'
    def onkey(event):
        if event.key=='d':
            ev.append(int(event.xdata))
            axs.plot(event.xdata,event.ydata,'r.')
        if event.key=='m':
            ev_miss.append(int(event.xdata))
            axs.plot(event.xdata,event.ydata,'g.')
        #ev.append(int(event.xdata))
        #axs.plot(event.xdata,event.ydata,'r.') #shift+click with mouse
    
    ev,ev_miss=[],[]
    cid=fig.canvas.mpl_connect('key_press_event',onkey)
    
    # rescale to original sample rate
    ev1=N.multiply(ev,float(Fs)/1000) #recalculate ms to idx
    ev_miss1=N.multiply(ev_miss,float(Fs)/1000) #recalculate ms to idx
    
    file_to_save=tkFileDialog.asksaveasfilename(title="Choose target filename")
    N.savetxt('%s_s%s_all_false_positive_DS_ind.txt' %(file_to_save,shank_No),ev1*decimation_factor)
    N.savetxt('%s_s%s_all_missed_DS_ind.txt' %(file_to_save,shank_No),ev_miss1*decimation_factor)
    N.savetxt('%s_s%s_all_peak_DS_ind.txt' %(file_to_save,shank_No),ind_peak_DS)
    N.savetxt('%s_s%s_all_raw_DS_range.txt' %(file_to_save,shank_No),raw_DS)
    N.savetxt('%s_s%s_all_ind_DS_range.txt' %(file_to_save,shank_No),ind_DS_range)
    Number=input('figure_number? ')
    fig.savefig('%s_s%s_DS_raw_%s.png' %(file_to_save,shank_No,Number))
    fig.savefig('%s_s%s_DS_raw_%s.eps' %(file_to_save,shank_No,Number),transparent=True)

    return ind_DS_range,raw_DS,ind_peak_DS,raw_batch,ind_immo,directory,fig,axs



#delete all found DS artefacts
def DS_artefact_del(ev,ind_DS_range,raw_DS,ind_peak_DS,raw_batch,axs,shank_No,decimation_factor=10,Fs=30000,plot_grand=True):
    ind_del=[]
    ind_DS_range_2=ind_DS_range
    raw_DS_2=raw_DS
    ind_peak_DS_2=ind_peak_DS
    for j in range(len(ev)):
        res=N.ravel(N.argwhere(ind_DS_range_2==ev[j]))
        if len(res)>0:
            ind_del.append(res[0]) 
            ind_DS_range_2=N.delete(ind_DS_range_2,res[0],0)
            raw_DS_2=N.delete(raw_DS_2,res[0],0)
            ind_peak_DS_2=N.delete(ind_peak_DS_2,res[0])
    raw_DS_2_filt = data_tools.filterData(raw_DS_2,low=10,high=100)
    if plot_grand==True:    
        ind_yrange=int(ma.ceil(float(len(raw_batch))/2))
        axs.vlines(N.divide(ind_peak_DS_2,float(Fs)/1000),-1000-2000*ind_yrange,1000,'r',linestyles='solid',alpha=0.5)
    '''
    axs[0].vlines(ind_peak_DS_2,min(raw_immo),max(raw_immo),'r',linestyles='solid',alpha=0.5)
    '''
    file_to_save=tkFileDialog.asksaveasfilename(title="Choose target filename")
    N.savetxt('%s_s%s_all_false_positive_DS_ind.txt' %(file_to_save,shank_No),ev)
    N.savetxt('%s_s%s_all_peak_DS_ind.txt' %(file_to_save,shank_No),ind_peak_DS_2)
    N.savetxt('%s_s%s_all_raw_DS_range.txt' %(file_to_save,shank_No),raw_DS_2)
    N.savetxt('%s_s%s_all_ind_DS_range.txt' %(file_to_save,shank_No),ind_DS_range_2)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig3,axs3=pl.subplots((int(len(ind_peak_DS_2)/8)+(len(ind_peak_DS_2)%8>0)),8,sharex=True,figsize=[16,12], subplot_kw=dict(frameon=False))
    fig4,axs4=pl.subplots((int(len(ind_peak_DS_2)/8)+(len(ind_peak_DS_2)%8>0)),8,sharex=True,figsize=[16,12], subplot_kw=dict(frameon=False))
    fig5,axs5=pl.subplots()
    fig2,axs2=pl.subplots()
    fig3.subplots_adjust(hspace=0.4)
    fig4.subplots_adjust(hspace=0.4)
    axs3=axs3.ravel()
    axs4=axs4.ravel()
    for j in range(len(ind_peak_DS_2)):
        axs3[j].plot(raw_DS_2[j])
        axs4[j].plot(raw_DS_2_filt[j])
        axs2.plot(raw_DS_2[j])
        axs5.plot(raw_DS_2_filt[j])
    
    fig3.savefig('%s_s%s_DS_single.png' %(file_to_save,shank_No))
    fig3.savefig('%s_s%s_DS_single.eps' %(file_to_save,shank_No),transparent=True)
    fig2.savefig('%s_s%s_DS.png' %(file_to_save,shank_No))
    fig2.savefig('%s_s%s_DS.eps' %(file_to_save,shank_No),transparent=True)
    fig4.savefig('%s_s%s_DS_single_100Hz.png' %(file_to_save,shank_No))
    fig4.savefig('%s_s%s_DS_single_100Hz.eps' %(file_to_save,shank_No),transparent=True)
    fig5.savefig('%s_s%s_DS_100Hz.png' %(file_to_save,shank_No))
    fig5.savefig('%s_s%s_DS_100Hz.eps' %(file_to_save,shank_No),transparent=True)
    
    return ind_peak_DS_2,ind_DS_range_2,raw_DS_2

def DS_add_missed(ev_miss,ind_peak_DS_2,ind_DS_range_2,raw_DS_2,raw_immo,raw_batch,low,high,axs,shank_No,Fs=30000,plot_grand=True):
    
    ind_peak_DS_3=ind_peak_DS_2
    raw_DS_3=raw_DS_2
    ind_DS_range_3=ind_DS_range_2
    raw_immo_filt_raw = data_tools.filterData(raw_immo,low=low,high=high)
    for i in range(len(ev_miss)):  
        result = N.ravel(N.argmax(raw_immo_filt_raw[int(ev_miss[i]-20*Fs/1000):int(ev_miss[i]+20*Fs/1000)]))
        result = ev_miss[i]+result[0]-20*Fs/1000
        ind_peak_DS_3=N.concatenate((ind_peak_DS_3,[result]))
        raw_DS_3=N.vstack((raw_DS_3,raw_immo_filt_raw[int(result-20*Fs/1000):int(result+20*Fs/1000)]))
        ind_DS_range_3=N.vstack((ind_DS_range_3,range(int(result-20*Fs/1000),int(result+20*Fs/1000))))
    raw_DS_3_filt = data_tools.filterData(raw_DS_3,low=10,high=100)
    if plot_grand==True:    
        ind_yrange=int(ma.ceil(float(len(raw_batch))/2))
        axs.vlines(N.divide(ind_peak_DS_3,float(Fs)/1000),-1000-2000*ind_yrange,1000,'k',linestyles='solid',alpha=0.5)
    '''
    axs[0].vlines(ind_peak_DS_2,min(raw_immo),max(raw_immo),'r',linestyles='solid',alpha=0.5)
    '''
    file_to_save=tkFileDialog.asksaveasfilename(title="Choose target filename")
    N.savetxt('%s_s%s_all_peak_DS_ind_final.txt' %(file_to_save,shank_No),ind_peak_DS_3)
    N.savetxt('%s_s%s_all_raw_DS_range_final.txt' %(file_to_save,shank_No),raw_DS_3)
    N.savetxt('%s_s%s_all_ind_DS_range_final.txt' %(file_to_save,shank_No),ind_DS_range_3)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    fig3,axs3=pl.subplots((int(len(ind_peak_DS_3)/8)+(len(ind_peak_DS_3)%8>0)),8,sharex=True,figsize=[16,12], subplot_kw=dict(frameon=False))
    fig4,axs4=pl.subplots((int(len(ind_peak_DS_3)/8)+(len(ind_peak_DS_3)%8>0)),8,sharex=True,figsize=[16,12], subplot_kw=dict(frameon=False))
    fig2,axs2=pl.subplots()
    fig5,axs5=pl.subplots()
    fig3.subplots_adjust(hspace=0.4)
    fig4.subplots_adjust(hspace=0.4)
    axs3=axs3.ravel()
    axs4=axs4.ravel()
    for j in range(len(ind_peak_DS_3)):
        axs3[j].plot(raw_DS_3[j])
        axs4[j].plot(raw_DS_3_filt[j])
        axs2.plot(raw_DS_3[j])
        axs5.plot(raw_DS_3_filt[j])
    
    fig3.savefig('%s_s%s_DS_single.png' %(file_to_save,shank_No))
    fig3.savefig('%s_s%s_DS_single.eps' %(file_to_save,shank_No),transparent=True)
    fig2.savefig('%s_s%s_DS.png' %(file_to_save,shank_No))
    fig2.savefig('%s_s%s_DS.eps' %(file_to_save,shank_No),transparent=True)
    fig4.savefig('%s_s%s_DS_single_100Hz.png' %(file_to_save,shank_No))
    fig4.savefig('%s_s%s_DS_single_100Hz.eps' %(file_to_save,shank_No),transparent=True)
    fig5.savefig('%s_s%s_DS_100Hz.png' %(file_to_save,shank_No))
    fig5.savefig('%s_s%s_DS_100Hz.eps' %(file_to_save,shank_No),transparent=True)
    
    return ind_peak_DS_3,ind_DS_range_3,raw_DS_3
#def spike_DS_PSTH(spike_train_immo,raw_immo_filt,ind_peak_DS,Fs,directory,filename):
    

def spike_DS_coupling_single(spike_train_immo,raw_immo_filt,ind_peak_DS,Fs,directory,filename):
    
    DS_start,DS_stop,ind_DS_range_prec,ind_DS_range_prec_pre=[],[],[],[]
    for n in range(len(ind_peak_DS)):
        start= N.argmin(raw_immo_filt[int(ind_peak_DS[n])-20*Fs/1000:int(ind_peak_DS[n])])+(int(ind_peak_DS[n])-20*Fs/1000)
        stop= N.argmin(raw_immo_filt[int(ind_peak_DS[n]):int(ind_peak_DS[n])+20*Fs/1000])+(int(ind_peak_DS[n]))
        DS_start.append(start)
        DS_stop.append(stop)
        ind_DS_range_prec.append(range(start,stop))
        ind_DS_range_prec_pre.append(range(start-len(range(start,stop))-1,start-1))
        
    spike_train_DS,spike_train_pre,spike_numb_DS,spike_numb_pre,spike_train=[],[],[],[],[]
    for m in range(len(ind_DS_range_prec)):
        res=N.take(spike_train_immo,ind_DS_range_prec[m])
        spike_numb_DS.append(N.sum(res))
        spike_train_DS.append(res)
        res_pre=N.take(spike_train_immo,ind_DS_range_prec_pre[m])
        spike_numb_pre.append(N.sum(res))
        spike_train_pre.append(res_pre)
        result=N.hstack((res_pre,res))
        spike_train.append(result)
    
    statistic, pvalue = stats.ttest_rel(spike_numb_DS,spike_numb_pre)
    if pvalue >0.05 or N.isnan(pvalue)==True:
        print 'no siginificance', pvalue
        fig,axs=pl.subplots(figsize=[16,6])
    else:
        print 'dentate spike correlated, significant',pvalue
        if plotData==True:
            
            #getting setup to export text correctly
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42
            
            fig,axs=pl.subplots(figsize=[16,6])
            for j in range(len(ind_peak_DS)):
                x=range(len(spike_train[j]))
                x=N.array(x)*1000/float(Fs)
                axs.plot(50*j+x,10*j+raw_immo_filt[ind_DS_range_prec_pre[j][0]:ind_DS_range_prec[j][-1]])
                axs.plot(50*j+x,10*j+200*spike_train[j][:])
            #axs.set_title('%s_DS correlation '%filename,fontsize=18)
            axs.set_xlabel('time (ms)',fontsize=14)
            axs.set_ylabel('(mV)',fontsize=14)
    return pvalue, fig
            
def PSTH_raster(trigger,spike_train,raw_immo,fname2,low=10,high=1000,trange=0.1,Fs=30000,steps=0.005):
    '''
    trigger: stimulation
    trange: for PSTH time range
    steps: bin steps
    fname2: name of timestamp
    '''
    raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high,Fs=Fs)
    spike_train_PSTH,raw_PSTH,su_trigger=[],[],[]
    for n in range(len(trigger)):
        res=spike_train[int(trigger[n]-trange*Fs):int(trigger[n]+trange*Fs)]
        raw_trigger=raw_immo_filt[int(trigger[n]-trange*Fs):int(trigger[n]+trange*Fs)]
        su_single=N.ravel(N.argwhere(res==1))-trange*Fs
        su_trigger.append(su_single)
        sum_bins=N.add.reduceat(res,N.arange(0,len(res),int(steps*Fs)))   #sum every n values
        spike_train_PSTH.append(sum_bins)
        raw_PSTH.append(raw_trigger)
    PSTH=N.sum(spike_train_PSTH,axis=0)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    fig,(ax1,ax2,ax3)=pl.subplots(3,1,sharex=True,figsize=[8,6])
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
    x=N.arange(-trange*Fs,trange*Fs)
    for i in range(len(raw_PSTH)):
        ax1.plot(x,raw_PSTH[i])
        ax2.eventplot(su_trigger,linewidths=2)
    binstart=N.arange(-trange*Fs,trange*Fs,int(steps*Fs))
    ax3.bar(binstart,PSTH,width=int(steps*Fs)*0.8,align='edge')
    uname=fname2.split('_')[0]
    fn=fname2.split('_')[1]
    fn=fn.split('.')[0]
    ax1.set_title('%s_%s_PSTH'%(uname,fn),size=15)
    ax1.set_ylabel(u"\u03BCV",fontsize=15)
    ax3.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)
    ax3.yaxis.set_tick_params(labelsize=12)
    ax2.set_ylabel("trial No.",fontsize=15)
    ax3.set_ylabel("spike No.",fontsize=15)
    
    return PSTH, raw_PSTH, su_trigger, x, fig
               
#all spike timing-DS coupling on one shank        
def spike_DS_coupling_all_batch(low=10,high=1000,Fs=30000):
    
    filename=tkFileDialog.askopenfilename(title='unit list spike train immo batch')
    directory, fname = os.path.split(filename)
    spike_train_batch=op.open_helper(filename)  
    filename2=tkFileDialog.askopenfilename(title='raw data channel immobility')
    directory2, fname2 = os.path.split(filename2)
    raw_immo=op.open_helper(filename2)
    raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high,Fs=Fs)
    
    os.chdir(directory2)
    posnew_immo = N.load('estimate speed from position_corrected posnew_immo.npy')
    speed_immo = N.load('estimate speed from position_speed_immo.npy')
    ind_immo = N.load('estimate speed from position_ind_immo.npy')
    
    dir1=tkFileDialog.askopenfilename(title='detected DS peak ind')          
    ind_peak_DS=op.open_helper(dir1)
    
    grand_result_p=[]
    for n in range(len(spike_train_batch)):
       print "Recording set %s of " %n, len(spike_train_batch)-1
       os.chdir(directory)                
       spike_train_immo=op.open_helper(spike_train_batch[n])
       filename=spike_train_batch[n]
       print 'unit %s selected' %spike_train_batch[n]
       print "Channel %s selected" %fname2
       pvalue, fig=spike_DS_coupling_single(spike_train_immo,raw_immo_filt,ind_peak_DS,Fs,directory,filename)
       result={'spike_train':spike_train_batch[n],'pvalue':pvalue}
       grand_result_p.append(result)
    os.chdir(directory)
    N.save('DS coupling pvalue',grand_result_p)
    
    return grand_result_p

def spike_DS_coupling_batch_for_all(spike_train_batch,raw_immo,ind_peak_DS,directory,directory2,fname2,file_to_save,fig_to_save,low=10,high=1000,Fs=30000,trange=0.1,steps=0.005):
    
    raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high,Fs=Fs)
    
    os.chdir(directory2)

    grand_result_p, grand_fig=[],[]
    grand_result_PSTH, grand_fig2=[],[]
    for n in range(len(spike_train_batch)):
       print "Recording set %s of " %n, len(spike_train_batch)-1
       os.chdir(directory2)                
       spike_train_immo=op.open_helper(spike_train_batch[n])
       filename=spike_train_batch[n]
       print 'unit %s selected' %spike_train_batch[n]
       print "Channel %s selected" %fname2
       pvalue, fig=spike_DS_coupling_single(spike_train_immo,raw_immo_filt,ind_peak_DS,Fs,directory,filename)
       c=filename.rsplit('_',-1)[0]
       fig.savefig('%s_%s_DS_couple_raster.png'%(fig_to_save,c))
       fig.savefig('%s_%s_DS_couple_raster.eps' %(fig_to_save,c),transparent=True)
       result={'spike_train':spike_train_batch[n],'pvalue':pvalue}
       grand_result_p.append(result)
       grand_fig.append(fig)
       PSTH, raw_PSTH, su_trigger, x, fig2=PSTH_raster(ind_peak_DS,spike_train_immo,raw_immo,spike_train_batch[n],low=low,high=high,trange=trange,Fs=Fs,steps=steps)
       fig2.savefig('%s_%s_DS_couple_PSTH.png'%(fig_to_save,c))
       fig2.savefig('%s_%s_DS_couple_PSTH.eps' %(fig_to_save,c),transparent=True)
       result2={'spike_train':spike_train_batch[n],'PSTH':PSTH,'raw_PSTH':raw_PSTH,'su_trigger':su_trigger,'x':x}
       grand_result_PSTH.append(result2)
       grand_fig2.append(fig2)
    os.chdir(directory)
    N.save('%s_DS_coupling_pvalue'%file_to_save,grand_result_p)
    N.save('%s_DS_coupling_PSTH'%file_to_save,grand_result_PSTH)
    
    return grand_result_p, grand_fig, grand_result_PSTH, grand_fig2

def bootstrap_resampling_DS_spike_coupling(spike_train_immo,raw_immo_filt,ind_peak_DS,Fs=30000,iterations=1000,ymax=250,plothist=True):
    '''Input:
        iterations: times of permutation for bootstraping analysis
        ymax: limitation of vline during plotting of 5%/95% lines
        Output:
            if the number of spikes during laser pulse periods < 1% of random spike number distribution, negative regulated
            if >99%, positive regulated
            if not, no significance
    '''
    DS_start,DS_stop,ind_DS_range_prec,ind_DS_range_prec_pre=[],[],[],[]
    for n in range(len(ind_peak_DS)):
        start= N.argmin(raw_immo_filt[int(ind_peak_DS[n])-20*Fs/1000:int(ind_peak_DS[n])])+(int(ind_peak_DS[n])-20*Fs/1000)
        stop= N.argmin(raw_immo_filt[int(ind_peak_DS[n]):int(ind_peak_DS[n])+20*Fs/1000])+(int(ind_peak_DS[n]))
        DS_start.append(start)
        DS_stop.append(stop)
        ind_DS_range_prec.append(range(start,stop))
        
    spike_train_DS, spike_numb_DS=[],[]
    for m in range(len(ind_DS_range_prec)):
        res=N.take(spike_train_immo,ind_DS_range_prec[m])
        spike_numb_DS.append(N.sum(res))
        spike_train_DS.append(res)
        nspike_DS_all=N.sum(spike_numb_DS)
    
    su_immo=N.ravel(N.argwhere(spike_train_immo==1))
    
    isi,spike_numb_shuff = [],[]
    isi.append(su_immo[0])
    for n in range(len(su_immo)-1):
        isi.append(su_immo[n+1]-su_immo[n])
    # Perform permutation of the inter-spike intervals, iteration times. 
    #Extract laser epochs and keep record of the rate.
    nspike_shuff=[]
    for i in range(iterations):
        r=N.random.permutation(isi)
        su_shuff=[]
        for j in range(1,len(r)):
            su_shuff.append(N.sum(r[:j]))
        su_shuff.append(N.sum(r))
        
        spike_numb_shuff = []
        for k in range(len(ind_DS_range_prec)):
            #find indices in the range of pulse to get spike numbers
            spike_numb_shuff.append(len(N.intersect1d(su_shuff,ind_DS_range_prec[k][:])))
        nspike_shuff.append(N.sum(spike_numb_shuff))
        #print("--- %s seconds ---" % (time.time() - start_time))
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    sign_Pulse=[]
    f=pl.figure(figsize=[8,6])
    if nspike_DS_all<N.percentile(nspike_shuff,1):
        print 'light sensitive, negative regulated'
        sign_Pulse.append(-1)
        if plothist==True:
            #f=pl.figure(figsize=[8,6])
            pl.hist(nspike_shuff)
            pl.vlines(nspike_DS_all,ymin=0,ymax=ymax,colors='r',label='spike No.')
            pl.vlines(N.percentile(nspike_shuff,1),ymin=0,ymax=ymax,colors='k',label='1%',linestyle='dashed')
            pl.vlines(N.percentile(nspike_shuff,99),ymin=0,ymax=ymax,colors='k',label='99%',linestyle='dashed')
            pl.xlabel('spike counts of laser epochs',Fontsize=14)
            pl.ylabel('No. of iterations',Fontsize=14)
            pl.title('bootstrap resampling')

        elif nspike_DS_all>N.percentile(nspike_shuff,99):
            print 'light sensitive, positive regulated'
        sign_Pulse.append(1)
        if plothist==True:
            #f=pl.figure(figsize=[8,6])
            pl.hist(nspike_shuff)
            pl.vlines(nspike_DS_all,ymin=0,ymax=ymax,colors='r',label='spike No.')
            pl.vlines(N.percentile(nspike_shuff,1),ymin=0,ymax=ymax,colors='k',label='1%',linestyle='dashed')
            pl.vlines(N.percentile(nspike_shuff,99),ymin=0,ymax=ymax,colors='k',label='99%',linestyle='dashed')
            pl.xlabel('spike counts of laser epochs',Fontsize=14)
            pl.ylabel('No. of iterations',Fontsize=14)
            pl.title('bootstrap resampling')
    else:
        print 'no significance'
        sign_Pulse.append(0)
    return sign_Pulse, nspike_DS_all, nspike_shuff, f

#all spike timing-DS coupling bootstrap on one shank        
def bootstrap_resampling_DS_spike_coupling_batch(low=10,high=1000,Fs=30000,iterations=1000,ymax=250,plothist=True):
    
    filename=tkFileDialog.askopenfilename(title='unit list spike train immo batch')
    directory, fname = os.path.split(filename)
    spike_train_batch=op.open_helper(filename)  
    filename2=tkFileDialog.askopenfilename(title='raw data channel immobility')
    directory2, fname2 = os.path.split(filename2)
    raw_immo=op.open_helper(filename2)
    raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high,Fs=Fs)
    
    os.chdir(directory2)
    
    dir1=tkFileDialog.askopenfilename(title='detected DS peak ind')          
    ind_peak_DS=op.open_helper(dir1)
    
    grand_result_sign_Pulse=[]
    for n in range(len(spike_train_batch)):
       print "Recording set %s of " %n, len(spike_train_batch)-1
       os.chdir(directory)                
       spike_train_immo=op.open_helper(spike_train_batch[n])
       filename=spike_train_batch[n]
       print 'unit %s selected' %spike_train_batch[n]
       print "Channel %s selected" %fname2
       sign_Pulse, nspike_DS_all, nspike_shuff, f=bootstrap_resampling_DS_spike_coupling(spike_train_immo,raw_immo_filt,ind_peak_DS,Fs=Fs,iterations=iterations,ymax=ymax,plothist=plothist)
       result={'spike_train':spike_train_batch[n],'sign_Pulse':sign_Pulse}
       grand_result_sign_Pulse.append(result)
    os.chdir(directory)
    N.save('DS_sign_value_for_resampling',grand_result_sign_Pulse)
    
    return grand_result_sign_Pulse

def bootstrap_resampling_DS_spike_coupling_batch_for_all(spike_train_batch,raw_immo,ind_peak_DS,directory,directory2,fname3,file_to_save,fig_to_save,low=10,high=1000,Fs=30000,iterations=1000,ymax=250,plothist=True):
    
    raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high,Fs=Fs)
    
        
    grand_result_sign_Pulse,grand_f=[],[]
    for n in range(len(spike_train_batch)):
       print "Recording set %s of " %n, len(spike_train_batch)-1
       os.chdir(directory2)                
       spike_train_immo=op.open_helper(spike_train_batch[n])
       filename=spike_train_batch[n]
       print 'unit %s selected' %spike_train_batch[n]
       print "Channel %s selected" %fname3
       sign_Pulse, nspike_DS_all, nspike_shuff, f=bootstrap_resampling_DS_spike_coupling(spike_train_immo,raw_immo_filt,ind_peak_DS,Fs=Fs,iterations=iterations,ymax=ymax,plothist=plothist)
       c=filename.rsplit('_',-1)[0]
       f.savefig('%s_%s_DS_hist_resampling.png'%(fig_to_save,c))
       f.savefig('%s_%s_DS_hist_resampling.eps' %(fig_to_save,c),transparent=True)
       pl.close('all')
       result={'spike_train':spike_train_batch[n],'sign_Pulse':sign_Pulse}
       grand_result_sign_Pulse.append(result)
       grand_f.append(f)
    
    N.save('%s_DS_sign_value_for_resampling'%file_to_save,grand_result_sign_Pulse)
    
    return grand_result_sign_Pulse,grand_f

def unit_DS_corr_all_animals(condition,low=10,high=1000,Fs=30000,iterations=1000,ymax=250,trange=0.1,steps=0.005,plothist=True):
    
    '''
    need a batch list for path 'position data directory'
    need a batch list for path 'all unti timestamp file'
    adapt the path for fig_to_save and file_to_save
    a path for final grand_result dictionary
    '''
    
    dir1=tkFileDialog.askopenfilename(title='path list for raw channel directory')
    directory, fname = os.path.split(dir1)
    path_raw_batch=op.open_helper(dir1)
    dir2=tkFileDialog.askopenfilename(title="path list for all unit list spike train immo")
    directory2, fname2 = os.path.split(dir2)
    path_su_batch=op.open_helper(dir2) 
    
    Path_raw_exist,Path_su_exist=[],[]
    for n in range(len(Path_raw_exist)):
        print 'raw path %s' %path_raw_batch[n]
        directory3, fname3 = os.path.split(path_raw_batch[n])
        os.chdir(directory3)
        isFile=os.path.isfile('all_peak_DS_ind.batch' )
        Path_raw_exist.append(isFile)
        print isFile
        print 'unit path %s' %path_su_batch[n]
        isFileunit=os.path.isfile(path_su_batch[n])
        Path_su_exist.append(isFileunit)
        print isFileunit
    
    if N.array(Path_raw_exist).all()==True and N.array(Path_su_exist).all()==True:
        sumtotal=0
        for n in range(len(Path_raw_exist)):
            spike_train_batch=op.open_helper(path_su_batch[n])
            sumtotal=sumtotal+len(spike_train_batch)
        
        grand_result_all = []
        subsum=0
        for n in range(len(path_raw_batch)):
            print 'Raw path %s' %path_raw_batch[n]
            print 'unit path %s' %path_su_batch[n]
            raw_immo=op.open_helper(path_raw_batch[n])
            directory3, fname3 = os.path.split(path_raw_batch[n])
            os.chdir(directory3)
            ind_peak_DS_batch=op.open_helper('all_peak_DS_ind.batch')
            ind_peak_DS=op.open_helper(ind_peak_DS_batch[n])
            spike_train_batch=op.open_helper(path_su_batch[n])
            print 'Recording from %s' %subsum
            subsum=subsum+len(spike_train_batch)
            print 'to %s of %s' %(subsum,sumtotal)
            directory2, fname2 = os.path.split(path_su_batch[n])
            c=directory2.rsplit('_',-1)[-1]
            c=c.rsplit('_',-1)[-1]
            d=directory2.rsplit('/',4)[-3]
            fig_to_save='%s/%s_%s_%s' %(directory2,d,c,condition)
            os.chdir(directory2)
            if os.path.isdir('Data saved')==False:
                os.mkdir('Data saved')
            file_to_save='%s/%s/%s_%s_%s' %(directory2,'Data saved',d,c,condition)
            grand_result_sign_Pulse, grand_f=bootstrap_resampling_DS_spike_coupling_batch_for_all(spike_train_batch,raw_immo,ind_peak_DS,directory3,directory2,fname3,file_to_save,fig_to_save,low=low,high=high,Fs=Fs,iterations=iterations,ymax=ymax,plothist=plothist)
            grand_result_p, grand_fig, grand_result_PSTH, grand_fig2=spike_DS_coupling_batch_for_all(spike_train_batch,raw_immo,ind_peak_DS,directory3,directory2,fname3,file_to_save,fig_to_save,low=low,high=high,Fs=Fs,trange=trange,steps=steps)
            pl.close('all')
    return grand_f, grand_fig, grand_fig2