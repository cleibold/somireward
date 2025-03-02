#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:45:18 2020

@author: Mei & Jonas
"""

import numpy as N
import matplotlib.pyplot as pl
import tkFileDialog
import open_helper as op
import os
from scipy import stats
from scipy import ndimage
import barplot_annotate_brackets as ba
import ANOVARM as AN
import matplotlib
import pandas as pd
import seaborn as sns
import collections
import neuronpy.util.spiketrain as sp


def extract_laser_pulses (laser250ms=True,thres=2,Fs=30000,plotData=True):
    
    ''' Related to Fig1d, S2
        Inputs:
        files: ADC recording of laser pulse
        raw: raw .continuous data
        su: timestamp of sorted unit
        Keyword arguments:
        laser250ms: if true, recordings with 250 ms laser pulse; if false, with 50 ms pulse.
        thres: Threshold to detect laser pulses
        Fs: sampling frequency (in Hz)
        Outputs:
        laser pulse interval matrix for all detections
        the same length of interval before pulse start for all detections
    '''

    # Directories of the data and target.
    if laser250ms==True:
        target_dir="/data/example_raster_plot_optoID/191212_#235_lin/#235_lin_2019-12-12_15-36-11_laser1"
    else:
        target_dir="/data/example_raster_plot_optoID/191212_#235_lin/#235_lin_2019-12-12_15-27-40_laser"
    
    os.chdir(target_dir)
    
    files=op.open_helper('100_ADC5.continuous')
    raw=op.open_helper('100_CH3.continuous')
    os.chdir('..')
    if laser250ms==True:
        su=op.open_helper('17.0_laser1.txt')
    else:
        su=op.open_helper('17.0_laser.txt')

    print 'total number of spikes:',len(su)
    Pulse_start, Pulse_end = [],[]
    for n in range(len(files)-1):
        #detect upward threshold crossings
        if files[n] < thres and files[n+1] >= thres:
            Pulse_start.append(n+1)
    for i in range(len(files)-1):
        #detect downward threshold crossings
        if files[i] >= thres and files[i+1] < thres:
            Pulse_end.append(i+1)
            
    if Pulse_start[0]>=Pulse_end[0]:
        Pulse_end=Pulse_end[1:]
    elif Pulse_start[-1]>=Pulse_end[-1]:
        Pulse_start=Pulse_start[:-2]
        
    most_com_len_interv = N.bincount(N.subtract(Pulse_end,Pulse_start)).argmax()
    
    raw_Pulse, pre = [], []
    for l in range(len(Pulse_start)):
        #select raw data in laser pulse range plus the same time interval before laser start
        raw_Pulse = N.append(raw_Pulse,raw[Pulse_start[l]:(Pulse_start[l]+most_com_len_interv)])
        pre = N.append(pre,raw[(Pulse_start[l]-most_com_len_interv-1):(Pulse_start[l]-1)])
        
    raw_Pulse = N.array(raw_Pulse).reshape((len(Pulse_start),most_com_len_interv))
    pre = N.array(pre).reshape((len(Pulse_start),most_com_len_interv))
    
    spike_ind_all, spike_numb_Pulse, spike_numb_pre = [],[],[]
    for k in range(len(Pulse_start)):
        #find indices in the range of pulse to get spike numbers
        result=N.ravel(N.ravel(N.argwhere((N.logical_and(su>=(Pulse_start[k]-most_com_len_interv-1), su<=(Pulse_start[k]+2*most_com_len_interv))))))
        spike_ind_all.append(N.array(N.take(su,result)-Pulse_start[k]))
        #spike_ind_all is for pre, Pulse and post_Pulse
        spike_numb_Pulse.append(len(N.argwhere((N.logical_and(su>=Pulse_start[k], su<=(Pulse_start[k]+most_com_len_interv))))))
        spike_numb_pre.append(len(N.argwhere(N.logical_and(su>=(Pulse_start[k]-most_com_len_interv-1), su<=(Pulse_start[k]-1)))))
        
    statistic, pvalue = stats.ttest_rel(spike_numb_Pulse,spike_numb_pre)
    if pvalue >0.05 or N.isnan(pvalue)==True:
        print 'no siginificance'
        #getting setup to export text correctly
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        if plotData == True:
            #x = N.arange((-most_com_len_interv-1),most_com_len_interv,1)/30.
            f,(ax1,ax2) = pl.subplots(2,1,sharex=True,figsize=[8,4])
            spike_ind_plot = []
            for m in range(len(spike_ind_all)):
                if len(spike_ind_all[m]) > 0:
                    spike_ind_plot = spike_ind_all[m:]
                    #spike_ind_plot = spike_ind_all[m:]
                    break
                
            x=range(-most_com_len_interv-1,2*most_com_len_interv)
            ax1.eventplot(spike_ind_plot,linewidths=2)
            #get the number of units
            uname='u17'
            if laser250ms==True:
                fn='laser250mspulse'
            else:
                fn='laser50mspulse'
            ax1.set_title('%s_%s_ChR2 activation'%(uname,fn),size=15)
            ax1.set_ylabel("trial No.",fontsize=15)
            ax2.plot(x,files[(Pulse_start[k]-most_com_len_interv-1):(Pulse_start[k]+2*most_com_len_interv)])
            ax2.vlines(5*Fs/1000,min(files[(Pulse_start[k]-most_com_len_interv-1):(Pulse_start[k]+2*most_com_len_interv)])-0.2,max(files[(Pulse_start[k]-most_com_len_interv-1):(Pulse_start[k]+2*most_com_len_interv)])+0.2,colors='r',linestyle='dashed')
            ax2.set_ylabel("laser pulse (V)",fontsize=15)
    else:
        print 'light sensitive, significant',pvalue
        #getting setup to export text correctly
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        if plotData == True:
            #x = N.arange((-most_com_len_interv-1),most_com_len_interv,1)/30.
            f,(ax1,ax2) = pl.subplots(2,1,sharex=True,figsize=[8,4])
            spike_ind_plot = []
            for m in range(len(spike_ind_all)):
                if len(spike_ind_all[m]) > 0:
                    spike_ind_plot = spike_ind_all[m:]
                    #spike_ind_plot = spike_ind_all[m:]
                    break
                
            x=range(-most_com_len_interv-1,2*most_com_len_interv)
            #get the number of units
            uname='u17'
            if laser250ms==True:
                fn='laser250mspulse'
            else:
                fn='laser50mspulse'
            ax1.eventplot(spike_ind_plot,linewidths=2)
            ax1.set_title('%s_%s_ChR2 activation'%(uname,fn),size=15)
            ax1.set_ylabel("trial No.",fontsize=15)
            ax2.plot(x,files[(Pulse_start[k]-most_com_len_interv-1):(Pulse_start[k]+2*most_com_len_interv)])
            ax2.vlines(5*Fs/1000,min(files[(Pulse_start[k]-most_com_len_interv-1):(Pulse_start[k]+2*most_com_len_interv)])-0.2,max(files[(Pulse_start[k]-most_com_len_interv-1):(Pulse_start[k]+2*most_com_len_interv)])+0.2,colors='r',linestyle='dashed')
            ax2.set_ylabel("laser pulse (V)",fontsize=15)
            
    return Pulse_start, Pulse_end, raw_Pulse, pre, spike_ind_all, spike_numb_Pulse, spike_numb_pre, pvalue
    
def scatter_plot_3D(condition,density=True,hist_inh=True,symlog=False,inh=True,SOM=True,local=True,proj=True,plotSOM=True,ext=False,IN=False,SOMcolor='orange',inhcolor='tab:blue',edgecolors='#8C8C8C',SOMedge='#B22222'):
    
    ''' Related to Fig1e, S3
        Inputs:
        filename: path to spike kinetic data folder
        Keyword arguments:
        condition: for title name
        Outputs:
        spike kinetic scatter plot with histogramm of individual unit types
    '''
    target_dir='/data/spike_kinetic_scatter_plot'                
    os.chdir(target_dir)
    
    if SOM==True:
        dur_SOM=N.loadtxt('dur_all_SOM.txt')
        buri_SOM=N.loadtxt('burst_ind_all_SOM.txt')
    
    if inh==True:
        dur_SOM_inh=N.loadtxt('dur_all_SOM_inhibited.txt')
        buri_SOM_inh=N.loadtxt('burst_ind_all_SOM_inhibited.txt')
    
    if ext==True:
        dur_ext=N.loadtxt('dur_ext.txt')
        buri_ext=N.loadtxt('burst_ind_ext.txt')
    
    if IN==True:
        dur_IN=N.loadtxt('dur_inh.txt')
        buri_IN=N.loadtxt('burst_ind_inh.txt')
        
    buri_all=N.loadtxt('burst_ind_all.txt')
    dur_all=N.loadtxt('dur_all.txt')
    
    if proj==True:
        dur_SOM_proj=N.loadtxt('dur_proj.txt')
        buri_SOM_proj=N.loadtxt('burst_proj.txt')
    
    if local==True:
        dur_SOM_local=N.loadtxt('dur_local.txt')
        buri_SOM_local=N.loadtxt('burst_local.txt')
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    #scatter plot for buri index
    fig = pl.figure(figsize=[12,9])
    ax = fig.add_subplot(111) 
    ax.scatter(dur_all,buri_all,s=20,c='#A9A9A9',marker='o',label='total:N=%s'%len(dur_all),edgecolors=edgecolors)  #all  units
    if symlog==True:
        ax.set_yscale('symlog')
    else:
        ax.set_yscale('log')
    
    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(ax)
    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histx.set_ylabel('spike No.',fontsize=14)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    ax_histy.set_xlabel('spike No.',fontsize=14)
    if symlog==True:
        ax_histy.set_yscale('symlog')

    # now determine nice limits by hand:
    binwidth_tp = 0.02
    xmax = N.max(N.abs(dur_all))
    ymax = N.max(N.abs(buri_all))
    xlim = (int(xmax/binwidth_tp) + 1)*binwidth_tp
    
    x_bins = N.arange(0, xlim + binwidth_tp, binwidth_tp)
    buri_y_bins = N.logspace(-4,3)
    #ax.scatter(dur_SOM_inh,buri_SOM_inh,s=20,c='#A9A9A9',marker='o')  #all SOM inhibited units
    if inh==True:
        ax.scatter(dur_SOM_inh,buri_SOM_inh,s=30,c=inhcolor,marker='o',alpha=0.5,label='SOM inh:N=%s'%len(dur_SOM_inh))  #all SOM inhibited units
        if hist_inh==True:
            ax_histx.hist(dur_SOM_inh, bins=x_bins,color=inhcolor,alpha=0.8)
            if symlog==True:
                ax_histy.set_yscale('symlog')
            ax_histy.hist(buri_SOM_inh, bins=buri_y_bins, color=inhcolor,alpha=0.8,orientation='horizontal')
    
        #ax.scatter(dur_FS,buri_FS,s=80,c='g',linewidth=5,marker='o',alpha=0.5)  #all FS
        #ax_histx.hist(dur_FS, bins=x_bins,color='g',alpha=0.8)
        #ax_histy.hist(buri_FS, bins=buri_y_bins, color='g',orientation='horizontal',alpha=0.8)
    
    if SOM==True and local==True and proj==True:
        if ext==True:
            ax.scatter(dur_ext,buri_ext,s=40,c='Tab:blue',marker='o',label='ext:N=%s'%len(dur_ext),edgecolors='b')  #all excitatory
            ax_histx.hist(dur_ext, bins=x_bins,color='b',alpha=0.5)
            ax_histy.hist(buri_ext, bins=buri_y_bins, color='b',orientation='horizontal',alpha=0.5)
        if IN==True:
            ax.scatter(dur_IN,buri_IN,s=40,c='m',marker='o',label='IN:N=%s'%len(dur_IN),edgecolors=edgecolors)  #all DG SOMIs
            ax_histx.hist(dur_IN, bins=x_bins,color='m',alpha=0.5)
            ax_histy.hist(buri_IN, bins=buri_y_bins, color='m',orientation='horizontal',alpha=0.5)
        ax.scatter(dur_SOM_proj,buri_SOM_proj,s=40,c='orange',linewidth=5,marker='o',label='SOM proj:N=%s'%len(dur_SOM_proj),edgecolor='#FF8C00')   #all projecting SOMIs
        
        ax.scatter(dur_SOM_local,buri_SOM_local,s=40,c='g',linewidth=5,marker='o',label='SOM local:N=%s'%len(dur_SOM_local),edgecolor='#006400')   #all local SOMIs
        if plotSOM==True:
            ax.scatter(dur_SOM,buri_SOM,s=40,c='r',marker='s',label='SOM:N=%s'%len(dur_SOM))  #all DG SOMIs
            ax_histx.hist(dur_SOM, bins=x_bins,color='r',alpha=0.5)
            ax_histy.hist(buri_SOM, bins=buri_y_bins, color='r',orientation='horizontal',alpha=0.5)
        ax_histx.hist(dur_SOM_proj, bins=x_bins,color='orange',alpha=0.8)
        ax_histy.hist(buri_SOM_proj, bins=buri_y_bins, color='orange',orientation='horizontal',alpha=0.8)
        
        ax_histx.hist(dur_SOM_local, bins=x_bins,color='g',alpha=0.8)
        ax_histy.hist(buri_SOM_local, bins=buri_y_bins, color='g',orientation='horizontal',alpha=0.8)
        
    else:
        if ext==True:
            ax.scatter(dur_ext,buri_ext,s=30,c='Tab:blue',marker='o',label='ext:N=%s'%len(dur_ext),edgecolors='b')  #all excitatory
            ax_histx.hist(dur_ext, bins=x_bins,color='b',alpha=0.5)
            ax_histy.hist(buri_ext, bins=buri_y_bins, color='b',orientation='horizontal',alpha=0.5)
        if IN==True:
            ax.scatter(dur_IN,buri_IN,s=30,c='m',marker='o',label='IN:N=%s'%len(dur_IN),edgecolors=edgecolors)  #all DG SOMIs
            ax_histx.hist(dur_IN, bins=x_bins,color='m',alpha=0.5)
            ax_histy.hist(buri_IN, bins=buri_y_bins, color='m',orientation='horizontal',alpha=0.5)
        ax.scatter(dur_SOM,buri_SOM,s=40,c=SOMcolor,marker='o',label='SOM:N=%s'%len(dur_SOM),edgecolors=SOMedge)  #all DG SOMIs
        ax_histx.hist(dur_SOM, bins=x_bins,color=SOMcolor,alpha=0.5)
        ax_histy.hist(buri_SOM, bins=buri_y_bins, color=SOMcolor,orientation='horizontal',alpha=0.5)
        
    ax.set_xlabel('trough to peak duration (ms)',fontsize=14)
    ax.set_ylabel('bursting index a/b',fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax_histx.set_title('scatter distribution of SOMIs vs. non SOMIs_bursting index',fontsize=18) 
    if symlog==True:
        ax.set_ylim([-0.1,500])
    else:
        ax.set_ylim([0.001,500])
    ax.set_xlim([0.1,1.1])
    ax.legend()
   # fig.savefig('%s_scatter plot for buri index_SOMI2.png'%condition )
   # fig.savefig('%s_scatter plot for buri index_SOMI2.eps'%condition ,transparent=True) 
    
    #cumulative hist for dur, asym, buri
    binwidth = 0.02
    binwidth_asy=0.05
    SOM_xmax = N.max(N.abs(dur_SOM))
    SOM_ymax = N.max(buri_SOM)
    SOM_ymin = N.min(buri_SOM)
    SOM_xlim = (int(SOM_xmax/binwidth) + 1)*binwidth
    SOM_ylim = (int(SOM_ymax/binwidth_asy) + 1)*binwidth_asy
    SOM_ylim_min = (int(SOM_ymin/binwidth_asy) - 1)*binwidth_asy
    
    SOM_x_bins = N.arange(0, SOM_xlim + binwidth, binwidth)
    SOM_y_bins = N.arange(SOM_ylim_min, SOM_ylim + binwidth_asy, binwidth_asy)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    fig = pl.figure(figsize=[12,9])
    ax = fig.add_subplot(221)
    if SOM==True and local==True and proj==True:
        ax.hist(dur_SOM,bins=SOM_x_bins,color='r',cumulative=True,alpha=0.5,density=density,label='DG_SOMIs')
        ax.hist(dur_SOM_proj,bins=SOM_x_bins,color='orange',cumulative=True,histtype='step',density=density,label='proj_SOMIs')
        ax.hist(dur_SOM_local,bins=SOM_x_bins,color='g',cumulative=True,histtype='step',density=density,label='local_SOMIs')
    else:
        ax.hist(dur_SOM,bins=SOM_x_bins,color=SOMcolor,cumulative=True,alpha=0.5,density=density,label='DG_SOMIs')
    ax.set_xlabel('trough to peak duration (ms)',fontsize=14)
    ax.set_ylabel('fraction of units',fontsize=14)
    ax.set_title('distribution of TP duration_SOMI',fontsize=18) 
    
    ax = fig.add_subplot(223)
    buri_SOM_y_bins = N.logspace(-4,2)
    #ax.hist(buri_SOM,bins=buri_y_bins,color='r',cumulative=True,histtype='step',density=True,label='DG_SOMIs')
    if SOM==True and local==True and proj==True:
        ax.hist(buri_SOM,bins=buri_SOM_y_bins,color='r',cumulative=True,alpha=0.5,density=density,label='DG_SOMIs')
        ax.hist(buri_SOM_proj,bins=buri_SOM_y_bins,color='orange',cumulative=True,histtype='step',density=density,label='proj_SOMIs')
        ax.hist(buri_SOM_local,bins=buri_SOM_y_bins,color='g',cumulative=True,histtype='step',density=density,label='local_SOMIs')
    else:
        ax.hist(buri_SOM,bins=SOM_y_bins,color=SOMcolor,cumulative=True,alpha=0.5,density=density,label='DG_SOMIs')
    ax.set_xscale('log')
    ax.set_xlabel('bursting index a/b',fontsize=14)
    ax.set_ylabel('fraction of units',fontsize=14)
    #ax.set_title('distribution of bur index_SOMI',fontsize=18) 
    
    #fig.savefig('%s_cumulative hist_TP_asym_buri.png'%condition )
    #fig.savefig('%s_cumulative hist_TP_asym_buri.eps'%condition ,transparent=True) 
    
    fig = pl.figure(figsize=[12,9])
    ax = fig.add_subplot(221)
    #ax.hist(dur_SOM,bins=SOM_x_bins,color='r',cumulative=True,histtype='step',density=True,label='DG_SOMIs')
    if SOM==True and local==True and proj==True and inh==True:
        ax.hist(dur_SOM,bins=x_bins,color='r',cumulative=True,alpha=0.5,density=density,label='DG_SOMIs')
        ax.hist(dur_SOM_proj,bins=x_bins,color='orange',cumulative=True,histtype='step',density=density,label='proj_SOMIs')
        ax.hist(dur_SOM_local,bins=x_bins,color='g',cumulative=True,histtype='step',density=density,label='local_SOMIs')
        ax.hist(dur_SOM_inh,bins=x_bins,color=inhcolor,cumulative=True,histtype='step',density=density,label='local_SOMIs')
    else:
        ax.hist(dur_SOM,bins=x_bins,color=SOMcolor,cumulative=True,alpha=0.5,density=density,label='DG_SOMIs')
    ax.set_xlabel('trough to peak duration (ms)',fontsize=14)
    ax.set_ylabel('fraction of units',fontsize=14)
    ax.set_title('distribution of TP duration_SOMI',fontsize=18) 
    
    ax = fig.add_subplot(223)
    #ax.hist(buri_SOM,bins=buri_y_bins,color='r',cumulative=True,histtype='step',density=True,label='DG_SOMIs')
    if SOM==True and local==True and proj==True and inh==True:
        ax.hist(buri_SOM,bins=buri_y_bins,color='r',cumulative=True,alpha=0.5,density=density,label='DG_SOMIs')
        ax.hist(buri_SOM_proj,bins=buri_y_bins,color='orange',cumulative=True,histtype='step',density=density,label='proj_SOMIs')
        ax.hist(buri_SOM_local,bins=buri_y_bins,color='g',cumulative=True,histtype='step',density=density,label='local_SOMIs')
        ax.hist(buri_SOM_inh,bins=buri_y_bins,color=inhcolor,cumulative=True,histtype='step',density=density,label='local_SOMIs')
    else:
        ax.hist(buri_SOM,bins=buri_y_bins,color=SOMcolor,cumulative=True,alpha=0.5,density=density,label='DG_SOMIs')
    ax.set_xscale('log')
    ax.set_xlabel('bursting index a/b',fontsize=14)
    ax.set_ylabel('fraction of units',fontsize=14)
    #ax.set_title('distribution of bur index_SOMI',fontsize=18) 
    
    #fig.savefig('%s_cumulative hist_TP_asym_buri_all.png'%condition )
    #fig.savefig('%s_cumulative hist_TP_asym_buri_all.eps'%condition,transparent=True) 


def Summary_bar_plot_with_sem_2variables_check_normalitydistribution(ylabel,xticks,title,pairedT=True,logy=False,jitterplot=True,jitter=0.05,colorsingle='#A9A9A9',markersize=9,absolute=False,minusvar2=False):
    
    ''' Related to Fig1h, Fig2d,2f, Fig3f,5e, S3b, S8b, S10c
        Inputs:
        var1, var2: variables to compare, paired or unpaired.
        Outputs:
        barplot of 2 variables with statistical analysis
    '''
    
    filename=str(tkFileDialog.askopenfilename(title='var1')) 
    directory, fname = os.path.split(filename)
    var1=op.open_helper(filename)
    var1=N.ravel(var1)
    filename=str(tkFileDialog.askopenfilename(title='var2'))    
    var2=op.open_helper(filename)
    var2=N.ravel(var2)
    if absolute==True:
        var1=abs(var1)
        var2=abs(var2)
    elif minusvar2==True:
        var2=-var2
    
    mean=[N.mean(var1),N.mean(var2)]
    sem=[stats.sem(var1),stats.sem(var2)]
    statistic,pvalue1=stats.shapiro(var1)
    statistic,pvalue2=stats.shapiro(var2)
    if pvalue1>0.05 and pvalue2>0.05:
        print 'datasets normally distributed'
        if pairedT==True:
            res=N.hstack((var1.reshape((len(var1),1)),var2.reshape((len(var2),1))))
            statistic, pvalue = stats.ttest_rel(var1,var2)
        else:
            statistic, pvalue = stats.ttest_ind(var1,var2)
            
    else:
        print 'datasets not normally distributed'
        if pairedT==True:
            res=N.hstack((var1.reshape((len(var1),1)),var2.reshape((len(var2),1))))
            statistic, pvalue = stats.wilcoxon(var1,var2)
        else:
            #statistic, pvalue = stats.mannwhitneyu(var1,var2)   #The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distribution underlying sample x is the same as the distribution underlying sample y
            statistic, pvalue = stats.ranksums(var1,var2)
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = pl.figure(figsize=[8,6])
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax = fig.add_subplot(121)
    if logy==True:
        ax.set_yscale('log')
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=1.5, color='k')
    if pairedT==True:
        if jitterplot==True:
            df = pd.DataFrame(collections.OrderedDict({xticks[0]: var1, xticks[1]: var2,}))
            jitter = jitter
            df_x_jitter = pd.DataFrame(N.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
            df_x_jitter += N.arange(len(df.columns))
            for col in df:
                ax.plot(df_x_jitter[col], df[col], 'o', color=colorsingle,markersize=markersize)
            for idx in df.index:
                ax.plot(df_x_jitter.loc[idx,[xticks[0],xticks[1]]], df.loc[idx,[xticks[0],xticks[1]]], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        else:
            for n in range(len(res)):
                ax.plot([0,1],res[n],'o--',color=colorsingle,markersize=markersize, alpha=0.5)
    else:
        df1 = pd.DataFrame({xticks[0]:var1})
        df2 = pd.DataFrame({xticks[1]:var2})
        df=pd.concat([df1,df2], ignore_index=True, axis=1)
        splot=sns.swarmplot(x="variable", y="value", data=df.melt(),ax=ax,size=6,alpha=0.8)
        #splot=sns.stripmplot(x="variable", y="value", data=df.melt(),ax=ax,size=6,alpha=0.8,color='#A9A9A9')
        if logy==True:
            splot.set(yscale='log')
    
    ax.boxplot([var1,var2],meanline=True,showmeans=True,meanprops=meanlineprops,medianprops=medianprops,positions=[0,1],widths=0.3)
    ax.set_xticklabels(xticks,fontsize=14)
    max1=max(var1)
    max2=max(var2)
    maxvalue=max([max1,max2])
    #maxvalue=N.matrix.max(N.matrix([var1,var2]))
    heights = [maxvalue, maxvalue-2*maxvalue/50, maxvalue-3*maxvalue/50]
    bars = N.arange(len(heights)+1)
    ba.barplot_annotate_brackets(0,1,'p=%s'%pvalue,bars,heights)
    pl.yticks(fontsize=16)
    #ax.set_xlabel('paired ttest pvalue %s'%pvalue)
    if pvalue1>0.05 and pvalue2>0.05:
        if pairedT==True:
            ax.set_xlabel('paired t test,n= %s'%len(var1),fontsize=14)
        else:
            ax.set_xlabel('indep t test,n= %s vs %s'%(len(var1),len(var2)),fontsize=14)
    else:
        if pairedT==True:
            ax.set_xlabel('wilcoxon for paired,n= %s'%len(var1),fontsize=14)
        else:
            ax.set_xlabel('rank sum for indep,n= %s vs %s'%(len(var1),len(var2)),fontsize=14)
    pl.ylabel(ylabel,fontsize=18)
    pl.xticks(N.arange(0,2,1),xticks,fontsize=18)
    pl.yticks(fontsize=18)
    #pl.ylim([-10,110])
    #pl.ylim([-5,5])
    #pl.ylim([0.03,210])
    pl.title(title,fontsize=18)
    if pairedT==True:
        ax = fig.add_subplot(122)
        if jitterplot==True:
            df = pd.DataFrame(collections.OrderedDict({xticks[0]: var1, xticks[1]: var2,}))
            jitter = jitter
            df_x_jitter = pd.DataFrame(N.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
            df_x_jitter += N.arange(len(df.columns))
            for col in df:
                ax.plot(df_x_jitter[col], df[col], 'o', color=colorsingle,markersize=markersize)
            for idx in df.index:
                ax.plot(df_x_jitter.loc[idx,[xticks[0],xticks[1]]], df.loc[idx,[xticks[0],xticks[1]]], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        else:
            for n in range(len(res)):
                ax.plot([0,1],res[n],'o--',color=colorsingle,alpha=0.5)
        ax.boxplot([var1,var2],meanline=True,showmeans=True,meanprops=meanlineprops,medianprops=medianprops,positions=[0,1],widths=0.3)
        ax.set_yscale('log')
        ax.set_xticklabels(xticks,fontsize=14)
        ax.set_ylabel('mean firing frequency (Hz)',fontsize=16)
    pl.yticks(fontsize=16)
    os.chdir(directory)
    #fig.savefig('%s.png' %title)
    #fig.savefig('%s.eps'%title,transparent=True) 

def summary_heatmap_plot(result,dir,plotData=True,Tlength=150,vmax=50,normalized=False,normalized_noreorg=True,zscore=False,reorganize_place=False,reorganize_place_max=True,reorganize_fir=False,meanplot=False,smooth=True,cmap='jet'):
    
    fig_to_save=tkFileDialog.asksaveasfilename(title="Choose target figure name")
    directory1, fname1 = os.path.split(fig_to_save)
    
    if meanplot==True:
        mean1=N.nanmean(result,axis=0)
        sem1=stats.sem(result)
    
    if smooth==True:
        sigma_rew=1.5 
        for n in range(len(result)):
            result[n,:]=ndimage.gaussian_filter(result[n,:],sigma_rew,truncate=2) 
            
    if normalized==True:
        order=[]
        for i in range(len(result)):
            if reorganize_place_max==True:
                order.append(N.ravel(N.argwhere(result==max(result[i][:]))))
            else:
                order.append(N.ravel(N.argwhere(result==min(result[i][:]))))
        for j in range(len(order)-1):
            for l in range(len(order)-j):
                if order[j+l][1]<order[j][1]:
                    order[j+l],order[j]=order[j],order[j+l]
                    dir[j+l],dir[j]=dir[j],dir[j+l]
        result_reorg=[]
        for m in range(len(order)):
            result_reorg.append(result[order[m][0]])
            #result_reorg=N.concatenate((result_reorg,result[order[m][0]]))
        os.chdir(os.path.split(dir[0])[0])
        namefile1 = open('list_norm_reorg.txt','w+')
        for i in dir:
            namefile1.write(os.path.split(i)[1] + '\n')
        namefile1.close()  
        N.save('order_reorganize_place',order)
        Norm=[]
        for i in range(len(result_reorg)):
            scale=max(result_reorg[i][:])
            Norm.append(N.divide(result_reorg[i][:],scale))
        N.save('order_reorganize_place',order)
        N.save('raw_reorganize_place',result_reorg)
    
    if normalized_noreorg==True:
        Norm=[]
        for i in range(len(result)):
            scale=max(result[i][:])
            Norm.append(N.divide(result[i][:],scale))
    
    if normalized_noreorg==True and zscore==True:
        Norm=[]
        for n in range(len(result)):
            Norm.append(stats.zscore(result[n][:]))
        
        
    if reorganize_place==True:
       order=[]
       for i in range(len(result)):
           if reorganize_place_max==True:
               order.append(N.ravel(N.argwhere(result==max(result[i][:]))))
           else:
               order.append(N.ravel(N.argwhere(result==min(result[i][:]))))
       for j in range(len(order)-1):
           for l in range(len(order)-j):
               if order[j+l][1]<order[j][1]:
                   order[j+l],order[j]=order[j],order[j+l]
                   dir[j+l],dir[j]=dir[j],dir[j+l]
       result_reorg=[]
       for m in range(len(order)):
           result_reorg.append(result[order[m][0]])
           #result_reorg=N.concatenate((result_reorg,result[order[m][0]]))
       os.chdir(os.path.split(dir[0])[0])
       namefile1 = open('list_reorg.txt','w+')
       for i in dir:
           namefile1.write(os.path.split(i)[1] + '\n')
       namefile1.close()
       N.save('order_reorganize_place',order)
       N.save('raw_reorganize_place',result_reorg)

    if reorganize_fir==True:
       order1=[]
       for i in range(len(result)):
           order1.append([i,N.mean(result[i][:])])
       for j in range(len(order1)-1):
           for l in range(len(order1)-j):
               if order1[j+l][1]<order1[j][1]:
                   order1[j+l],order1[j]=order1[j],order1[j+l]
                   dir[j+l],dir[j]=dir[j],dir[j+l]
       result_reorg_fir=[]
       for m in range(len(order1)):
           result_reorg_fir.append(result[order1[m][0]])
           #result_reorg=N.concatenate((result_reorg,result[order[m][0]]))
       os.chdir(os.path.split(dir[0])[0])
       namefile2 = open('list_reorg_fir.txt','w+')
       for i in dir:
           namefile2.write(os.path.split(i)[1] + '\n')
       namefile2.close()
    
    if normalized==True:
        contour_levels=N.linspace(0,1,100)
    elif normalized_noreorg==True:
        if zscore==False:
            contour_levels=N.linspace(0,1,100)
        else:
           contour_levels=N.linspace(0,N.argmax(Norm)+1,100) 
    else:
        contour_levels=N.linspace(0,N.argmax(result)+10,100)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    if plotData==True:
       if N.isnan(Tlength)==True:
           #f,ax=pl.subplots(figsize=(8,12),sharex=True)
           f,ax=pl.subplots(figsize=(8,4),sharex=True)
           #f,ax=pl.subplots(figsize=(8,8),sharex=True)
           #f,ax=pl.subplots(figsize=(8,6),sharex=True)
           f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
           if normalized==True:
               #im=ax.imshow(Norm,cmap=cmap,interpolation="nearest",origin='upper',aspect='auto',vmin=0,vmax=1)
               im=ax.imshow(Norm,cmap=cmap,interpolation="nearest",origin='upper',extent=[-5,8,0,len(result)],aspect='auto',vmin=0,vmax=1)
               #im=ax.imshow(Norm,cmap=cmap,interpolation="nearest",origin='upper',extent=[-3,6,0,len(result)],aspect='auto',vmin=0,vmax=1)
           elif normalized_noreorg==True:
               #im=ax.imshow(Norm,cmap='jet',interpolation="nearest",origin='upper',extent=[-5,3,0,len(result)],aspect='auto',vmin=0,vmax=1)
               #im=ax.imshow(Norm,cmap='jet',interpolation="nearest",origin='upper',extent=[-3,5,0,len(result)],aspect='auto',vmin=0,vmax=1)
               #im=ax.imshow(Norm,cmap='jet',interpolation="nearest",origin='upper',extent=[-3,6,0,len(result)],aspect='auto',vmin=0,vmax=1)
               im=ax.imshow(Norm,cmap=cmap,interpolation="nearest",origin='upper',extent=[-5,8,0,len(result)],aspect='auto',vmin=0,vmax=1)
               #ax.grid(color='k', linewidth=1,axis='y')
               #ax.set_yticks(range(len(Norm)))
               #im=ax.imshow(Norm,cmap=cmap,interpolation="nearest",origin='upper',extent=[-8,5,0,len(result)],aspect='auto',vmin=0,vmax=1)
               #im=ax.imshow(Norm,cmap='jet',interpolation="nearest",origin='upper',extent=[-1,29,0,len(result)],aspect='auto',vmin=0,vmax=1)
               #im=ax.imshow(Norm,cmap='jet',interpolation="nearest",origin='upper',extent=[-250,500,0,len(result)],aspect='auto',vmin=0,vmax=1)
               #im=ax.imshow(Norm,cmap='jet',interpolation="nearest",origin='upper',extent=[-50,100,0,len(result)],aspect='auto',vmin=0,vmax=1)
               #im=ax.imshow(Norm,cmap=cmap,interpolation="nearest",origin='upper',extent=[-100,100,0,len(result)],aspect='auto',vmin=0,vmax=1)
               if zscore==True:
                   im=ax.imshow(Norm,cmap=cmap,interpolation="nearest",origin='upper',extent=[-5,8,0,len(result)],aspect='auto',vmin=N.amin(Norm),vmax=N.amax(Norm))
               
           elif reorganize_place==True:
               #im=ax.imshow(result_reorg,cmap=cmap,interpolation="nearest",origin='upper',aspect='auto',vmin=0,vmax=vmax)
               im=ax.imshow(Norm,cmap=cmap,interpolation="nearest",origin='upper',extent=[-5,8,0,len(result)],aspect='auto',vmin=0,vmax=1)
           elif reorganize_fir==True:
               im=ax.imshow(result_reorg_fir,cmap=cmap,interpolation="nearest",origin='upper',aspect='auto',vmin=0,vmax=vmax)
           else:
               #im=ax.imshow(result,cmap=cmap,interpolation="nearest",origin='upper',extent=[-5,3,0,len(result)],aspect='auto',vmin=0,vmax=vmax)
               im=ax.imshow(result,cmap=cmap,interpolation="nearest",origin='upper',extent=[-3,6,0,len(result)],aspect='auto',vmin=0,vmax=vmax)
               #im=ax.imshow(result,cmap=cmap,interpolation="nearest",origin='upper',extent=[-3,5,0,len(result)],aspect='auto',vmin=0,vmax=vmax)
               #im=ax.imshow(result,cmap=cmap,interpolation="nearest",origin='upper',extent=[-5,8,0,len(result)],aspect='auto',vmin=0,vmax=vmax)
               #im=ax.imshow(result,cmap=cmap,interpolation="nearest",origin='upper',extent=[-8,5,0,len(result)],aspect='auto',vmin=0,vmax=vmax)
           #ax.set_xlabel('track position (au)',fontsize=14)
           ax.set_xlabel('time (s)',fontsize=14)
           #ax.set_xlim([-3,6])
           ax.set_ylabel('Cell ID.',fontsize=14)
           ax.set_title('%s_summary'%fig_to_save.split('/')[-1],size=14)
           #ax.set_xticks(N.arange(0,Tlength+binsize,10*binsize))
           # create an axes on the right side of ax. The width of cax will be 5%
           # of ax and the padding between cax and ax will be fixed at 0.05 inch.
           divider = make_axes_locatable(ax)
           cax = divider.append_axes("right", size="5%", pad=0.05)
           cbar = ax.figure.colorbar(im, cax=cax)
           if normalized==True:
               cbar.ax.set_ylabel('normalized firing rate', rotation=-90, va="bottom",fontsize=14)
           else:
               cbar.ax.set_ylabel('firing rate (Hz)', rotation=-90, va="bottom",fontsize=14)
       else:
           f,ax=pl.subplots(figsize=(8,4),sharex=True)
           #f,ax=pl.subplots(figsize=(4,4),sharex=True)
           #f,ax=pl.subplots(figsize=(8,12),sharex=True)
           f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
           if normalized==True:
               im=ax.imshow(Norm,cmap=cmap,interpolation="nearest",origin='upper',extent=[0,Tlength,0,len(result)],aspect='auto',vmin=0,vmax=1)
           elif reorganize_place==True:
               im=ax.imshow(result_reorg,cmap=cmap,interpolation="nearest",origin='upper',extent=[0,Tlength,0,len(result)],aspect='auto',vmin=0,vmax=vmax)
           elif reorganize_fir==True:
               im=ax.imshow(result_reorg_fir,cmap=cmap,interpolation="nearest",origin='upper',extent=[0,Tlength,0,len(result)],aspect='auto',vmin=0,vmax=vmax)
           elif normalized_noreorg==True:
               im=ax.imshow(Norm,cmap=cmap,interpolation="nearest",origin='upper',extent=[0,Tlength,0,len(result)],aspect='auto',vmin=0,vmax=vmax)
           else:
               im=ax.imshow(result,cmap=cmap,interpolation="nearest",origin='upper',extent=[0,Tlength,0,len(result)],aspect='auto',vmin=0,vmax=vmax)
           ax.set_xlabel('track position (cm)',fontsize=14)
           ax.set_ylabel('Cell ID.',fontsize=14)
           ax.set_title('%s_summary'%fig_to_save.split('/')[-1],size=14)
           #ax.set_xticks(N.arange(0,Tlength+binsize,10*binsize))
           # create an axes on the right side of ax. The width of cax will be 5%
           # of ax and the padding between cax and ax will be fixed at 0.05 inch.
           divider = make_axes_locatable(ax)
           cax = divider.append_axes("right", size="5%", pad=0.05)
           cbar = ax.figure.colorbar(im, cax=cax)
           if normalized==True:
               cbar.ax.set_ylabel('normalized firing rate', rotation=-90, va="bottom",fontsize=14)
           else:
               cbar.ax.set_ylabel('firing rate (Hz)', rotation=-90, va="bottom",fontsize=14)
       if meanplot==True:
           ax = f.add_subplot(212)
           #mean1=N.nanmean(result,axis=0)
           #sem1=stats.sem(result)
           divider = make_axes_locatable(ax)
           cax = divider.append_axes("right", size="5%", pad=0.05)
           cbar = ax.figure.colorbar(im, cax=cax)
           #x=N.arange(-5,7.9,0.1) 
           #x=N.arange(0,400,5) 
           x=N.arange(-3,5.9,0.1) 
           #ax.set_title('proj vs local SOMI_speed correlation',size=14)
           #for n in range(len(result)):
           #    ax.plot(x,result[n],alpha=0.5,c='#A9A9A9')
           ax.fill_between(x,mean1+sem1,mean1-sem1,facecolor='orange', alpha=0.5)
           ax.plot(x,mean1,'r-')
           #ax.set_xlim([-5,8])
           ax.set_xlim([-3,6])
           #ax.set_xlim([0,400])
       #f.savefig('%s_Summary.png' %fig_to_save)
       #
       f.savefig('%s_Summary.eps' %fig_to_save,transparent=True) 
        
def autocorrelogram(Fs=30000,maxlag=100,plotData=True,saveFig=False):
    
    '''Computes the autocorrelation function of a spike train.
    Inputs:
    indices: Unit spike indices
    file_to_save: file name and target folder for saving the figure.
    fname: file name of the target spike file.
    Output:
    Autocorrelation function

    '''
    target_dir='/data/example_raster_plot_optoID'                
    os.chdir(target_dir)
    indices=op.open_helper('SU_126.0_timestamps.txt')
    file_to_save=target_dir
    fname='SU_126.0_timestamps.txt'
    # Convert spike indices to ms.
    msFs=Fs/1000.0
    # Create spike indices of unit ms.
    ind1=[]
    for n in range(len(indices)):
          ind1.append(indices[n]/msFs)
    
    
    raw=sp.filter_correlogram(ind1,dt=1,shift=maxlag)
    acorr=raw[0]
    acorr[len(acorr)/2]=0
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if plotData==True:
        fig=pl.figure(figsize=[8,6])
        x=N.linspace(-maxlag,maxlag,len(acorr))
        pl.bar(x,acorr,width=1)
        #_=pl.plot(x,acorr,"k")
        pl.xlim(-maxlag,maxlag)
        uname=fname.split('_')[1]
        pl.title('u%s_autocorrelogram '%uname ,fontsize=18)
    
    if saveFig==True:
        fig.savefig('%s_u%s_autocorrelogram.png' %(file_to_save,uname))
        fig.savefig('%s_u%s_autocorrelogram.eps' %(file_to_save,uname),transparent=True) 
    return acorr

def barplot_stacked_percentage_catergories(names,c=2,title='OFF_linear_regression_all_loc',barWidth = 0.5):
    ##c: catergories
    '''
    related to S7e, S9c, S10C
    
    '''
    
    r = N.arange(c)
    color=['#FF6103','#FF9912','#FFC125','#A1A1A1']
    raw_data= {'higher2': [31,11], 'between1_2': [16,18],'smaller1': [14,5], 'nosig': [0,27]} #speed
    #raw_data= {'higher2': [16,24], 'between1_2': [15,15],'smaller1': [21,19], 'nosig': [9,3]} #acceleration
    #color=['#0000FF','#1E90FF','#00EEEE','#A1A1A1']
    #color=['#FF6103','#FF9912']
    #raw_data= {'higher2': [15,8], 'between1_2': [6,6],'smaller1': [5,1], 'nosig': [0,11]}  #speed
    #raw_data= {'higher2': [11,15], 'between1_2': [8,4],'smaller1': [4,6], 'nosig': [3,1]}  #acceleration
    #color=['#6E8B3D','#458B00','#76EE00','#A1A1A1']
    #color=['#6E8B3D','#A1A1A1']
    #raw_data= {'higher2': [5,5], 'between1_2': [10,5],'smaller1': [24,5], 'nosig': [6,30]} #speed
    #raw_data= {'higher2': [3,3], 'between1_2': [8,6],'smaller1': [18,27], 'nosig': [16,9]} #acceleration
    #raw_data= {'higher2': [21,10,7], 'between1_2': [40,16,38]}  #spike_rate speed cross corr
    #raw_data= {'rew_air_resp': [5,3], 'rew_air_no_resp': [4,6]}  #senstim response of SOMI local rew vs airpuf
    #raw_data= {'rew_air_resp': [9,9], 'rew_air_no_resp': [14,14]}  #senstim response of SOMI proj rew vs airpuf
    
    df = pd.DataFrame(raw_data)
    totals = [i+j+k+l for i,j,k,l in zip(df['higher2'], df['between1_2'], df['smaller1'], df['nosig'])]
    #totals = [i+j for i,j in zip(df['higher2'], df['between1_2'])]
    higher2 = [100* float(i) / j  for i,j in zip(df['higher2'], totals)]
    between1_2 = [100*float(i) / j for i,j in zip(df['between1_2'], totals)]
    smaller1 = [100*float(i) / j  for i,j in zip(df['smaller1'], totals)]
    nosig = [100*float(i) / j  for i,j in zip(df['nosig'], totals)]
    
    #names = (namec1,namec2)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig=pl.figure(figsize=[5,6])
    ax1=pl.bar(r, higher2, color=color[0], edgecolor='white', width=barWidth,label="r higher 0.2")
    # Create orange Bars
    ax2=pl.bar(r, between1_2, bottom=higher2, color=color[1], edgecolor='white', width=barWidth,label="r between 0.1 and 0.2")
    # Create blue Bars
    ax3=pl.bar(r, smaller1, bottom=[i+j for i,j in zip(higher2, between1_2)], color=color[2], edgecolor='white', width=barWidth,label="r smaller 0.1")
    ax4=pl.bar(r, nosig, bottom=[i+j+k for i,j,k in zip(higher2, between1_2, smaller1)], color=color[3], edgecolor='white', width=barWidth,label="r no sig")
    # Custom x axis
    pl.xticks(r, names,fontsize=16)
    pl.ylabel("% of SOMIs",fontsize=16)
    pl.ylim([0,105])
    pl.title(title,fontsize=16)
    #pl.yticks(fontsize=16)
    pl.legend()
    
    for r1, r2, r3, r4 in zip(ax1,ax2,ax3,ax4):
    #for r1, r2 in zip(ax1,ax2):
        h1=r1.get_height()
        h2=r2.get_height()
        h3=r3.get_height()
        h4=r4.get_height()
        pl.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%f" % h1, ha="center", va="bottom", color="white")
        pl.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%f" % h2, ha="center", va="bottom", color="white")
        pl.text(r3.get_x() + r3.get_width() / 2., h1 + h2 + h3 / 2., "%f" % h3, ha="center", va="bottom", color="white")
        pl.text(r4.get_x() + r4.get_width() / 2., h1 + h2 + h3 + h4 / 2., "%f" % h4, ha="center", va="bottom", color="white")
    
    #os.chdir(directory)
    #fig.savefig('%s.png' %title)
    #fig.savefig('%s.eps'%title,transparent=True) 
    
def piechart_all():
    
    '''
    Related to Fig2g, Fig3e, Fig5a, S2c,f, S7c, S8d
    '''
    labels='ON','OFF','NON'
    labels1='local','proj','SOMI'
    sizes_local=[5,14,5]            
    sizes_proj=[11,14,6]            
    sizes_dec=[14,15,15]
    sizes_inc=[5,11,9]
    
    labels_learned='dec','others'
    labels_not_learned='inc','others'
    sizes_learned=[11,6]
    sizes_not_learned=[26,27]
    
    labels='sig','others'
    sizes_reward=[12,21]
    sizes_airpuf=[9,42]
    sizes_sound=[6,27]
    sizes_visual=[8,25]
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    fig = pl.figure(figsize=[9,9])
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax = fig.add_subplot(221)
    ax.pie(sizes_local,labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    ax.set_title('local SOMI',fontsize=16)
    ax = fig.add_subplot(222)
    ax.pie(sizes_proj,labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    ax.set_title('proj SOMI',fontsize=16)
    ax = fig.add_subplot(223)
    ax.pie(sizes_dec,labels=labels1, autopct='%1.1f%%',shadow=True, startangle=90)
    ax.set_title('stop triggered SOMI',fontsize=16)
    ax = fig.add_subplot(224)
    ax.pie(sizes_inc,labels=labels1, autopct='%1.1f%%',shadow=True, startangle=90)
    ax.set_title('start triggered SOMI',fontsize=16)
    
    fig = pl.figure(figsize=[9,9])
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax = fig.add_subplot(221)
    ax.pie(sizes_learned,labels=labels_learned, autopct='%1.1f%%',shadow=True, startangle=90)
    ax.set_title('SOMI in learned mice',fontsize=16)
    ax = fig.add_subplot(222)
    ax.pie(sizes_not_learned,labels=labels_not_learned, autopct='%1.1f%%',shadow=True, startangle=90)
    ax.set_title('SOMI in not learned mice',fontsize=16)
    
    fig = pl.figure(figsize=[9,9])
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax = fig.add_subplot(221)
    ax.pie(sizes_reward,labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    ax.set_title('senstim reward',fontsize=16)
    ax = fig.add_subplot(222)
    ax.pie(sizes_airpuf,labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    ax.set_title('senstim airpuff',fontsize=16)
    ax = fig.add_subplot(223)
    ax.pie(sizes_visual,labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    ax.set_title('senstim visual',fontsize=16)
    ax = fig.add_subplot(224)
    ax.pie(sizes_sound,labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    ax.set_title('senstim sound',fontsize=16)
    
    #fig_to_save=tkFileDialog.asksaveasfilename(title="Choose target figure name")
    #fig.savefig('%s_piechart_summary.png' %fig_to_save)
    #fig.savefig('%s_piechart_summary.eps' %fig_to_save,transparent=True)


def Donutplot():
    
    '''
    Related to S5b
    '''
    
    '''
    group_names=['ON', 'OFF', 'Non-mod']
    #group_size=[27,59,45]
    group_size=[26,59,45]
    subgroup_names=['local', 'proj', 'DG-ident-only', 'local', 'proj','DG-ident-only', 'local', 'proj','DG-ident-only']
    subgroup_size=[4,13,10,12,32,15,10,20,15]
    
    group_names=['pos', 'pos&neg','neg', 'Not_tuned']
    #group_size=[27,59,45]
    group_size=[17,55,17,10]
    #subgroup_names=['OFF', 'On', 'No-mod', 'OFF', 'ON','No-mod', 'OFF', 'ON','No-mod', 'OFF', 'ON','No-mod']
    #subgroup_size=[5,5,7,23,14,18,8,5,4,1,1,8]
    subgroup_names=['local', 'proj', 'DG-ident-only', 'local', 'proj','DG-ident-only', 'local', 'proj','DG-ident-only', 'local', 'proj','DG-ident-only']
    subgroup_size=[5,5,7,14,31,10,3,7,7,1,3,6] 
    # Create colors
    a, b, c, d=[pl.cm.Blues, pl.cm.Oranges,pl.cm.Greens,  pl.cm.Greys]
    '''
    group_names=['pos/&neg','neg', 'Not_tuned']
    #group_size=[27,59,45]
    #group_size=[72,17,10]
    group_size=[67,16,10]
    #subgroup_names=['OFF', 'On', 'No-mod', 'OFF', 'ON','No-mod', 'OFF', 'ON','No-mod']
    #subgroup_size=[28,19,25,8,5,4,1,1,8]
    subgroup_names=['local', 'proj','DG-ident-only', 'local', 'proj','DG-ident-only', 'local', 'proj','DG-ident-only']
    #subgroup_size=[19,36,17,3,7,7,1,3,6] 
    subgroup_size=[19,36,12,3,7,6,1,3,6] 
    # Create colors
    a, b, c=[pl.cm.Oranges, pl.cm.Blues, pl.cm.Greys]
     
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    fig, ax = pl.subplots()
    ax.axis('equal')
    #mypie, _ = ax.pie(group_size, radius=1, labels=group_names, labeldistance=0.6, colors=[a(0.6), b(0.6), c(0.6), d(0.6)] )
    mypie, _ = ax.pie(group_size, radius=1, labels=group_names, labeldistance=0.6, colors=[a(0.6), b(0.6), c(0.6)] )
    #mypie, _ = ax.pie(group_size, labels=group_names, colors=[a(0.6), b(0.6), c(0.6)] )
    pl.setp( mypie, width=0.6, edgecolor='white')
     
    # Second Ring (Inside)
    #mypie2, _ = ax.pie(subgroup_size, radius=1+0.3, labels=subgroup_names, labeldistance=0.8, colors=[a(0.5), a(0.4), a(0.3), b(0.5), b(0.4), b(0.3), c(0.5), c(0.4), c(0.3),d(0.5), d(0.4), d(0.3)])
    mypie2, _ = ax.pie(subgroup_size, radius=1+0.3, labels=subgroup_names, labeldistance=0.8, colors=[a(0.5), a(0.4), a(0.3), b(0.5), b(0.4), b(0.3), c(0.5), c(0.4), c(0.3)])
    pl.setp( mypie2, width=0.3, edgecolor='white')
    pl.margins(0,0)
     
    # show it
    pl.show()
    
    fig_to_save=tkFileDialog.asksaveasfilename(title="Choose target figure name")
    fig.savefig('%s_Donutpiechart_summary.png' %fig_to_save)
    fig.savefig('%s_Donutpiechart_summary.eps' %fig_to_save,transparent=True)   

    
def summary_reward_corr(Fs=30000,speed=True,smooth=False,speedhighresol=True,zscore=False):
    
    '''
    Related to Fig2b,e, Fig3c,d, Fig4c,d, Fig5b, S4a,b,e, S5a, S8a, S9b, S10a,b, S11b, S12a, S13a,b,c
    '''
    
    dir1 = tkFileDialog.askopenfilename(title='Choose file1')
    #directory, fname = os.path.split(dir1)
    #result1=op.open_helper(dir1)
    result1=op.open_helper(dir1)
    dir2 = tkFileDialog.askopenfilename(title='Choose file2')
    #directory, fname = os.path.split(dir2)
    result2=op.open_helper(dir2)
    fig_to_save=tkFileDialog.asksaveasfilename(title="Choose target figure name")
    if speed==True:
        dir1 = tkFileDialog.askopenfilename(title='Choose speedfile1')
        #directory, fname = os.path.split(dir1)
        speed1=op.open_helper(dir1)
        dir2 = tkFileDialog.askopenfilename(title='Choose speedfile2')
        #directory, fname = os.path.split(dir2)
        speed2=op.open_helper(dir2)
        speedmean1=N.nanmean(speed1,axis=0)
        speedsem1=stats.sem(speed1, nan_policy='omit')
        #std1=N.nanstd(result1,axis=0)
        speedmean2=N.nanmean(speed2,axis=0)
        speedsem2=stats.sem(speed2, nan_policy='omit')
    result1plot=[]
    result2plot=[]
    if zscore==True:
        for n in range(len(result1)):
            result1plot.append(stats.zscore(result1[n,:]))
        for m in range(len(result2)):
            result2plot.append(stats.zscore(result2[m,:])) 
        result1=result1plot
        result2=result2plot
    if smooth==True:
         sigma_rew=1.5 
         result1=ndimage.gaussian_filter(result1,sigma_rew)  #gaussian kernel filter to smooth the data
         result2=ndimage.gaussian_filter(result2,sigma_rew) 
         
    #x=N.arange(-len(result1[0])+1,1)
    #x2=N.arange(-len(result2[0])+1,1)
    x=N.arange(-5,7.9,0.1)+0.05  
    #x2=N.arange(-5,7.9,0.1)+0.05 
    #x=N.arange(-3,5.9,0.1)+0.05 
    #x=N.arange(0,400,5)+2.5 
    x2=x
    #x=N.arange(0,400,8.34)+4.16 
    #x=N.arange(-3,4.9,0.2)
    #x=N.arange(1,7)    
    #x=N.arange(-3,5,0.2)+0.1
    #x=N.arange(-3,5,0.1)+0.05
    #x=N.arange(-6,8,0.01)+0.005
    #x=N.arange(-5.99,7.99,0.01)+0.005
    #x=N.arange(-100,101,1)+0.5
    mean1=N.nanmean(result1,axis=0)
    sem1=stats.sem(result1, nan_policy='omit')
    #std1=N.nanstd(result1,axis=0)
    mean2=N.nanmean(result2,axis=0)
    sem2=stats.sem(result2, nan_policy='omit')
    #std2=N.nanstd(result2,axis=0)
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    #f,(ax1,ax2)=pl.subplots(2,1,sharex=True,figsize=[8,6],sharey=True,gridspec_kw={'height_ratios': [2, 1]})
    #f,(ax1,ax2)=pl.subplots(2,1,sharex=True,figsize=[8,6],sharey=True)
    f,(ax1,ax2)=pl.subplots(2,1,sharex=True,sharey=True,figsize=[8,6])
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    f.text(0.04, 0.5, 'Averaged firing frequency (Hz)', va='center', rotation='vertical',fontsize=14)
    ax1.set_title('learned vs not learned lick_reward correlation',size=14)
    ax1.plot(x,mean1,'r-')
    ax1.fill_between(x,mean1+sem1,mean1-sem1,facecolor='orange', alpha=0.5)
    #ax1.set_xlim([-5,8])
    #ax1.set_xlim([-3,6])
    ax1.set_ylim([10,45])
      
    #ax1.plot(x,mean2,'b-')
    ax1.plot(x2,mean2,'b-')
    ax1.fill_between(x2,mean2+sem2,mean2-sem2,facecolor='darkgreen', alpha=0.5)
    ax1.set_xlabel('time (s)',fontsize=14)
    '''
    ax2.plot(x,mean2,'b-')
    ax2.fill_between(x,mean2+sem2,mean2-sem2,facecolor='darkgreen', alpha=0.5)
    ax2.set_xlabel('time (s)',fontsize=14)
    '''  
    #ax1.set_xlim([-3,6])
    
    #ax2.set_xlim([-2,2])
    
    if speed==True:
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        #f,(ax1,ax2)=pl.subplots(2,1,sharex=True,figsize=[8,6],sharey=True)
        f,(ax1,ax2)=pl.subplots(2,1,sharex=True,figsize=[8,6])
        #f,(ax1,ax2)=pl.subplots(1,2,sharex=True,figsize=[8,6])
        #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
        f.text(0.04, 0.5, 'Averaged firing frequency (Hz)', va='center', rotation='vertical',fontsize=14)
        ax1.set_title('learned vs not learned lick_reward correlation',size=14)
        if speedhighresol==True:
            ax1.plot(x*Fs,mean1,'r-')
            ax1.fill_between(x*Fs,mean1+sem1,mean1-sem1,facecolor='orange', alpha=0.5)
            #ax1.set_xlim([-5*Fs,8*Fs])
            ax2.plot(x*Fs,mean2,'b-')
            ax2.fill_between(x*Fs,mean2+sem2,mean2-sem2,facecolor='darkgreen', alpha=0.5)
        else:
            ax1.plot(x,mean1,'r-')
            ax1.fill_between(x,mean1+sem1,mean1-sem1,facecolor='orange', alpha=0.5)
            ax2.plot(x,mean2,'b-')
            ax2.fill_between(x,mean2+sem2,mean2-sem2,facecolor='darkgreen', alpha=0.5)
        ax2.set_xlabel('time (s)',fontsize=14)
        ax3 = ax1.twinx()
        if speedhighresol==True:
            #x=range(int(-3*Fs-1),int(2*3*Fs))
            x=range(int(-5*Fs-1),int(8*Fs))
            ax3.fill_between(x[::50],speedmean1[::50]+speedsem1[::50],speedmean1[::50]-speedsem1[::50],facecolor='#A9A9A9', alpha=0.5)
            #ax3.set_xlim([-3*Fs,6*Fs])
        else:
            ax3.fill_between(x,speedmean1+speedsem1,speedmean1-speedsem1,facecolor='#A9A9A9', alpha=0.5)
        ax3.plot(x,speedmean1,color='k')
        ax3.set_ylabel('Speed (cm/s)',color='k',fontsize=14)
        ax3.tick_params(axis='y', labelcolor='k')
        ax3.set_ylim([2,18])  ###for lick in trans
        ax4 = ax2.twinx()
        if speedhighresol==True:
            ax4.fill_between(x[::50],speedmean2[::50]+speedsem2[::50],speedmean2[::50]-speedsem2[::50],facecolor='#A9A9A9', alpha=0.5)
            #ax4.set_xlim([-3*Fs,6*Fs])
        else:
            ax4.fill_between(x,speedmean2+speedsem2,speedmean2-speedsem2,facecolor='#A9A9A9', alpha=0.5)
        ax4.plot(x,speedmean2,color='k')
        ax4.set_ylabel('Speed (cm/s)',color='k',fontsize=14)
        ax4.tick_params(axis='y', labelcolor='k')
        #ax4.set_ylim([2,18])  ###for lick in trans
        #ax1.set_ylim([0,6.5])
        #ax2.set_ylim([0,6.5])
        #ax1.set_ylim([2,12])
        #ax2.set_ylim([2,12])
        #ax3.set_ylim([3,17])
        #ax4.set_ylim([3,17])
        #ax1.set_xlim([-3*Fs,6*Fs])
        #ax1.set_ylim([-0.6,1.2])
        #ax2.set_xlim([-3*Fs,6*Fs])
        #ax2.set_ylim([-0.6,1.2])
        #ax1.set_xlim([-5,8])
        #ax1.set_xlim([-2*Fs,3*Fs])
        #ax1.set_xlim([-1.5*Fs,1.5*Fs])
    #f.savefig('%s_reward_corr_summary.png' %fig_to_save)
    #f.savefig('%s_reward_corr_summary.eps' %fig_to_save,transparent=True)       


def Summary_rate_prerepost(ylabels,xticklabels,sharey=False,ylim=[0,22]):
    
    '''
    Related to Fig2c,h, Fig4b,e,f, S4c, S12b
    '''
    dir1 = tkFileDialog.askopenfilename(title='Choose file1')
    directory, fname = os.path.split(dir1)
    result1=op.open_helper(dir1)
    dir2 = tkFileDialog.askopenfilename(title='Choose file2')
    #directory, fname = os.path.split(dir2)
    result2=op.open_helper(dir2)
    fig_to_save=tkFileDialog.asksaveasfilename(title="Choose target figure name")
    
    mean1=N.nanmean(result1,axis=0)
    sem1=stats.sem(result1, nan_policy='omit')
    mean2=N.nanmean(result2,axis=0)
    sem2=stats.sem(result2, nan_policy='omit')
    pre1,re1,post1=[],[],[]
    for n in range(len(result1)):
        pre1.append(result1[n][0])
        re1.append(result1[n][1])
        post1.append(result1[n][2])
    pre2,re2,post2=[],[],[]
    for n in range(len(result2)):
        pre2.append(result2[n][0])
        re2.append(result2[n][1])
        post2.append(result2[n][2])
    
    uname=fname.split('_')[0]
    fn=fname.split('_')[1]
    fn=fn.split('.')[0]    
    pvalue1, Num_df1, DEN_df1, F_value1, pvalue112, pvalue113, pvalue123 = AN.ANOVA_RM_1way_or_kruskal(pre1,re1,post1,fig_to_save,uname,fn,script='exrewpulse')
    pvalue2, Num_df2, DEN_df2, F_value2, pvalue212, pvalue213, pvalue223 = AN.ANOVA_RM_1way_or_kruskal(pre2,re2,post2,fig_to_save,uname,fn,script='exrewpulse')
    if N.isnan(Num_df1)==False:
        print 'file1 RM one way ANOVA test, F(%s,%s):%s, p=%s ' %(Num_df1,DEN_df1,F_value1,pvalue1)
    else:
        print 'file1 Friedman test, p=%s ' %pvalue1
    if pvalue112 >0.05 or N.isnan(pvalue112)==True:
        print 'file1 no siginificance pre vs. reward'
    else:
        print 'file1 reward sensitive, significant',pvalue112
        
    if N.isnan(Num_df1)==False:
        print 'file2 RM one way ANOVA test, F(%s,%s):%s, p=%s ' %(Num_df2,DEN_df2,F_value2,pvalue2)
    else:
        print 'file2 Friedman test, p=%s ' %pvalue2
    if pvalue212 >0.05 or N.isnan(pvalue212)==True:
        print 'file2 no siginificance pre vs. reward'
    else:
        print 'file2 reward sensitive, significant',pvalue212
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = pl.figure(figsize=[8,6])
    maxvalue=N.matrix.max(N.matrix([pre1,re1,post1]))
    heights = [maxvalue, maxvalue-3*maxvalue/10, maxvalue-2*maxvalue/10, maxvalue-maxvalue/10]
    bars = N.arange(len(heights)+1)
    ax = fig.add_subplot(121)
    for n in range(len(result1)):
        ax.plot([1,2,3],result1[n],'o--',color='orange',alpha=0.5)
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=1.5, color='k')
    ax.boxplot(result1,meanline=True,showmeans=True,meanprops=meanlineprops,medianprops=medianprops)
    ax.set_xticklabels(xticklabels,fontsize=16)
    ax.set_ylabel(ylabels,fontsize=16)
    pl.yticks(fontsize=16)
    ba.barplot_annotate_brackets(1,2,pvalue112,bars,heights)
    ba.barplot_annotate_brackets(2,3,pvalue123,bars,heights)
    ba.barplot_annotate_brackets(1,3,pvalue113,bars,heights,dh=0.2)
    if sharey==True:
        ax.set_ylim(ylim)
    
    ax = fig.add_subplot(122)
    for n in range(len(result2)):
        ax.plot([1,2,3],result2[n],'o--',color='orange',alpha=0.5)
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=1.5, color='k')
    maxvalue=N.matrix.max(N.matrix([pre2,re2,post2]))
    heights = [maxvalue, maxvalue-3*maxvalue/10, maxvalue-2*maxvalue/10, maxvalue-maxvalue/10]
    bars = N.arange(len(heights)+1)
    ax.boxplot(result2,meanline=True,showmeans=True,meanprops=meanlineprops,medianprops=medianprops)
    ax.set_xticklabels(xticklabels,fontsize=16)
    ax.set_ylabel(ylabels,fontsize=16)
    pl.yticks(fontsize=16)
    ba.barplot_annotate_brackets(1,2,pvalue212,bars,heights)
    ba.barplot_annotate_brackets(2,3,pvalue223,bars,heights)
    ba.barplot_annotate_brackets(1,3,pvalue213,bars,heights,dh=0.2)
    if sharey==True:
        ax.set_ylim(ylim)
    #fig.savefig('%s_reward_barplot_Summary.png' %fig_to_save)
    #fig.savefig('%s_reward_barplot_Summary.eps' %fig_to_save,transparent=True)

def persons_r_2sets_of_data(smooth=False):
    dir1=tkFileDialog.askopenfilename(title='dataset1')
    directory, fname = os.path.split(dir1)
    #a=op.open_helper(dir1)[:,50:79]
    a=op.open_helper(dir1) 
    dir2=tkFileDialog.askopenfilename(title="dataset1")
    directory2, fname2 = os.path.split(dir2)
    #b=op.open_helper(dir2) [:,30:59]
    b=op.open_helper(dir2)
    file_to_save=tkFileDialog.asksaveasfilename(title="Choose target filename") 
    
    if smooth==True:
         sigma_rew=1.5 
         a=ndimage.gaussian_filter(a,sigma_rew,truncate=2)  
         b=ndimage.gaussian_filter(b,sigma_rew,truncate=2)  
         
    Pearsonr,p_all=[],[]
    for n in range(len(a)):
        pearsonr,p=stats.pearsonr(a[n],b[n])
        Pearsonr.append(pearsonr)
        p_all.append(p)
    
    os.chdir(directory)
    N.save('%s_pearson_correlation'%file_to_save,Pearsonr)
    N.save('%s_pearson_correlation_Pvalue'%file_to_save,p_all)
    
    return Pearsonr,p_all