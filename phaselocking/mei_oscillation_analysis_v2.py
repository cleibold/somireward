#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:21:09 2018

@author: jonas Mei
"""
import numpy as N
import js_data_tools as data_tools
import scipy.signal as si
import tkFileDialog
import open_helper as op
import js_stat_tools as stat
import scipy.stats as st
import os
import matplotlib.pyplot as pl
import speed_detection_vr_V2_mei as speed

from scipy import stats
import matplotlib


def spike_oscillation_coupling_single_above_thres(su,filt,env,low=1,high=5,thres=2,Fs=30000):
    ''' Implementation of spike-oscillation coupling following the description
    of Tamura et al,Neuron 2016. Measures pairwise phase consistency, mean resultant length,
    and test for deviation from uniformity using Rayleigh's test.
    thres: defined the threshold (in SD above the mean) of the envelope of the signal.
    Inputs:
    su : Spike indices of single unit
    raw : raw recording'''
    
    if len(su)<10:
        
        result={'circ':N.nan,'ang':N.nan,'ppc':N.nan,'p':N.nan,'mrl':N.nan}
    else:
        
        thr=N.mean(env)+thres*N.std(env)
        # Convert spike indices to integers and remove spikes that are <1s away from edges.    
        indices=[]
        for n in range(len(su)):
            if Fs<su[n]<(max(su)-Fs):
                indices.append(int(su[n]))
                
        ang_delta=[]
        used_indices=[]
        for n in indices:
            if env[n]>thr:
            
                local_delta=filt[n-Fs:n+Fs]                
                # Compute Hilbert transforms.
                hil=si.hilbert(local_delta)                
                ip_delta=N.angle(hil,deg=False)                               
                # Extract angles for each spike.        
                ang_delta.append(ip_delta[len(ip_delta)/2])   #where spike indice is
                used_indices.append(n)                      
        
        ang_delta=N.asarray(ang_delta)
        spike_number=len(ang_delta)
        if len(ang_delta)<10:
            result={'circ':N.nan,'ang':N.nan,'ppc':N.nan,'p':N.nan,'mrl':N.nan}
        else:
            
            # Test for circular uniformity using Rayleigh's test.
            p_delta=stat.rayleigh_test(ang_delta)                        
            
            # Get pair-wise angular distances.
            ad_delta=[]        
            
            for n in range(len(ang_delta)-1):        
                for i in range(n+1,len(ang_delta),1):                                
                    ad_delta.append(N.pi-N.abs(N.abs(ang_delta[n]-ang_delta[i])-N.pi))
                    
            
            ad_delta=(2.0/(spike_number*(spike_number-1)))*N.sum(ad_delta)                      
            #compute mrl if more than 50 spikes were found.
            
            
            if len(ang_delta)>51 and p_delta<0.05:
                r_total=[]
                for n in range(1000):
                    local=N.random.choice(ang_delta,size=50,replace=False)
                    
                    r=data_tools.mrl(local)
                    r_total.append(r)
                                
                mrl=N.mean(r_total)
            else:
                mrl=N.nan
            
            
            circmean=st.circmean(ang_delta,low=-N.pi,high=N.pi)
            if p_delta<0.05:
                circ=circmean
                ppc_delta=(N.pi-(2*ad_delta))/N.pi
            else:
                circ=N.nan
                ppc_delta=N.nan
            
            result={'circ':circ,'ang':ang_delta,'ppc':ppc_delta,'p':p_delta,'mrl':mrl,'used indices':used_indices}
    return result



def spike_oscillation_coupling_single_batch_starter(low=1,high=5,Fs=30000,thr=1.5,restrict_to=False):
    
    filename=tkFileDialog.askopenfilename(title='unit list timestamp batch')
    directory, fname = os.path.split(filename)
    su_batch=op.open_helper(filename)  
    filename2=tkFileDialog.askopenfilename(title='raw data channel list batch')
    directory2, fname2 = os.path.split(filename2)
    raw_batch=op.open_helper(filename2)
 
    grand_result=[]
    #raw=data_tools.filterData(raw,low=low,high=high,Fs=Fs)
    
    #print "Computing envelope"
    #env=data_tools.envelope(raw)
    for n in range(len(su_batch)):       
  
        print "Recording set %s of " %n, len(su_batch)-1
        os.chdir(directory)                
        su=op.open_helper(su_batch[n])
        
        os.chdir(directory2)
        raw=op.open_helper(raw_batch[n])
        raw=data_tools.filterData(raw,low=low,high=high,Fs=Fs)
        
        print "Computing envelope"
        env=data_tools.envelope(raw)
                    
        
        res=spike_oscillation_coupling_single_above_thres(su,raw,env,thres=thr,low=low,high=high,Fs=Fs)
        
        grand_result.append(res)
        
    return grand_result

def spike_oscillation_coupling_all_animals(condition,Fs=30000,thr=1.5,restrict_to=False,savecondition=True):
    
    '''
    need a batch list for path 'raw data channel list batch'
    need a batch list for path 'all unti timestamp file'
    adapt the path for fig_to_save and file_to_save
    a path for final grand_result dictionary
    '''
 
    dir1=tkFileDialog.askopenfilename(title='raw data channel list batch')
    directory, fname = os.path.split(dir1)
    path_raw_batch=op.open_helper(dir1)
    dir2=tkFileDialog.askopenfilename(title="path list for all unit timestamp file")
    directory2, fname2 = os.path.split(dir2)
    path_su_batch=op.open_helper(dir2) 
    
    Path_raw_exist,Path_su_exist,Path_equal_length=[],[],[]
    for n in range(len(path_raw_batch)):
        print 'raw channel list path %s' %path_raw_batch[n]
        directory3, fname3 = os.path.split(path_raw_batch[n])
        raw_batch=op.open_helper(path_raw_batch[n])
        os.chdir(directory3)
        isFile=os.path.isfile(fname3)
        Path_raw_exist.append(isFile)
        print isFile
        print 'unit path %s' %path_su_batch[n]
        isFileunit=os.path.isfile(path_su_batch[n])
        Path_su_exist.append(isFileunit)
        su_batch=op.open_helper(path_su_batch[n])
        print isFileunit
        if len(raw_batch)==len(su_batch):
            Path_equal_length.append(True)
        else:
            print 'length of raw channels & batch: %s' %False
            Path_equal_length.append(False)
    
    if N.array(Path_raw_exist).all()==True and N.array(Path_su_exist).all()==True and N.array(Path_equal_length).all()==True:
        sumtotal=0
        for n in range(len(Path_su_exist)):
            su_batch=op.open_helper(path_su_batch[n])
            sumtotal=sumtotal+len(su_batch)
        
        grand_result_all = []
        subsum=0
        for n in range(len(path_raw_batch)):
            print 'Raw path %s' %path_raw_batch[n]
            print 'unit path %s' %path_su_batch[n]
            raw_batch=op.open_helper(path_raw_batch[n])
            directory3, fname3 = os.path.split(path_raw_batch[n])
            su_batch=op.open_helper(path_su_batch[n])
            print 'Recording from %s' %subsum
            subsum=subsum+len(su_batch)
            print 'to %s of %s' %(subsum,sumtotal)
            directory2, fname2 = os.path.split(path_su_batch[n])
            c=directory2.rsplit('_',-1)[-1]
            c=c.rsplit('_',-1)[-1]
            d=directory2.rsplit('/',4)[-3]
            if savecondition==True:
                os.chdir(directory2)
                if os.path.isdir(condition)==False:
                    os.mkdir(condition)
                fig_to_save='%s/%s/%s_%s_%s' %(directory2,condition,d,c,condition)
            else:
                fig_to_save='%s/%s_%s_%s' %(directory2,d,c,condition)
            os.chdir(directory2)
            if os.path.isdir('Data saved')==False:
                os.mkdir('Data saved')
            pl.ioff()
            file_to_save='%s/%s/%s_%s_%s' %(directory2,'Data saved',d,c,condition)
            fig_to_save_theta=''.join((fig_to_save,'_theta'))
            grand_result_theta=spike_oscillation_coupling_single_batch_for_all_animals(su_batch,raw_batch,directory2,directory3,low=4,high=12,Fs=Fs,thr=thr,restrict_to=restrict_to)
            spike_oscillation_coupling_plot_for_all_animals(grand_result_theta,su_batch,directory2,fig_to_save_theta,low=4,high=12,bins=18)
            N.save('%s_spike_theta_coupling'%file_to_save,grand_result_theta)
            pl.close('all')
            grand_result_gammalow=spike_oscillation_coupling_single_batch_for_all_animals(su_batch,raw_batch,directory2,directory3,low=30,high=60,Fs=Fs,thr=thr,restrict_to=restrict_to)
            fig_to_save_gammalow=''.join((fig_to_save,'_gammalow'))
            spike_oscillation_coupling_plot_for_all_animals(grand_result_gammalow,su_batch,directory2,fig_to_save_gammalow,low=30,high=60,bins=18)
            N.save('%s_spike_low_gamma_coupling'%file_to_save,grand_result_gammalow)
            pl.close('all')
            grand_result_gammamid=spike_oscillation_coupling_single_batch_for_all_animals(su_batch,raw_batch,directory2,directory3,low=60,high=90,Fs=Fs,thr=thr,restrict_to=restrict_to)
            fig_to_save_gammamid=''.join((fig_to_save,'_gammamid'))
            spike_oscillation_coupling_plot_for_all_animals(grand_result_gammamid,su_batch,directory2,fig_to_save_gammamid,low=60,high=90,bins=18)
            N.save('%s_spike_mid_gamma_coupling'%file_to_save,grand_result_gammamid)
            pl.close('all')
            grand_result_gammahigh=spike_oscillation_coupling_single_batch_for_all_animals(su_batch,raw_batch,directory2,directory3,low=90,high=120,Fs=Fs,thr=thr,restrict_to=restrict_to)
            fig_to_save_gammahigh=''.join((fig_to_save,'_gammahigh'))
            spike_oscillation_coupling_plot_for_all_animals(grand_result_gammahigh,su_batch,directory2,fig_to_save_gammahigh,low=90,high=120,bins=18)
            N.save('%s_spike_high_gamma_coupling'%file_to_save,grand_result_gammahigh)
            pl.close('all')
            
def spike_oscillation_coupling_single_batch_for_all_animals(su_batch,raw_batch,directory,directory2,low=1,high=5,Fs=30000,thr=1.5,restrict_to=False):
    
    '''
    filename=tkFileDialog.askopenfilename(title='unit list timestamp batch')
    directory, fname = os.path.split(filename)
    su_batch=op.open_helper(filename)  
    filename2=tkFileDialog.askopenfilename(title='raw data channel list batch')
    directory2, fname2 = os.path.split(filename2)
    raw_batch=op.open_helper(filename2)
    '''
 
    grand_result=[]
    #raw=data_tools.filterData(raw,low=low,high=high,Fs=Fs)
    
    #print "Computing envelope"
    #env=data_tools.envelope(raw)
    for n in range(len(su_batch)):       
  
        print "Recording set %s of " %n, len(su_batch)-1
        os.chdir(directory)                
        su=op.open_helper(su_batch[n])
        
        os.chdir(directory2)
        raw=op.open_helper(raw_batch[n])
        raw=data_tools.filterData(raw,low=low,high=high,Fs=Fs)
        
        print "Computing envelope"
        env=data_tools.envelope(raw)
                    
        
        res=spike_oscillation_coupling_single_above_thres(su,raw,env,thres=thr,low=low,high=high,Fs=Fs)
        
        grand_result.append(res)
        
    return grand_result


def spike_oscillation_coupling_plot_all(low=30,high=120,bins=18):
    
    filename=tkFileDialog.askopenfilename(title='spike_oscillation_coupling_single_batch_starter_all')
    directory, fname = os.path.split(filename)
    grand_result=op.open_helper(filename)
    filename2=tkFileDialog.askopenfilename(title='unit list timestamp batch')
    directory2, fname2 = os.path.split(filename2)
    su_batch=op.open_helper(filename2)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    os.chdir(directory)
    fig,axs=pl.subplots((int(len(grand_result)/4)+(len(grand_result)%4>0)),4,sharex=True,figsize=[24,18])
    fig.subplots_adjust(hspace=0.4)
    fig.text(0.5, 0.04, 'Phase of LFP frequency between %s and %s Hz' %(low,high), ha='center',fontsize=15)
    fig.text(0.08, 0.5, 'averaged spike No.', va='center', rotation='vertical',fontsize=15)
    axs=axs.ravel()
    
    for n in range(len(grand_result)):
        if N.isnan(N.min(grand_result[n]['ang']))==True:
            axs[n].plot(grand_result[n]['ang'])
            axs[n].set_title('NAN_unit_%s'%su_batch[n])
        else:
            #axs[n].hist(grand_result[n]['ang'],bins=bins,color='#0504aa',alpha=0.7, rwidth=0.85)
            axs[n].hist(grand_result[n]['ang'],bins=bins,color='#0504aa',alpha=0.7, rwidth=0.85,label='circ: %s'%grand_result[n]['circ'])
            #axs[n].plot(N.arange(-N.pi,N.pi,0.1),3*N.cos(N.arange(-N.pi,N.pi,0.1))+10)
            axs[n].set_title('unit_%s'%su_batch[n])
            axs[n].legend(loc="best")
            axs[n].set_xlabel('mrl: %s  ppc: %s and p: %s ' %(grand_result[n]['mrl'],grand_result[n]['ppc'],grand_result[n]['p']))
            pl.plot(N.arange(-N.pi,N.pi,0.1),3*N.cos(N.arange(-N.pi,N.pi,0.1)))
      

def spike_oscillation_coupling_plot_for_all_animals(grand_result,su_batch,directory,fig_to_save,low=30,high=120,bins=18):
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    os.chdir(directory)
    fig,axs=pl.subplots((int(len(grand_result)/4)+(len(grand_result)%4>0)),4,sharex=True,figsize=[24,18])
    fig.subplots_adjust(hspace=0.4)
    fig.text(0.5, 0.04, 'Phase of LFP frequency between %s and %s Hz' %(low,high), ha='center',fontsize=15)
    fig.text(0.08, 0.5, 'averaged spike No.', va='center', rotation='vertical',fontsize=15)
    axs=axs.ravel()
    
    for n in range(len(grand_result)):
        if N.isnan(N.min(grand_result[n]['ang']))==True:
            axs[n].plot(grand_result[n]['ang'])
            axs[n].set_title('NAN_unit_%s'%su_batch[n])
        else:
            #axs[n].hist(grand_result[n]['ang'],bins=bins,color='#0504aa',alpha=0.7, rwidth=0.85)
            axs[n].hist(grand_result[n]['ang'],bins=bins,color='#0504aa',alpha=0.7, rwidth=0.85,label='circ: %s'%grand_result[n]['circ'])
            #axs[n].plot(N.arange(-N.pi,N.pi,0.1),3*N.cos(N.arange(-N.pi,N.pi,0.1))+10)
            axs[n].set_title('unit_%s'%su_batch[n])
            axs[n].legend(loc="best")
            #axs[n].set_xlabel('mrl: %s  ppc: %s and p: %s ' %(grand_result[n]['mrl'],grand_result[n]['ppc'],grand_result[n]['p']))
            pl.plot(N.arange(-N.pi,N.pi,0.1),3*N.cos(N.arange(-N.pi,N.pi,0.1)))
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
            
    for n in range(len(grand_result)):
        if N.isnan(N.min(grand_result[n]['ang']))==True:
            fig1,axes=pl.subplots(figsize=(8,6))
            axes.plot(grand_result[n]['ang'])
            axes.set_title('NAN_unit_%s'%su_batch[n])
        else:
            result = grand_result[n]
            #getting setup to export text correctly
            fig1,axes=pl.subplots(figsize=(8,6))
            axes.hist(result['ang'],bins=bins,color='#0504aa',alpha=0.7, rwidth=0.85,label='circ: %s'%result['circ'])
            axes.set_title('unit_%s'%su_batch[n],fontsize=15)
            axes.set_xlabel('Phase of LFP frequency between %s and %s Hz' %(low,high),fontsize=15)
            axes.set_ylabel('Averaged spike No.',fontsize=15)
            axes.legend(loc="upper right",fontsize=15)
            #pl.text(1, 18, 'mrl: %s  ppc: %s and p: %s ' %(grand_result[n]['mrl'],grand_result[n]['ppc'],grand_result[n]['p']), fontsize=15)
            axes.text(1.2, 18, 'mrl: %s' %grand_result[n]['mrl'], fontsize=15)
            axes.text(1.2, 16, 'ppc: %s' %grand_result[n]['ppc'], fontsize=15)
            axes.text(1.2, 15, 'p: %s ' %grand_result[n]['p'], fontsize=15) 
        fig1.savefig('%s_%s_spike_oscillation_coupling.png' %(fig_to_save,su_batch[n]))
        fig1.savefig('%s__%s_spike_oscillation_coupling.eps' %(fig_to_save,su_batch[n]),transparent=True)
        pl.close('fig1')
    
    fig.savefig('%s_spike_oscillation_coupling.png' %fig_to_save)
    fig.savefig('%s_spike_oscillation_coupling.eps' %fig_to_save,transparent=True)

        
def spike_oscillation_coupling_hist_single_unit(n,low=4,high=12,bins=18):
    
    filename=tkFileDialog.askopenfilename(title='spike_oscillation_coupling_single_batch_starter_all.npy')
    directory, fname = os.path.split(filename)
    grand_result=op.open_helper(filename)
    filename2=tkFileDialog.askopenfilename(title='single unit timestamp list')
    directory2, fname2 = os.path.split(filename2)
    su=op.open_helper(filename2)
    
    result = grand_result[n]
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    pl.figure(figsize=(8,6))
    pl.hist(result['ang'],bins=bins,color='#0504aa',alpha=0.7, rwidth=0.85,label='circ: %s'%result['circ'])
    pl.title('unit_%s'%filename2,fontsize=15)
    pl.xlabel('Phase of LFP frequency between %s and %s Hz' %(low,high),fontsize=15)
    pl.ylabel('Averaged spike No.',fontsize=15)
    pl.legend(loc="upper right",fontsize=15)
    #pl.text(1, 18, 'mrl: %s  ppc: %s and p: %s ' %(grand_result[n]['mrl'],grand_result[n]['ppc'],grand_result[n]['p']), fontsize=15)
    pl.text(1.2, 18, 'mrl: %s' %grand_result[n]['mrl'], fontsize=15)
    pl.text(1.2, 17, 'ppc: %s' %grand_result[n]['ppc'], fontsize=15)
    pl.text(1.2, 16, 'p: %s ' %grand_result[n]['p'], fontsize=15)   
    

def spike_oscillation_coupling_hist_single_unit_theta_gamma(n,bins=18):
    
    filename=tkFileDialog.askopenfilename(title='spike_oscillation_coupling_single_batch_starter_all_theta.npy')
    directory, fname = os.path.split(filename)
    filename2=tkFileDialog.askopenfilename(title='single unit timestamp list')
    directory2, fname2 = os.path.split(filename2)
    su=op.open_helper(filename2)
    
    os.chdir(directory)
    grand_result1 = N.load('spike_oscillation_coupling_single_batch_starter_all_theta.npy')
    grand_result3 = N.load('spike_oscillation_coupling_single_batch_starter_all_gamma.npy')
    
    result1 = grand_result1[n]
    result3 = grand_result3[n]
    low1=4
    low2=30
    high1=12
    high2=120
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    f,(ax1,ax2)=pl.subplots(2,1,figsize=[8,6])
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    ax1.hist(result1['ang'],bins=bins,color='#0504aa',alpha=0.7, rwidth=0.85,label='circ: %s'%result1['circ'])
    ax1.grid(axis='y', alpha=0.75) 
    ax1.set_xlabel('mrl: %s  ppc: %s and p: %s ' %(grand_result1[n]['mrl'],grand_result1[n]['ppc'],grand_result1[n]['p']),fontsize=15)
    ax1.set_ylabel('Averaged spike No.',fontsize=15)
    ax1.set_title('Phase of LFP frequency between %s and %s Hz_%s' %(low1,high1,fname2),fontsize=15)
    ax1.legend(loc="upper right",fontsize=15)
    #ax1.text(1.2, 18, 'mrl: %s' %grand_result1[n]['mrl'], fontsize=15)
    #ax1.text(1.2, 17, 'ppc: %s' %grand_result1[n]['ppc'], fontsize=15)
    #ax1.text(1.2, 16, 'p: %s ' %grand_result1[n]['p'], fontsize=15) 
    
    ax2.hist(result3['ang'],bins=bins,color='#0504aa',alpha=0.7, rwidth=0.85,label='circ: %s'%result3['circ'])
    ax2.grid(axis='y', alpha=0.75)
    ax2.set_xlabel('mrl: %s  ppc: %s and p: %s ' %(grand_result3[n]['mrl'],grand_result3[n]['ppc'],grand_result3[n]['p']),fontsize=15)
    ax2.set_ylabel('Averaged spike No.',fontsize=15)
    ax2.set_title('Phase of LFP frequency between %s and %s Hz_%s' %(low2,high2,fname2),fontsize=15)
    ax2.legend(loc="upper right",fontsize=15)
    #ax2.text(1.2, 18, 'mrl: %s' %grand_result3[n]['mrl'], fontsize=15)
    #ax2.text(1.2, 17, 'ppc: %s' %grand_result3[n]['ppc'], fontsize=15)
    #ax2.text(1.2, 16, 'p: %s ' %grand_result3[n]['p'], fontsize=15) 
    
def spike_oscillation_coupling_polar_single_unit_theta_gamma(n,bins=18):
    
    filename=tkFileDialog.askopenfilename(title='spike_oscillation_coupling_single_batch_starter_all_theta.npy')
    directory, fname = os.path.split(filename)
    filename2=tkFileDialog.askopenfilename(title='single unit timestamp list')
    directory2, fname2 = os.path.split(filename2)
    su=op.open_helper(filename2)
    
    os.chdir(directory)
    grand_result1 = N.load('spike_oscillation_coupling_single_batch_starter_all_theta.npy')
    grand_result3 = N.load('spike_oscillation_coupling_single_batch_starter_all_gamma.npy')
    
    result1 = grand_result1[n]
    result3 = grand_result3[n]
    low1=4
    low2=30
    high1=12
    high2=120
    
    hist1,bin_edges1 = N.histogram(result1['ang'],bins=bins)
    hist3,bin_edges3 = N.histogram(result3['ang'],bins=bins)
    
    bins_middle1,bins_middle3 = [],[]
    for n in range(len(bin_edges1)-1):
        bins_middle1.append((bin_edges1[n+1]-bin_edges1[n])/2+bin_edges1[n])
        bins_middle3.append((bin_edges3[n+1]-bin_edges3[n])/2+bin_edges3[n])
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    f,(ax1,ax2)=pl.subplots(1,2,figsize=[8,6],subplot_kw=dict(polar=True))
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    ax1.bar(bins_middle1,hist1,width=2*N.pi/bins,bottom=0,color='k')
    #ax1.bar(bins_middle1,hist1,width=2*N.pi/bins,bottom=0,color='#0504aa')
    #ax1.xaxis.set_major_formatter()
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    #ax1.set_xlabel('mrl: %s  ppc: %s and p: %s ' %(grand_result1[n]['mrl'],grand_result1[n]['ppc'],grand_result1[n]['p']),fontsize=15)
    #ax1.set_ylabel('Averaged spike No.',fontsize=15)
    ax1.set_title('spike-phase correlation of unit_%s' %fname2, va='bottom',fontsize='large')
    ax1.set_xlabel(r"$\theta$ phase (between %s and %s Hz)" %(low1,high1))
    
    
    ax2.bar(bins_middle3,hist3,width=2*N.pi/bins,bottom=0,color='k')
    #ax2.bar(bins_middle3,hist3,width=2*N.pi/bins,bottom=0,color='#0504aa')
    ax2.set_title('spike-phase correlation of unit_%s' %fname2, va='bottom',fontsize='large')
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_xlabel(r"$\gamma$ phase (between %s and %s Hz)" %(low2,high2))

def get_values_from_object(variablename):
    
    filename=tkFileDialog.askopenfilename(title='spike_oscillation_coupling_single_batch_starter_all_theta.npy')
    directory, fname = os.path.split(filename)
    os.chdir(directory)
    grand_result=op.open_helper(fname)
    
    result=[]
    uname=fname.split('_')[0]
    fn=fname.split('_')[1]
    c=fname.split('_')[2]
    for n in range(len(grand_result)):
        result.append(grand_result[n][variablename])
    N.savetxt('%s_%s_%s_%s.txt'%(uname,fn,c,variablename),result)

def get_values_from_object_for_batch(grand_result,directory,fname,variablename):
    
    os.chdir(directory)
    result=[]
    uname=fname.split('_')[0]
    fn=fname.split('_')[1]
    c=fname.split('_')[2]
    d=fname.split('.')[0]
    d=d.split('_')[-2]
    e=fname.split('_')[-3]
    for n in range(len(grand_result)):
        result.append(grand_result[n][variablename])
    N.savetxt('%s_%s_%s_%s_%s_%s.txt'%(uname,fn,c,d,e,variablename),result)

def get_values_from_object_batch():
    
    filename=tkFileDialog.askopenfilename(title='spike_oscillation_coupling_single_batch_starter_all_theta.npy')
    directory, fname = os.path.split(filename)
    os.chdir(directory)
    #grand_result=op.open_helper(fname)
    
    for file in os.listdir(directory):
        if file.endswith("spike_osc_corr_spike_mid_gamma_coupling.npy"):
            grand_result=op.open_helper(file)
            get_values_from_object_for_batch(grand_result,directory,file,'ppc')
            #get_values_from_object_for_batch(grand_result,directory,fname,'ang')
            get_values_from_object_for_batch(grand_result,directory,file,'mrl')
            get_values_from_object_for_batch(grand_result,directory,file,'circ')
            get_values_from_object_for_batch(grand_result,directory,file,'p')
    for file in os.listdir(directory):
        if file.endswith("spike_osc_corr_spike_low_gamma_coupling.npy"):
            grand_result=op.open_helper(file)
            get_values_from_object_for_batch(grand_result,directory,file,'ppc')
            #get_values_from_object_for_batch(grand_result,directory,fname,'ang')
            get_values_from_object_for_batch(grand_result,directory,file,'mrl')
            get_values_from_object_for_batch(grand_result,directory,file,'circ')
            get_values_from_object_for_batch(grand_result,directory,file,'p')
    for file in os.listdir(directory):
        if file.endswith("spike_osc_corr_spike_high_gamma_coupling.npy"):
            grand_result=op.open_helper(file)
            get_values_from_object_for_batch(grand_result,directory,file,'ppc')
            #get_values_from_object_for_batch(grand_result,directory,fname,'ang')
            get_values_from_object_for_batch(grand_result,directory,file,'mrl')
            get_values_from_object_for_batch(grand_result,directory,file,'circ')
            get_values_from_object_for_batch(grand_result,directory,file,'p')
    for file in os.listdir(directory):
        if file.endswith("spike_osc_corr_spike_theta_coupling.npy"):
            grand_result=op.open_helper(file)
            get_values_from_object_for_batch(grand_result,directory,file,'ppc')
            #get_values_from_object_for_batch(grand_result,directory,fname,'ang')
            get_values_from_object_for_batch(grand_result,directory,file,'mrl')
            get_values_from_object_for_batch(grand_result,directory,file,'circ')
            get_values_from_object_for_batch(grand_result,directory,file,'p')
    
    
#present all channels LFP theta & gamma
def display_oscillations_all_channels(low_t=4,high_t=12,low_g=30,high_g=120,startt=0,Fs=30000,column=2,figwidth=8,figlength=12,filterData=True,plotData=True,printDataSingle=True,printData=True):
    ''' Display filtered LFP data (theta &gamma) for all channels on one shank at the same time
    '''
    total_t=(startt+30)*Fs
    total_g=(startt+30)*Fs
    #total_g=(startt+1)*Fs/10
    
    filename2=tkFileDialog.askopenfilename(title='channel list of shank x batch')
    directory2, fname2 = os.path.split(filename2)
    raw_channel_batch=op.open_helper(filename2)
    
    
    
    grand_result_theta,grand_result_gamma=[],[]
    intf_batch,first_peak_batch,first_amp_batch,maxpoint_batch,second_amp_batch,minpoint_batch=[],[],[],[],[],[]
    dur_batch,asym_batch=[],[]
    for i in range(len(raw_channel_batch)):
        print "Recording set %s of " %i, len(raw_channel_batch)-1
        os.chdir(directory2)                
        raw=op.open_helper(raw_channel_batch[i])
        raw_t=data_tools.filterData(raw,low=low_t,high=high_t,Fs=Fs)
        raw_g=data_tools.filterData(raw,low=low_g,high=high_g,Fs=Fs)
        grand_result_theta.append(raw_t)
        grand_result_gamma.append(raw_g)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if plotData==True:
        fig,axs=pl.subplots((int(len(grand_result_theta)/column)+(len(grand_result_theta)%column>0)),column,sharex=True,sharey=True,figsize=[figwidth,figlength])
        fig.subplots_adjust(hspace=0)
        fig.text(0.5, 0.04, 'theta', ha='center',fontsize=15)
        axs=axs.ravel()
        for m in range(len(grand_result_theta)):
            #axs[m].plot(grand_result_theta[m],'k')
            axs[m].plot(grand_result_theta[m][startt*Fs:total_t])
            
        
        fig,axs=pl.subplots((int(len(grand_result_gamma)/column)+(len(grand_result_gamma)%column>0)),column,sharex=True,sharey=True,figsize=[figwidth,figlength])
        fig.subplots_adjust(hspace=0)
        fig.text(0.5, 0.04, 'gamma', ha='center',fontsize=15)
        axs=axs.ravel()
        for m in range(len(grand_result_gamma)):
            axs[m].plot(grand_result_gamma[m][startt*Fs:total_t])

        
    return grand_result_theta,grand_result_gamma


#new version moved to onkey function
#pur algorithm, some detections are false positive    
def dentate_spike_detection(immo_thres=1,thres=2,low=1,high=1000,Fs=30000,mid=5,end=15,start=-5,plotData=True):
    
    '''
    immo_thres: threshold of speed for immobility
    thres: for detecting dentate spike event, baseline+thres*SD
    low: low frequency for bandpass filterung
    high: high frequency for bandpass filterung
    mid: middle of the time window to detect the maximum exceeding thres
    start, end: start and end of the time window to detect the maximum exceeding thres    
    '''
    
    #to get the directory where variables are saved
    dir1=tkFileDialog.askopenfilename(title='position data directory')          
    directory, fname = os.path.split(dir1)
    dir1=op.open_helper(dir1)
    
    os.chdir(directory)
    posnew_immo = N.load('estimate speed from position_corrected posnew_immo.npy')
    speed_immo = N.load('estimate speed from position_speed_immo.npy')
    ind_immo = N.load('estimate speed from position_ind_immo.npy')
    spike_train_immo = N.load('estimate speed from position_spike_train_immo.npy')
    
    #posnew_immo,speed_immo,ind_immo=speed.create_unit_place_field_speed_corr_immobility(thres=immo_thres)
    
    
    filename=tkFileDialog.askopenfilename(title='raw channel data .continuous')
    raw=op.open_helper(filename)

    
    raw_immo = N.take(raw,ind_immo)
    raw_immo_filt = data_tools.filterData(raw_immo,low=low,high=high)
    thr=N.mean(raw_immo)+thres*N.std(raw_immo)
    
    #detect dentate spike according to Nokia et al. 2017 & threshold according to Szabo et al. 2017
    ind_above_thr,ind_DS,ind_peak_DS,raw_DS,ind_peak_above_thr,res,ind_DS_range=[],[],[],[],[],[],[]
    for i in range(len(raw_immo))[1:]:        
        #find the first index exceeing threshold positive deflection
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
            if N.amax(raw_immo_filt[i+mid*Fs/1000:i+end*Fs/1000])>thr and abs(N.amax(raw_immo_filt[i+start*Fs/1000:i+mid*Fs/1000])-N.amax(raw_immo[i+mid*Fs/1000+1:i+end*Fs/1000]))>thr:
                ind_DS.append(i)
                result=N.ravel(N.argmax(raw_immo_filt[i+start*Fs/1000:i+end*Fs/1000]))
                result = i+result[0]
                ind_peak_DS.append(result)
                raw_DS.append(raw_immo_filt[result-20*Fs/1000:result+20*Fs/1000])
                ind_DS_range.append(range(result-20*Fs/1000,result+20*Fs/1000))
    if len(ind_peak_DS)==0:
        print 'no dentate spike detected by threshold of %s X SD' %thres
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    if len(ind_peak_DS)>0 and plotData == True:
        fig,axs=pl.subplots((int(len(ind_peak_DS)/4)+(len(ind_peak_DS)%4>0)),4,sharex=True)
        fig.subplots_adjust(hspace=0.4)
        axs=axs.ravel()
        for j in range(len(ind_peak_DS)):
            axs[j].plot(raw_DS[j])
            fig2=pl.figure(2)
            fig2 = pl.plot(raw_DS[j])
    
    return ind_peak_DS, raw_DS, ind_DS_range

def delete_manually_non_DS(index):
    
    for i in range(len(index)):
        ind_peak_DS = N.delete(ind_peak_DS,index[i])
        raw_DS = N.delete(raw_DS,index[i],0)
        ind_DS_range = N.delete(ind_DS_range,index[i])
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig,axs=pl.subplots((int(len(ind_peak_DS)/4)+(len(ind_peak_DS)%4>0)),4,sharex=True)
    fig.subplots_adjust(hspace=0.4)
    axs=axs.ravel()
    for j in range(len(ind_peak_DS)):
        axs[j].plot(raw_DS[j])
        fig2=pl.figure(2)
        fig2 = pl.plot(raw_DS[j])
    
    return ind_peak_DS_new, raw_DS, ind_DS_range

def dentate_spike_all_on_one_shank(immo_thres=0.5,thres=2,low=1,high=600,Fs=30000,mid=5,end=15,start=-5,plotData=True):
    
    filename2=tkFileDialog.askopenfilename(title='channel list of individual shank .batch')
    directory2, fname2 = os.path.split(filename2)
    raw_batch=op.open_helper(filename2)
    
    ind_peak_DS, raw_DS, ind_DS_range =dentate_spike_detection(immo_thres=immo_thres,thres=thres,low=low,high=high,Fs=Fs,mid=mid,end=end,start=start,plotData=True)
    grand_result,raw_DS=[],[]
    for n in range(len(raw_batch)):
        print "Recording set %s of " %n, len(raw_batch)-1       
        os.chdir(directory2)
        raw=op.open_helper(raw_batch[n])
        raw=data_tools.filterData(raw,low=low,high=high,Fs=Fs)
        print "Channel %s filtered" %raw_batch[n]
        for l in range(len(ind_peak_DS)):
            res=raw[ind_peak_DS[l]-50*Fs/1000:ind_peak_DS[l]+50*Fs/1000]
            raw_DS.append(res)
            result={'filtered_raw_DS':raw_DS}
            grand_result.append(result)
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    for i in range(len(ind_peak_DS)):
        fig[i],axs=pl.subplots(len(grand_result),1,sharex=True)
        axs=axs.ravel()
        for j in range(len(grand_result)):
            raw_DS.append(grand_result[j]['filtered_raw_DS'][i])
            axs[j].plot(raw_DS[j])

    

def spike_DS_coupling_single(immo_thres=0.5,thres=2,low=1,high=600,Fs=30000,mid=5,end=15,start=-5,columns=8,plotData=True):
    
    
    filename=tkFileDialog.askopenfilename(title='single unit timestamp .txt')
    #directory, fname = os.path.split(filename)
    su=op.open_helper(filename) 
    
    ind_peak_DS, raw_DS, ind_DS_range =dentate_spike_detection(immo_thres=immo_thres,thres=thres,low=low,high=high,Fs=Fs,mid=mid,end=end,start=start,plotData=True)
    
    su_DS,ind_pre_range,su_pre,spike_numb_DS,spike_numb_pre=[],[],[],[],[]
    ind_pre_range=N.subtract(ind_DS_range,40*Fs/30000+1)
    for m in range(len(ind_DS_range)):
        res=N.intersect1d(su,ind_DS_range[m])
        spike_numb_DS.append(len(res))
        su_DS.append(res)
        res_pre=N.intersect1d(su,ind_pre_range[m])
        spike_numb_pre.append(len(res_pre))
        su_pre.append(res_pre)
    
    statistic, pvalue = stats.ttest_rel(spike_numb_DS,spike_numb_pre)
    if pvalue >0.05 or N.isnan(pvalue)==True:
        print 'no siginificance'
    else:
        print 'dentate spike correlated, significant',pvalue
        #getting setup to export text correctly
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        if plotData==True:
           fig,axs=pl.subplots((int(len(ind_peak_DS)/columns)+(len(ind_peak_DS)%columns>0)),columns,sharex=True)
           fig.subplots_adjust(hspace=0.4)
           axs=axs.ravel()
           for j in range(len(ind_peak_DS)):
               axs[j].plot(raw_DS[j])
               axs[j].eventplot(su_DS[j],lineoffsets=N.amax(raw_DS[j]+2))
               fig2=pl.figure(2)
               fig2=pl.plot(raw_DS[j])
               fig2=pl.eventplot(su_DS)
    
    return spike_numb_DS,spike_numb_pre,su_DS,su_pre       


def spike_DS_coupling_all_batch(immo_thres=0.5,thres=2,low=1,high=600,Fs=30000,mid=5,end=15,start=-5,plotData=True):
    
    filename=tkFileDialog.askopenfilename(title='unit list timestamp batch')
    directory, fname = os.path.split(filename)
    su_batch=op.open_helper(filename)  
    filename2=tkFileDialog.askopenfilename(title='raw data channel list batch')
    directory2, fname2 = os.path.split(filename2)
    raw_batch=op.open_helper(filename2)
    
    grand_result_DS=[]
    
    for n in range(len(su_batch)):
       print "Recording set %s of " %n, len(su_batch)-1
       os.chdir(directory)                
       su=op.open_helper(su_batch[n])
       
       os.chdir(directory2)
       raw=op.open_helper(raw_batch[n])
       raw=data_tools.filterData(raw,low=low,high=high,Fs=Fs)
       print 'unit %s selected' %fname
       print "Channel %s selected" %su_batch[n]
       ind_peak_DS,raw_DS,ind_DS_range= dentate_spike_detection(raw,immo_thres=immo_thres,thres=thres,low=low,high=high,Fs=Fs,mid=mid,end=end,start=start,plotData=plotData)
       
       su_DS,ind_pre_range=[],[]
       ind_pre_range=N.subtract(ind_DS_range,40*Fs/30000)
       for m in range(len(ind_DS_range)):
           res=N.intersect1d(su,ind_DS_range[m])
           su_DS.append(res)
           
       result = {'ind_peak_DS':ind_peak_DS,'raw_DS':raw_DS,'ind_DS_range':ind_DS_range,'su_DS':su_DS}
       grand_result_DS.append(result)
            
            

    
    
   
    

        
        
            
            
            

