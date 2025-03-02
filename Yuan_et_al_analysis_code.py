#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:20:09 2024

@author: Mei & Jonas
"""

import js_data_tools as data_tools
import numpy as N
import matplotlib.pyplot as pl
import tkFileDialog
import open_helper as op
import os
from scipy import stats
from scipy import ndimage
import math as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable
import barplot_annotate_brackets as ba
import ANOVARM as AN
import matplotlib
import pandas as pd
import mountain_io as mio
import scipy.interpolate as sci
import neuronpy.util.spiketrain as sp
import scipy.stats as st
import scipy.signal as si
import math


#%% function for spike waveform, autocorrelogram, mean firing rate for unit classification

### function for spike waveform, trough_to_peak_duration, asymmetry index, mean firing rate
su_id=88        #example unit id
before=60       #time window for spike waveform detection before (6 ms) and after (8 ms) peak 
after=80
total=int(before+after)

target_dir="/data/example_raw_for_position_extraction_spike_waveform/200715_#423_re"
os.chdir(target_dir)
firings=mio.readmda('firings.curated.mda')
firings=N.transpose(firings)
raw=op.open_helper('100_CH62.continuous')
    
def display_unit_kinetics_mountainlab_for_batch(su_id,firings,raw,Fs=30000,filtered=True,plotData=True,printData=True):
    ''' Analyse duration of the average waveform in the dominant channel. Waveforms are interpolated 10 times
        Inputs:
        su_max_amp_n: id of channel with max amp in shank batch file
        su_id: number specifying the su to be measured.
        file dialog: open firings.curated.mda
    '''
    before=60
    after=80
    total=int(before+after)
    
    
    su=[]
    for n in range(len(firings)):
        if firings[n][2]==su_id:
            su.append(int(firings[n][1]))
            channel=int((firings[n][0])-1) # Account for 1 and zero-indexing difference.
    
    f_all=float(len(su))/float((float(len(raw))/float(Fs)))
    wf=[]
    if filtered==True:
        filtered=data_tools.filterData(raw,low=300,high=5000,Fs=Fs)
    else:
        filtered=raw
        #filtered=raw[channel]   mei changed
    for n in su:
        #if before<n<after:
        if len(raw)>n+after:
            wf.extend(filtered[n-before:n+after])
    wf=N.mean(N.reshape(wf,[-1,total]),axis=0)
    
    if N.all(N.isnan(wf))==True:
        print 'no spike of u%s in chosen raw file' %su_id
        dur=N.nan
        asym=N.nan
    else:

        # Interpolate the data by a factor of 10.
        xnew=N.linspace(0,total,total*10)       
        f=sci.UnivariateSpline(range(total),wf,s=0)
        intf=f(xnew)     
    
        baseline=N.mean(intf[:20])
        intf=intf-baseline
        minpoint=N.argmin(intf)
        maxpoint=minpoint+N.argmax(intf[minpoint:(minpoint+30*10)]) 
        dur=(((maxpoint-minpoint)/10.0)/float(Fs))*1000
        if minpoint<len(intf)-50:
            for n in range(minpoint,0,-1):
                if intf[n+1]<intf[n]>intf[n-1]:
                    first_peak=n
                    break
                else:
                    first_peak=0
                   
                
            first_amp=intf[first_peak]
            second_amp=intf[maxpoint]
            asym=(second_amp-first_amp)/(second_amp+first_amp)
        else:
            print 'small spike amplitude of u%s in chosen raw file' %su_id
            dur=N.nan
            asym=N.nan
            first_peak=0
            first_amp=0
            second_amp=0
    
        if plotData==True:
            _=pl.plot(intf,"k")
            _=pl.vlines(first_peak,0,first_amp,color="r")
            _=pl.vlines(maxpoint,0,second_amp,color="r")
            _=pl.axvline(x=minpoint,color="b")
            _=pl.title('unit_%s'%(su_id))
            
            
            #pl.savefig('%s_spike_kinetic.png' %(su_id))
            #pl.savefig('%s_spike_kinetic.eps' %(su_id),transparent=True)
            #pl.close()
        if printData==True:
            print ("{}".format(su_id)) + " trough-peak duration: {0:.3f}".format(dur)
            print ("{}".format(su_id)) + " asym: {0:.3f}".format(asym)
        
    return dur,asym,f_all

### function for bursting index for all units on one shank
def burst_index_all_batch(Fs=30000,binwidth=1,maxlag=300,saveData=False,plotData=False,saveFig=False):

    target_dir="/data/example_raw_for_position_extraction_spike_waveform/200721_#386_lin/output/ms3--all_shank2"
    os.chdir(target_dir)
    su_batch=op.open_helper('all_timestamp_SU.batch')
    file_to_save=tkFileDialog.asksaveasfilename(title="Choose target filename") 
    
    grand_result,burst_ind_all=[],[]
    for n in range(len(su_batch)):
        print "Recording set %s of " %n, len(su_batch)-1
        os.chdir(target_dir)
        su=op.open_helper(su_batch[n])
        print "unit %s " %su_batch[n]
        if len(su)<100:
            print 'too few spikes; %s' %len(su)
            grand_result.append({'suid':su_batch[n],'burst_ind':N.nan})
            burst_ind_all.append(N.nan)
            n=n+1
        else:
            burst_ind=burst_ind_for_batch(su,file_to_save,su_batch[n],Fs=Fs,maxlag=maxlag,plotData=plotData,saveFig=saveFig)
            burst_ind_suid = {'suid':su_batch[n],'burst_ind':burst_ind}
            burst_ind_all.append(burst_ind)
            grand_result.append(burst_ind_suid)
            print ( " burst index: {0:.3f}".format(burst_ind))
        
    
    if saveData == True:
        #write the shank number which should be added on the saved filename
        N.save('%s_burst_index_with_suid_batch' %file_to_save,grand_result) 
        N.savetxt('%s_burst_index_without_suid_batch.txt' %file_to_save,burst_ind_all) 
        
    return grand_result

def burst_ind_for_batch(indices,file_to_save,fname,Fs=30000,maxlag=300,plotData=True,saveFig=True):
    '''
    after getting autocorrelogram, calculate the average spike number of 3-5 ms divided by average spike number of 200-300 ms.
    '''
    acorr=autocorrelogram(indices,file_to_save,fname,Fs=Fs,maxlag=maxlag,plotData=plotData,saveFig=saveFig)
    # get average spike number of 3-5 s lag from autocorrelogram
    spike_numb1=float(N.sum(acorr[(len(acorr)/2+3):(len(acorr)/2+5+1)])+N.sum(acorr[(len(acorr)/2-5-1):(len(acorr)/2-3)]))/float(6) 
    # get average spike number of 200-300 s lag from autocorrelogram
    spike_numb2=float(N.sum(acorr[(len(acorr)/2+200):(len(acorr)/2+300)])+N.sum(acorr[(len(acorr)/2-300):(len(acorr)/2-200)]))/float(200) 
    if spike_numb2==0:
        print 'zero spikes with isi between 200 and 300 ms'
        burst_ind=N.nan
    else:
        burst_ind=float(spike_numb1)/float(spike_numb2)
    
    return burst_ind

def autocorrelogram(indices,file_to_save,fname,Fs=30000,maxlag=100,plotData=True,saveFig=False):
    '''
    Computes the autocorrelation function of a spike train.
    Inputs:
    ch1: Raw data trace
    indices: Unit spike indices
    Output:
    Autocorrelation function
    '''

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
        pl.xlim(-maxlag,maxlag)
        uname=fname.split('_')[1]
        pl.title('u%s_autocorrelogram '%uname ,fontsize=18)
    
    if saveFig==True:
        fig.savefig('%s_u%s_autocorrelogram.png' %(file_to_save,uname))
        fig.savefig('%s_u%s_autocorrelogram.eps' %(file_to_save,uname),transparent=True) 
    return acorr

#%% function for optoID -- see 'extract_laser_pulses' function in Summary_Figure_plot.py

#%% scatter plot for trough-to-peak duration and bursting index. -- see 'scatter_plot_3D' function in Summary_Figure_plot.py

#%% function for monosynaptic connection

def test_e_i_synpses_combined_variant8_hollowed_gaussian(Fs=30000,min_spike_number=500,saveData=False):
    '''Testing pairwise correlation of presynaptic pyr onto postsanyptic INs using
    neuronpy's util.spiketrain.filter_correlogram and partially hollowed gaussian kernel
    
    Inclusion criteria:
    minimum spike number.
    
    Input: cells_batch: .batch file with path to all spike train timestamps
    

    Outputs a dictionary with:
    presynaptic id, postsynaptic id, stp, anatomical location of pre and post, tested pairs, connected pairs
    
    '''
    target_dir='/data/example_raw_for_position_extraction_spike_waveform/200721_#386_lin/output'
    os.chdir(target_dir)
    cells_batch=op.open_helper('all_shanks_timestamp_SU.batch')
    
    # Compute xcorrs of all possible pyr-in pairs.
    name_pre_in,name_post_in=[],[]    
    index_pre_in,index_post_in=[],[]
    
    name_pre_ex,name_post_ex=[],[]    
    index_pre_ex,index_post_ex=[],[]
    
    tested,connected_in,connected_ex=0,0,0
    counter=0
    count_pre_in,count_post_in=[],[]
    count_pre_ex,count_post_ex=[],[]
    name_pre_ex,name_post_ex=[],[]    
    index_pre_ex,index_post_ex=[],[]
        
    stp_in,stp_ex=[],[]
    for n in range(len(cells_batch)-1):
        indices1=op.open_helper(cells_batch[n])
        
        for i in range(n+1,len(cells_batch),1):            
            counter+=1
            indices2=op.open_helper(cells_batch[i])
            #print "iteration", counter, "of", comb
            
            
            if len(indices1)>min_spike_number and len(indices2)>min_spike_number:
                tested+=1                
                # Compute stp based on all spikes 
                gau,raw,con,p_in,p_ex,stpi,stpe=check_i_e_synaptic_connection_variant8_hollowed_convolution(indices1,indices2,Fs=Fs,printData=False,plotData=False)            
                if p_in==1:
                    connected_in+=1
                    # Extract anatomical information about the detected pair.
                    print("inh. connection detected")
                    name_pre_in.append(cells_batch[n])
                    name_post_in.append(cells_batch[i])
                    count_pre_in.append(len(indices1))
                    count_post_in.append(len(indices2))
                    index_pre_in.append(n)
                    index_post_in.append(i)                    
                    stp_in.append(stpi)
                if p_ex==1:
                    connected_ex+=1
                    print("exc. connection detected")
                    name_pre_ex.append(cells_batch[n])
                    name_post_ex.append(cells_batch[i])
                    count_pre_ex.append(len(indices1))
                    count_post_ex.append(len(indices2))
                    index_pre_ex.append(n)
                    index_post_ex.append(i)                    
                    stp_ex.append(stpe)
                    
               
    res_in={'stp':stp_in,'count pre':count_pre_in,'tested':tested,'connected':connected_in,'name_pre':name_pre_in,'name_post':name_post_in,'index_pre':index_pre_in,'index_post':index_post_in}
    res_ex={'stp':stp_ex,'count pre':count_pre_ex,'tested':tested,'connected':connected_ex,'name_pre':name_pre_ex,'name_post':name_post_ex,'index_pre':index_pre_ex,'index_post':index_post_ex}
    res={'inhibition':res_in,'excitation':res_ex}
    #filename=target_dir.split('/')[-2]
    #N.save('%s/%s_ei_synaptic_connection'%(directory,filename),res)
    return res

def check_i_e_synaptic_connection_variant8_hollowed_convolution(ind1,ind2,Fs=30000,printData=True,plotData=True):
    ''' Test for synaptic connection following the method described in English et al., 2017 using a
        partially hollowed gaussian kernel

    
        Inputs:
        ind1,ind2: spike timestamps of the putative pre and postsynaptic neurons
    '''
        
    # Convert spike indices to ms.
    msFs=Fs/1000.0
       
    raw=sp.filter_correlogram(ind2,ind1,dt=0.4*msFs,shift=50*msFs)
    raw=raw[0]
    
    # Create hollowed window with 10 ms SD and 0.6 hollow fraction.
    gau=si.gaussian(len(raw),std=25)
    gau=gau/float(N.sum(gau)) # Normalize gau to sum 1.
    gau[len(gau)/2]=N.max(gau)*0.4 # Insert the hollow fraction of 0.6.
    # Extend both edges of raw following Start%Abbes 2009 to avoid edge effects.
    temp=[]
    for n in range(50,0,-1):
        temp.append(raw[n])
    temp.extend(raw)
    for n in range(200,251,1):
        temp.append(raw[n])
        
    con=N.convolve(gau,temp,mode='same') # convolve
    final_con=[]
    for n in range(50,301,1): # Remove edges to retain same duration as raw.
        final_con.append(con[n])
    con=final_con  
       
    # Using the poisson distribution to estimate significant coupling.
    # Target range: raw[128:136], corresponds to 1.2 - 4 ms following Fernadnez-Ruiz,    
    
    sign_tester_in,sign_tester_ex=[],[]
    for n in range(128,136,1): # Iterate over the test range.
        base=con[n]
        value=raw[n]
        if st.poisson.pmf(value,base) <0.0001: # Check if the value is below the 99.9999 percentile of the poisson distribution at the predicted rate.
            # Additionally exclude neurons with zero-lag correlation.
            if not st.poisson.pmf(raw[125],con[125])<0.001:
                if value<base: # Check if the measured value lies below the baseline distribution.
                    # Additionally exclude neurons with zero-lag correlation.                
                    sign_tester_in.append(1)
                else:
                    
                    sign_tester_ex.append(1)
        else:
            sign_tester_in.append(0)
            sign_tester_ex.append(0)
            
    # Check if more than one singificant trough has been detected and if they are sequential in at least two bins.
    stp_in,stp_ex=[],[]
    p_in,p_ex=0,0
    # test inhibition.
    sign_tester=sign_tester_in
    stp=[]
    if N.sum(sign_tester)>1:
        for n in range(len(sign_tester)-1):
            if sign_tester[n]==1 and sign_tester[n+1]==1:
                p_in=1       

    if p_in==1:
        for m in range(128,136,1):
            stp.append((raw[m]-con[m])/len(ind1)) 
    stp_in=N.sum(stp)
    # test excitation
    sign_tester=sign_tester_ex
    stp=[]
    if N.sum(sign_tester)>1:
        for n in range(len(sign_tester)-1):
            if sign_tester[n]==1 and sign_tester[n+1]==1:
                p_ex=1       
    if p_ex==1:
        for m in range(128,136,1):
            stp.append((raw[m]-con[m])/len(ind1)) 
    stp_ex=N.sum(stp)
    
    
    if printData==True:
        
        if p_in==1:
            print("Significant inhib. inhibition")
        elif p_ex==1:
                print("Significant exc. inhibition")
        else:
            print("NO COUPLING")
          
    return gau,raw,con,p_in,p_ex,stp_in,stp_ex

def load_dict_ei_connetion(Fs=30000,savedataframe=True,plotSeparate=False):
    filename2=tkFileDialog.askopenfilename(title='ei_synaptic_connection')
    directory2, fname2 = os.path.split(filename2)
    os.chdir(directory2)
    res=op.open_helper(filename2)
    c=directory2.rsplit('/',-1)[-2]
    
    result=dict(enumerate(res.flatten(),1))
    df2=pd.DataFrame(result)
    df_ext=pd.DataFrame(df2[1]['excitation'])
    df_inh=pd.DataFrame(df2[1]['inhibition'])
    
    ext_batch_pre=df_ext['name_pre']
    ext_batch_post=df_ext['name_post']
    
    # Convert spike indices to ms.
    msFs=Fs/1000.0
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if plotSeparate==False:
        fig = pl.figure('%s_ext'%(c),figsize=[12,12])
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None,hspace=0.5)
        figzoom = pl.figure('%s_ext_zoom'%(c),figsize=[12,12])
        figzoom.subplots_adjust(left=None, bottom=None, right=None, top=None,hspace=0.5)
        #pl.title('%s_ext'%(c))
        a=int(math.ceil(float(len(ext_batch_pre))/4))
    print 'excitation', len(ext_batch_pre)
    for n in range(len(ext_batch_pre)):
        ind1=op.open_helper(ext_batch_pre[n])
        print 'pre:', ext_batch_pre[n]
        pre_a=ext_batch_pre[n].rsplit('/',-1)[-2]
        pre_b=pre_a.rsplit('_',-1)[-1]
        for m in pre_b:
            if m.isdigit():
                pre_s=m
        pre_d=ext_batch_pre[n].rsplit('/',-1)[-1]
        pre_e=pre_d.rsplit('_',-1)[-2]
        ind2=op.open_helper(ext_batch_post[n])
        print 'post', ext_batch_post[n]
        post_a=ext_batch_post[n].rsplit('/',-1)[-2]
        post_b=post_a.rsplit('_',-1)[-1]
        for m in post_b:
            if m.isdigit():
                post_s=m
        post_d=ext_batch_post[n].rsplit('/',-1)[-1]
        post_e=post_d.rsplit('_',-1)[-2]
        raw=sp.filter_correlogram(ind2,ind1,dt=0.4*msFs,shift=50*msFs)
        raw=raw[0]
        if plotSeparate==False:
            if a>1:
                ax=fig.add_subplot(4,a,n+1)
                ax1=figzoom.add_subplot(4,a,n+1)
            else:
                ax=fig.add_subplot(4,1,n+1)
                ax1=figzoom.add_subplot(4,1,n+1)
        else:
            fig=pl.figure()
            ax=fig.add_subplot(2,1,1)
            ax1=fig.add_subplot(2,1,2)
        x=N.arange(-50,50.1,0.4)
        ax.bar(x,raw,width=0.4,align='edge',color='#0504aa',alpha=0.7)
        ax.vlines(0,0,max(raw)+1,linestyle='dashed',color='#A9A9A9')
        ax.set_xlabel('time (ms)')
        #ax1.set_xlim([-5,50])
        ax.set_title('s%s_u%s vs s%s_u%s '%(pre_s,pre_e,post_s,post_e))
        ax1.bar(x,raw,width=0.4,align='edge',color='#0504aa',alpha=0.7)
        ax1.vlines(0,0,max(raw)+1,linestyle='dashed',color='#A9A9A9')
        ax1.set_xlabel('time (ms)')
        ax1.set_xlim([-10,10])
        ax1.set_title('s%s_u%s vs s%s_u%s '%(pre_s,pre_e,post_s,post_e))
        if plotSeparate==True:
            fig.savefig('s%s_u%s_vs_s%s_u%s.png'%(pre_s,pre_e,post_s,post_e))  
            fig.savefig('s%s_u%s_vs_s%s_u%s.eps'%(pre_s,pre_e,post_s,post_e),transparent=True)
            pl.close(fig)
    
    inh_batch_pre=df_inh['name_pre']
    inh_batch_post=df_inh['name_post']
    fig1 = pl.figure('%s_inh'%(c),figsize=[12,12])
    fig1.subplots_adjust(left=None, bottom=None, right=None, top=None,hspace=0.5)
    fig1zoom = pl.figure('%s_inh_zoom'%(c),figsize=[12,12])
    fig1zoom.subplots_adjust(left=None, bottom=None, right=None, top=None,hspace=0.5)
    #pl.title('%s_ext'%(c))
    print 'inhibition',len(inh_batch_pre)
    a=int(math.ceil(float(len(inh_batch_pre))/4))
    for n in range(len(inh_batch_pre)):
        ind1=op.open_helper(inh_batch_pre[n])
        print 'pre:', inh_batch_pre[n]
        pre_a=inh_batch_pre[n].rsplit('/',-1)[-2]
        pre_b=pre_a.rsplit('_',-1)[-1]
        for m in pre_b:
            if m.isdigit():
                pre_s=m
        pre_d=inh_batch_pre[n].rsplit('/',-1)[-1]
        pre_e=pre_d.rsplit('_',-1)[-2]
        ind2=op.open_helper(inh_batch_post[n])
        print 'post', inh_batch_post[n]
        post_a=inh_batch_post[n].rsplit('/',-1)[-2]
        post_b=post_a.rsplit('_',-1)[-1]
        for m in post_b:
            if m.isdigit():
                post_s=m
        post_d=inh_batch_post[n].rsplit('/',-1)[-1]
        post_e=post_d.rsplit('_',-1)[-2]
        raw_inh=sp.filter_correlogram(ind2,ind1,dt=0.4*msFs,shift=50*msFs)
        raw_inh=raw_inh[0]
        if a>1:
            ax=fig1.add_subplot(4,a,n+1)
            ax1=fig1zoom.add_subplot(4,a,n+1)
        else:
            ax=fig1.add_subplot(4,1,n+1)
            ax1=fig1zoom.add_subplot(4,1,n+1)
        x=N.arange(-50,50.1,0.4)
        ax.bar(x,raw_inh,width=0.4,align='edge',color='#0504aa',alpha=0.7)
        ax.vlines(0,0,max(raw_inh)+1,linestyle='dashed',color='#A9A9A9')
        ax.set_xlabel('time (ms)')
        #ax1.set_xlim([-5,50])
        ax.set_title('s%s_u%s vs s%s_u%s '%(pre_s,pre_e,post_s,post_e))
        ax1.bar(x,raw_inh,width=0.4,align='edge',color='#0504aa',alpha=0.7)
        ax1.vlines(0,0,max(raw_inh)+1,linestyle='dashed',color='#A9A9A9')
        ax1.set_xlabel('time (ms)')
        ax1.set_xlim([-10,10])
        ax1.set_title('s%s_u%s vs s%s_u%s '%(pre_s,pre_e,post_s,post_e))
    if plotSeparate==False:
        fig.savefig('%s_ei_connection_ext.png'%(c))  
        fig.savefig('%s_ei_connection_ext.eps'%(c),transparent=True)
        figzoom.savefig('%s_ei_connection_ext_zoom.png'%(c))  
        figzoom.savefig('%s_ei_connection_ext_zoom.eps'%(c),transparent=True)
    fig1.savefig('%s_ei_connection_inh.png'%(c))  
    fig1.savefig('%s_ei_connection_inh.eps'%(c),transparent=True)
    fig1zoom.savefig('%s_ei_connection_inh_zoom.png'%(c))  
    fig1zoom.savefig('%s_ei_connection_inh_zoom.eps'%(c),transparent=True)
    
    if savedataframe==True:
        df_ext.to_csv('%s_ei_synaptic_connection_ext.csv'%c)
        df_inh.to_csv('%s_ei_synaptic_connection_inh.csv'%c)

#%% function for LFP waveforms in relation to unit spiking timestamps

def spike_oscillation_coupling_single_batch_starter(low=4,high=12,Fs=30000,thr=1.5,restrict_to=False):
    
    target_dir='/data/example_raw_for_position_extraction_spike_waveform/200721_#386_lin'
    os.chdir(target_dir)
    os.chdir('output/ms3--all_shank2')
    su_batch=op.open_helper('all_timestamp_test.batch') 
    os.chdir(target_dir)
    directory2='#386_2020-07-21_16-05-13_test'
    os.chdir(directory2)
    raw_batch=op.open_helper('all_channels_raw_list_shank2.batch')
 
    grand_result=[]
    for n in range(len(su_batch)):       
  
        print "Recording set %s of " %n, len(su_batch)-1
        os.chdir(target_dir)
        os.chdir('output/ms3--all_shank2')                
        su=op.open_helper(su_batch[n])
        
        os.chdir(target_dir)
        os.chdir(directory2)
        raw=op.open_helper(raw_batch[n])
        raw=data_tools.filterData(raw,low=low,high=high,Fs=Fs)
        
        print "Computing envelope"
        env=data_tools.envelope(raw)
        
        res=spike_oscillation_coupling_single_above_thres(su,raw,env,thres=thr,low=low,high=high,Fs=Fs)
        
        grand_result.append(res)
        
    return grand_result

def spike_oscillation_coupling_single_above_thres(su,filt,env,low=1,high=5,thres=2,Fs=30000):
    ''' 
    Implementation of spike-oscillation coupling following the description
    of Tamura et al,Neuron 2016. Measures pairwise phase consistency, mean resultant length,
    and test for deviation from uniformity using Rayleigh's test.
    thres: defined the threshold (in SD above the mean) of the envelope of the signal.
    Fs: sampling rate
    Inputs:
    su : Spike indices of single unit
    filt : raw recording
    env:envelope
    low & high: frequency boader for band pass filter
    '''
    
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
            p_delta=rayleigh_test(ang_delta)                        
            
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

def rayleigh_test(angles):
    '''
        Performs Rayleigh Test for non-uniformity of circular data.
        Compares against Null hypothesis of uniform distribution around circle
        Assume one mode and data sampled from Von Mises.
        Use other tests for different assumptions.
        Uses implementation close to CircStats Matlab toolbox, maths from [Biostatistical Analysis, Zar].
    '''

    no = len(angles)

    # Compute Rayleigh's R
    R = no*angle_population_R(angles)

    # Compute Rayleight's z
    z = R**2. / no

    # Compute pvalue (Zar, Eq 27.4)
    pvalue = N.exp(N.sqrt(1. + 4*no + 4*(no**2. - R**2)) - 1. - 2.*no)

    return pvalue

### Dentate spike detection -- see 'onkey.py'

#%% OFF/ON/NON-SOMI classification & speed modulation & firing heatmap on position axis

def unit_place_corr_batch_plot_memory_efficient(thres=3,thr_artifact=3,thr_pos_intertrial=40,Fs=30000,high=2,binsize=5,iterations=1000,Tlength=400,bins=30,vmin=-1,bins_loc=25,step=1,tdur=3,tstart=5,binspers=5,t1=1,t2=2,thres_immo=0.5,thres_loc=2,plotData=True,plotrewpos=True,saveData=False,vmaxauto=True,vmaxValue=N.nan,step100ms=True,set_acc_max=True,acc_max=10):
    
    directory2='/data/example_raw_for_position_extraction_spike_waveform/200721_#386_lin/output/ms3--all_shank2'
    os.chdir(directory2)
    su_batch=op.open_helper('all_timestamp_test_SOMI.batch') 
    directory='/data/example_raw_for_position_extraction_spike_waveform/200721_#386_lin/#386_2020-07-21_16-05-13_test'
    
    os.chdir(directory)
    posnew = N.load('estimate speed from position_corrected posnew.npy')
    rect = N.load('estimate speed from position_rect.npy')
    speed = N.load('estimate speed from position_speed.npy')
    ADC=op.open_helper('100_ADC3.continuous')
    ADC2=op.open_helper('100_ADC2.continuous')
    
    os.chdir(directory2)
    if os.path.isdir('Data saved')==False:
        os.mkdir('Data saved')
    fig_to_save=tkFileDialog.asksaveasfilename(title="Choose target figure name") 
    file_to_save=tkFileDialog.asksaveasfilename(title="Choose target filename")  
    su_all= create_rasterplot_of_all_sorted_units(su_batch,posnew,rect,speed,ADC,ADC2,fig_to_save)
    os.chdir('Data saved')
    c=directory.rsplit('_',-1)[-1]
    name=file_to_save.rsplit('/',-1)[-1]
    g=open("all_unit_speed_corr_%s.batch" %(c),"w")
    
    posnew_loc,speed_loc,ind_loc= [],[],[]
    for i in range(len(speed)):
        if speed[i]>thres:
            posnew_loc.append(posnew[i])
            speed_loc.append(speed[i])
            ind_loc.append(i)
        
    binstart= N.arange(posnew_loc[N.argmin(posnew_loc)],posnew_loc[N.argmax(posnew_loc)],binsize)

    #put indices in bins respectively
    tstart_nosmooth,tstop_nosmooth=[],[]    
    for i in range(len(binstart)-1):
        result=N.ravel(N.argwhere(N.logical_and(posnew_loc>=binstart[i],posnew_loc<binstart[i+1])))
        if len(result)==0:
            break
        else:
            i_in_result=N.ravel(N.argwhere(N.diff(result)>1))
            i_start_in_result=(i_in_result+1)
            i_start_in_result=N.insert(i_start_in_result,0,0)
            i_in_result=N.insert(i_in_result,len(i_in_result),(len(result)-1))
            tstop_nosmooth.append(N.take(result,i_in_result))
            tstart_nosmooth.append(N.take(result,i_start_in_result))
    result=N.ravel(N.argwhere(posnew_loc>=binstart[-1]))
    i_in_result=N.ravel(N.argwhere(N.diff(result)>1))
    i_start_in_result=(i_in_result+1)
    i_start_in_result=N.insert(i_start_in_result,0,0)
    i_in_result=N.insert(i_in_result,len(i_in_result),(len(result)-1))
    tstop_nosmooth.append(N.take(result,i_in_result))
    tstart_nosmooth.append(N.take(result,i_start_in_result))
    
    pos_filt_all_trial,pos_filt_per_trial,res=smoothing_position_data_Tlength_memory_efficient (posnew_loc,thres=thres,thr_artifact=thr_artifact,thr_pos_intertrial=thr_pos_intertrial,Tlength=Tlength,Fs=Fs,high=high)
    
    grand_result_speed,grand_result_acc,grand_result_speed_loc=[],[],[]
    for n in range(len(su_batch)):
        print "Recording set %s of " %n, len(su_batch)-1
        print su_batch[n]
        os.chdir(target_dir)
        os.chdir(directory2)
        su=op.open_helper(su_batch[n])
        for i in range(len(su)):
            if su[i]>len(speed):
                su=su[:i-1]
        
        result_speed_all, r, p=create_unit_speed_corr_all_in_s(su,speed,su_batch[n],file_to_save,fig_to_save,Fs=Fs,plotData=plotData,step100ms=step100ms)
        result_acc_all,r_pos,p_pos,r_neg,p_neg=create_unit_acceleration_corr_all(su,speed,su_batch[n],file_to_save,fig_to_save,Fs=Fs,set_acc_max=set_acc_max,acc_max=acc_max,plotData=plotData)
        grand_result_speed.append(result_speed_all)
        grand_result_acc.append(result_acc_all)
        result_speed_loc,r_loc,p_loc=create_unit_speed_corr_locomotion_in_s(su,posnew,rect,speed,su_batch[n],file_to_save,fig_to_save,thres=thres,Fs=Fs,plotData=plotData,step100ms=step100ms)
        grand_result_speed_loc.append(result_speed_loc)
        spike_freq, speed_x, spike_freq_loc, speed_x_loc=create_unit_speed_corr_all_vs_loc(su,posnew,rect,speed,su_batch[n],file_to_save,fig_to_save,Fs=Fs,bins=bins,vmin=vmin,thres=thres,bins_loc=bins_loc,step=step,plotData=plotData,saveData=saveData)
        g.write('%s_%s_mean_fir_freq_all.npy' %(name,su_batch[n]))
        g.write('\n')
        spike_train = N.zeros(len(speed))
        N.put(spike_train,su.astype(int),1) 
        spike_train_loc=N.take(spike_train,ind_loc)
        Mean_Fir_rate, binstart,pos_filt_all_trial,Fir_rate=create_unit_place_field_correlation(pos_filt_all_trial,spike_train_loc,posnew_loc,res,su_batch[n],file_to_save,fig_to_save,Fs=Fs,binsize=binsize,plotData=plotData,saveData=saveData)
        Fir_rate=create_heatmap_place_field (pos_filt_all_trial,pos_filt_per_trial,spike_train_loc,posnew_loc,res,su_batch[n],file_to_save,fig_to_save,directory,Fs=Fs,binsize=binsize,Tlength=Tlength,plotData=plotData,plotrewpos=plotrewpos,vmaxauto=vmaxauto,vmaxValue=vmaxValue,saveData=saveData,cmap='jet',speedmap=False)
        create_heatmap_transition_immo_loc(su,speed,su_batch[n],file_to_save,fig_to_save,ADC,tdur=tdur,tstart=tstart,t1=t1,t2=t2,thres_immo=thres_immo,thres_loc=thres_loc,Fs=Fs,plotData=plotData,saveData=saveData)
        create_heatmap_transition_immo_loc_high_resol(su,speed,su_batch[n],file_to_save,fig_to_save,tdur=tdur,tstart=tstart,binspers=binspers,t1=t1,t2=t2,thres_immo=thres_immo,thres_loc=thres_loc,Fs=Fs,plotData=plotData,saveData=saveData)
        pl.close('all')
        
    g.close()
    
    if saveData==True:
        N.save('%s_speed_corr_all'%file_to_save,grand_result_speed)
        N.save('%s_acc_corr_all'%file_to_save,grand_result_acc)
        N.save('%s_speed_corr_loc'%file_to_save,grand_result_speed_loc)        

#################################################################
        
#################################################################
def create_rasterplot_of_all_sorted_units(su,posnew,rect,speed,ADC3,ADC2,fig_to_save,Fs=30000,subplots=True,plotraw=True,filtered=True):
    '''
    rasterplot of all units on one shank in relation to position, speed and raw recording trace
    '''   
 
    if plotraw==True:
        dir1=tkFileDialog.askopenfilename(title='raw ,continuous')
        directory, fname = os.path.split(dir1)
        raw=op.open_helper(dir1)
        if filtered==True:
            raw=data_tools.filterData(raw,low=0.2,high=1000,Fs=Fs)
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    f,axes=pl.subplots(figsize=[12,6])
    for i in range(len(su)):
        print "Recording set %s of " %i, len(su)-1
        su_single=op.open_helper(su[i])
        axes.eventplot(su_single,lineoffsets=i,linelengths=0.8)
    axes.set_ylabel("sorted unit No.")
    axes.set_title('%s' %fig_to_save)
    f.savefig('%s_unit_raster.png' %fig_to_save)
    f.savefig('%s_unit_raster.eps' %fig_to_save,transparent=True)
    
    if subplots==True:
        if plotraw==True:
            f,(ax1,ax2,ax3,ax4,ax5)=pl.subplots(5,1,sharex=True,figsize=[8,8],gridspec_kw={'height_ratios': [3,1,1,1,1]})
        else:
            f,(ax1,ax2,ax3,ax4)=pl.subplots(4,1,sharex=True,figsize=[8,6],gridspec_kw={'height_ratios': [3,1,1,1]})
        for i in range(len(su)):
            print "Recording set %s of " %i, len(su)-1
            su_single=op.open_helper(su[i])
            ax1.eventplot(su_single,lineoffsets=i,linelengths=0.8)
        ax1.set_ylabel("sorted unit No.")
        ax2.plot(posnew)
        ax2.set_ylabel("Track position (au)")
        ax3.plot(rect)
        ax3.set_ylabel("Cumulative position (cm)")
        ax4.plot(speed)
        ax4.set_ylabel("Speed (cm/s)")
        if plotraw==True:
            ax5.plot(raw)
            ax5.set_ylabel("raw")
        f.savefig('%s_all_pos_rect_speed.png' %fig_to_save)
        f.savefig('%s_all_pos_rect_speed.eps' %fig_to_save,transparent=True)
        
        if plotraw==True:
            f,(ax1,ax2,ax3,ax4,ax5,ax6)=pl.subplots(6,1,sharex=True,figsize=[12,10],gridspec_kw={'height_ratios': [3,1,1,1,1,2]})
        else:    
            f,(ax1,ax2,ax3,ax4,ax5)=pl.subplots(5,1,sharex=True,figsize=[12,9],gridspec_kw={'height_ratios': [3,1,1,1,1]})
        for i in range(len(su)):
            print "Recording set %s of " %i, len(su)-1
            su_single=op.open_helper(su[i])
            ax1.eventplot(su_single,lineoffsets=i,linelengths=0.8)
        ax1.set_ylabel("sorted unit No.")
        ax2.plot(posnew)
        ax2.set_ylabel("Track position (au)")
        ax3.plot(speed)
        ax3.set_ylabel("Speed (cm/s)")
        ax4.plot(ADC3)
        ax4.set_ylabel("reward")
        ax5.plot(ADC2)
        ax5.set_ylabel("licks")
        if plotraw==True:
            ax6.plot(raw)
            ax6.set_ylabel("raw")
        f.savefig('%s_all_pos_rect_speed_rew.png' %fig_to_save)
        f.savefig('%s_all_pos_rect_speed_rew.eps' %fig_to_save,transparent=True)


def smoothing_position_data_Tlength_memory_efficient (posnew_loc,thres=3,thr_artifact=3,thr_pos_intertrial=40,Tlength=400,Fs=30000,high=2):
    '''
    posnew_loc: real-time position during locomotion
    thres: threshold of speed for locomotion
    thr_artifact: for strange jump between the end of track and the start
    thr_pos_intertrial: intertrial position difference threshold
    high: the highest frequency for low pass filter filtering the position data
    '''
    
    #get the break of each trial
    res=[]
    for l in range(1,len(posnew_loc)):
        if posnew_loc[l]-posnew_loc[l-1]>-thr_pos_intertrial and posnew_loc[l]-posnew_loc[l-1]<-thr_artifact:
            posnew_loc[l]=posnew_loc[l-1]
        if posnew_loc[l]-posnew_loc[l-1]<-thr_pos_intertrial: 
            res.append(l)
    
    #seperate each trial in one array
    pos_per_trial=[]
    pos_per_trial.append(posnew_loc[:res[0]])
    for j in range(len(res)-1):
        pos_per_trial.append(posnew_loc[res[j]:res[j+1]])
    pos_per_trial.append(posnew_loc[res[-1]:])
    
    #filter each array
    pos_filt_per_trial=[]
    for k in range(len(pos_per_trial)):
        pos_filt_per_trial.append(data_tools.lowpass(pos_per_trial[k][:],high=high)) 
    
    scale_all=float(Tlength)/float(max(pos_filt_per_trial[0][:]))
    for l in range(1,len(pos_filt_per_trial)-1):
        pos_filt_per_trial[l][:]=pos_filt_per_trial[l][:]-min(pos_filt_per_trial[l][:])
        scale=float(Tlength)/float(max(pos_filt_per_trial[l][:]))
        pos_filt_per_trial[l][:]=N.asarray(pos_filt_per_trial[l][:])*scale
    pos_filt_per_trial[0][:]=pos_filt_per_trial[0][:]-min(pos_filt_per_trial[1][:])
    pos_filt_per_trial[0][:]=N.asarray(pos_filt_per_trial[0][:])*scale_all
    pos_filt_per_trial[-1][:]=pos_filt_per_trial[-1][:]-min(pos_filt_per_trial[-1][:])
    pos_filt_per_trial[-1][:]=N.asarray(pos_filt_per_trial[-1][:])*scale_all
        
    
    #put all trials together
    pos_filt_all_trial=[]
    for m in range(len(pos_filt_per_trial)):
        pos_filt_all_trial=N.concatenate((pos_filt_all_trial,pos_filt_per_trial[m][:]))
    
    return pos_filt_all_trial,pos_filt_per_trial,res


def create_unit_speed_corr_all_in_s(su,speed,fname2,file_to_save,fig_to_save,Fs=30000,plotData=True,saveData=True,step100ms=True):
    '''
    binsize in ms
    calculate spike frequency in each speed unit (per second);
    put all frequencies in speed bins;
    average them.
    
    '''
    #create spike train with su as indices of 1, others of 0
    spike_train = N.zeros(len(speed))
    N.put(spike_train,su.astype(int),1)    
    
    #get speed data in second bin & firing frequency each second
    Fir_freq,mean_speed_pers = [],[]
    for n in range(0,len(speed)-Fs,Fs):
        Fir_freq.append(sum(spike_train[n:n+Fs]))
        mean_speed_pers.append(N.mean(speed[n:n+Fs]))
    mean_fir_freq=N.mean(Fir_freq)
    slope,intercept,r,p,stderr=stats.linregress(mean_speed_pers,Fir_freq)
    print 'unit %s' %fname2
    print "mean frequency all: %s " %mean_fir_freq
    
    #get speed data in 10 ms bin & firing frequency each 10 ms for linear fit 10ms bins
    Fir_freq_ms,mean_speed_per10ms = [],[]
    for n in range(0,len(speed)-Fs/100,Fs/100):
        Fir_freq_ms.append(sum(spike_train[n:n+Fs/100])*100)
        mean_speed_per10ms.append(N.mean(speed[n:n+Fs/100]))
    Fir_freq_ms_smooth=ndimage.gaussian_filter(Fir_freq_ms,1,mode='nearest')
    
    step=0.1     #every 100ms
    Fir_freq_100ms,mean_speed_per100ms = [],[]
    for n in range(0,len(speed)-int(step*Fs),int(step*Fs)):
        Fir_freq_100ms.append(sum(spike_train[n:n+int(step*Fs)])/step)      #sum of every 100 ms divided by step/Fs=*10
        mean_speed_per100ms.append(N.mean(speed[n:n+int(step*Fs)]))   #cm/s
    Fir_freq_smooth=ndimage.gaussian_filter(Fir_freq_100ms,1,mode='nearest')
    
    #find bins of speed range; math.ceil to get the smallest integer greater than float    
    bins = int(ma.ceil(mean_speed_pers[N.argmax(mean_speed_pers)]-mean_speed_pers[N.argmin(mean_speed_pers)]))
    if step100ms==True:
        bins = ma.ceil((mean_speed_per100ms[N.argmax(mean_speed_per100ms)]-mean_speed_per100ms[N.argmin(mean_speed_per100ms)])/step)
        slope,intercept,r,p,stderr=stats.linregress(mean_speed_per100ms,Fir_freq_smooth)
        spike_freq,bin_edges,binnumber= stats.binned_statistic(mean_speed_per100ms,Fir_freq_smooth,statistic='mean',bins=bins)
        spike_freq_std,bin_edges_std,binnumber_std= stats.binned_statistic(mean_speed_per100ms,Fir_freq_smooth,statistic='std',bins=bins)
    else:
        slope,intercept,r,p,stderr=stats.linregress(mean_speed_pers,Fir_freq)
        spike_freq,bin_edges,binnumber= stats.binned_statistic(mean_speed_pers,Fir_freq,statistic='mean',bins=bins)
        spike_freq_std,bin_edges_std,binnumber_std= stats.binned_statistic(mean_speed_pers,Fir_freq,statistic='std',bins=bins)
    spike_freq_sem=spike_freq_std/N.sqrt(len(spike_freq_std))
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    if plotData==True:
        bins_middle = []
        for n in range(len(bin_edges)-1):
            bins_middle.append((bin_edges[n+1]-bin_edges[n])/2+bin_edges[n])
        
        speed_x=[]
        for j in range(len(mean_speed_pers)):
            idx=(N.abs(bins_middle-mean_speed_pers[j])).argmin()
            speed_x.append(bins_middle[idx])
        
        f,(ax1,ax2)=pl.subplots(2,1,sharex=True,figsize=[8,6])
        f.text(0.04, 0.5, 'Averaged firing frequency (Hz)', va='center', rotation='vertical',fontsize=14)
        ax1.bar(bin_edges[:-1],spike_freq,width=0.8,align='edge',color='#0504aa',alpha=0.7)
        ax1.set_title('%s_speed correlation_all'%fname2,size=14)
        ax1.grid(axis='y', alpha=0.75)
        if step100ms==True:
            ax1.plot(mean_speed_per100ms,intercept+slope*N.array(mean_speed_per100ms),'r--')
            ax2.plot(mean_speed_per100ms,intercept+slope*N.array(mean_speed_per100ms),'r--')
        else:
            ax2.plot(speed_x,Fir_freq,'o',color='#A9A9A9')
            ax1.plot(mean_speed_pers,intercept+slope*N.array(mean_speed_pers),'r--')
        ax2.plot(bins_middle,spike_freq,'r-')
        ax2.fill_between(bins_middle,spike_freq+spike_freq_sem,spike_freq-spike_freq_sem,facecolor='orange', alpha=0.5)
        ax2.set_xlabel('Speed (cm/s)',fontsize=14)
        c=fname2.rsplit('_',-1)[0]
        f.savefig('%s_u%s_speed_all.png' %(fig_to_save,c))
        f.savefig('%s_u%s_speed_all.eps' %(fig_to_save,c),transparent=True)
        f1, ax = pl.subplots(figsize=(8, 6))
        ax.scatter(mean_speed_per100ms,Fir_freq_smooth,linewidth=0.05,marker='o',alpha=0.8) 
        ax.set_title('%s_speed correlation_scatter'%fname2,size=14)
        ax.set_xlabel('Speed (cm/s)',fontsize=14)
        ax.set_ylabel('Firing frequency (Hz)',fontsize=14)
        ax.plot(mean_speed_per100ms,intercept+slope*N.array(mean_speed_per100ms),'r--')
        f1.text(0.5, 0.8, 'r=%s' %r, fontsize=14)
        f1.text(0.5, 0.75, 'p=%s' %p, fontsize=14)
        f1.savefig('%s_u%s_speed_scatter.png' %(fig_to_save,c))
        f1.savefig('%s_u%s_speed_scatter.eps' %(fig_to_save,c),transparent=True)
        result={'su_id':c,'bins_middle':bins_middle,'spike_freq_binned':spike_freq,'mean_speed_pers':mean_speed_pers,'Fir_freq':Fir_freq,'intercept':intercept,'slope':slope,'spike_freq_std':spike_freq_std,'r':r,'p':p}
    
    if saveData==True:
        N.save('%s_%s_smoothed_spike_train_per_10ms' %(file_to_save,fname2),Fir_freq_ms_smooth) 
        N.save('%s_%s_mean_speed_per_10ms' %(file_to_save,fname2),mean_speed_per10ms) 
        
    return result, r, p

def create_unit_speed_corr_locomotion_in_s(su,posnew,rect,speed,fname2,file_to_save,fig_to_save,thres=3,Fs=30000,plotData=True,step100ms=True):
    
    '''
    calculate spike frequency in each speed unit (per second);
    put all frequencies in speed bins;
    average them.
    '''
    
    #create spike train with su as indices of 1, others of 0
    spike_train = N.zeros(len(speed))
    N.put(spike_train,su.astype(int),1) 
    
    speed_loc,ind_loc,spike_train_loc = [],[],[]
    for i in range(len(speed)):
        if speed[i]>thres:
            speed_loc.append(speed[i])
            ind_loc.append(i)
            spike_train_loc.append(spike_train[i]) 
       
    
    step=0.1     #every 100ms
    Fir_freq_100ms,mean_speed_per100ms = [],[]
    for n in range(0,len(speed_loc)-int(step*Fs),int(step*Fs)):
        Fir_freq_100ms.append(sum(spike_train_loc[n:n+int(step*Fs)])/step)      #sum of every 100 ms divided by step/Fs=*10
        mean_speed_per100ms.append(N.mean(speed_loc[n:n+int(step*Fs)]))   #cm/s
    Fir_freq_smooth=ndimage.gaussian_filter(Fir_freq_100ms,1,mode='nearest')
    
    #get speed data in second bin & firing frequency each second
    Fir_freq,mean_speed_pers= [],[]
    for n in range(0,len(speed_loc)-Fs,Fs):
        Fir_freq.append(sum(spike_train_loc[n:n+Fs]))
        mean_speed_pers.append(N.mean(speed_loc[n:n+Fs]))
    mean_fir_freq=N.mean(Fir_freq)
    print "mean frequency locomotion: %s " %mean_fir_freq
       
    #find bins of speed range; math.ceil to get the smallest integer greater than float    
    bins = int(ma.ceil(mean_speed_pers[N.argmax(mean_speed_pers)]-mean_speed_pers[N.argmin(mean_speed_pers)]))
    
    if step100ms==True:
        bins=bins*10
        slope,intercept,r,p,stderr=stats.linregress(mean_speed_per100ms,Fir_freq_smooth)
        spike_freq,bin_edges,binnumber= stats.binned_statistic(mean_speed_per100ms,Fir_freq_smooth,statistic='mean',bins=bins)
        spike_freq_std,bin_edges_std,binnumber_std= stats.binned_statistic(mean_speed_per100ms,Fir_freq_smooth,statistic='std',bins=bins)
    else:
        slope,intercept,r,p,stderr=stats.linregress(mean_speed_pers,Fir_freq) 
        spike_freq,bin_edges,binnumber= stats.binned_statistic(mean_speed_pers,Fir_freq,statistic='mean',bins=bins)
        spike_freq_std,bin_edges_std,binnumber_std= stats.binned_statistic(mean_speed_pers,Fir_freq,statistic='std',bins=bins)
    spike_freq_sem=spike_freq_std/N.sqrt(len(spike_freq_std))    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    if plotData==True:
        bins_middle = []
        for n in range(len(bin_edges)-1):
            bins_middle.append((bin_edges[n+1]-bin_edges[n])/2+bin_edges[n])
        
        speed_x=[]
        for j in range(len(mean_speed_pers)):
            idx=(N.abs(bins_middle-mean_speed_pers[j])).argmin()
            speed_x.append(bins_middle[idx])
        
        f,(ax1,ax2)=pl.subplots(2,1,sharex=True,figsize=[8,6])
        f.text(0.04, 0.5, 'Averaged firing frequency (Hz)', va='center', rotation='vertical',fontsize=14)
        ax1.bar(bins_middle,spike_freq,width=0.8,color='#0504aa',alpha=0.7)
        
        f.text(0.5, 0.8, 'r=%s' %r, fontsize=14)
        f.text(0.5, 0.75, 'p=%s' %p, fontsize=14)
        ax1.set_title('%s_speed correlation_locomotion'%fname2,size=14)
        ax1.grid(axis='y', alpha=0.75)
        if step100ms==True:
            ax1.plot(mean_speed_per100ms,intercept+slope*N.array(mean_speed_per100ms),'r--') 
            ax2.plot(mean_speed_per100ms,intercept+slope*N.array(mean_speed_per100ms),'r--') 
        else:
            ax2.plot(speed_x,Fir_freq,'o',color='#A9A9A9')
            ax1.plot(mean_speed_pers,intercept+slope*N.array(mean_speed_pers),'r--')
        ax2.plot(bins_middle,spike_freq,'ro-')
        ax2.fill_between(bins_middle,spike_freq+spike_freq_sem,spike_freq-spike_freq_sem,facecolor='orange', alpha=0.5)
        ax2.set_xlabel('Speed (cm/s)',fontsize=14)
        c=fname2.rsplit('_',-1)[0]
        f.text(0.5, 0.8, 'r=%s' %r, fontsize=14)
        f.text(0.5, 0.75, 'p=%s' %p, fontsize=14)
        f.savefig('%s_u%s_speed_loc.png' %(fig_to_save,c))
        f.savefig('%s_u%s_speed_loc.eps' %(fig_to_save,c),transparent=True)
        result={'su_id':c,'bins_middle':bins_middle,'spike_freq_binned':spike_freq,'mean_speed_pers':mean_speed_pers,'Fir_freq':Fir_freq,'intercept':intercept,'slope':slope,'spike_freq_std':spike_freq_std,'r':r,'p':p}
    
    
    return result,r,p


def create_unit_acceleration_corr_all(su,speed,fname2,file_to_save,fig_to_save,Fs=30000,set_acc_max=True,acc_max=10,plotData=True,saveData=True):
    '''
    get cell firing -- acceleration correlation
    
    '''
    #create spike train with su as indices of 1, others of 0
    spike_train = N.zeros(len(speed))
    N.put(spike_train,su.astype(int),1)    
    
    #get speed data in second bin & firing frequency each 100ms
    Fir_freq,mean_speed_per100ms,mean_acc_per100ms,Fir_freq_set_acc_max,Fir_freq_acc = [],[],[],[],[]
    step=0.1     #every 100ms
    for n in range(0,len(speed)-int(step*Fs),int(step*Fs)):
        Fir_freq.append(sum(spike_train[n:n+int(step*Fs)])/step)      #sum of every 100 ms divided by step/Fs=*10
        mean_speed_per100ms.append(N.mean(speed[n:n+int(step*Fs)]))   #cm/s
        if set_acc_max==True:
            acc=N.mean(N.diff(speed[n:n+int(step*Fs)])*Fs)
            if abs(acc)<=acc_max:
                mean_acc_per100ms.append(acc)
                Fir_freq_set_acc_max.append(sum(spike_train[n:n+int(step*Fs)])/step)      #sum of every 100 ms divided by step/Fs=*10
        else:
            mean_acc_per100ms.append(N.mean(N.diff(speed[n:n+int(step*Fs)])*Fs)) 
            Fir_freq_acc.append(sum(spike_train[n:n+int(step*Fs)])/step) 
    if abs(acc)<=acc_max:
        Fir_freq_smooth=ndimage.gaussian_filter(Fir_freq_set_acc_max,1,mode='nearest')
        mean_fir_freq=N.mean(Fir_freq_set_acc_max)
    else:
        Fir_freq_smooth=ndimage.gaussian_filter(Fir_freq,1,mode='nearest')
        mean_fir_freq=N.mean(Fir_freq)
    
    
    ind,ind_neg=[],[]
    for l in range(len(mean_acc_per100ms)):
        if mean_acc_per100ms[l]>0:
            ind.append(l)
        else:
            ind_neg.append(l)
    mean_acc_pos=N.take(mean_acc_per100ms,ind)
    Fir_freq_pos=N.take(Fir_freq_smooth,ind)
    mean_acc_neg=N.take(mean_acc_per100ms,ind_neg)
    Fir_freq_neg=N.take(Fir_freq_smooth,ind_neg)
    slope_pos,intercept_pos,r_pos,p_pos,stderr_pos=stats.linregress(mean_acc_pos,Fir_freq_pos)
    slope_neg,intercept_neg,r_neg,p_neg,stderr_neg=stats.linregress(mean_acc_neg,Fir_freq_neg)
    
    if ma.ceil(mean_acc_per100ms[N.argmax(mean_acc_per100ms)]-mean_acc_per100ms[N.argmin(mean_acc_per100ms)])>50:
        bins=50
        binwidth=ma.ceil(mean_acc_per100ms[N.argmax(mean_acc_per100ms)]-mean_acc_per100ms[N.argmin(mean_acc_per100ms)])/bins
    elif set_acc_max==True:
        bins=4*acc_max
        binwidth=ma.ceil(mean_acc_per100ms[N.argmax(mean_acc_per100ms)]-mean_acc_per100ms[N.argmin(mean_acc_per100ms)])/bins
    else:
        bins=ma.ceil(mean_acc_per100ms[N.argmax(mean_acc_per100ms)]-mean_acc_per100ms[N.argmin(mean_acc_per100ms)])
        binwidth=0.8
    spike_freq,bin_edges,binnumber= stats.binned_statistic(mean_acc_per100ms,Fir_freq_smooth,statistic='mean',bins=int(bins))
    spike_freq_std,bin_edges_std,binnumber_std= stats.binned_statistic(mean_acc_per100ms,Fir_freq_smooth,statistic='std',bins=int(bins))
    spike_freq_sem=spike_freq_std/N.sqrt(len(spike_freq_std))
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    if plotData==True:
        bins_middle,bins_middle_pos = [],[]
        for n in range(len(bin_edges)-1):
            bins_middle.append((bin_edges[n+1]-bin_edges[n])/2+bin_edges[n])
            if bins_middle[n]>0:
                bins_middle_pos.append(bins_middle[n])
        bins_middle_neg=N.setdiff1d(bins_middle,bins_middle_pos)
        
        acc_x=[]
        for j in range(len(mean_acc_per100ms)):
            idx=(N.abs(bins_middle-mean_acc_per100ms[j])).argmin()
            acc_x.append(bins_middle[idx])
        f,(ax1,ax2)=pl.subplots(2,1,sharex=True,figsize=[8,6])
        f.text(0.04, 0.5, 'Averaged firing frequency (Hz)', va='center', rotation='vertical',fontsize=14)
        ax1.bar(bin_edges[:-1],spike_freq,width=binwidth,align='edge',color='#0504aa',alpha=0.7)
        ax1.set_title('%s_acceleration correlation_all'%fname2,size=14)
        ax2.plot(bins_middle,spike_freq,'r-')
        ax2.plot(bins_middle_pos,intercept_pos+slope_pos*N.array(bins_middle_pos),'r--')
        ax2.plot(bins_middle_neg,intercept_neg+slope_neg*N.array(bins_middle_neg),'r--')
        ax2.fill_between(bins_middle,spike_freq+spike_freq_sem,spike_freq-spike_freq_sem,facecolor='orange', alpha=0.5)
        ax2.set_xlabel('Acceleration (cm/$s^-2$)',fontsize=14)
        c=fname2.rsplit('_',-1)[0]
        f.savefig('%s_u%s_acceleration_all.png' %(fig_to_save,c))
        f.savefig('%s_u%s_acceleration_all.eps' %(fig_to_save,c),transparent=True)
        f1, ax = pl.subplots(figsize=(8, 6))
        ax.scatter(mean_acc_per100ms,Fir_freq_smooth,linewidth=0.1,marker='.',alpha=0.8) 
        ax.plot(bins_middle_pos,intercept_pos+slope_pos*N.array(bins_middle_pos),'r--')
        ax.plot(bins_middle_neg,intercept_neg+slope_neg*N.array(bins_middle_neg),'r--')
        f1.text(0.15, 0.8, 'r_neg=%s' %r_neg, fontsize=12)
        f1.text(0.15, 0.75, 'p_neg=%s' %p_neg, fontsize=12)
        f1.text(0.5, 0.8, 'r_pos=%s' %r_pos, fontsize=12)
        f1.text(0.5, 0.75, 'p_pos=%s' %p_pos, fontsize=12)
        ax.set_title('%s_acceleration correlation_scatter'%fname2,size=14)
        ax.set_xlabel('Acceleration (cm/$s^-2$)',fontsize=14)
        ax.set_ylabel('Firing frequency (Hz)',fontsize=14)
        f1.savefig('%s_u%s_acceleration_scatter.png' %(fig_to_save,c))
        f1.savefig('%s_u%s_acceleration_scatter.eps' %(fig_to_save,c),transparent=True)
        result={'su_id':c,'bins_middle':bins_middle,'spike_freq_binned':spike_freq,'mean_acc_per100ms':mean_acc_per100ms,'Fir_freq':Fir_freq,'intercept_pos':intercept_pos,'intercept_neg':intercept_neg,'slope_pos':slope_pos,'slope_neg':slope_neg,'spike_freq_std':spike_freq_std,'r_pos':r_pos,'r_neg':r_neg,'p_pos':p_pos,'p_neg':p_neg}
    
    return result,r_pos,p_pos,r_neg,p_neg


def create_unit_speed_corr_all_vs_loc(su,posnew,rect,speed,fname2,file_to_save,fig_to_save,Fs=30000,bins=30,vmin=-1,thres=3,bins_loc=25,step=1,plotData=True,saveData=True):
    
    '''
    binsize in ms
    calculate spike frequency in each speed unit (per second);
    put all frequencies in speed bins;
    average them.
    
    '''
    
    #create spike train with su as indices of 1, others of 0
    spike_train = N.zeros(len(speed))
    N.put(spike_train,su.astype(int),1)    
    
    #get speed data in second bin & firing frequency each second
    Fir_freq,mean_speed_pers = [],[]
    for n in range(0,len(speed)-int(step*Fs),int(step*Fs)):
        Fir_freq.append(sum(spike_train[n:n+int(step*Fs)]))
        mean_speed_pers.append(N.mean(speed[n:n+int(step*Fs)]))
    mean_fir_freq=N.mean(Fir_freq)
    print "mean frequency all: %s " %mean_fir_freq
    bins_new=int(bins/float(step))
    spike_freq,bin_edges,binnumber= stats.binned_statistic(mean_speed_pers,Fir_freq,statistic='mean',bins=bins_new,range=[vmin,vmin+bins])
    spike_freq_std,bin_edges_std,binnumber_std= stats.binned_statistic(mean_speed_pers,Fir_freq,statistic='std',bins=bins_new,range=[vmin,vmin+bins])

    posnew_loc,speed_loc,rect_loc,ind_loc,spike_train_loc = [],[],[],[],[]
    for i in range(len(speed)):
        if speed[i]>thres:
            posnew_loc.append(posnew[i])
            speed_loc.append(speed[i])
            rect_loc.append(rect[i])
            ind_loc.append(i)
            spike_train_loc.append(spike_train[i]) 
       
    #get speed data in second bin & firing frequency each second
    Fir_freq_loc,mean_speed_pers_loc = [],[]
    for n in range(0,len(speed_loc)-int(step*Fs),int(step*Fs)):
        Fir_freq_loc.append(sum(spike_train_loc[n:n+int(step*Fs)]))
        mean_speed_pers_loc.append(N.mean(speed_loc[n:n+int(step*Fs)]))
    mean_fir_freq_loc=N.mean(Fir_freq_loc)
    print "mean frequency locomotion: %s " %mean_fir_freq_loc
    bins_loc_new=int(bins_loc/float(step))
    spike_freq_loc,bin_edges_loc,binnumber_loc= stats.binned_statistic(mean_speed_pers_loc,Fir_freq_loc,statistic='mean',bins=bins_loc_new,range=[thres,thres+bins_loc])
    spike_freq_std_loc,bin_edges_std_loc,binnumber_std_loc= stats.binned_statistic(mean_speed_pers_loc,Fir_freq_loc,statistic='std',bins=bins_loc_new,range=[thres,thres+bins_loc])
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    if plotData==True:
        bins_middle = []
        for n in range(len(bin_edges)-1):
            bins_middle.append((bin_edges[n+1]-bin_edges[n])/2+bin_edges[n])
        
        speed_x=[]
        for j in range(len(mean_speed_pers)):
            idx=(N.abs(bins_middle-mean_speed_pers[j])).argmin()
            speed_x.append(bins_middle[idx])
            
        bins_middle_loc = []
        for n in range(len(bin_edges_loc)-1):
            bins_middle_loc.append((bin_edges_loc[n+1]-bin_edges_loc[n])/2+bin_edges_loc[n])
        
        speed_x_loc=[]
        for j in range(len(mean_speed_pers_loc)):
            idx=(N.abs(bins_middle_loc-mean_speed_pers_loc[j])).argmin()
            speed_x_loc.append(bins_middle_loc[idx])
        
        f,(ax1,ax2)=pl.subplots(2,1,sharex=False,figsize=[8,6])
        f.text(0.04, 0.5, 'Averaged firing frequency (Hz)', va='center', rotation='vertical',fontsize=14)
        ax1.set_title('%s_speed correlation_locomotion'%fname2,size=14)
        ax1.plot(bins_middle,spike_freq,'r-')
        ax1.fill_between(bins_middle,spike_freq+spike_freq_std,spike_freq-spike_freq_std,facecolor='orange', alpha=0.5)
        ax2.plot(bins_middle_loc,spike_freq_loc,'b-')
        ax2.fill_between(bins_middle_loc,spike_freq_loc+spike_freq_std_loc,spike_freq_loc-spike_freq_std_loc,facecolor='darkgreen', alpha=0.5)
        ax2.set_xlabel('Speed (cm/s)',fontsize=14)
        c=fname2.rsplit('_',-1)[0]
        f.savefig('%s_u%s_speed_all_vs_loc.png' %(fig_to_save,c))
        f.savefig('%s_u%s_speed_all_vs_loc.eps' %(fig_to_save,c),transparent=True)
    
    if saveData==True:
        #write the shank number which should be added on the saved filename
        N.save('%s_%s_mean_fir_freq_all' %(file_to_save,fname2),spike_freq) 
        N.save('%s_%s_binedges_all' %(file_to_save,fname2),bin_edges) 
        N.save('%s_%s_speedx_all' %(file_to_save,fname2),speed_x)    
        N.save('%s_%s_mean_fir_freq_loc' %(file_to_save,fname2),spike_freq_loc) 
        N.save('%s_%s_binedges_loc' %(file_to_save,fname2),bin_edges_loc) 
        N.save('%s_%s_speedx_loc' %(file_to_save,fname2),speed_x_loc) 
        
    return spike_freq, speed_x, spike_freq_loc, speed_x_loc


def create_unit_place_field_correlation(pos_filt_all_trial,spike_train_loc,posnew_loc,res,fname2,file_to_save,fig_to_save,Fs=30000,binsize=3,plotData=True,saveData=True):
    
    binstart= N.arange(pos_filt_all_trial[N.argmin(pos_filt_all_trial)],pos_filt_all_trial[N.argmax(pos_filt_all_trial)]+0.001,binsize)

    #put indices in bins respectively
    tstart,tstop=[],[]    
    for i in range(len(binstart)-1):
        #find all indices in the range of bins
        result=N.ravel(N.argwhere(N.logical_and(pos_filt_all_trial>=binstart[i],pos_filt_all_trial<binstart[i+1])))
        i_in_result=N.ravel(N.argwhere(N.diff(result)>1)) # find the break of each trial for certain bins
        i_start_in_result=(i_in_result+1)
        i_start_in_result=N.insert(i_start_in_result,0,0) #add the first index of result
        i_in_result=N.insert(i_in_result,len(i_in_result),(len(result)-1))
        tstop.append(N.take(result,i_in_result)) #get teh raw index of each bin
        tstart.append(N.take(result,i_start_in_result))
    
   
    #calculate firing rate in each bin of each time range
    Fir_rate,Mean_Fir_rate,Mean_Fir_std=[],[],[]
    t_range=N.subtract(tstop,tstart)
    for j in range(len(tstart)):
        istart=tstart[j][:]
        istop=tstop[j][:]
        fir_rate_each=[]
        for k in range(len(istart)):
            fir_rate_each.append(sum(spike_train_loc[istart[k]:istop[k]])*Fs/t_range[j][k])
        Mean_Fir_rate.append(N.mean(fir_rate_each))
        Mean_Fir_std.append(N.std(fir_rate_each))
        Fir_rate.append(fir_rate_each)
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if plotData==True: 
        f,(ax1,ax2)=pl.subplots(2,1,sharex=True,figsize=[8,6])
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
        ax1.bar(binstart[:-1],Mean_Fir_rate,width=binsize-0.1,align='edge',color='#0504aa',alpha=0.7)        
        ax1.set_title('%s_place field preference_locomotion'%fname2,size=14)
        ax2.set_xlabel('Track position (cm)',fontsize=14)
        f.text(0.04, 0.5, 'Averaged firing frequency (Hz)', va='center', rotation='vertical',fontsize=14)
        for n in range(len(Fir_rate)):
            ax2.plot(N.linspace((binstart[n]+float(binsize)/2.0),(binstart[n]+float(binsize)/2.0),num=len(Fir_rate[n])),Fir_rate[n],'o',color='#A9A9A9',alpha=0.8) #make plots not overlapping
        ax2.plot(binstart[:-1]+float(binsize)/2.0,Mean_Fir_rate,'ro-')
        ax2.fill_between(binstart[:-1]+float(binsize)/2.0,N.array(Mean_Fir_rate)+N.array(Mean_Fir_std),N.array(Mean_Fir_rate)-N.array(Mean_Fir_std),facecolor='orange', alpha=0.5)
        f2,ax=pl.subplots(figsize=[8,4])
        ax.plot((N.array(range(len(pos_filt_all_trial)))/float(Fs)),pos_filt_all_trial)
        ax.set_xlabel('time (s)',fontsize=14)
        ax.set_ylabel('Track position (cm)',fontsize=14)
        c=fname2.rsplit('_',-1)[0]
        f.savefig('%s_u%s_place.png' %(fig_to_save,c))
        f.savefig('%s_u%s_place.eps' %(fig_to_save,c),transparent=True)
    
    
    if saveData==True:

        #write the unit and shank number which should be added on the saved filename
        N.save('%s_%s_Place_map_Mean_Fir_rate' %(file_to_save,fname2),Mean_Fir_rate) 
        N.save('%s_%s_place_binstart' %(file_to_save,fname2),binstart) 
        
    return Mean_Fir_rate, binstart,pos_filt_all_trial, Fir_rate

def create_heatmap_place_field (pos_filt_all_trial,pos_filt_per_trial,spike_train_loc,posnew_loc,res,fname2,file_to_save,fig_to_save,directory,Fs=30000,binsize=3,Tlength=400,plotData=True,plotrewpos=True,vmaxauto=True,vmaxValue=N.nan,saveData=True,cmap='jet',speedmap=False):
    '''
    res: break index of pos
    pos_filt_all_trial:filtered position
    
    '''
    
    if plotrewpos==True:    #if in heatmap of firing-place correlates vlines of reward position
        os.chdir(directory)
        pos_reward_start1_Tlength=N.load('pos_reward_start1_Tlength.npy')
        pos_reward_start2_Tlength=N.load('pos_reward_start2_Tlength.npy')
    
    #select the trials out with complete position
    start_complete,stop_complete=[],[]
    pos_start=N.take(pos_filt_all_trial,res)
    pos_stop=N.take(pos_filt_all_trial,N.subtract(res,1))
    pos_start=pos_start[:-1]
    pos_stop=pos_stop[1:]
    result_start=N.ravel(N.argwhere(pos_start<(min(pos_filt_all_trial)+binsize)))
    result_stop=N.ravel(N.argwhere(pos_stop>(max(pos_filt_all_trial)-binsize)))
    result=N.intersect1d(result_start,result_stop)
    start_complete=N.take(res[:-1],result)
    stop_complete=N.take(N.subtract(res,1)[1:],result)
    
    #put complete trials together
    pos_filt_complete,pos_filt_each,spike_train_each,spike_train_complete=[],[],[],[]
    for k in range(len(start_complete)):
        pos_filt_each.append(pos_filt_all_trial[start_complete[k]:stop_complete[k]])
        spike_train_each.append(spike_train_loc[start_complete[k]:stop_complete[k]])
        pos_filt_complete=N.concatenate((pos_filt_complete,pos_filt_all_trial[start_complete[k]:stop_complete[k]]))
        spike_train_complete=N.concatenate((spike_train_complete,spike_train_loc[start_complete[k]:stop_complete[k]]))
        
    binstart= N.arange(pos_filt_all_trial[N.argmin(pos_filt_all_trial)],pos_filt_all_trial[N.argmax(pos_filt_all_trial)]+0.001,binsize)
    
    #find start/stop indice of each bin for each trial
    tstart,tstop=[],[]
    for i in range(len(pos_filt_each)):
        tstart_each_track,tstop_each_track=[],[]
        for j in range(len(binstart)-1):
            result=N.ravel(N.argwhere(N.logical_and(pos_filt_each[i][:]>=binstart[j],pos_filt_each[i][:]<binstart[j+1])))
            tstart_each_track.append(result[0])
            tstop_each_track.append(result[-1])
        tstart.append(tstart_each_track)
        tstop.append(tstop_each_track)

    #calculate firing rate in each bin of each time range
    Fir_rate=[]
    t_range=N.subtract(tstop,tstart)
    if speedmap==True:
        for j in range(len(tstart)):
            istart=tstart[j][:]
            istop=tstop[j][:]
            ispike_train=spike_train_each[j][:]
            fir_rate_each=[]
            for k in range(len(istart)):
                fir_rate_each.append(N.mean(ispike_train[istart[k]:istop[k]]))
            Fir_rate.append(fir_rate_each)
            Mean_Fir_rate=N.mean(Fir_rate,axis=0)
            SEM_Fir_rate=stats.sem(Fir_rate)
    else:
        for j in range(len(tstart)):
            istart=tstart[j][:]
            istop=tstop[j][:]
            ispike_train=spike_train_each[j][:]
            fir_rate_each=[]
            for k in range(len(istart)):
                fir_rate_each.append(sum(ispike_train[istart[k]:istop[k]])*Fs/t_range[j][k])
            Fir_rate.append(fir_rate_each)
            Mean_Fir_rate=N.mean(Fir_rate,axis=0)
            SEM_Fir_rate=stats.sem(Fir_rate,axis=0)

    Fir_rate_plot=Fir_rate
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    if plotData==True:
        f,(ax1,ax2)=pl.subplots(2,1,sharex=True,figsize=(8,6),gridspec_kw={'height_ratios': [2, 1]})
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        if vmaxauto==True:
            im=ax1.imshow(Fir_rate_plot,cmap=cmap,interpolation="nearest",origin='lower',extent=[0,Tlength,0,len(Fir_rate_plot)],aspect='auto',vmin=0)
        elif vmaxauto==False and N.isnan(vmaxValue)==True:
            im=ax1.imshow(Fir_rate_plot,cmap=cmap,interpolation="nearest",origin='lower',extent=[0,Tlength,0,len(Fir_rate_plot)],aspect='auto',vmin=0,vmax=max(Mean_Fir_rate)+3*max(SEM_Fir_rate))
        else:
            im=ax1.imshow(Fir_rate_plot,cmap=cmap,interpolation="nearest",origin='lower',extent=[0,Tlength,0,len(Fir_rate_plot)],aspect='auto',vmin=0,vmax=vmaxValue)
        ax2.set_xlabel('track position (cm)',fontsize=14)
        ax1.set_ylabel('trial No.',fontsize=14)
        ax1.set_title('%s_firing in place_locomotion'%fname2,size=14)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = ax1.figure.colorbar(im, cax=cax)
        if speedmap==True:
            cbar1.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
        else:
            cbar1.ax.set_ylabel('firing rate (Hz)', rotation=-90, va="bottom",fontsize=14)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = ax2.figure.colorbar(im, cax=cax)
        if speedmap==True:
            cbar2.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
            ax2.set_ylabel('speed (cm/s)',fontsize=15)
        else:
            cbar2.ax.set_ylabel('firing rate (Hz)', rotation=-90, va="bottom",fontsize=14)
            ax2.set_ylabel('firing rate (Hz)',fontsize=15)
        ax2.fill_between(binstart[:-1]+float(binsize)/2.0,Mean_Fir_rate+SEM_Fir_rate,Mean_Fir_rate-SEM_Fir_rate,facecolor='orange', alpha=0.5)
        ax2.plot(binstart[:-1]+float(binsize)/2.0,Mean_Fir_rate,'r')
        ax2.set_xlim([binstart[0],binstart[-1]])
        
        if plotrewpos==True: 
            ax1.vlines(pos_reward_start1_Tlength,0,len(Fir_rate_plot),colors='r',linestyle='dashed')
            ax1.vlines(pos_reward_start2_Tlength,0,len(Fir_rate_plot),colors='r',linestyle='dashed')
            ax1.vlines(pos_reward_start1_Tlength-Tlength/16,0,len(Fir_rate_plot),colors='g',linestyle='dashed')
            ax1.vlines(pos_reward_start2_Tlength-Tlength/16,0,len(Fir_rate_plot),colors='g',linestyle='dashed')
            ax2.vlines(pos_reward_start1_Tlength,min(Mean_Fir_rate-SEM_Fir_rate),max(Mean_Fir_rate+SEM_Fir_rate),colors='r',linestyle='dashed')
            ax2.vlines(pos_reward_start2_Tlength,min(Mean_Fir_rate-SEM_Fir_rate),max(Mean_Fir_rate+SEM_Fir_rate),colors='r',linestyle='dashed')
            ax2.vlines(pos_reward_start1_Tlength-Tlength/16,min(Mean_Fir_rate-SEM_Fir_rate),max(Mean_Fir_rate+SEM_Fir_rate),colors='g',linestyle='dashed')
            ax2.vlines(pos_reward_start2_Tlength-Tlength/16,min(Mean_Fir_rate-SEM_Fir_rate),max(Mean_Fir_rate+SEM_Fir_rate),colors='g',linestyle='dashed')
        c=fname2.rsplit('_',-1)[0]
        f.savefig('%s_u%s_place_heatmap.png' %(fig_to_save,c))
        f.savefig('%s_u%s_place_heatmap.eps' %(fig_to_save,c),transparent=True)
        
    if saveData==True:

        #write the unit and shank number which should be added on the saved filename
        N.save('%s_%s_Place_map_Mean_Fir_rate' %(file_to_save,fname2),Mean_Fir_rate) 
        N.save('%s_%s_place_binstart' %(file_to_save,fname2),binstart) 

    return Fir_rate

def create_heatmap_transition_immo_loc(su,speed,fname2,file_to_save,fig_to_save,files,tdur=3,tstart=5,t1=1,t2=2,thres_immo=0.5,thres_loc=2,Fs=30000,plotData=True,saveData=True):
    '''select data during transition & rasterplot'''
    
    #get spike train for firing rate calculation
    spike_train = N.zeros(len(speed))
    N.put(spike_train,su.astype(int),1)
    
    #select the trainsition from immo to loc or loc to immo
    ind,ind_loctoimmo,speed_loctoimmo,su_loctoimmo,spike_train_loctoimmo,speed_loctoimmo_4mean,rew_loctoimmo=[],[],[],[],[],[],[]
    ind2,ind_itol,speed_itol,su_itol,spike_train_itol,speed_itol_4mean,rew_itol=[],[],[],[],[],[],[]
    for i in range(len(speed)-tdur*Fs-1):
        if i+(tdur+1)*Fs<len(speed):
            if speed[i]>thres_immo and speed[i+1]<=thres_immo and speed[i-tdur*Fs]>=thres_loc and speed[i+tdur*Fs]<=thres_immo and speed[i-int(t1*Fs)]>=thres_loc and speed[i+int(t1*Fs)]<=thres_immo and speed[i-int(t2*Fs)]>=thres_loc and speed[i+int(t2*Fs)]<=thres_immo:
                ind.append(i)
                ind_loctoimmo.append(range(i-tstart*Fs,i+tdur*Fs))
                speed_loctoimmo.append(speed[range(i-tstart*Fs,i+tdur*Fs)])
                su_i=N.intersect1d(su,range(i-tstart*Fs,i+tdur*Fs))-i
                su_loctoimmo.append(su_i)
                speed_loctoimmo_4mean.append(speed[range(i-tstart*Fs,i+(tdur+1)*Fs)])
                spike_train_loctoimmo.append(spike_train[range(i-tstart*Fs,i+(tdur+1)*Fs)])
                rew_loctoimmo.append(files[range(i-tstart*Fs,i+tdur*Fs)])
                
            elif speed[i]<=thres_immo and speed[i+1]>thres_immo and speed[i+tdur*Fs]>=thres_loc and speed[i-tdur*Fs]<=thres_immo and speed[i+int(t1*Fs)]>=thres_loc and speed[i-int(t1*Fs)]<=thres_immo and speed[i+int(t2*Fs)]>=thres_loc and speed[i-int(t2*Fs)]<=thres_immo:
                ind2.append(i)
                ind_itol.append(range(i-tdur*Fs,i+tstart*Fs))
                speed_itol.append(speed[range(i-tdur*Fs,i+tstart*Fs)])
                su_i=N.intersect1d(su,range(i-tdur*Fs,i+tstart*Fs))-i
                su_itol.append(su_i)
                speed_itol_4mean.append(speed[range(i-tdur*Fs,i+(tstart+1)*Fs)])
                spike_train_itol.append(spike_train[range(i-tdur*Fs,i+(tstart+1)*Fs)])
                rew_itol.append(files[range(i-tdur*Fs,i+tstart*Fs)])

    if len(ind)!=0:
        #calculating mean firing rate and speed per second range
        Fir_freq_loctoimmo,speed_heatmap=[],[]
        for n in range(len(speed_loctoimmo)):
            fir_loctoimmo,speed_binned_loctoimmo=[],[]
            for j in range(0,len(speed_loctoimmo_4mean[n][:])-Fs,Fs):
                fir_loctoimmo.append(sum(spike_train_loctoimmo[n][j:j+Fs]))
                speed_binned_loctoimmo.append(N.mean(speed_loctoimmo_4mean[n][j:j+Fs]))
            Fir_freq_loctoimmo.append(fir_loctoimmo)
            speed_heatmap.append(speed_binned_loctoimmo)
            mean_fir_freq_loctoimmo=N.mean(Fir_freq_loctoimmo,axis=0)
            mean_speed_loctoimmo=N.mean(speed_heatmap,axis=0)
    else:
       print 'no transition fitting criterion for loc to immo' 
    
    if len(ind2)!=0:
        Fir_freq_itol,speed_heatmap_itol=[],[]
        for n in range(0,len(speed_itol)):
            fir_itol,speed_binned_itol=[],[]
            for j in range(0,len(speed_itol_4mean[n][:])-Fs,Fs):
                fir_itol.append(sum(spike_train_itol[n][j:j+Fs]))
                speed_binned_itol.append(N.mean(speed_itol_4mean[n][j:j+Fs]))
            Fir_freq_itol.append(fir_itol)
            speed_heatmap_itol.append(speed_binned_itol)
            mean_fir_freq_itol=N.mean(Fir_freq_itol,axis=0)
            mean_speed_itol=N.mean(speed_heatmap_itol,axis=0)
    else:
       print 'no transition fitting criterion for itol' 
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    if N.array(speed_loctoimmo).ndim == 1:
        speed_loctoimmo=N.expand_dims(speed_loctoimmo,axis=0)
    if N.array(speed_itol).ndim == 1:
        speed_itol=N.expand_dims(speed_itol,axis=0)
 
    #heatmap of speed vs rasterplot of spike timestamps
    if plotData==True:
        f,(ax1,ax2,ax3)=pl.subplots(3,1,figsize=(8,9))
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
        ax1.eventplot(su_loctoimmo,linelengths=0.8)
        ax1.set_xlim([-tstart*Fs,tdur*Fs])
        im=ax2.imshow(speed_loctoimmo,cmap='Greys',interpolation="nearest",origin='lower',extent=[-tstart,tdur,0,len(speed_loctoimmo)],aspect='auto',vmin=-1,vmax=25)
        ax3.set_xlabel('time (s)',fontsize=14)
        ax2.set_ylabel('trial No.',fontsize=14)
        ax3.set_ylabel('trial No.',fontsize=14)
        ax1.set_title('%s_firing during transition_loc to immo'%fname2,size=14)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = ax2.figure.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = ax1.figure.colorbar(im, cax=cax1)
        cbar1.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
        ax3.set_xlim([-tstart,tdur])
        x=N.arange(-tstart,tdur,float(1)/Fs)
        for k in range(0,len(rew_loctoimmo)):
                ax3.plot(x,rew_loctoimmo[k]+int(k))
        c=fname2.rsplit('_',-1)[0]
        f.savefig('%s_%s_u%s_transition_ltoi.png' %(fig_to_save,fname2,c))
        f.savefig('%s_%s_u%s_transition_ltoi.eps' %(fig_to_save,fname2,c),transparent=True)
        
        f,(ax1,ax2,ax3)=pl.subplots(3,1,figsize=(8,9))
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
        ax1.eventplot(su_itol,linelengths=0.8)
        ax1.set_xlim([-tdur*Fs,tstart*Fs])
        im=ax2.imshow(speed_itol,cmap='Greys',interpolation="nearest",origin='lower',extent=[-tdur,tstart,0,len(speed_itol)],aspect='auto',vmin=-1,vmax=25)
        ax3.set_xlabel('time (s)',fontsize=14)
        ax2.set_ylabel('trial No.',fontsize=14)
        ax3.set_ylabel('trial No.',fontsize=14)
        ax1.set_title('%s_firing during transition_immo_to_loc'%fname2,size=14)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = ax2.figure.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = ax1.figure.colorbar(im, cax=cax1)
        cbar1.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
        cbar1.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
        ax3.set_xlim([-tdur,tstart])
        for k in range(0,len(rew_itol)):
                ax3.plot(N.arange(-tdur,tstart,float(1)/Fs),rew_itol[k]+int(k))
        f.savefig('%s_%s_u%s_transition_itol.png' %(fig_to_save,fname2,c))
        f.savefig('%s_%s_u%s_transition_itol.eps' %(fig_to_save,fname2,c),transparent=True)
        
        if len(ind)!=0 and len(ind2)!=0:
            f,(ax1,ax2)=pl.subplots(2,1,figsize=(8,4),sharey=True)
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
            ax1.set_title('%s_speed during transition'%fname2,size=14)
            ylimit_speed=[int(min(min(mean_speed_loctoimmo),min(mean_speed_itol)))-0.2,ma.ceil(max(max(mean_speed_loctoimmo),max(mean_speed_itol)))+0.2]
            ylimit_freq=[int(min(min(mean_fir_freq_loctoimmo),min(mean_fir_freq_itol)))-0.2,ma.ceil(max(max(mean_fir_freq_loctoimmo),max(mean_fir_freq_itol)))+0.2]
            for n in range(len(speed_loctoimmo)):
                ax1.plot(range(-tstart*Fs,tdur*Fs),speed_loctoimmo[n][:])
            for n in range(len(speed_itol)):
                ax2.plot(range(-tdur*Fs,tstart*Fs),speed_itol[n][:])
            f,(ax1,ax2)=pl.subplots(2,1,figsize=(8,6))
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
            ax1.set_title('%s_speed during transition_ltoi vs itol'%fname2,size=14)
            ax1.bar(N.subtract(range(-tstart,tdur),-0.5),mean_speed_loctoimmo,width=0.9,color='#0504aa',alpha=0.7)
            ax1.set_ylabel('speed (cm/s)',color='#0504aa',fontsize=14)
            ax1.tick_params(axis='y', labelcolor='#0504aa')
            ax3 = ax1.twinx()
            ax3.plot(N.subtract(range(-tstart,tdur),-0.5),mean_fir_freq_loctoimmo,color='darkorange',marker='o')
            ax3.set_ylabel('mean firing freq (Hz)',color='darkorange',fontsize=14)
            ax3.set_xlim([-tstart,tdur])
            ax3.tick_params(axis='y', labelcolor='darkorange')
            ax2.bar(N.subtract(range(-tdur,tstart),-0.5),mean_speed_itol,width=0.9,color='#0504aa',alpha=0.7)
            ax2.tick_params(axis='y', labelcolor='#0504aa') 
            ax2.set_ylabel('speed (cm/s)',color='#0504aa',fontsize=14)
            ax2.set_xlabel('time (s)',fontsize=14)
            ax4 = ax2.twinx()
            ax4.set_xlim([-tdur,tstart])
            ax4.plot(N.subtract(range(-tdur,tstart),-0.5),mean_fir_freq_itol,color='darkorange',marker='o')
            ax4.set_ylabel('mean firing freq (Hz)',color='darkorange',fontsize=14)
            ax4.tick_params(axis='y', labelcolor='darkorange')
            ax3.set_ylim(ylimit_freq)
            ax4.set_ylim(ylimit_freq)
            ax1.set_ylim(ylimit_speed)
            ax2.set_ylim(ylimit_speed)
            f.savefig('%s_%s_u%s_transition_meanfir.png' %(fig_to_save,fname2,c))
            f.savefig('%s_%s_u%s_transition_meanfir.eps' %(fig_to_save,fname2,c),transparent=True)
    
    #save the mean firing rate and mean speed in second bins
    if saveData==True:
        #write the unit and shank number which should be added on the saved filename
        if len(ind)!=0:
            N.save('%s_%s_mean_fir_freq_loctoimmo' %(file_to_save,fname2),mean_fir_freq_loctoimmo) 
            N.save('%s_%s_mean_speed_loctoimmo' %(file_to_save,fname2),mean_speed_loctoimmo)
        if len(ind2)!=0:
            N.save('%s_%s_mean_fir_freq_itol' %(file_to_save,fname2),mean_fir_freq_itol) 
            N.save('%s_%s_mean_speed_itol' %(file_to_save,fname2),mean_speed_itol)
    
def create_heatmap_transition_immo_loc_high_resol(su,speed,fname2,file_to_save,fig_to_save,tdur=3,tstart=5,binspers=10,t1=1,t2=2,thres_immo=0.5,thres_loc=2,Fs=30000,plotData=True,saveData=True):
    '''select data during transition & rasterplot'''
    
    #get spike train for firing rate calculation
    spike_train = N.zeros(len(speed))
    N.put(spike_train,su.astype(int),1) 
    
    #select the trainsition from immo to loc or loc to immo
    ind,ind_loctoimmo,speed_loctoimmo,su_loctoimmo,spike_train_loctoimmo,speed_loctoimmo_4mean=[],[],[],[],[],[]
    ind2,ind_itol,speed_itol,su_itol,spike_train_itol,speed_itol_4mean=[],[],[],[],[],[]
    for i in range(len(speed)-tdur*Fs-1):
        if i+(tdur+1)*Fs<len(speed):
            if speed[i]>thres_immo and speed[i+1]<=thres_immo and speed[i-tdur*Fs]>=thres_loc and speed[i+tdur*Fs]<=thres_immo and speed[i-int(t1*Fs)]>=thres_loc and speed[i+int(t1*Fs)]<=thres_immo and speed[i-int(t2*Fs)]>=thres_loc and speed[i+int(t2*Fs)]<=thres_immo:
                ind.append(i)
                ind_loctoimmo.append(range(i-tstart*Fs,i+tdur*Fs))
                speed_loctoimmo.append(speed[range(i-tstart*Fs,i+tdur*Fs)])
                su_i=N.intersect1d(su,range(i-tstart*Fs,i+tdur*Fs))-i
                su_loctoimmo.append(su_i)
                speed_loctoimmo_4mean.append(speed[range(i-tstart*Fs,i+(tdur+1)*Fs)])
                spike_train_loctoimmo.append(spike_train[range(i-tstart*Fs,i+(tdur+1)*Fs)])
                
            elif speed[i]<=thres_immo and speed[i+1]>thres_immo and speed[i+tdur*Fs]>=thres_loc and speed[i-tdur*Fs]<=thres_immo and speed[i+int(t1*Fs)]>=thres_loc and speed[i-int(t1*Fs)]<=thres_immo and speed[i+int(t2*Fs)]>=thres_loc and speed[i-int(t2*Fs)]<=thres_immo:
                ind2.append(i)
                ind_itol.append(range(i-tdur*Fs,i+tstart*Fs))
                speed_itol.append(speed[range(i-tdur*Fs,i+tstart*Fs)])
                su_i=N.intersect1d(su,range(i-tdur*Fs,i+tstart*Fs))-i
                su_itol.append(su_i)
                speed_itol_4mean.append(speed[range(i-tdur*Fs,i+(tstart+1)*Fs)])
                spike_train_itol.append(spike_train[range(i-tdur*Fs,i+(tstart+1)*Fs)])
    
    if len(ind)!=0:
    #calculating mean firing rate and speed per 200 mili second range if binspers=5
        Fir_freq_loctoimmo,speed_heatmap=[],[]
        for n in range(len(speed_loctoimmo)):
            fir_loctoimmo,speed_binned_loctoimmo=[],[]
            for j in range(0,len(speed_loctoimmo_4mean[n][:])-Fs,Fs/binspers):
                fir_loctoimmo.append(sum(spike_train_loctoimmo[n][j:j+Fs/binspers])*binspers)
                speed_binned_loctoimmo.append(N.mean(speed_loctoimmo_4mean[n][j:j+Fs/binspers]))
            Fir_freq_loctoimmo.append(fir_loctoimmo)
            speed_heatmap.append(speed_binned_loctoimmo)
            mean_fir_freq_loctoimmo=N.mean(Fir_freq_loctoimmo,axis=0)
            sem_fir_freq_loctoimmo=stats.sem(Fir_freq_loctoimmo,axis=0)
            mean_speed_loctoimmo=N.mean(speed_heatmap,axis=0)
            sem_speed_loctoimmo=stats.sem(speed_heatmap,axis=0)
    else:
       print 'no transition fitting criterion for loc to immo' 
    
    if len(ind2)!=0:
        Fir_freq_itol,speed_heatmap_itol=[],[]
        for n in range(0,len(speed_itol)):
            fir_itol,speed_binned_itol=[],[]
            for j in range(0,len(speed_itol_4mean[n][:])-Fs,Fs/binspers):
                fir_itol.append(sum(spike_train_itol[n][j:j+Fs/binspers])*binspers)
                speed_binned_itol.append(N.mean(speed_itol_4mean[n][j:j+Fs/binspers]))
            Fir_freq_itol.append(fir_itol)
            speed_heatmap_itol.append(speed_binned_itol)
            mean_fir_freq_itol=N.mean(Fir_freq_itol,axis=0)
            sem_fir_freq_itol=stats.sem(Fir_freq_loctoimmo,axis=0)
            mean_speed_itol=N.mean(speed_heatmap_itol,axis=0)
            sem_speed_itol=stats.sem(speed_heatmap_itol,axis=0)
    else:
       print 'no transition fitting criterion for itol' 
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    if N.array(speed_loctoimmo).ndim == 1:
        speed_loctoimmo=N.expand_dims(speed_loctoimmo,axis=0)
    if N.array(speed_itol).ndim == 1:
        speed_itol=N.expand_dims(speed_itol,axis=0)
        
    #heatmap of speed vs rasterplot of spike timestamps
    if plotData==True:
        f,(ax1,ax2)=pl.subplots(2,1,figsize=(8,4))
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
        ax1.eventplot(su_loctoimmo)
        ax1.set_xlim([-tstart*Fs,tdur*Fs])
        im=ax2.imshow(speed_loctoimmo,cmap='jet',interpolation="nearest",origin='lower',extent=[-tstart,tdur,0,len(speed_loctoimmo)],aspect='auto')
        ax2.set_xlabel('time (s)',fontsize=14)
        ax2.set_ylabel('trial No.',fontsize=14)
        ax1.set_title('%s_firing during transition_loc to immo'%fname2,size=14)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = ax2.figure.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = ax1.figure.colorbar(im, cax=cax1)
        cbar1.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
        c=fname2.rsplit('_',-1)[0]
        f.savefig('%s_u%s_transition_ltoi_high_resol.png' %(fig_to_save,c))
        f.savefig('%s_u%s_transition_ltoi_high_resol.eps' %(fig_to_save,c),transparent=True)
        
        f,(ax1,ax2)=pl.subplots(2,1,figsize=(8,4))
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
        ax1.eventplot(su_itol)
        ax1.set_xlim([-tdur*Fs,tstart*Fs])
        im=ax2.imshow(speed_itol,cmap='jet',interpolation="nearest",origin='lower',extent=[-tdur,tstart,0,len(speed_itol)],aspect='auto')
        ax2.set_xlabel('time (s)',fontsize=14)
        ax2.set_ylabel('trial No.',fontsize=14)
        ax1.set_title('%s_firing during transition_immo_to_loc'%fname2,size=14)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = ax2.figure.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = ax1.figure.colorbar(im, cax=cax1)
        cbar1.ax.set_ylabel('speed (cm/s)', rotation=-90, va="bottom",fontsize=14)
        f.savefig('%s_u%s_transition_itol_high_resol.png' %(fig_to_save,c))
        f.savefig('%s_u%s_transition_itol_high_resol.eps' %(fig_to_save,c),transparent=True)
        
        if len(ind)!=0 and len(ind2)!=0:
            f,(ax1,ax2)=pl.subplots(2,1,figsize=(8,4))
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
            ax1.set_title('%s_speed during transition'%fname2,size=14)
            for n in range(len(speed_loctoimmo)):
                ax1.plot(range(-tstart*Fs,tdur*Fs),speed_loctoimmo[n][:])
            for n in range(len(speed_itol)):
                ax2.plot(range(-tdur*Fs,tstart*Fs),speed_itol[n][:])
               
            mean_fir_freq_loctoimmo=ndimage.gaussian_filter(mean_fir_freq_loctoimmo,2,mode='nearest',truncate=2)
            sem_fir_freq_loctoimmo=ndimage.gaussian_filter(sem_fir_freq_loctoimmo,2,mode='nearest',truncate=2)
            mean_fir_freq_itol=ndimage.gaussian_filter(mean_fir_freq_itol,2,mode='nearest',truncate=2)
            sem_fir_freq_itol=ndimage.gaussian_filter(sem_fir_freq_itol,2,mode='nearest',truncate=2)
            mean1=mean_fir_freq_loctoimmo
            sem1=sem_fir_freq_loctoimmo
            mean2=mean_fir_freq_itol
            sem2=sem_fir_freq_itol
            
            f,(ax1,ax2)=pl.subplots(2,1,figsize=(8,6),sharey=True)
            f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
            ax1.set_title('%s_speed during transition_ltoi vs itol'%fname2,size=14)
            ax1.plot(N.subtract(N.arange(-tstart,tdur,float(1)/binspers),-(float(1)/binspers)/2),mean_speed_loctoimmo,color='#0504aa',alpha=0.7)
            ax1.fill_between(N.subtract(N.arange(-tstart,tdur,float(1)/binspers),-(float(1)/binspers)/2),mean_speed_loctoimmo+sem_speed_loctoimmo,mean_speed_loctoimmo-sem_speed_loctoimmo,facecolor='#0504aa', alpha=0.5)
            
            ax1.set_ylabel('speed (cm/s)',color='#0504aa',fontsize=14)
            ax1.tick_params(axis='y', labelcolor='#0504aa')
            ax3 = ax1.twinx()
            ax3.plot(N.subtract(N.arange(-tstart,tdur,float(1)/binspers),-(float(1)/binspers)/2),mean_fir_freq_loctoimmo,color='darkorange')
            ax3.fill_between(N.subtract(N.arange(-tstart,tdur,float(1)/binspers),-(float(1)/binspers)/2),mean1+sem1,mean1-sem1,facecolor='orange', alpha=0.5)
            
            ax3.set_ylabel('mean firing freq (Hz)',color='darkorange',fontsize=14)
            ax3.tick_params(axis='y', labelcolor='darkorange')
            ax2.plot(N.subtract(N.arange(-tdur,tstart,float(1)/binspers),-(float(1)/binspers)/2),mean_speed_itol,color='#0504aa',alpha=0.7)
            ax2.fill_between(N.subtract(N.arange(-tdur,tstart,float(1)/binspers),-(float(1)/binspers)/2),mean_speed_itol+sem_speed_itol,mean_speed_itol-sem_speed_itol,facecolor='#0504aa', alpha=0.5)
            
            ax2.tick_params(axis='y', labelcolor='#0504aa') 
            ax2.set_ylabel('speed (cm/s)',color='#0504aa',fontsize=14)
            ax2.set_xlabel('time (s)',fontsize=14)
            ax4 = ax2.twinx()
            ax4.plot(N.subtract(N.arange(-tdur,tstart,float(1)/binspers),-(float(1)/binspers)/2),mean_fir_freq_itol,color='darkorange')
            ax4.fill_between(N.subtract(N.arange(-tdur,tstart,float(1)/binspers),-(float(1)/binspers)/2),mean2+sem2,mean2-sem2,facecolor='orange', alpha=0.5)
            
            ax4.set_ylabel('mean firing freq (Hz)',color='darkorange',fontsize=14)
            ax4.tick_params(axis='y', labelcolor='darkorange')
            f.savefig('%s_u%s_%s_transition_meanfir_high_resol.png' %(fig_to_save,c,fname2))
            f.savefig('%s_u%s_%s_transition_meanfir_high_resol.eps' %(fig_to_save,c,fname2),transparent=True)
    
    #save the mean firing rate and mean speed in second bins
    if saveData==True:
        if len(ind)!=0:
            N.save('%s_%s_mean_fir_freq_loctoimmo_high_resol' %(file_to_save,fname2),mean_fir_freq_loctoimmo) 
            N.save('%s_%s_mean_speed_loctoimmo_high_resol' %(file_to_save,fname2),mean_speed_loctoimmo)
        if len(ind2)!=0:
            N.save('%s_%s_mean_fir_freq_itol_high_resol' %(file_to_save,fname2),mean_fir_freq_itol) 
            N.save('%s_%s_mean_speed_itol_high_resol' %(file_to_save,fname2),mean_speed_itol)  

#%% mean activity change of OFF- or ON-SOMIs during running onset/offset phase
def compare_pre_post_transition_itol_locotoimmo(run_onset=True,binspers=5,tpre=3,transitioninsec=3):
    '''
    Keyword arguments: 
    run_onset: if true, load running onsets firing of all OFF & ON-SOMIs; otherwise data during running offsets
    Inputs:
    mean firing rate at on/offset phases of all OFF- & ON-SOMIs
    '''
    
    target_dir="/data/Fig3"
    os.chdir(target_dir)
    fig_to_save=tkFileDialog.asksaveasfilename(title="Choose target figname") 
    
    ### select data of running onsets or offsets
    if run_onset==True: 
        ON=op.open_helper('INC_mean_fir_itol_Summary.npy')
        OFF=op.open_helper('DEC_mean_fir_itol_Summary.npy')
    else:
        ON=op.open_helper('INC_mean_fir_locotoimmo_Summary.npy')
        OFF=op.open_helper('DEC_mean_fir_locotoimmo_Summary.npy')
    
    ### get mean activity difference between immobility and locomotion during onset/offset
    PRE_ON,POST_ON,DIFF_ON=[],[],[]
    for n in range(len(ON)):
        pre=N.mean(ON[n][(transitioninsec-tpre)*binspers:transitioninsec*binspers])
        post=N.mean(ON[n][transitioninsec*binspers:(transitioninsec+tpre)*binspers])
        PRE_ON.append(pre)
        POST_ON.append(post)
        DIFF_ON.append(post-pre)
        
    PRE_OFF,POST_OFF,DIFF_OFF=[],[],[]
    for n in range(len(OFF)):
        pre=N.mean(OFF[n][(transitioninsec-tpre)*binspers:transitioninsec*binspers])
        post=N.mean(OFF[n][transitioninsec*binspers:(transitioninsec+tpre)*binspers])
        PRE_OFF.append(pre)
        POST_OFF.append(post)
        DIFF_OFF.append(post-pre)
    
    ###  save activity difference between immobility and locomotion  
    N.save('%s_mean_fir_rate_diff_transition_pre_post_ON' %fig_to_save,DIFF_ON)
    N.save('%s_mean_fir_rate_diff_transition_pre_post_OFF' %fig_to_save,DIFF_OFF)
    
#%% lick/reward modulation of SOMIs FAM/NOV/ORI for SOMIs

#### detect reward position, reward onset, lick indices
def detect_rewardzone_pos_split_trials(thres=3,Fs=30000,Tlength=400,thr_artifact=3,thr_pos_intertrial=40,plotData=True):
    
    target_dir='/data/example_raw_for_position_extraction_spike_waveform/200721_#386_lin'
    os.chdir(target_dir)
    directory='#386_2020-07-21_16-05-13_test'
    os.chdir(directory)
    reward=op.open_helper('100_ADC3.continuous') 
    lick=op.open_helper('100_ADC2.continuous')
    posnew = N.load('estimate speed from position_corrected posnew.npy')
    
    most_com_pos_reward_start1, most_com_pos_reward_start2, reward_start_ind, pos_reward_start1_Tlength, pos_reward_start2_Tlength, dur_reward=reward_pos_relation(reward, posnew,thres=thres,Fs=Fs,Tlength=Tlength,plotData=plotData)
    rewardzone_ind_start=N.array(reward_start_ind)-2*Fs
    rewardzone_ind_end=N.array(reward_start_ind)+2*Fs
    res_start,res_end,res_mid=split_each_trial_pos(posnew,thr_artifact=thr_artifact,thr_pos_intertrial=thr_pos_intertrial)
    lick_start, lick_end =extract_lick_pulses_for_bootstrap(lick,thres=thres,Fs=Fs) 
    
    os.chdir(target_dir)
    os.chdir(directory)
    N.save('most_com_pos_reward_start1',most_com_pos_reward_start1)
    N.save('most_com_pos_reward_start2',most_com_pos_reward_start2)
    N.save('reward_start_ind',reward_start_ind)
    N.save('pos_reward_start1_Tlength',pos_reward_start1_Tlength)
    N.save('pos_reward_start2_Tlength',pos_reward_start2_Tlength)
    N.save('rewardzone_ind_start',rewardzone_ind_start)
    N.save('rewardzone_ind_end',rewardzone_ind_end)
    N.save('res_start',res_start)
    N.save('res_end',res_end)
    N.save('res_mid',res_mid)
    N.save('lick_start',lick_start)
    N.save('lick_end',lick_end)
    
    return most_com_pos_reward_start1, most_com_pos_reward_start2,rewardzone_ind_start,reward_start_ind,rewardzone_ind_end,res_start,res_end

#################################################
    
#################################################
def reward_pos_relation(reward, posnew,thres=3,Fs=30000,Tlength=400, plotData=True):
    #find out where the reward zone is

    reward_start_ind, reward_end_ind = [],[]
    for n in range(len(reward)-1):
        #detect upward threshold crossings
        if reward[n] < thres and reward[n+1] >= thres:
            reward_start_ind.append(n+1)
    for i in range(len(reward)-1):
        #detect downward threshold crossings
        if reward[i] >= thres and reward[i+1] < thres:
            reward_end_ind.append(i+1)
    
    reward_start_all=reward_start_ind[::4]
    reward_end_all=reward_end_ind[3::4]

    if reward_start_all[0]>=reward_end_all[0]:
        reward_end_all=reward_end_all[1:]
    elif reward_start_all[-1]>=reward_end_all[-1]:
        reward_start_all=reward_start_all[:-2]
    
    pos_reward_start=N.take(posnew,reward_start_all)
    dur_reward=N.subtract(reward_end_all,reward_start_all)/float(Fs)
    most_com_pos_reward_start1 = N.bincount(pos_reward_start[::2]).argmax()
    most_com_pos_reward_start2 = N.bincount(pos_reward_start[1::2]).argmax()
    pos_reward_start1_Tlength=most_com_pos_reward_start1*Tlength/float(max(posnew)-min(posnew))
    pos_reward_start2_Tlength=most_com_pos_reward_start2*Tlength/float(max(posnew)-min(posnew))
    
    if plotData==True:
        f,(ax1,ax2)=pl.subplots(2,1,sharex=True,figsize=[8,6])
        ax1.plot(posnew)
        ax2.plot(reward)
        for l in range(len(pos_reward_start)):
            ax1.vlines(reward_start_all[l],0,55,color='r',alpha=0.5)
            ax1.vlines(reward_end_all[l],0,55,color='g',alpha=0.5)
        ax1.hlines(most_com_pos_reward_start1,0,len(posnew),color='orange',linestyles='dashed')
        ax1.hlines(most_com_pos_reward_start2,0,len(posnew),color='orange',linestyles='dashed')
        f.savefig('reward.png' )
        f.savefig('reward.eps',transparent=True) 
        for n in range(len(reward_start_all)):
            reward_start_all[n]=int(reward_start_all[n]+0.94*Fs)
    
    return most_com_pos_reward_start1, most_com_pos_reward_start2, reward_start_all, pos_reward_start1_Tlength, pos_reward_start2_Tlength, dur_reward
    

def split_each_trial_pos(posnew,thr_artifact=3,thr_pos_intertrial=40):
    
    #get the break of each trial
    res=[]
    for l in range(1,len(posnew)):
        if posnew[l]-posnew[l-1]>-thr_pos_intertrial and posnew[l]-posnew[l-1]<-thr_artifact:
            posnew[l]=posnew[l-1]
        if posnew[l]-posnew[l-1]<-thr_pos_intertrial: 
            res.append(l)
    res_start=N.ravel(res[:-1])
    res_start=N.insert(res_start,0,0)
    res_end=N.ravel(res[:])
    
    res_mid,ind_del=[],[]
    pos_mid=min(posnew)+(max(posnew)-min(posnew))/2
    for n in range(len(res_start)):
        result=N.ravel(N.argwhere(posnew[res_start[n]:res_end[n]]==pos_mid))
        if len(result)>0:
            ind=result[-1]+res_start[n]
            res_mid.append(ind)
        else:
            ind_del.append(n)
    
    #keep the size of res_start, res_end, res_mid the same.
    res_start=N.delete(res_start,ind_del)
    res_end=N.delete(res_end,ind_del)
    
    return res_start,res_end,res_mid

def extract_lick_pulses_for_bootstrap(ADC,thres=3,Fs=30000):
    ''' Inputs:
        ADC: ADC recording of laser pulse
        Keyword arguments:
        thres: Threshold to detect laser pulses
        Fs: sampling frequency (in Hz)
        Outputs:
        laser pulse interval matrix for all detections
        spike number for all detected laser pulse intervals
    '''
    
    Pulse_start, Pulse_end = [],[]
    for n in range(len(ADC)-1):
        #detect upward threshold crossings
        if ADC[n] < thres and ADC[n+1] >= thres:
            Pulse_start.append(n+1)
        elif ADC[n] >= thres and ADC[n+1] < thres:
            Pulse_end.append(n+1)
    
    
    if Pulse_start[0]>=Pulse_end[0]:
        Pulse_end=Pulse_end[1:]
    elif Pulse_start[-1]>=Pulse_end[-1]:
        Pulse_start=Pulse_start[:-1]
    
    Pulse_interval = N.column_stack([Pulse_start, Pulse_end])  
    
    return Pulse_start, Pulse_end    

#######################################################################

### detect ORI indices in NOV (shifted reward sites)
def extract_ind_original_reward_pos(Fs=30000):
    
    directory = '/data/example_raw_for_position_extraction_spike_waveform/200402_#328_lin/#328_2020-04-02_15-56-34_trainshift'
    directory2 = '/data/example_raw_for_position_extraction_spike_waveform/200402_#328_lin/#328_2020-04-02_14-57-40_VRtest1'
    
    os.chdir(directory)
    posnew = N.load('estimate speed from position_corrected posnew.npy')
    os.chdir(directory2)
    most_com_pos_reward_start1=N.load('most_com_pos_reward_start1.npy')
    most_com_pos_reward_start2=N.load('most_com_pos_reward_start2.npy')
    
    result=N.ravel(N.ravel(N.argwhere(posnew==most_com_pos_reward_start1)))
    i_in_result=N.ravel(N.argwhere(N.diff(result)>Fs)) # find the break of each trial for certain bins
    i_start_in_result=(i_in_result+1)
    i_start_in_result=N.insert(i_start_in_result,0,0) #add the first index of result
    indstart=N.take(result,i_start_in_result)
    result2=N.ravel(N.ravel(N.argwhere((posnew==most_com_pos_reward_start2))))
    i_in_result2=N.ravel(N.argwhere(N.diff(result2)>Fs)) # find the break of each trial for certain bins
    i_start_in_result2=(i_in_result2+1)
    i_start_in_result2=N.insert(i_start_in_result2,0,0) #add the first index of result
    indstart2=N.take(result2,i_start_in_result2)
    ind_org_reward_pos=N.concatenate((indstart,indstart2))
    
    os.chdir(directory)
    N.save('index_original_reward_position',ind_org_reward_pos)

def extract_ind_original_rewardzone_start_end(Fs=30000):
    
    directory = '/data/example_raw_for_position_extraction_spike_waveform/200402_#328_lin/#328_2020-04-02_15-56-34_trainshift'
    directory2 = '/data/example_raw_for_position_extraction_spike_waveform/200402_#328_lin/#328_2020-04-02_14-57-40_VRtest1'
    
    os.chdir(directory)
    posnew = N.load('estimate speed from position_corrected posnew.npy')
    os.chdir(directory2)
    most_com_pos_reward_start1=N.load('most_com_pos_reward_start1.npy')
    most_com_pos_reward_start2=N.load('most_com_pos_reward_start2.npy')
    
    pos_rzone_start=(max(posnew)-min(posnew))/16
    result=N.ravel(N.ravel(N.argwhere(posnew==most_com_pos_reward_start1-pos_rzone_start)))
    i_in_result=N.ravel(N.argwhere(N.diff(result)>Fs)) # find the break of each trial for certain bins
    i_start_in_result=(i_in_result+1)
    i_start_in_result=N.insert(i_start_in_result,0,0) #add the first index of result
    indstart=N.take(result,i_start_in_result)
    result2=N.ravel(N.ravel(N.argwhere((posnew==most_com_pos_reward_start2-pos_rzone_start))))
    i_in_result2=N.ravel(N.argwhere(N.diff(result2)>Fs)) # find the break of each trial for certain bins
    i_start_in_result2=(i_in_result2+1)
    i_start_in_result2=N.insert(i_start_in_result2,0,0) #add the first index of result
    indstart2=N.take(result2,i_start_in_result2)
    ind_org_rewardzone_start=N.concatenate((indstart,indstart2))
    
    result=N.ravel(N.ravel(N.argwhere(posnew==most_com_pos_reward_start1)))
    i_in_result=N.ravel(N.argwhere(N.diff(result)>Fs)) # find the break of each trial for certain bins
    i_start_in_result=(i_in_result+1)
    i_start_in_result=N.insert(i_start_in_result,0,0) #add the first index of result
    indstart=N.take(result,i_start_in_result)
    result2=N.ravel(N.ravel(N.argwhere((posnew==most_com_pos_reward_start2))))
    i_in_result2=N.ravel(N.argwhere(N.diff(result2)>Fs)) # find the break of each trial for certain bins
    i_start_in_result2=(i_in_result2+1)
    i_start_in_result2=N.insert(i_start_in_result2,0,0) #add the first index of result
    indstart2=N.take(result2,i_start_in_result2)
    ind_org_reward_pos=N.concatenate((indstart,indstart2))
    ind_org_rewardzone_end=ind_org_reward_pos+3*Fs
    
    os.chdir(directory)
    N.save('index_original_rewardzone_start',ind_org_rewardzone_start)
    N.save('index_original_rewardzone_end',ind_org_rewardzone_end)
    
    return ind_org_rewardzone_start, ind_org_rewardzone_end

############################################################
    
### reward-site modulation lick/firing FAM/NOV/ORI & base-pre-post lick/firing rate quantification
def reward_rewardzone_firrate_prerepost_batch(thres=3,Fs=30000,iterations=1000,ymax=250,Tlength=400,thr_artifact=3,thr_pos_intertrial=40,tprepost=5,binspers=10,plotData=True,plotSpeed=False,saveData=False,lick=True,scan=False,shift=False,originshift=False,suc_trial=False,failure_trial=False):
    
    if shift==False and originshift==False:
        ### for FAM
        directory='/data/example_raw_for_position_extraction_spike_waveform/200721_#386_lin/#386_2020-07-21_16-05-13_test'
        directory2='/data/example_raw_for_position_extraction_spike_waveform/200721_#386_lin/output/ms3--all_shank2'
        os.chdir(directory2)
        su_batch=op.open_helper('all_timestamp_test_SOMI.batch')
    elif shift==True or originshift==True:
        ### for NOV (new reward sites in translocated environment) or ORI
        directory = '/data/example_raw_for_position_extraction_spike_waveform/200402_#328_lin/#328_2020-04-02_15-56-34_trainshift'
        directory2 = '/data/example_raw_for_position_extraction_spike_waveform/200402_#328_lin/output/ms3--all_shank2'
        os.chdir(directory2)
        su_batch=op.open_helper('all_timestamp_trainshift.batch')

    fig_to_save=tkFileDialog.asksaveasfilename(title="Choose target figure name") 
    file_to_save=tkFileDialog.asksaveasfilename(title="Choose target filename")
    lick_to_save=tkFileDialog.asksaveasfilename(title="Choose target lick figure name")
    
    grand_result1=lick_rewardzone_spike_corr_batch_for_batch(directory,directory2,su_batch,fig_to_save,file_to_save,lick_to_save,thres=thres,Fs=Fs,iterations=iterations,ymax=ymax,saveData=saveData,lick=lick,scan=scan,shift=shift,originshift=originshift)
    grand_result2=extract_reward_pulses_batch_for_batch(directory,directory2,su_batch,fig_to_save,file_to_save,thres=thres,Fs=Fs,Tlength=Tlength,thr_artifact=thr_artifact,thr_pos_intertrial=thr_pos_intertrial,tprepost=tprepost,binspers=binspers,plotData=plotData,plotSpeed=plotSpeed,saveData=saveData,originshift=originshift)

    return grand_result1, grand_result2

######################################################################
    
######################################################################
def lick_rewardzone_spike_corr_batch_for_batch(directory,directory2,su_batch,fig_to_save,file_to_save,lick_to_save,thres=3,Fs=30000,iterations=1000,ymax=250,saveData=True,lick=True,scan=True,shift=False,originshift=False):
    
    os.chdir(directory)
    if originshift==False:
        reward_start_ind=N.load('reward_start_ind.npy')
        rewardzone_ind_start=N.load('rewardzone_ind_start.npy')
        rewardzone_ind_end=N.load('rewardzone_ind_end.npy')
    else:
        reward_start_ind=N.load('index_original_reward_position.npy')
        rewardzone_ind_start=N.load('index_original_rewardzone_start.npy')
        rewardzone_ind_end=N.load('index_original_rewardzone_end.npy')
    res_start=N.load('res_start.npy')
    res_end=N.load('res_end.npy')
    res_mid=N.load('res_mid.npy')
    lick_start=N.load('lick_start.npy')
    
    if len(reward_start_ind)>3 and len(rewardzone_ind_start)>3 and len(rewardzone_ind_end)>3:
        if res_mid[0]>res_end[0]:
            res_start=res_start[1:]
            res_end=res_end[1:]
            
        if len(rewardzone_ind_start)>len(rewardzone_ind_end):
            rewardzone_ind_start=rewardzone_ind_start[:-1]
            if reward_start_ind[-1]>rewardzone_ind_end[-1]:
                reward_start_ind=reward_start_ind[:-1]
        elif len(rewardzone_ind_end)>len(rewardzone_ind_start):
            rewardzone_ind_end=rewardzone_ind_end[1:]
            if reward_start_ind[0]<rewardzone_ind_start[0]:
                reward_start_ind=reward_start_ind[1:]
        if len(rewardzone_ind_end)>len(rewardzone_ind_start):
            rewardzone_ind_end=rewardzone_ind_end[1:]
            if reward_start_ind[0]<rewardzone_ind_start[0]:
                reward_start_ind=reward_start_ind[1:]
        
        trial_ind_reward,lick_each_trial,ind=[],[],[]
        scan_ind_range=range(len(res_start))
        for n in range(len(reward_start_ind)):
            print 'n %s of total %s'%(n+1,len(reward_start_ind))
            for i in scan_ind_range:
                if reward_start_ind[n] in range(res_start[i],res_mid[i]):
                    trial_ind_reward.append(range(res_start[i],res_mid[i]))
                    lick_each=N.intersect1d(lick_start,range(res_start[i],res_mid[i]))
                    lick_each_trial.append(lick_each)
                    ind.append(n)
                    break
                elif reward_start_ind[n] in range(res_mid[i]+1,res_end[i]):
                    trial_ind_reward.append(range(res_mid[i]+1,res_end[i]))
                    lick_each=N.intersect1d(lick_start,range(res_mid[i]+1,res_end[i]))
                    lick_each_trial.append(lick_each)
                    ind.append(n)
                    break
            if scan==True:
                scan_ind_range=N.arange(i-2,len(res_start))
            print i
        rewardzone_ind_start=N.take(rewardzone_ind_start,ind) 
        rewardzone_ind_end=N.take(rewardzone_ind_end,ind) 
        reward_start_ind=N.take(reward_start_ind,ind)
        
        grand_result,mean_su_rate_prerepost=[],[]
        if lick==True:
            mean_rate_lick,lick_rate_pre,lick_rate_reward,lick_rate_rest,lick_rate_rewardzone,mean_rate_rewardzone,mean_rate_rest=licking_rate_pre_post_reward_pos(rewardzone_ind_start,reward_start_ind,rewardzone_ind_end,trial_ind_reward,lick_each_trial,directory,lick_to_save,Fs=Fs,saveData=saveData)
            N.save('%s_mean_lick_rate_prerepost'%lick_to_save,mean_rate_lick)
        for n in range(len(su_batch)):
            os.chdir(directory2)
            su=op.open_helper(su_batch[n])
            print "unit %s " %su_batch[n]
            fname2=su_batch[n]
            mean_rate,su_rate_pre,su_rate_reward,su_rate_rest,su_rate_rewardzone,mean_rate_rewardzone,mean_rate_rest=spike_rate_pre_reward_rest_zone(su,rewardzone_ind_start,reward_start_ind,rewardzone_ind_end,trial_ind_reward,fig_to_save,fname2,Fs=Fs,saveData=saveData,shift=shift,originshift=originshift)  
            mean_su_rate_prerepost.append(mean_rate)
            os.chdir(directory2)
            result={'su_id':fname2,'mean_rate_su_pre_reward_post':mean_rate,'mean_rate_rewardzone':mean_rate_rewardzone,'mean_rate_rest':mean_rate_rest}
            grand_result.append(result)
            pl.close('all')
        if saveData == True:
            N.save('%s_lick_rewardzone_spike_corr'%file_to_save,grand_result)
            N.save('%s_mean_fir_rate_prerepost'%file_to_save,mean_su_rate_prerepost)
    else:
        grand_result=[]
    return grand_result

def extract_reward_pulses_batch_for_batch(directory,directory2,su_batch,fig_to_save,file_to_save,thres=3,Fs=30000,Tlength=750,thr_artifact=3,thr_pos_intertrial=40,tprepost=5,binspers=10,plotData=True,plotSpeed=True,saveData=True,originshift=False):
    
    os.chdir(directory)
    posnew = N.load('estimate speed from position_corrected posnew.npy')
    if originshift==False:
        reward_start_ind=N.load('reward_start_ind.npy')
    else:
        reward_start_ind=N.load('index_original_reward_position.npy')
    ADC = op.open_helper('100_ADC3.continuous')
    
    if len(reward_start_ind)>3:
        os.chdir(directory2)
        if os.path.isdir('Data saved')==False:
            os.mkdir('Data saved')
        os.chdir('Data saved')
        c=directory.rsplit('_',-1)[-1]
        name=file_to_save.rsplit('/',-1)[-1]
        g=open("all_unit_reward_corr_%s.batch" %c,"w")
        grand_result=[]
        for n in range(len(su_batch)):
            os.chdir(directory2)
            su=op.open_helper(su_batch[n])
            print "unit %s " %su_batch[n]
            fname2=su_batch[n]
            if len(su)>0:
                spike_ind_all, spike_numb_Pulse, spike_numb_pre, pvalue,mean_fir_freq_rew=extract_reward_pulse_for_batch (su,posnew,ADC,fname2,fig_to_save,reward_start_ind,directory,thres=thres,Fs=Fs,trange=3,tprepost=tprepost,binspers=binspers,plotData=plotData,plotSpeed=plotSpeed)
                result={'su_id':fname2,'reward_start':reward_start_ind,'spike_ind_all':spike_ind_all,'spike_numb_Pulse':spike_numb_Pulse,'spike_numb_pre':spike_numb_pre,'pvalue':pvalue}
                grand_result.append(result)
                c=fname2.rsplit('_',-1)[0]
                os.chdir(directory2)
                os.chdir('Data saved')
                N.save('%s_u%s_mean_fir_freq_reward' %(file_to_save,c),mean_fir_freq_rew)
                g.write('%s_u%s_mean_fir_freq_reward.npy' %(name,c))
                g.write('\n')
                pl.close('all')
        g.close()
        if saveData == True:
            N.save('%s_extract_reward_pulse_batch' %file_to_save,grand_result)
    else:
        grand_result=[]
        
    return grand_result

####################################################################    
def licking_rate_pre_post_reward_pos(rewardzone_ind_start,reward_start_ind,rewardzone_ind_end,trial_ind_reward,lick_each_trial,fname,fig_to_save,Fs=30000,saveData=True):
    

    spike_numb_pre,spike_numb_reward,spike_numb_rest,lick_rate_pre,lick_rate_reward,lick_rate_rest,lick_rate_rewardzone,success_count=[],[],[],[],[],[],[],[]
    for m in range(len(rewardzone_ind_start)):
        #find indices in the range of pulse to get spike numbers
        pre=N.argwhere((N.logical_and(lick_each_trial[m]>=rewardzone_ind_start[m], lick_each_trial[m]<=(reward_start_ind[m]))))
        reward=N.argwhere((N.logical_and(lick_each_trial[m]>=reward_start_ind[m], lick_each_trial[m]<=(rewardzone_ind_end[m]))))
        spike_numb_pre.append(len(pre))
        spike_numb_reward.append(len(reward))
        lick_pre=N.take(lick_each_trial[m],pre)
        lick_reward=N.take(lick_each_trial[m],reward)
        res = N.setdiff1d(lick_each_trial[m],lick_pre)
        lick_rest = N.setdiff1d(res,lick_reward)
        spike_numb_rest.append(len(lick_rest))
        lick_rate_pre.append(len(pre)*Fs/float(reward_start_ind[m]-rewardzone_ind_start[m]))
        lick_rate_reward.append(len(reward)*Fs/float(rewardzone_ind_end[m]-reward_start_ind[m]))
        lick_rate_rest.append(len(lick_rest)*Fs/float(len(trial_ind_reward[m])-(rewardzone_ind_end[m]-rewardzone_ind_start[m])))
        lick_rate_rewardzone.append((len(pre)+len(reward))*Fs/float(rewardzone_ind_end[m]-rewardzone_ind_start[m]))
    for n in range(len(lick_rate_pre)):
        if lick_rate_pre[n]>lick_rate_rest[n]:
            success_count.append(n)
    suc_rate=float(len(success_count))/len(lick_rate_pre)
    print 'performance score %s' %suc_rate
    mean_rate,sem=[],[]
    mean_rate.append(N.mean(lick_rate_pre))
    mean_rate.append(N.mean(lick_rate_reward))
    mean_rate.append(N.mean(lick_rate_rest))
    sem.append(stats.sem(lick_rate_pre))
    sem.append(stats.sem(lick_rate_reward))
    sem.append(stats.sem(lick_rate_rest))
    mean_rate_rewardzone=N.mean(lick_rate_rewardzone)
    mean_rate_rest=N.mean(lick_rate_rest)
    sem_rewardzone=stats.sem(lick_rate_rewardzone)
    sem_rest=stats.sem(lick_rate_rest)
    print 'mean licking rate of reward zone pre_reward_base %s Hz' %mean_rate
    uname=fname.split('_')[-1]
    fn=fig_to_save.split('/')[-1]
    if len(lick_rate_pre)>=3 and len(lick_rate_reward)>=3 and len(lick_rate_rest)>=3:
        pvalue, Num_df, DEN_df, F_value, pvalue12, pvalue13, pvalue23 = AN.ANOVA_RM_1way_or_kruskal(lick_rate_pre,lick_rate_reward,lick_rate_rest,fig_to_save,uname,fn,script='lickprerepost')
    
        if N.isnan(Num_df)==False:
            if pvalue>0.05:
                print 'lick rates no significant, RM one way ANOVA, F(%s,%s):%s, p=%s ' %(Num_df,DEN_df,F_value,pvalue)
            else:
                print 'lick rate significant change pre reward base, RM one way ANOVA, F(%s,%s):%s, p=%s ' %(Num_df,DEN_df,F_value,pvalue)
                print 'p value between groups: pre-base %s' %pvalue13
                print 'p value between groups: pre-reward %s' %pvalue12
                print 'p value between groups: reward-base %s' %pvalue23
        else:
            if pvalue>0.05:
                print 'lick rates no significant, Kuskal-Wallis test, p=%s ' %pvalue
            else:
                print 'lick rate significant change pre reward base, Kuskal-Wallis test, p=%s ' %pvalue
                print 'p value between groups: pre-base %s' %pvalue13
                print 'p value between groups: pre-reward %s' %pvalue12
                print 'p value between groups: reward-base %s' %pvalue23
        
        statistic,pvalue4=stats.ttest_rel(lick_rate_rewardzone,lick_rate_rest)
    
     
        heights = [max(mean_rate+sem), max(mean_rate+sem)+1, max(mean_rate+sem)+2, max(mean_rate+sem)+3]
        bars = N.arange(len(heights)+1)
        
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        
        f,(ax1,ax2)=pl.subplots(1,2,figsize=[8,6],gridspec_kw={'width_ratios': [3, 2]})
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.6)
        ax1.bar([1,2,3],mean_rate,yerr=sem,width=0.6,edgecolor='#0504aa',alpha=0.8,facecolor='None',linewidth=3, capsize=10)
        for n in range(len(lick_rate_pre)):
            ax1.plot([1,2,3],[lick_rate_pre[n],lick_rate_reward[n],lick_rate_rest[n]],'o-',color='#A9A9A9',alpha=0.5)
        ax1.set_ylabel('licking rate (Hz)',fontsize=18)
        ax1.set_xticks([1,2,3])
        ax1.set_xticklabels(['pre','reward','base'])
        ax1.tick_params(axis='both', which='major', labelsize=18)
        f.text(0.15, 0.9, '%s_%s_lick rate - reward relationship'%(fn,uname), va='center', fontsize=18)
        pl.axes(ax1)
        ba.barplot_annotate_brackets(1,2,pvalue12,bars,heights,maxasterix=3)
        ba.barplot_annotate_brackets(2,3,pvalue23,bars,heights,maxasterix=3)
        ba.barplot_annotate_brackets(1,3,pvalue13,bars,heights,dh=0.2,maxasterix=3)
        if N.isnan(Num_df)==False:
            f.text(0.2, 0.02, 'RM oneway ANOVA, F(%s,%s):%s, p=%s' %(Num_df,DEN_df,F_value,pvalue), va='center', fontsize=14)
        else:
            f.text(0.2, 0.02, 'Kuskal_wallis test, p=%s' %pvalue, va='center', fontsize=14)
        
        ax2.bar([1,2],[mean_rate_rewardzone,mean_rate_rest],yerr=[sem_rewardzone,sem_rest],width=0.6,edgecolor='#0504aa',alpha=0.8,facecolor='None',linewidth=3, capsize=10)
        for n in range(len(lick_rate_pre)):
            ax2.plot([1,2],[lick_rate_rewardzone[n],lick_rate_rest[n]],'o-',color='#A9A9A9',alpha=0.5)
        ax1.set_ylabel('licking rate (Hz)',fontsize=18)
        ax1.set_xticks([1,2,3])
        ax1.set_xticklabels(['pre','reward','base'])
        ax1.tick_params(axis='both', which='major', labelsize=18)
        
        ax2.set_xticks([1,2])
        ax2.set_xticklabels(['rewardzone','base'])
        ax2.tick_params(axis='both', which='major', labelsize=18)
        pl.axes(ax2)
        ba.barplot_annotate_brackets(1,2,'p=%s'%pvalue4,bars,heights)
        
        f.savefig('%s_%s_licking_rate_reward.png' %(fig_to_save,uname))
        f.savefig('%s_%s_licking_rate_reward.eps' %(fig_to_save,uname),transparent=True)
        
        fig = pl.figure(figsize=[8,6])
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        ax = fig.add_subplot(121)
        for n in range(len(lick_rate_pre)):
            ax.plot([1,2,3],[lick_rate_pre[n],lick_rate_reward[n],lick_rate_rest[n]],'o--',color='orange',alpha=0.5)
        medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
        meanlineprops = dict(linestyle='--', linewidth=1.5, color='k')
        ax.boxplot([lick_rate_pre,lick_rate_reward,lick_rate_rest],meanline=True,showmeans=True,meanprops=meanlineprops,medianprops=medianprops)
        ax.set_xticklabels(['pre','reward','base'],fontsize=16)
        ax.set_ylabel('licking frequency (Hz)',fontsize=16)
        fig.text(0.15, 0.9, '%s_%s_lick rate - reward relationship'%(fn,uname), va='center', fontsize=18)
        if N.isnan(Num_df)==False:
            fig.text(0.2, 0.02, 'RM oneway ANOVA, F(%s,%s):%s, p=%s' %(Num_df,DEN_df,F_value,pvalue), va='center', fontsize=14)
        else:
            fig.text(0.2, 0.02, 'Kuskal_wallis test, p=%s' %pvalue, va='center', fontsize=14)
        pl.yticks(fontsize=16)
        if pvalue<0.05:
            ba.barplot_annotate_brackets(1,2,pvalue12,bars,heights)
            ba.barplot_annotate_brackets(2,3,pvalue23,bars,heights)
            ba.barplot_annotate_brackets(1,3,pvalue13,bars,heights,dh=0.2)
        ax = fig.add_subplot(122)
        for n in range(len(lick_rate_pre)):
            ax.plot([1,2],[lick_rate_rewardzone[n],lick_rate_rest[n]],'o--',color='orange',alpha=0.5)
        ax.boxplot([lick_rate_rewardzone,lick_rate_rest],meanline=True,showmeans=True,meanprops=meanlineprops,medianprops=medianprops)
        ax.set_xticklabels(['rewardzone','base'],fontsize=16)
        ba.barplot_annotate_brackets(1,2,'p=%s'%pvalue4,bars,heights)
        pl.yticks(fontsize=16)
        fig.savefig('%s_%s_licking_rate_reward_barplot.png' %(fig_to_save,uname))
        fig.savefig('%s_%s_licking_rate_reward_barplot.eps' %(fig_to_save,uname),transparent=True)
    
    
    if saveData==True:
        N.save('%s_%s_Mean_lick_rate_pre_reward_base' %(fig_to_save,uname),mean_rate)
        N.save('%s_%s_sem_lick_rate_pre_reward_base' %(fig_to_save,uname),sem)
        N.save('%s_%s_Mean_lick_rate_rewardzone' %(fig_to_save,uname),mean_rate_rewardzone)
        N.save('%s_%s_sem_lick_rate_rewardzone' %(fig_to_save,uname),sem_rewardzone)
        N.save('%s_%s_Mean_lick_rate_base' %(fig_to_save,uname),mean_rate_rest)
        N.save('%s_%s_sem_lick_rate_base' %(fig_to_save,uname),sem_rest)
        N.save('%s_%s_performance_score' %(fig_to_save,uname),suc_rate)
        N.save('%s_%s_lick_pre' %(fig_to_save,uname),lick_rate_pre)
        N.save('%s_%s_lick_base' %(fig_to_save,uname),lick_rate_rest)
        N.save('%s_%s_lick_reward' %(fig_to_save,uname),lick_rate_reward)
    
    return mean_rate,lick_rate_pre,lick_rate_reward,lick_rate_rest,lick_rate_rewardzone,mean_rate_rewardzone,mean_rate_rest


def spike_rate_pre_reward_rest_zone(su,rewardzone_ind_start,reward_start_ind,rewardzone_ind_end,trial_ind_reward,fig_to_save,fname,Fs=30000,saveData=True,shift=False,originshift=False):
    
    su_each_trial=[]
    for n in range(len(trial_ind_reward)):
        su_each=N.intersect1d(su,trial_ind_reward[n])
        su_each_trial.append(su_each)
            
    spike_numb_pre,spike_numb_reward,spike_numb_rest,su_rate_pre,su_rate_reward,su_rate_rest,su_rate_rewardzone=[],[],[],[],[],[],[]
    for m in range(len(rewardzone_ind_start)):
        #find indices in the range of pulse to get spike numbers
        pre=N.ravel(N.argwhere((N.logical_and(su_each_trial[m]>=rewardzone_ind_start[m], su_each_trial[m]<=(reward_start_ind[m])))))
        reward=N.ravel(N.argwhere((N.logical_and(su_each_trial[m]>=reward_start_ind[m], su_each_trial[m]<=(rewardzone_ind_end[m])))))
        su_pre=N.take(su_each_trial[m],pre)
        su_reward=N.take(su_each_trial[m],reward)
        spike_numb_pre.append(len(pre))
        spike_numb_reward.append(len(reward))
        if shift==True:
            su_rest=N.ravel(N.argwhere(N.logical_and(rewardzone_ind_end[m]<su_each_trial[m],su_each_trial[m]<rewardzone_ind_end[m]+3*Fs)))
            spike_numb_rest.append(su_rest)   ## for shifted, take only 3 s after rewardzone ends
        elif originshift==True:
            su_rest=N.ravel(N.argwhere(N.logical_and(trial_ind_reward[m][0]<su_each_trial[m],su_each_trial[m]<rewardzone_ind_start[m])))
            spike_numb_rest.append(su_rest)  ##for originshift, take the parts before rewardzone starts
        else:
            res = N.setdiff1d(su_each_trial[m],su_pre)
            su_rest = N.setdiff1d(res,su_reward)
        spike_numb_rest.append(len(su_rest))
        su_rate_pre.append(len(pre)*Fs/float(reward_start_ind[m]-rewardzone_ind_start[m]))
        su_rate_reward.append(len(reward)*Fs/float(rewardzone_ind_end[m]-reward_start_ind[m]))
        if shift==True:
            su_rate_rest.append(len(su_rest)*Fs/float(3*Fs))
        elif originshift==True:
            su_rate_rest.append(len(su_rest)*Fs/float(rewardzone_ind_start[m]-trial_ind_reward[m][0]))
        else:
            su_rate_rest.append(len(su_rest)*Fs/float(len(trial_ind_reward[m])-(rewardzone_ind_end[m]-rewardzone_ind_start[m])))
        su_rate_rewardzone.append((len(pre)+len(reward))*Fs/float(rewardzone_ind_end[m]-rewardzone_ind_start[m]))
    mean_rate,sem=[],[]
    mean_rate.append(N.mean(su_rate_pre))
    mean_rate.append(N.mean(su_rate_reward))
    mean_rate.append(N.mean(su_rate_rest))
    sem.append(stats.sem(su_rate_pre))
    sem.append(stats.sem(su_rate_reward))
    sem.append(stats.sem(su_rate_rest))
    mean_rate_rewardzone=N.mean(su_rate_rewardzone)
    mean_rate_rest=N.mean(su_rate_rest)
    sem_rewardzone=stats.sem(su_rate_rewardzone)
    sem_rest=stats.sem(su_rate_rest)
    print 'mean rate of reward zone pre_reward_rest %s Hz' %mean_rate
    
    uname=fname.split('_')[0]
    fn=fname.split('_')[1]
    fn=fn.split('.')[0]
    pvalue, Num_df, DEN_df, F_value, pvalue12, pvalue13, pvalue23 = AN.ANOVA_RM_1way_or_kruskal(su_rate_pre,su_rate_reward,su_rate_rest,fig_to_save,uname,fn,script='fprerepost')
    statistic4,pvalue4=stats.ttest_rel(su_rate_rewardzone,su_rate_rest)
    if N.isnan(Num_df)==False:
        if pvalue>0.05:
            print 'no significant, pvalue one way ANOVA %s' %pvalue
        else:
            print 'reward pre reward rest related, RM one way ANOVA, F(%s,%s):%s, p=%s ' %(Num_df,DEN_df,F_value,pvalue)
            print 'p value between groups: pre-rest %s' %pvalue13
            print 'p value between groups: pre-reward %s' %pvalue12
            print 'p value between groups: reward-rest %s' %pvalue23
    else:
        if pvalue>0.05:
            print 'firing rates no significant, Kuskal-Wallis test, p=%s ' %pvalue
        else:
            print 'firing rate significant change pre reward rest, Kuskal-Wallis test, p=%s ' %pvalue
            print 'p value between groups: pre-rest %s' %pvalue13
            print 'p value between groups: pre-reward %s' %pvalue12
            print 'p value between groups: reward-rest %s' %pvalue23
    if pvalue4>0.05:
        print 'no significant between reward zone and rest, pvalue4 %s' %pvalue4
    else:
        print 'rewardzone total related, pvalue4 %s' %pvalue4    
    maxvalue=N.matrix.max(N.matrix([su_rate_pre,su_rate_reward,su_rate_rest]))
    heights = [maxvalue, maxvalue-3*maxvalue/10, maxvalue-2*maxvalue/10, maxvalue-maxvalue/10]
    maxvalue2=N.matrix.max(N.matrix([su_rate_rewardzone,su_rate_rest]))
    heights2 = [maxvalue2, maxvalue2-2*maxvalue2/50, maxvalue2-3*maxvalue2/50]
    
    #compensate the problem with open .eps in Coreldraw 2020; getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    bars = N.arange(len(heights)+1)
    f,(ax1,ax2)=pl.subplots(1,2,figsize=[8,6],gridspec_kw={'width_ratios': [3, 2]})
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    for n in range(len(su_rate_pre)):
        ax1.plot([1,2,3],[su_rate_pre[n],su_rate_reward[n],su_rate_rest[n]],'o-',color='#A9A9A9',alpha=0.5)
    ax1.bar([1,2,3],mean_rate,yerr=sem,width=0.6,edgecolor='#0504aa',alpha=0.8,facecolor='None',linewidth=3, capsize=10)
    pl.axes(ax1)
    ba.barplot_annotate_brackets(1,2,pvalue12,bars,heights)
    ba.barplot_annotate_brackets(2,3,pvalue23,bars,heights)
    ba.barplot_annotate_brackets(1,3,pvalue13,bars,heights,dh=0.2)
    for n in range(len(su_rate_pre)):
        ax2.plot([1,2],[su_rate_rewardzone[n],su_rate_rest[n]],'o-',color='#A9A9A9',alpha=0.5)
    ax2.bar([1,2],[mean_rate_rewardzone,mean_rate_rest],yerr=[sem_rewardzone,sem_rest],width=0.6,edgecolor='#0504aa',alpha=0.8,facecolor='None',linewidth=3, capsize=10)
    ax1.set_ylabel('Firing rate (Hz)',fontsize=18)
    ax1.set_xticks([1,2,3])
    ax1.set_xticklabels(['pre','reward','rest'])
    ax1.tick_params(axis='both', which='major', labelsize=18)
    f.text(0.15, 0.92, 'u%s_%s_spike timing - reward relationship' %(uname,fn), va='center', fontsize=18)
    ax2.set_xticks([1,2])
    ax2.set_xticklabels(['rewardzone','rest'])
    ax2.tick_params(axis='both', which='major', labelsize=18)
    if N.isnan(Num_df)==False:
        f.text(0.2, 0.02, 'RM oneway ANOVA, F(%s,%s):%s, p=%s' %(Num_df,DEN_df,F_value,pvalue), va='center', fontsize=14)
    else:
        f.text(0.2, 0.02, 'Kuskal_wallis test, p=%s' %pvalue, va='center', fontsize=14)
    pl.axes(ax2)
    ba.barplot_annotate_brackets(1,2,'p=%s'%pvalue4,bars,heights2)
    f.savefig('%s_u%s_%s_firing_rate_reward.png' %(fig_to_save,uname,fn))
    f.savefig('%s_u%s_%s_firing_rate_reward.eps' %(fig_to_save,uname,fn),transparent=True)
    
    
    fig = pl.figure(figsize=[8,6])
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    ax = fig.add_subplot(121)
    for n in range(len(su_rate_pre)):
        ax.plot([1,2,3],[su_rate_pre[n],su_rate_reward[n],su_rate_rest[n]],'o--',color='orange',alpha=0.5)
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=1.5, color='k')
    ax.boxplot([su_rate_pre,su_rate_reward,su_rate_rest],meanline=True,showmeans=True,meanprops=meanlineprops,medianprops=medianprops)
    ax.set_xticklabels(['pre','reward','rest'],fontsize=16)
    ax.set_ylabel('firing frequency (Hz)',fontsize=16)
    fig.text(0.15, 0.92, 'u%s_%s_spike timing - reward relationship' %(uname,fn), va='center', fontsize=18)
    if N.isnan(Num_df)==False:
        fig.text(0.2, 0.02, 'RM oneway ANOVA, F(%s,%s):%s, p=%s' %(Num_df,DEN_df,F_value,pvalue), va='center', fontsize=14)
    else:
        fig.text(0.2, 0.02, 'Kuskal_wallis test, p=%s' %pvalue, va='center', fontsize=14)
    pl.yticks(fontsize=16)
    ba.barplot_annotate_brackets(1,2,pvalue12,bars,heights)
    ba.barplot_annotate_brackets(2,3,pvalue23,bars,heights)
    ba.barplot_annotate_brackets(1,3,pvalue13,bars,heights,dh=0.2)
    ax = fig.add_subplot(122)
    for n in range(len(su_rate_pre)):
        ax.plot([1,2],[su_rate_rewardzone[n],su_rate_rest[n]],'o--',color='orange',alpha=0.5)
    ax.boxplot([su_rate_rewardzone,su_rate_rest],meanline=True,showmeans=True,meanprops=meanlineprops,medianprops=medianprops)
    ax.set_xticklabels(['rewardzone','rest'],fontsize=16)
    ba.barplot_annotate_brackets(1,2,'p=%s'%pvalue4,bars,heights2)
    pl.yticks(fontsize=16)
    fig.savefig('%s_u%s_%s_firing_rate_reward_barplot.png' %(fig_to_save,uname,fn))
    fig.savefig('%s_u%s_%s_firing_rate_reward_barplot.eps' %(fig_to_save,uname,fn),transparent=True)
    
    if saveData==True:
        N.save('%s_u%s_%s_Mean_fir_rate_pre_reward_post' %(fig_to_save,uname,fn),mean_rate)
        N.save('%s_u%s_%s_sem_fir_rate_pre_reward_post' %(fig_to_save,uname,fn),sem)
        N.save('%s_u%s_%s_Mean_fir_rate_rewardzone_rest' %(fig_to_save,uname,fn),mean_rate_rewardzone)
        N.save('%s_u%s_%s_sem_fir_rate_rewardzone_rest' %(fig_to_save,uname,fn),sem_rewardzone)
    
    return mean_rate,su_rate_pre,su_rate_reward,su_rate_rest,su_rate_rewardzone,mean_rate_rewardzone,mean_rate_rest


def extract_reward_pulse_for_batch (su,posnew,files,fname,fig_to_save,reward_start_ind,directory3,thres=3,Fs=30000,trange=3,tprepost=5,binspers=10,plotData=True,plotSpeed=True):
    
    ''' Inputs:
        files: ADC recording of laser pulse
        raw: raw .continuous data
        su: timestamp of sorted unit
        Keyword arguments:
        thres: Threshold to detect laser pulses
        Fs: sampling frequency (in Hz)
        low, high: bandpass filter
        Outputs:
        laser pulse interval matrix for all detections
        the same length of interval before pulse start for all detections
    '''
    if plotSpeed==True:
        os.chdir(directory3)
        for file in os.listdir(directory3):
            if file.endswith("mean_speed_reward.npy"):
                meanspeed=N.load(file) 
            if file.endswith("sem_speed_reward.npy"):
                semspeed=N.load(file) 
    
    
    ind_delete=[]
    for i in range(len(su)):
        if su[i]>len(posnew):
            ind_delete.append(i)
    su=N.delete(su,ind_delete)
    #get spike train for firing rate calculation
    spike_train = N.zeros(len(posnew))
    N.put(spike_train,su.astype(int),1)
    
    spike_ind_all, spike_numb_Pulse, spike_numb_pre, spike_numb_post,spike_train_rew,spike_train_plot = [],[],[],[],[],[]
    for k in range(len(reward_start_ind)):
        #find indices in the range of pulse to get spike numbers
        result=N.ravel(N.ravel(N.argwhere((N.logical_and(su>=(reward_start_ind[k]-trange*Fs-1), su<=(reward_start_ind[k]+2*trange*Fs))))))
        spike_ind_all.append(N.array(N.take(su,result)-reward_start_ind[k]))
        spike_numb_Pulse.append(len(N.argwhere((N.logical_and(su>=reward_start_ind[k], su<=(reward_start_ind[k]+trange*Fs))))))
        spike_numb_pre.append(len(N.argwhere(N.logical_and(su>=(reward_start_ind[k]-trange*Fs-1), su<=(reward_start_ind[k]-1)))))
        spike_numb_post.append(len(N.argwhere((N.logical_and(su>=reward_start_ind[k]+trange*Fs, su<=(reward_start_ind[k]+2*trange*Fs-1))))))
        if reward_start_ind[k]+((trange+tprepost)*Fs-1)<=len(spike_train):
            spike_train_rew.append(spike_train[range(reward_start_ind[k]-tprepost*Fs-1,(reward_start_ind[k]+((trange+tprepost)*Fs-1)))])
            spike_train_plot.append(spike_train[range(reward_start_ind[k]-trange*Fs-1,(reward_start_ind[k]+2*trange*Fs))])
    
    uname=fname.split('_')[0]
    fn=fname.split('_')[1]
    fn=fn.split('.')[0]    
    #calculating mean firing rate per 200 mili second range if binspers=5
    Fir_freq_rew=[]
    for n in range(len(spike_train_rew)):
        fir_rew=[]
        for j in range(0,len(spike_train_rew[n][:])-Fs/binspers,Fs/binspers):
            fir_rew.append(sum(spike_train_rew[n][j:j+Fs/binspers])*binspers)
        Fir_freq_rew.append(fir_rew)
    mean_fir_freq_rew=N.mean(Fir_freq_rew,axis=0)
    N.save('%s_u%s_mean_fir_freq_reward_last5runs' %(fig_to_save,uname),mean_fir_freq_rew)  
    N.save('%s_u%s_singletrial_fir_freq_reward_last5runs' %(fig_to_save,uname),Fir_freq_rew) 
    Fir_freq_plot=[]
    for n in range(len(spike_train_plot)):
        fir_plot=[]
        for j in range(0,len(spike_train_plot[n][:])-Fs/binspers,Fs/binspers):
            fir_plot.append(sum(spike_train_plot[n][j:j+Fs/binspers])*binspers)
        Fir_freq_plot.append(fir_plot)
    mean_fir_freq_plot=N.mean(Fir_freq_plot,axis=0)
    sem_fir_freq_plot=stats.sem(Fir_freq_plot)
    
    pvalue3, Num_df, DEN_df, F_value, pvalue12, pvalue13, pvalue23 = AN.ANOVA_RM_1way_or_kruskal(spike_numb_pre,spike_numb_Pulse,spike_numb_post,fig_to_save,uname,fn,script='exrewpulse')
    if N.isnan(Num_df)==False:
        print 'RM one way ANOVA test, F(%s,%s):%s, p=%s ' %(Num_df,DEN_df,F_value,pvalue3)
    else:
        print 'Kuskal-Wallis test, p=%s ' %pvalue3
    if pvalue12 >0.05 or N.isnan(pvalue12)==True:
        print 'no siginificance pre vs. reward'
    else:
        print 'reward sensitive, significant',pvalue12
    
    #getting setup to export text correctly
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    if plotData == True:
        f,(ax1,ax2,ax3) = pl.subplots(3,1,sharex=True,figsize=[8,8])
        spike_ind_plot = []
        for m in range(len(spike_ind_all)):
            if len(spike_ind_all[m]) > 0:
                spike_ind_plot = spike_ind_all[m:]
                break
            
        x=range(int(-trange*Fs-1),int(2*trange*Fs))
        ax1.eventplot(spike_ind_plot,linewidths=2)
        ax1.vlines(int(trange*Fs),0,len(spike_ind_plot),colors='r',linestyle='dashed')
        
        ax1.set_title('%s_firing correlates to reward'%uname,size=15)
        ax1.set_ylabel("trial No.",fontsize=15)
        for k in range(len(reward_start_ind)):
            if reward_start_ind[k]+2*trange*Fs<=len(files) and reward_start_ind[k]-trange*Fs-1>0:
                ax2.plot(x,files[int(reward_start_ind[k]-trange*Fs-1):int(reward_start_ind[k]+2*trange*Fs)])
                ax2.vlines(int(trange*Fs),min(files[int(reward_start_ind[k]-trange*Fs-1):int(reward_start_ind[k]+2*trange*Fs)]),max(files[int(reward_start_ind[k]-trange*Fs-1):int(reward_start_ind[k]+2*trange*Fs)]),colors='r',linestyle='dashed')
                break
        ax2.set_ylabel("reward pulse (V)",fontsize=15)
        ax2.set_ylabel("reward pulse (V)",fontsize=15)
        if N.isnan(Num_df)==False:
            ax2.set_xlabel('RM oneway ANOVA  F(%s,%s):%s, p=%s' %(Num_df,DEN_df,F_value,pvalue3),fontsize=15)
        else:
            ax2.set_xlabel('Kuskal-Wallis test, p=%s' %pvalue3,fontsize=15)
        ax1.set_xlabel("Holm-Bon test pvalue %s"%pvalue12,fontsize=15)  
        if plotSpeed==True:
            ax3.set_ylabel('speed (cm/s)',fontsize=15)
            ax3.set_xlim([x[0]-1,x[-1]+1])
            ax4 = ax3.twinx()
            ax4.fill_between(N.arange(-trange*Fs-1,2*trange*Fs,Fs*float(1)/binspers)[:-1],mean_fir_freq_plot+sem_fir_freq_plot,mean_fir_freq_plot-sem_fir_freq_plot,facecolor='orange', alpha=0.5)
            ax4.plot(N.arange(-trange*Fs-1,2*trange*Fs,Fs*float(1)/binspers)[:-1],mean_fir_freq_plot,color='r')
            ax3.fill_between(x[::50],meanspeed[::50]+semspeed[::50],meanspeed[::50]-semspeed[::50],facecolor='#A9A9A9', alpha=0.5)
            ax3.plot(x,meanspeed,color='k')
            ax4.set_ylabel('firing frequency (Hz)',color='r',fontsize=14)
            ax4.tick_params(axis='y', labelcolor='r')
        else:
            ax3.set_ylabel('firing frequency (Hz)',fontsize=14)
            ax3.fill_between(N.arange(-trange*Fs-1,2*trange*Fs,Fs*float(1)/binspers)[:-1],mean_fir_freq_plot+sem_fir_freq_plot,mean_fir_freq_plot-sem_fir_freq_plot,facecolor='orange', alpha=0.5)
            ax3.plot(N.arange(-trange*Fs-1,2*trange*Fs,Fs*float(1)/binspers)[:-1],mean_fir_freq_plot,color='r')
            ax3.set_xlim([x[0]-1,x[-1]+1])
            
        f.savefig('%s_u%s_reward_raster.png' %(fig_to_save,uname))
        f.savefig('%s_u%s_reward_raster.eps' %(fig_to_save,uname),transparent=True) 
            
    return spike_ind_all, spike_numb_Pulse, spike_numb_pre, pvalue12,mean_fir_freq_rew


#%% Sensory stimulation analyses for SOMIs   
def Sen_stimuli_ind(thres=3):
    
    target_dir='/data/example_raw_for_position_extraction_spike_waveform/200404_#327_lin/#327_2020-04-04_14-20-50_senstim'
    os.chdir(target_dir)
    reward=op.open_helper('100_ADC3.continuous')
    lick=op.open_helper('100_ADC2.continuous') 
    airpuf=op.open_helper('100_ADC4.continuous') 
    sound=op.open_helper('100_ADC7.continuous')
    visual=op.open_helper('100_ADC6.continuous')
    
    #detect reward start
    reward_start_ind,lick_start,airpuf_start,sound_start,visual_start,sound_stop = [],[],[],[],[],[]
    for n in range(len(reward)-1):
        #detect upward threshold crossings
        if reward[n] < thres and reward[n+1] >= thres:
            reward_start_ind.append(n+1) 
        if lick[n] < thres and lick[n+1] >= thres:
            lick_start.append(n+1)
        if airpuf[n] < thres and airpuf[n+1] >= thres:
            airpuf_start.append(n+1)
        if sound[n] < thres and sound[n+1] >= thres:
            sound_start.append(n+1)
        if sound[n] > thres and sound[n+1] <= thres:
            sound_stop.append(n+1)
        if visual[n] < thres and visual[n+1] >= thres:
            visual_start.append(n+1)
    
    RES=[]
    RES.append(sound_start[0])
    for i in range(len(sound_start)-1):
        if sound_start[i+1]-sound_stop[i]>100:
            RES.append(sound_start[i+1])
    
        
    os.chdir(target_dir)
    N.save('reward_start_ind_sen',reward_start_ind)
    N.save('lick_start',lick_start)
    N.save('airpuf_start',airpuf_start)
    N.save('sound_start',RES)
    N.save('visual_start',visual_start)

def extract_senstim_pulses_batch(thres=3,Fs=30000,tprepost=3,binspers=10,plotData=True,plotSpeed=False,saveData=True):
    
    directory='/data/example_raw_for_position_extraction_spike_waveform/200404_#327_lin/output/ms3--all_shank3'
    os.chdir(directory)
    su_batch=op.open_helper('all_timestamp_senstim.batch') 
    directory3='/data/example_raw_for_position_extraction_spike_waveform/200404_#327_lin/#327_2020-04-04_14-20-50_senstim'
    os.chdir(directory3)
    posnew = op.open_helper('100_ADC1.continuous')
    reward=op.open_helper('100_ADC3.continuous')
    airpuf=op.open_helper('100_ADC4.continuous') 
    sound=op.open_helper('100_ADC7.continuous')
    visual=op.open_helper('100_ADC6.continuous')
    reward_start_ind=N.load('reward_start_ind_sen.npy')
    airpuf_start=N.load('airpuf_start.npy')
    sound_start=N.load('sound_start.npy')
    visual_start=N.load('visual_start.npy')
    
    os.chdir(directory)
    if os.path.isdir('Data saved')==False:
        os.mkdir('Data saved')
    fig_to_save=tkFileDialog.asksaveasfilename(title="Choose target figure name") 
    file_to_save=tkFileDialog.asksaveasfilename(title="Choose target filename") 
    fig_to_save_rew=''.join((fig_to_save,'_rew'))
    fig_to_save_airpuf=''.join((fig_to_save,'_airpuf'))
    fig_to_save_sound=''.join((fig_to_save,'_sound'))
    fig_to_save_visual=''.join((fig_to_save,'_visual'))
    grand_result_rew,grand_result_airpuf,grand_result_sound,grand_result_visual=[],[],[],[]
    for n in range(len(su_batch)):
        os.chdir(directory)
        su=op.open_helper(su_batch[n])
        print "unit %s " %su_batch[n]
        fname2=su_batch[n]
        c=fname2.rsplit('_',-1)[0]
        os.chdir('Data saved')
        if len(reward_start_ind)>0:
            print 'reward corr'
            spike_ind_all_rew, spike_numb_Pulse_rew, spike_numb_pre_rew, pvalue,mean_fir_freq_rew=extract_reward_pulse_for_batch (su,posnew,reward,fname2,fig_to_save_rew,reward_start_ind,directory3,thres=thres,Fs=Fs,trange=3,tprepost=tprepost,binspers=binspers,plotData=plotData,plotSpeed=plotSpeed)
            result_rew={'su_id':fname2,'spike_ind_all_rew':spike_ind_all_rew,'spike_numb_Pulse_rew':spike_numb_Pulse_rew,'spike_numb_pre_rew':spike_numb_pre_rew,'pvalue':pvalue}
            grand_result_rew.append(result_rew)
            N.save('%s_u%s_mean_fir_freq_reward' %(file_to_save,c),mean_fir_freq_rew)
        else:
            print 'no reward stimuli'
            grand_result_rew=N.nan
        if len(airpuf_start)>0:
            print 'airpuf corr'
            spike_ind_all_airpuf, spike_numb_Pulse_airpuf, spike_numb_pre_airpuf, pvalueairpuf,mean_fir_freq_airpuf=extract_reward_pulse_for_batch (su,posnew,airpuf,fname2,fig_to_save_airpuf,airpuf_start,directory3,thres=thres,Fs=Fs,trange=3,tprepost=tprepost,binspers=binspers,plotData=plotData,plotSpeed=plotSpeed)
            result_airpuf={'su_id':fname2,'spike_ind_all_airpuf':spike_ind_all_airpuf,'spike_numb_Pulse_airpuf':spike_numb_Pulse_airpuf,'spike_numb_pre_airpuf':spike_numb_pre_airpuf,'pvalueairpuf':pvalueairpuf}
            grand_result_airpuf.append(result_airpuf)
            N.save('%s_u%s_mean_fir_freq_airpuf' %(file_to_save,c),mean_fir_freq_airpuf)
        else:
            print 'no airpuf stimuli'
            grand_result_airpuf=N.nan
        if len(sound_start)>0:
            print 'sound corr'
            spike_ind_all_sound, spike_numb_Pulse_sound, spike_numb_pre_sound, pvalue_sound,mean_fir_freq_sound=extract_reward_pulse_for_batch (su,posnew,sound,fname2,fig_to_save_sound,sound_start,directory3,thres=thres,Fs=Fs,trange=3,tprepost=tprepost,binspers=binspers,plotData=plotData,plotSpeed=plotSpeed)
            result_sound={'su_id':fname2,'spike_ind_all_sound':spike_ind_all_sound,'spike_numb_Pulse_sound':spike_numb_Pulse_sound,'spike_numb_pre_sound':spike_numb_pre_sound,'pvalue_sound':pvalue_sound}
            grand_result_sound.append(result_sound)
            N.save('%s_u%s_mean_fir_freq_sound' %(file_to_save,c),mean_fir_freq_sound)
        else:
            print 'no sound stimuli'
            grand_result_sound=N.nan
        if len(visual_start)>0:
            print 'visual corr'
            spike_ind_all_visual, spike_numb_Pulse_visual, spike_numb_pre_visual, pvalue_visual,mean_fir_freq_visual=extract_reward_pulse_for_batch (su,posnew,visual,fname2,fig_to_save_visual,visual_start,directory3,thres=thres,Fs=Fs,trange=3,tprepost=tprepost,binspers=binspers,plotData=plotData,plotSpeed=plotSpeed)
            result_visual={'su_id':fname2,'spike_ind_all_visual':spike_ind_all_visual,'spike_numb_Pulse_visual':spike_numb_Pulse_visual,'spike_numb_pre_visual':spike_numb_pre_visual,'pvalue_visual':pvalue_visual}
            grand_result_visual.append(result_visual)
            N.save('%s_u%s_mean_fir_freq_visual' %(file_to_save,c),mean_fir_freq_visual)
        else:
            print 'no visual stimuli'
            grand_result_visual=N.nan
        
    if saveData == True:
        N.save('%s_extract_senstim_reward_batch' %file_to_save,grand_result_rew)
        N.save('%s_extract_senstim_airpuf_batch' %file_to_save,grand_result_airpuf)
        N.save('%s_extract_senstim_sound_batch' %file_to_save,grand_result_sound)
        N.save('%s_extract_senstim_visual_batch' %file_to_save,grand_result_visual)
        
    return grand_result_rew,grand_result_airpuf,grand_result_sound,grand_result_visual

#%% time delay of SOMI (or FSI) response at reward sites
    
def SOMI_reward_response_delay(smooth=True,zscore=True):
    ## result is the concatenated matrix of all SOMIs of E/NE
    target_dir='/data/Fig2'
    os.chdir(target_dir)
    result=op.open_helper('H_all_ordered_noysmooth_Summary.npy')
    if smooth==True:
         sigma_rew=1.5 #consistent with Df script
         for n in range(len(result)):
             if zscore==True:
                 a=stats.zscore(result[n,:])
                 result[n,:]=ndimage.gaussian_filter(a,sigma_rew,truncate=2)
             else:
                 result[n,:]=ndimage.gaussian_filter(result[n,:],sigma_rew,truncate=2)
    
    delayr1,peak=[],[]
    for n in range(len(result)):
        if abs(max(result[n,50:79]))>abs(min(result[n,50:79])):
            delayr1.append(N.ravel(N.argwhere(result[n,50:79]==max(result[n,50:79]))))
            peak.append(max(result[n,50:79])-N.mean(result[n,0:49]))
        else:
            delayr1.append(N.ravel(N.argwhere(result[n,50:79]==min(result[n,50:79]))))
            peak.append(min(result[n,50:79])-N.mean(result[n,0:49]))
    
    delayr1=N.array(delayr1)/float(10)
    
    '''
    ###for FSI trough detection:
    delayr1,peak=[],[]
    for n in range(len(result)):
        if all(result[n] == 0)==False and min(result[n,50:79])<N.mean(result[n,0:49])-0.1: 
            delayr1.append(N.ravel(N.argwhere(result[n,50:79]==min(result[n,60:79])))[0])
            peak.append(min(result[n,50:79])-N.mean(result[n,0:49]))
    
    delayr1=N.array(delayr1)/float(10)
    '''
    return delayr1

