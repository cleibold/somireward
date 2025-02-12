#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:28:34 2023

@author: jonas
"""

import os
import numpy as N
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import scipy.stats as st


'''
General linear model to measure the contribution of behavioural variables (speed, licking and reward delivery) on SOMI spiking, separately for low and high performer groups.

Spikes of each cell are binned during the time before reward delivery. Speed is given as the mean value of the speed trace,
licks are binned the same as as spikes. Reward time is shifted to optimally match the delay in SOMI spiking after reward (separately for low and high performer groups).

The overall explained variance (EV) is measured with sklearn's 'r2'. Reduction models, in which 1 of the variables is randomly scrambled in time (number defined by iterations), are used to quantify
the reduction in explained variance when 1 beahvioural variable is removed.

Statistical comparisons are included in the final result.

Output structure:
    Type: dict
    low/high: disct
        score, red speed, red lick, red rew # EV and reduction when the named variable is removed.
    stats: dict
        p,t,used test, comparison # p-value, statistic, the used test and which pair is compared. All comparisons are between low and high groups.
        
'''

N.random.seed(42)

# Directories of the data and target.

base_dir = '/Users/sven/Work/Projects/DecodingAnalysisMei'

target_dir=base_dir + "LinearModelJonas/GLM_analysis"
data_dir_high=base_dir + "/data/Expert"
data_dir_low=base_dir + "/data/Non-expert"

# target_dir="/media/jonas/data3/Mei_linear_model/reanalysis_corrected_timing/analysis/GLM_analysis"
# data_dir_high="/media/jonas/data3/Mei_linear_model/data/with_pc/High_performer"
# data_dir_low="/media/jonas/data3/Mei_linear_model/data/with_pc/Low_performer"

# Parameters.
latency=0.94                 # in s
Fs=30000                    # Sampling frequency in Hz.
latency_dp=int(latency*Fs)  # The latency in sampling points to correct the detected reward delivery points in time.
binsize=0.1                 # Bin size for spike train/behavioural signal binning.
cell_type='SOMI' # String: SOMI, FSI or PC
session_type='fam' # String: fam or nov

binwidth=7500 # In dp
bins=12
bin_start,bin_end=[],[]
for n in range(-binwidth*bins,0,binwidth):
    bin_start.append(n)
    bin_end.append(n+binwidth)

# Params for decoding.
max_iter=10000000 # Iterations of the solver. Ridge only
cv=5 # Number of folds for cross-validation.
iterations=500 # Number of random shuffles to assess reduction in explained variance.
sig_thres=95
evaluation_metric="explained_variance"

params={'binwidth':binwidth,
        '# bins':bins*2,
        'max iter':max_iter,
        'cv':cv,
        'iterations':iterations,
        'used code':os.path.basename(__file__)}



# I/O functions.
def load_txt(filename): # Loads the raw spike times.
    f=open(filename, "r")
    g=f.read()
    g=g.split()
    h=N.asarray(g)
    data = N.empty((len(h)))
    for n in range(len(h)):
        temp=float(h[n])
        data[n]=int(temp)
    return data

# Get concatenated trial x cell arrays.

def get_binned_activity_during_trial_single_cell(local_spikes,rew_times):
    # Iterate over reward points and extract the summed spikes for each bin relative to reward sites.
    binned_spikes=[]
    for r in range(len(rew_times)):   
        for bin_number in range(len(bin_start)):            
            current_start=bin_start[bin_number]
            current_end=bin_end[bin_number]            
            binned_spikes.append(((rew_times[r]+current_start < local_spikes) & (local_spikes < rew_times[r]+current_end)).sum())

    return N.asarray(binned_spikes)

# Speed binning function.
def get_binned_speed_and_licks(bin_start,bin_end):
    # Get the currently valid reward times.
    os.chdir("pos_fam")
    rew_times_raw=N.load("reward_start_ind.npy")  
    # Correct rew times.
    rew_times=rew_times_raw+latency_dp
    speed=N.load("estimate speed from position_speed.npy") 
    acc=N.diff(speed)
    licks=N.load("lick_start.npy")

    #before_reward=N.zeros((bins))
    #after_reward=N.ones((bins))
    #reward_on=N.append(before_reward,after_reward)
    reward_on=N.zeros((bins))
    binned_speed,binned_acc,binned_licks,binned_reward=[],[],[],[]
    for r in range(len(rew_times)):   
        for bin_number in range(len(bin_start)):            
            current_start=bin_start[bin_number]
            current_end=bin_end[bin_number]            
            binned_speed.append(N.mean(speed[rew_times[r]+current_start:rew_times[r]+current_end]))
            binned_acc.append(N.mean(acc[rew_times[r]+current_start:rew_times[r]+current_end]))
            binned_licks.append(((rew_times[r]+current_start < licks) & (licks < rew_times[r]+current_end)).sum())
        binned_reward.extend(reward_on)
            
    os.chdir("..")       
    return N.asarray(binned_speed),N.asarray(binned_acc),N.asarray(binned_licks),N.asarray(binned_reward),N.asarray(rew_times)
    
# Spike binning function - for all cells of all mice in the data directories.
def get_average_spike_bins(bin_start,bin_end,rew_times):

    # Check whether individual shanks are present.
    # Select the neurons of session session_type and type cell_type.
    unit_counter=0
    cell_names=[]
    folder="ms3--all_shank1"
    if os.path.exists(folder):
        os.chdir(folder)
        all_files_raw=os.listdir()
        all_files=[]
        for fname in all_files_raw:
            if session_type in fname:
                if cell_type in fname:
                    all_files.append(fname)  
                    cell_names.append("%s_%s" %(folder,fname))
        unit_counter+=len(all_files)
        os.chdir("..")

    folder="ms3--all_shank2"
    if os.path.exists(folder):
        os.chdir(folder)
        all_files_raw=os.listdir()
        all_files=[]
        for fname in all_files_raw:
            if session_type in fname:
                if cell_type in fname:
                    all_files.append(fname)    
                    cell_names.append("%s_%s" %(folder,fname))
        unit_counter+=len(all_files)
        os.chdir("..")

    folder="ms3--all_shank3"
    if os.path.exists(folder):
        os.chdir(folder)
        all_files_raw=os.listdir()
        all_files=[]
        for fname in all_files_raw:
            if session_type in fname:
                if cell_type in fname:
                    all_files.append(fname)       
                    cell_names.append("%s_%s" %(folder,fname))
        unit_counter+=len(all_files)
        os.chdir("..")

    folder="ms3--all_shank4"
    if os.path.exists(folder):
        os.chdir(folder)
        all_files_raw=os.listdir()
        all_files=[]
        for fname in all_files_raw:
            if session_type in fname:
                if cell_type in fname:
                    all_files.append(fname)   
                    cell_names.append("%s_%s" %(folder,fname))
        unit_counter+=len(all_files)
        os.chdir("..")

    binned_spikes=N.empty((unit_counter,len(rew_times)*bins))
    
    # Load all timestamp files.
    current_cell_id=0
             
    # Check whether individual shanks are present and get a list of spike files.
    folder="ms3--all_shank1"
    if os.path.exists(folder):
        os.chdir(folder)
        all_files_raw=os.listdir()
        all_files=[]
        for fname in all_files_raw:
            if session_type in fname:
                if cell_type in fname:
                    all_files.append(fname)               
        for files in range(len(all_files)):
            local_spikes=load_txt(all_files[files])
            binned_spikes[current_cell_id]=get_binned_activity_during_trial_single_cell(local_spikes,rew_times)
            current_cell_id+=1                    
        os.chdir("..")
    folder="ms3--all_shank2"
    if os.path.exists(folder):
        os.chdir(folder)
        all_files_raw=os.listdir()
        all_files=[]
        for fname in all_files_raw:
            if session_type in fname:
                if cell_type in fname:
                    all_files.append(fname)  

        for files in range(len(all_files)):
            local_spikes=load_txt(all_files[files])
            binned_spikes[current_cell_id]=get_binned_activity_during_trial_single_cell(local_spikes,rew_times)
            current_cell_id+=1  
        os.chdir("..")
    folder="ms3--all_shank3"
    if os.path.exists(folder):
        os.chdir(folder)
        all_files_raw=os.listdir()
        all_files=[]
        for fname in all_files_raw:
            if session_type in fname:
                if cell_type in fname:
                    all_files.append(fname)  

        for files in range(len(all_files)):
            local_spikes=load_txt(all_files[files])
            binned_spikes[current_cell_id]=get_binned_activity_during_trial_single_cell(local_spikes,rew_times)
            current_cell_id+=1  
        os.chdir("..")
    folder="ms3--all_shank4"
    if os.path.exists(folder):
        os.chdir(folder)
        all_files_raw=os.listdir()
        all_files=[]
        for fname in all_files_raw:
            if session_type in fname:
                if cell_type in fname:
                    all_files.append(fname)  

        for files in range(len(all_files)):
            local_spikes=load_txt(all_files[files])
            binned_spikes[current_cell_id]=get_binned_activity_during_trial_single_cell(local_spikes,rew_times)
            current_cell_id+=1  
        os.chdir("..")   

        
    return st.zscore(binned_spikes,axis=1),binned_spikes,cell_names # Return X and y for decoding

# Decoding function.
'''
def decode(model,X,y):
    
    score=cross_val_score(model,X.T,y,cv=fold_iterator,scoring=evaluation_metric)
    score=N.nanmean(score)
    
    return score
'''
def decode(X,y):
    # Define the model.
    clf=LinearRegression()
    kf=KFold(n_splits=cv,shuffle=True)
    
    score=[]
    coeffs=[]
    for train_ind, test_ind in kf.split(X.T,y):
        X_train, X_test = X.T[train_ind], X.T[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
               
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)

        #score.append(explained_variance_score(y_test,y_pred))    
        score.append(r2_score(y_test,y_pred))        
        coeffs.append(clf.coef_)

    return N.mean(score),N.mean(coeffs,axis=0)


def run_analysis_master(data_dir,random_iterations=False):
    os.chdir(data_dir)
    dir_list=os.listdir(data_dir)
    score_total=[]
    red_speed,red_acc,red_lick,red_rew=[],[],[],[]
    coeffs_total=[]
    Xs,ys=[],[]
    cell_names_total=[]
    n_sig_modulated=0
    n_unmodulated=0
    
    for n in range(len(dir_list)):
        os.chdir(dir_list[n]) # Go to the current directory.
        speed_binned,acc_binned,licks_binned,reward_binned,reward_times=get_binned_speed_and_licks(bin_start,bin_end)
        firing_binned,firing_binned_raw,cell_names_raw=get_average_spike_bins(bin_start,bin_end,reward_times) # This gives the concatenated binned activity of all neurons in X_raw

        # Create features array.
        X_raw=N.empty((3,len(speed_binned)))
        X_raw[0]=speed_binned
        X_raw[1]=licks_binned
        X_raw[2]=acc_binned
            
        # Iterate over neurons.
        for i in range(len(firing_binned)):
            
            y_raw=firing_binned[i]
            
            # Exclude trials with possible nan and inf entries.
            check1=N.ravel(N.isfinite(X_raw[0]))
            check2=N.ravel(N.isfinite(X_raw[1]))
            check3=N.ravel(N.isfinite(y_raw))
            
            ind=[]
            for k in range(len(check3)):
                if check1[k]==True and check2[k]==True and check3[k]==True:
                    ind.append(k)
            ind=N.asarray(ind)
            X=X_raw[:,ind]
            y=y_raw[ind]
         
            
            Xs.append(X_raw)
            ys.append(N.reshape(firing_binned_raw[i],[-1,bins]))
            
            score,coeffs=decode(X,y)
            
                    
            # Find neurons for which the firing rate is significantly explained by the behavioural variables.
            score_temp=[]
            for k in range(iterations):
                X_shift=N.empty_like(X)
                X_shift[0]=N.random.permutation(X[0])
                X_shift[1]=N.random.permutation(X[1])
                X_shift[2]=N.random.permutation(X[2])
                score_shift,_=decode(X_shift,y)
                score_temp.append(score_shift)
            if score>N.percentile(score_temp,sig_thres):
                n_sig_modulated+=1
                score_total.append(score)
                coeffs_total.append(coeffs)
                cell_names_total.append("%s_%s" %(dir_list[n],cell_names_raw[i])) # Keep a record of the currently used cell.
                
                if random_iterations==True:
               
                    score_temp=[]
                    for k in range(iterations):
                        X_shift=N.empty_like(X)
                        X_shift[0]=N.random.permutation(X[0])
                        X_shift[1]=X[1]
                        X_shift[2]=X[2]
                        score_shift,_=decode(X_shift,y)
                        score_temp.append(score_shift)
                    score_temp=N.nanmean(score_temp)
                    red_speed.append(score-score_temp)
                                   
                    score_temp=[]
                    for k in range(iterations):
                        X_shift=N.empty_like(X)
                        X_shift[0]=X[0]
                        X_shift[1]=N.random.permutation(X[1])
                        X_shift[2]=X[2]
                        score_shift,_=decode(X_shift,y)
                        score_temp.append(score_shift)
                    score_temp=N.nanmean(score_temp)
                    red_lick.append(score-score_temp)
                    
                    score_temp=[]
                    for k in range(iterations):
                        X_shift=N.empty_like(X)
                        X_shift[0]=X[0]
                        X_shift[1]=X[1]
                        X_shift[2]=N.random.permutation(X[2])
                        score_shift,_=decode(X_shift,y)
                        score_temp.append(score_shift)
                    score_temp=N.nanmean(score_temp)
                    red_acc.append(score-score_temp)
                    
            else:
                n_unmodulated+=1
      
        os.chdir(data_dir)
        
    res={'score':N.asarray(score_total),'coef':coeffs_total,'coef legend':['speed','licks','acc'],'cell ids':cell_names_total,
          'score wo speed':N.asarray(red_speed),'score wo lick':N.asarray(red_lick),'score wo acc':N.asarray(red_acc),
          'X':Xs,'y':ys,
          'n sig':n_sig_modulated,'n total':n_unmodulated}
    
    return res

### Executed part of the code.
high=run_analysis_master(data_dir_high,random_iterations=True)
low=run_analysis_master(data_dir_low)
res={'high':high,'low':low}



# Statistical comparisons.
# run pairwise comparisons.

def stat_comp(data1,data2):
    t,p1=st.shapiro(data1)
    t,p2=st.shapiro(data2)
    if p1>0.05 and p2>0.05:
        t,p=st.ttest_ind(data1,data2)
        used="t test"
    else:
        t,p=st.mannwhitneyu(data1,data2)
        used="MWU"
    return t,p,used

used_test,tt,pp,comp=[],[],[],[]
t,p,used=stat_comp(res['high']['score'],res['low']['score'])
tt.append(t)
pp.append(p)
comp.append('high v low score')
used_test.append(used) 
t,p,used=stat_comp(res['high']['score'],res['high']['score wo speed'])
tt.append(t)
pp.append(p)
comp.append('high v wo speed')
used_test.append(used)   
t,p,used=stat_comp(res['high']['score'],res['high']['score wo acc'])
tt.append(t)
pp.append(p)
comp.append('high v wo acc')
used_test.append(used)  
t,p,used=stat_comp(res['high']['score'],res['high']['score wo lick'])
tt.append(t)
pp.append(p)
comp.append('high v wo lick')
used_test.append(used) 

stat={'t':tt,'p':pp,'used test':used_test,'comparison':comp} 

used_test,tt,pp,comp=[],[],[],[]
t,p,used=stat_comp(res['high']['score wo speed'],res['low']['score wo speed'])
tt.append(t)
pp.append(p)
comp.append('high v low wo speed')
used_test.append(used)   
t,p,used=stat_comp(res['high']['score wo acc'],res['low']['score wo acc'])
tt.append(t)
pp.append(p)
comp.append('high v low wo acc')
used_test.append(used)  
t,p,used=stat_comp(res['high']['score wo lick'],res['low']['score wo lick'])
tt.append(t)
pp.append(p)
comp.append('high v low wo lick')
used_test.append(used) 

stat_across={'t':tt,'p':pp,'used test':used_test,'comparison':comp}      


res['statistics reduced models']=stat   
res['statistics high v low']=stat_across     
   
res['params']=params        
os.chdir(target_dir)
N.save("linear_modelling_of_SOMI_activity_before_reward.npy",res)                     