#!/usr/bin/env python
# coding: utf-8

## Decoding of reward expectation from spiking activity of individual SOMI units

import numpy as np
import matplotlib.pyplot as plt

import os
import glob
from pathlib import Path

from scipy.signal import decimate
from sklearn.neighbors import KernelDensity
from scipy.stats import ecdf 

#import adaptivekde # from Shimazaki & Shinomoto paper

###############################################################
# Constants

# Sampling rate, all timestamps are multiple of 1/sample_rate
sample_rate = 30 # in kHz (see Intro_to_Data_labeling.txt)
sample_rate_Hz = sample_rate * 1e3 # in Hz
#print(f'Sampling time step is {1/sample_rate} ms')

# Strings for folder names with data of expert and non-expert animals
expert_s = 'Expert'
non_expert_s = 'Non-expert'

behavior_folder = 'pos_fam/' # containing behavioral data
shank_folder_name = 'ms3--all_shank' # name of folders corresponding to the shanks without number

# strings indicating the different neuron types
somi_name = 'SOMI' # Somatostatin-expressing interneuron
fsi_name = 'FSI' # fast-spiking interneuron, most likely the Parvalbumin-expressing interneuron
pc_name = 'PC' # principal cell (granule cell or mossy cell)

unit_type_names = (pc_name, fsi_name, somi_name)

split_variant_names = ('even_odd', 'pairs_random', 'permute_random')

###############################################################

###############################################################
# Further constants

# Estimated latency of reward delievery
latency = 0.94 # in s 
latency_steps = round(latency * sample_rate_Hz)

# Define time windows for alignment of data and alignment function;
# before and after the reward (zone) start.
time_before = 6.0 # in s
time_after = time_before # in s, no reason to use different time

# transform to number of sampling time steps:
steps_before = round(time_before * sample_rate_Hz)
steps_after = round(time_after * sample_rate_Hz)

window_max = 4.0 # in s, maximum time window for binning before and after alignment time

min_pooled_rate = 0.5 # in Hz, minimum rate to consider unit for decoding analysis
###############################################################


###############################################################
# Functions for downsampling
# 
# For plotting, we need to decimate/down-sample the position and speed data. Here we use the function decimate() from scipy.signal.
def downsample(signal, dt):
    dec_factor = int(sample_rate * dt) # dt needs to have the same units as sample_rate
    signal_decim = decimate(signal, dec_factor, ftype='fir')
    #ts = dt * np.arange(signal_decim.shape[0]) # time points, with same units as dt
    return signal_decim

def get_timesteps(signal_decim, dt):
    return dt * np.arange(signal_decim.shape[0]) # time steps, with same units as dt

# Alternative reduction by boxed average (I think):
def reduce_mean(signal, dt):
    dec_factor = int(sample_rate * dt)
    last = dec_factor * (signal.shape[0] // dec_factor)
    signal_reduced = np.mean(signal[:last].reshape(-1,dec_factor), axis=1)
    ts = dt * (0.5 + np.arange(signal_reduced.shape[0])) # time points, with same units as dt
    return signal_reduced, ts

###############################################################


###############################################################
#
# Functions for data loading

def load_reward_times(folder = behavior_folder):
    # Load reward-related timestamps
    reward_start = np.load(folder + 'reward_start_ind.npy')
    rewardzone_start = np.load(folder + 'rewardzone_ind_start.npy')
    # yield arrays of int64
    return reward_start, rewardzone_start

def get_unit_paths(unit_type_names):
    # Use glob module:
    unit_path_list = [] # list of the units' spiking data paths
    unit_type_list = [] # list of the units' putative cell type

    num_units_type_shank = np.zeros((len(unit_type_names),4), dtype=int) # separated into shanks
    for i, cell_type in enumerate(unit_type_names):
        for shank in range(1,5):
            units_tmp = sorted(glob.glob(shank_folder_name + str(shank) + '/*_' + cell_type + '.txt'))
            num_units_type_shank[i,shank-1] = len(units_tmp)
            unit_path_list.extend(units_tmp)
            unit_type_list.extend([cell_type]*len(units_tmp)) # seems to work!

    num_units_type = num_units_type_shank.sum(axis=1) # number of units with different cell types,
    # in the order of unit_type_names
    return unit_path_list, unit_type_list, num_units_type

def load_spikes(unit_path_list):
    # Load spiking data from all recorded units
    # in the order of unit_path_list
    spikes_list = []
    for unit_name in unit_path_list:
        # check spike file is not empty:
        if Path(unit_name).stat().st_size:
            spikes_list.append(np.loadtxt(unit_name, ndmin=1).astype(np.int64))
            #spikes_list.append(np.loadtxt(unit_name).astype(np.int64))
        else:
            spikes_list.append(np.array([], dtype=np.int64))
    return spikes_list

# Using pathlib module
# p = Path('.')
# unit_path_list = sorted(p.glob(shank_folder_name + '?/*.txt'))
###############################################################

###############################################################
#
# Overview plot of recorded spiking activity
def plot_spikes_overview(spikes_list, unit_type_list, reward_start, rewardzone_start):
    spike_colors = ('tab:blue', 'tab:orange', 'tab:green')
    fig = plt.figure(dpi=150, figsize=(14,8))
    for i, spikes in enumerate(spikes_list):
        plt.plot(spikes/sample_rate_Hz, i*np.ones_like(spikes), ls='', 
                 marker='|', markersize=3,
                 color = spike_colors[unit_type_names.index(unit_type_list[i])])

    for i in range(reward_start.shape[0]):
        plt.axvline(reward_start[i]/sample_rate_Hz,
                    linestyle='--', linewidth = 0.5,
                    color = 'tab:red')

    for i in range(rewardzone_start.shape[0]):
        plt.axvline(rewardzone_start[i]/sample_rate_Hz,
                    linestyle=':', linewidth = 0.5,
                    color = 'tab:purple')

    plt.xlabel('time (s)')
    plt.ylabel('unit index')
    
    return fig

###############################################################


###############################################################
#
# Analyze spiking activity of individual units
# compute the average firing rate and perhaps variability, like CV and Fano factor
def get_isi_stats(spikes, plot_isi=False):
    intervals = np.diff(spikes)
    mu = intervals.mean()
    sigma = intervals.std()
    cv = sigma/mu # coefficient of variation
    rate = sample_rate_Hz/mu # in Hz
    if plot_isi:
        plt.figure('Interspike intervals', dpi=150, figsize=(6,4.5))
        plt.hist(intervals/sample_rate, bins='fd', 
                 density=True);
        plt.axvline(mu/sample_rate, linestyle='--', linewidth = 0.5,
                    color = 'tab:red')
        plt.xlabel('spike interval (ms)')
        plt.ylabel('prob. density')
    return rate, cv
        
# Average firing rate (in Hz):
# sample_rate_Hz/np.diff(spikes).mean()
# (len(spikes)-1)/(spikes[-1] - spikes[0])*sample_rate_Hz
# Interestingly, both methods yield the same result.

def get_spikes_stats(spikes, restr_cv=True, restr_cv_cutoff=5e3):
    num_spikes = len(spikes)
    if num_spikes < 2:
        return dict(num_spikes=num_spikes)
    intervals = np.diff(spikes)/sample_rate # in ms 
    mu = intervals.mean()
    med_interval = np.median(intervals)
    max_interval = intervals.max()
    min_interval = intervals.min()
    sigma = intervals.std()
    cv = sigma/mu
    rate = 1e3/mu # average rate during session, in Hz
    cv_restr = np.nan
    if restr_cv and np.sum(intervals<=restr_cv_cutoff) > 2:
        #num_restr_intervals = np.sum(intervals<=restr_cv_cutoff)
        #mu = intervals.mean(where = intervals<=restr_cv_cutoff)
        cv_restr = intervals.std(where = intervals<=restr_cv_cutoff)/intervals.mean(where = intervals<=restr_cv_cutoff)
    stats_dict = dict(num_spikes=num_spikes, rate=rate, cv=cv, mu=mu, med_interval=med_interval,
                      max_interval=max_interval, min_interval=min_interval, cv_restr=cv_restr)
    return stats_dict

###############################################################


###############################################################
#
# Select and align spikes in a window around reference steps:
def align_spikes(spikes, reward_start): #, steps_before, steps_after):
    spikes_aligned_list = []
    for start_step in reward_start:
        spikes_window = np.searchsorted(spikes, [start_step - steps_before, start_step + steps_after], side='left')
        spikes_aligned_list.append(spikes[spikes_window[0]:spikes_window[1]] - start_step)
    return spikes_aligned_list

def plot_spikes_aligned(spikes_aligned_list, start_aligned, spike_color='tab:blue'):
    fig = plt.figure(dpi=150, figsize=(7,5), layout="constrained")
    if start_aligned.max() < 0:
        align_color = 'tab:red'
        start_color = 'tab:purple'
    else:
        align_color = 'tab:purple'
        start_color = 'tab:red'
    plt.axvline(0, linestyle=':', linewidth = 1, color = align_color, alpha=0.75)
    for i, spikes_aligned in enumerate(spikes_aligned_list):
        plt.plot(spikes_aligned/sample_rate_Hz, i*np.ones_like(spikes_aligned), ls='', 
                 marker='|', markersize=3, color = spike_color)
        plt.plot(start_aligned[i]/sample_rate_Hz, i, ls='',
                 marker='d', markersize=5, color = start_color, fillstyle='none')
    plt.xlabel('time (s)')
    plt.ylabel('trial number')
    return fig

# Pool interspike intervals in the aligned spike data, 
# i.e., only from the windows around the reward(zone) start,
# and compute interval statistics for these:
def get_stats_aligned(spikes_aligned_list, plot_isi=False):
    intervals_list = []
    for spikes_aligned in spikes_aligned_list:
        intervals_list.append(np.diff(spikes_aligned))
    intervals = np.hstack(intervals_list)
    mu = intervals.mean()
    sigma = intervals.std()
    cv = sigma/mu # coefficient of variation
    mean_rate = sample_rate_Hz/mu # in Hz
    if plot_isi:
        plt.figure(dpi=150, figsize=(6,4.5))
        plt.hist(intervals/sample_rate, bins='fd', 
                 density=True);
        plt.axvline(mu/sample_rate, linestyle='--', linewidth = 0.5,
                    color = 'tab:red')
        plt.xlabel('spike interval (ms)')
        plt.ylabel('prob. density')
    return mean_rate, cv

###############################################################


###############################################################
#
# Kernel density estimator (KDE) of instanteous spike rate for the aligned spikes
# 
# Function to compute kernel density estimate of instantaneous spike rate
# for an array of spike timesteps
# and evaluated at times ts_rate (in seconds):
def get_kde_rate(spikes, bw, ts_rate):
    rate_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(spikes.reshape(-1, 1)/sample_rate_Hz)
    # Compute log-density from the kde for the sampling points in ts_rate:
    log_dens = rate_kde.score_samples(ts_rate.reshape(-1, 1))
    rate_est = spikes.shape[0]*np.exp(log_dens) # estimated (smoothed) instantaneous rate in Hz
    return rate_est

def get_rates_aligned(spikes_aligned_list, bw, ts_rate):
    rates_aligned = np.zeros((len(spikes_aligned_list),ts_rate.shape[0]))
    for i, spikes_aligned in enumerate(spikes_aligned_list):
        if spikes_aligned.shape[0] > 0:
            rates_aligned[i,:] = get_kde_rate(spikes_aligned, bw, ts_rate)
    return rates_aligned
# TODO: What should we do with trials with zero spikes?
# I think they should remain in there, both for the rate estimation and for the classification.

def create_ts_rate(dt_rate_ms = 10):
    num_ts = round((time_before+time_after)*1e3/dt_rate_ms)+1 # uses the constants from above
    ts_rate = np.linspace(-time_before, time_after, num=num_ts)
    dt_rate = 1e-3 * dt_rate_ms # in seconds
    return ts_rate, dt_rate

###############################################################


###############################################################
#
# Helper functions for trial splitting and binning
# Preprocessing: Binning of spikes and construction of expected spike counts
# from instantaneous rate

# Create true labels with 0 for "before" and 1 for "after"
# aranged as (num_trials x 2)-array
def get_true_labels(num_trials):
    labels = np.zeros((num_trials,2), dtype=int)
    labels[:,1] = 1
    return labels

def shuffle_labels(labels, rng):
    # TODO: How to use different rng instances?
    return rng.permutation(labels.flat).reshape((-1,2))

# Here, we completely shuffle the labels, hence the adjacent
# observed spike counts can be assigned the identical label.
# Alternatively, we could just by randomly switch the labels
# in each row of labels_true.
# I think the first approach is better/more appropriate.

def shuffle_trial_labels(labels, rng):
    # Only permute the labels in each trial,
    # Keeps structure of the data in trials, but less randomness?
    return rng.permuted(labels, axis=1)

# Split into even and odd trials:
# "even" includes index 0
# if num_trials is odd, the "even" trials contain one more
def split_even_odd(num_trials):
    inds_even = np.arange(0, num_trials, 2)
    inds_odd = np.arange(1, num_trials, 2)
    return inds_even, inds_odd

# Alternatively, split trials randomly in half:
def split_random(num_trials, rng):
    inds_permuted = rng.permutation(num_trials)
    inds_first = np.sort(inds_permuted[:(num_trials+1)//2])
    inds_second = np.sort(inds_permuted[(num_trials+1)//2:])
    return inds_first, inds_second

# The following is similar to split in even and odd trials,
# but selects of each pair of even and odd trials randomly
# if it is put in the first or the second half.
# Again if num_trials is odd, the "first" half contains one more trial
def split_pairs_random(num_trials, rng):
    num_pairs = num_trials//2
    switch_ind = rng.choice(2, num_pairs)
    # as in split_even_odd():
    inds_first = np.arange(0, num_trials, 2)
    inds_second = np.arange(1, num_trials, 2)
    inds_first[:num_pairs] += switch_ind
    inds_second -= switch_ind
    return inds_first, inds_second

# Binning of the spike trains
def create_bins(bin_width, bin_window = 4.0):
    # in seconds
    # time window for binning is before and after t0
    t0 = 0.0 # alignment time
    num_bins = round(bin_window/bin_width)
    # Create bin edges, in seconds, for before and after:
    bins_before = np.linspace(-bin_window, t0, num_bins+1)
    bins_after = np.linspace(t0, bin_window, num_bins+1)
    bins_all = np.hstack([bins_before[:-1], bins_after]) # assuming consecutive bins
    return num_bins, bins_before, bins_after, bins_all

def bin_spikes_aligned(spikes_aligned_list, bins_all):
    spike_counts_aligned = np.zeros((len(spikes_aligned_list), bins_all.shape[0]-1), dtype=int) # equal to 2*num_bins
    for i, spikes_aligned in enumerate(spikes_aligned_list):
        if spikes_aligned.shape[0] > 0:
            spike_counts, _ = np.histogram(spikes_aligned/sample_rate_Hz, bins_all)
            spike_counts_aligned[i,:] = spike_counts
            #spikes_test[num_test_trials+i,:] = spike_counts[num_bins_test:]
    return spike_counts_aligned

# Compute the rate spike counts by integrating the estimated rate
# from Gaussian kernel smoothing
def integrate_rate(rate, ts_rate, dt_rate, inds_bins):
    # indices of bin edges in ts_rate:
    # inds_bins = np.searchsorted(ts_rate, bins-0.5*dt_rate)
    rate_counts = np.zeros(inds_bins.shape[0]-1)
    for i in range(inds_bins.shape[0]-1):
        rate_counts[i] = np.trapz(rate[inds_bins[i]:inds_bins[i+1]+1], dx=dt_rate)
    return rate_counts

def get_rate_counts(rates_aligned, ts_rate, dt_rate, bins_all):
    # indices of bin edges in ts_rate:
    inds_bins = np.searchsorted(ts_rate, bins_all-0.5*dt_rate)
    rate_counts_aligned = np.zeros((rates_aligned.shape[0], inds_bins.shape[0]-1))
    for i, rate in enumerate(rates_aligned):
        rate_counts_aligned[i,:] = integrate_rate(rate, ts_rate, dt_rate, inds_bins)
    return rate_counts_aligned

# Alternatively, first compute cumulative integral of rate_train using cumulative_trapezoid().
# from scipy import integrate
# rate_train_integral = integrate.cumulative_trapezoid(rate_train, dx=dt_rate, initial=0)
# Then compute differences of values at bin edge indices:
# np.diff(rate_train_integral[inds_bins_before])
# np.diff(rate_train_integral[inds_bins_after])
# This alternative method seems to agree up to machine precision.
#

# Reduced/shorter bin window for decoding analysis
# Get the indices for before and after:
def get_window_idx(bin_window, bin_width, num_bins):
    num_bins_window = round(bin_window/bin_width)
    # generate appropriate index ranges for slicing
    # both have the same length len(idx_range_before)
    idx_range_before = range(num_bins-num_bins_window, num_bins)
    idx_range_after = range(num_bins, num_bins+num_bins_window)
    return idx_range_before, idx_range_after

###############################################################


###############################################################
#
# Training and testing
# Training uses only rate_counts_aligned or _train, the integrated smoothed rate count estimates
# and the given labels and the trial indices to be used for training:
def train_decoder(rate_counts_train, labels, train_inds):
    # training data, reshaped
    rate_counts_temp = np.reshape(rate_counts_train[train_inds],
                                  (-1, rate_counts_train.shape[1]//2), order='C')
    labels_temp = np.reshape(labels[train_inds], (-1,), order='C') # labels[train_inds].ravel()
    #print(labels_temp[:6])
    labels_before = (labels_temp==0) # 1-d boolean array
    labels_after = (labels_temp==1) # equal to np.logical_not(labels_before)
    if np.all(labels_before):
        expected_counts_before = np.mean(rate_counts_temp[labels_before], axis=0)
        expected_counts_after = np.nan * np.ones_like(expected_counts_before)
        print('expected_counts_after set to array filled with np.nan')
    elif np.all(labels_after):
        expected_counts_after = np.mean(rate_counts_temp[labels_after], axis=0)
        expected_counts_before = np.nan * np.ones_like(expected_counts_after)
        print('expected_counts_before set to array filled with np.nan')
    else:
        expected_counts_before = np.mean(rate_counts_temp[labels_before], axis=0)
        expected_counts_after = np.mean(rate_counts_temp[labels_after], axis=0)
    # TODO: check what happens to the np.nan arrays in the subsequent analysis
    return expected_counts_before, expected_counts_after

# Testing uses the spike_counts_aligned or _test, the binned spike count data
# and the given labels and the trial indices to be used for testing:
def test_decoding(spike_counts_test, labels, test_inds, expect_before, expect_after):
    # reshape testing data:
    spike_counts_temp = np.reshape(spike_counts_test[test_inds], (-1, spike_counts_test.shape[1]//2), order='C')
    labels_temp = np.reshape(labels[test_inds], (-1,), order='C')
    #print(labels_temp[:6])
    decoded_labels, _ = get_decoded_labels(spike_counts_temp, expect_before, expect_after)
    # compute accuracy
    frac_correct, num_correct = get_accuracy(decoded_labels, labels_temp)
    #print('Accuracy:', frac_correct)
    return frac_correct, num_correct

# Do the training for a subset, namely the training set, which we choose in some way. 
# For now, we use half of the trials: '2-fold cross-validation'.
# Directly assess accuracy using the the two splits:
def train_test_decoder(rate_counts_train, spike_counts_test, labels, inds_split):
    num_split = len(inds_split) # equal to 2, only two-fold
    frac_correct = np.zeros(num_split)
    frac_correct_train = np.zeros(num_split)
    frac_correct_rate_before = np.zeros(num_split) # using the time-dependent rate for 'before'
    for i in range(num_split):
        train_inds = inds_split[i]
        test_inds = inds_split[(i+1) % 2]
        # print('i =', i)
        # print(train_inds)
        # print(test_inds)
        expect_before, expect_after = train_decoder(rate_counts_train, labels, train_inds)
        frac_correct[i], _ = test_decoding(spike_counts_test, labels, test_inds,
                                           expect_before.mean(), expect_after)
        # Don't do the analysis using the time-dependent rate for 'before':
        # frac_correct_rate_before[i] = frac_correct[i]
        frac_correct_rate_before[i], _ = test_decoding(spike_counts_test, labels, test_inds,
                                                       expect_before, expect_after)
        # accuracy when using the training data, using train_inds:
        # Note: We do the decoding with training data only using the mean for before.
        frac_correct_train[i], _ = test_decoding(spike_counts_test, labels, train_inds,
                                                 expect_before.mean(), expect_after)
    return frac_correct, frac_correct_rate_before, frac_correct_train
       
# Functions for decoding
def log_prob_poisson(spike_counts, mean_vals):
    # Compute the log-probability of the observed spike counts
    # in the array spike_counts
    # under the assumption of independent Poisson distributions
    # with mean parameters given by the array mean_vals.
    # Both should have the same length.
    #
    # TODO: What should be done if an entry of mean_vals is zero?
    # If the corresponding count is zero than it is okay,
    # and we can set 0*log(0) = 0. Otherwise, the log-prob should be -inf,
    # as this cannot occur under the assumed model.
    # For now, we assume that np.min(mean_vals) > 0.
    # Furthermore, spike_counts should be non-negative integers.
    # Simple solution: add a very small eps>0 to mean_vals
    #
    # We can leave out the following term for the Bayes classification
    # because it does not depend on the parameters mean_vals
    # and is therefore identical for different models:
    # from scipy.special import factorial
    # log_fac_sum = np.sum(np.log(factorial(spike_counts)))
    return np.sum(spike_counts * np.log(mean_vals + 1e-16) - mean_vals)
    #return np.sum(spike_counts * np.log(mean_vals) - mean_vals) - log_fac_sum

# Because the features, spike counts in the different bins, in the Poisson model
# are independent, the resulting Bayes classifier corresponds to a naive Bayes classifier
# with Poisson distributed features. Perhaps, one can use approaches and implementations
# available for naive Bayes classification.
def bayes_classifier(spike_counts, expect_before, expect_after):
    # estimated expected spike counts
    # expect_before and expect_after
    # need to have the same length as spike_counts
    # or can be scalars
    # prediction of classifier: 0 for "before" and 1 for "after"
    log_prob_before = log_prob_poisson(spike_counts, expect_before)
    log_prob_after = log_prob_poisson(spike_counts, expect_after)
    log_probs = np.array([log_prob_before, log_prob_after]) # log-probabilities of before and after
    decoded_label = np.argmax(log_probs)
    return decoded_label, log_probs

def get_decoded_labels(spike_counts_decode, expect_before, expect_after):
    # Bayes decoding of a 2d array of spike counts using
    # the estimated expected spike counts before and after
    log_probs_decode = np.zeros((spike_counts_decode.shape[0],2))
    decoded_labels = np.zeros(spike_counts_decode.shape[0], dtype=int)
    for i, spike_counts in enumerate(spike_counts_decode):
        decoded_labels[i], log_probs_decode[i] = bayes_classifier(spike_counts, expect_before, expect_after)
    return decoded_labels, log_probs_decode

def get_accuracy(decoded_labels, labels):
    # compute fraction of correct classifications/decodings
    # both need to be integer arrays with the same length
    # We don't distinguish between the sort of errors, i.e.,
    # "before" wrongly classified as "after" or vice versa. 
    num_correct = np.sum(decoded_labels == labels) # both are binary (or, in general, integer)
    frac_correct = num_correct/labels.shape[0]
    return frac_correct, num_correct

def decoding_analysis(bin_window, bin_width, num_bins, rate_counts_aligned, spike_counts_aligned, true_labels, inds_split, rng_shuffle):
    rng_shuffle = np.random.default_rng(rng_shuffle)
    # Range of indices for the binned data for the given bin_window
    idx_range_before, idx_range_after = get_window_idx(bin_window, bin_width, num_bins)
    
    # Restrict the data arrays for the decoding analysis to the shorter bin_window:
    rate_counts_train = rate_counts_aligned[:, list(idx_range_before) + list(idx_range_after)]
    spike_counts_test = spike_counts_aligned[:, list(idx_range_before) + list(idx_range_after)]
    # num_bins_window = len(idx_range_before) # not used

    # Train and test the decoder using the data in bin_window for the true labels and the given split
    result_true_labels = np.array(train_test_decoder(rate_counts_train, spike_counts_test, true_labels, inds_split))
    #print('Decoding fraction correct:', result_true_labels)
    # 3 variants: frac_correct, frac_correct_rate_before, frac_correct_train
    # using mean rate for 'before' (default), 
    # using the time-dependent rate for 'before', using training trials (training accuracy?)
    frac_correct = result_true_labels.mean(axis=1)

    # Estimate significance of decoding accuracy with permutations/shuffles of decoding training and testing:
    num_shuffle = 1000
    fc_shuffled = np.zeros((num_shuffle,frac_correct.shape[0]))

    for i in range(num_shuffle):
        #labels = shuffle_labels(true_labels, rng_shuffle)
        labels = shuffle_trial_labels(true_labels, rng_shuffle)
        result = train_test_decoder(rate_counts_train, spike_counts_test, labels, inds_split)
        fc_shuffled[i,:] = list(map(np.mean, result))

    # Compute p-values:
    p_values = np.ones_like(frac_correct)
    for i, fc in enumerate(frac_correct):
        p_values[i] = 1/num_shuffle + np.mean(fc_shuffled[:,i] >= (fc-1e-8))
    # set off frac_correct by small negative number to not miss some values
    # due to floating point arithmetic
    
    # Alternative way to obtain the p-values using the
    # empirical cumulative distribution function (here: ecdf from scipy.stats)
    # p_vals_ecdf = np.ones_like(frac_correct)
    # ecdfs = []
    # for i in range(3):
    #     ecdfs.append(ecdf(fc_shuffled[:,i]))
    #     p_vals_ecdf[i] = ecdfs[i].sf.evaluate(frac_correct[i]-1e-8) # shift to the left to get the value before a jump

    # 3 variants: frac_correct, frac_correct_rate_before, frac_correct_train
    return frac_correct, p_values

def decoding_windows(windows, bin_width, num_bins,
                     rate_counts_aligned, spike_counts_aligned,
                     true_labels, inds_split, output=False):
    # We use the same seed for different window length for a unit.    
    shuffle_seed = np.random.SeedSequence().entropy # generate a new seed each time the function is called
    #shuffle_seed = 314119939430639218714413498984034137457 # from secrets.randbits(128)
    # TODO: But should we use different seeds for the analysis of different units and in different sessions?
    # I think different sessions is clear because of different number of trials and thus labels.    

    fc_list = []
    pv_list = []
    
    for window in windows:
        if output: print('Window:', window, 'seconds')
        frac_correct, p_values = decoding_analysis(window, bin_width, num_bins,
                                                   rate_counts_aligned, spike_counts_aligned,
                                                   true_labels, inds_split, rng_shuffle=shuffle_seed)
        if output:
            print('Average accuracy:', frac_correct)
            #print('p-values from shuffling:', p_values)
        fc_list.append(frac_correct)
        pv_list.append(p_values)

    fc_arr = np.array(fc_list)
    pv_arr = np.array(pv_list)
    return fc_arr, pv_arr

def create_windows(start=0.25, stop=3.75, step=0.125):
    # for decoding analysis
    # in seconds, step=2**-3
    num_wins = round((stop-start)/step) + 1
    return np.linspace(start, stop, num_wins)

def get_counts_aligned(spikes_aligned_list, bw_ms):
    # Estimate aligned rates using kernel density smoothing
    bw = 1e-3 * bw_ms # bandwidth in seconds
    dt_rate_ms = 1e3 * 2**-9 # default value: 1.953125 ms
    ts_rate, dt_rate = create_ts_rate(dt_rate_ms) # in seconds
    rates_aligned = get_rates_aligned(spikes_aligned_list, bw, ts_rate)
    
    # Binning of spikes and construction of expected spike counts
    bin_width = 2**2 * dt_rate # 2**2 integer multiple of dt_rate
    # should fit into win_step
    
    num_bins, bins_before, bins_after, bins_all = create_bins(bin_width, window_max)
    # num_bins is also the index of the bin in bins_all that starts at t0 = 0.0
    spike_counts_aligned = bin_spikes_aligned(spikes_aligned_list, bins_all)
    rate_counts_aligned = get_rate_counts(rates_aligned, ts_rate, dt_rate, bins_all)

    return (spike_counts_aligned, rate_counts_aligned,
            bin_width, num_bins,
            ts_rate, rates_aligned)


def decoding_spikes_aligned(spikes_aligned_list, bw_ms, split_variant, rng_split, print_output=True):
    # Perfom the full decoding analysis on a list of aligned spike trains for given parameters
    rng_split = np.random.default_rng(rng_split)
    
    num_trials = len(spikes_aligned_list)
    # Get true labels: 0 for "before" and 1 for "after"
    true_labels = get_true_labels(num_trials)
    
    # Estimate aligned rates using kernel density smoothing
    bw = 1e-3 * bw_ms # in seconds
    dt_rate_ms = 1e3 * 2**-9 # default value: 1.953125 ms
    ts_rate, dt_rate = create_ts_rate(dt_rate_ms) # in seconds
    rates_aligned = get_rates_aligned(spikes_aligned_list, bw, ts_rate)
    # TODO: plot rates_aligned, or return or save it?
    
    # Binning of spikes and construction of expected spike counts
    bin_width = 2**2 * dt_rate # 2**2 integer multiple of dt_rate
    # should fit into win_step
    
    num_bins, bins_before, bins_after, bins_all = create_bins(bin_width, window_max)
    # num_bins is also the index of the bin in bins_all that starts at t0 = 0.0
    spike_counts_aligned = bin_spikes_aligned(spikes_aligned_list, bins_all)
    rate_counts_aligned = get_rate_counts(rates_aligned, ts_rate, dt_rate, bins_all)
    
    # Window lengths for analysis:
    win_start = 0.25 #0.25 # in seconds
    win_stop = 3.75 #3.75 should be less than window_max
    win_step = max(0.125, bin_width) #0.125 in seconds, 2**-3
    win_arr = create_windows(win_start, win_stop, win_step)
    
    # Create a split of the trials:
    if print_output:
        print('Using split variant:', split_variant_names[split_variant])
    
    if split_variant == 0:
        # Split into even and odd trials:
        inds_split = split_even_odd(num_trials)
    elif split_variant == 1:
        # Split subsequent pairs of trials randomly:
        inds_split = split_pairs_random(num_trials, rng_split)
    elif split_variant == 2:
        # or simply split randomly into two halfs (using a random permutation):
        inds_split = split_random(num_trials, rng_split)
    
    # Perform decoding analysis for a collection of time windows:
    # 3 variants in this order: frac_correct, frac_correct_rate_before, frac_correct_train
    fc_arr, pv_arr = decoding_windows(win_arr, bin_width, num_bins, 
                                      rate_counts_aligned, spike_counts_aligned, 
                                      true_labels, inds_split, output=print_output)
    
    results_d = dict(win_arr=win_arr, fc_arr=fc_arr, pv_arr=pv_arr, bw_ms=bw_ms, split_variant=split_variant, bin_width=bin_width)
    return results_d, ts_rate, rates_aligned

def save_decoding_results(unit_decoding_s, results_d, results_folder):
    #unit_decoding_s = unit_id_s + f'_bw{bw_ms}' + f'_split_{split_variant_names[split_variant]}'
    decoding_results_s = 'decoding__' + unit_decoding_s
    fname = Path(results_folder) / decoding_results_s
    np.savez(fname, **results_d)
    return fname

def plot_decoding_results(win_arr, fc_arr, pv_arr, plot_train_fc=True):
    # Plot results of decoding analysis
    fig, axs = plt.subplots(2,2, dpi=150, figsize=(10,7.5),
                            sharex='col', sharey='row',
                            layout="constrained")
    
    for i, ax in enumerate(axs[0]):
        if i == 0 and plot_train_fc:
            ax.plot(win_arr, fc_arr[:,2], 'o--', color='C1', alpha=0.5, label='training accuracy')
        ax.plot(win_arr, fc_arr[:,i], marker='.', label='test accuracy')
        ax.axhline(0.65, linestyle=':', linewidth=0.5, color='k')
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('window length (s)')
        ax.set_ylabel('fraction correct')
        if i == 0:
            ax.legend()
            ax.set_title("using mean rate for 'before'")
        elif i == 1:
            ax.set_title("using time-dep. rate for 'before'")
    
    for i, ax in enumerate(axs[1]):
        ax.plot(win_arr, pv_arr[:,i], marker='.')
        ax.axhline(0.05, linestyle=':', linewidth=0.5, color='k')
        ax.set_xlabel('window length (s)')
        ax.set_ylabel('p-value')
    
    return fig

def plot_rates_aligned(unit_id_s, ts_rate, rates_aligned, bw_ms, rng):
    rates_aligned_mean = rates_aligned.mean(axis=0)
    rates_aligned_std = rates_aligned.std(axis=0)

    num_trials = rates_aligned.shape[0]
    num_example = min(5,num_trials)
    example_inds = rng.choice(num_trials, num_example, replace=False)
    
    fig, ax = plt.subplots(1,1, dpi=150, figsize=(5,4))
    ax.axvline(0, linestyle=':', linewidth = 0.5, color = 'k')
    for ind in example_inds:
        ax.plot(ts_rate, rates_aligned[ind], linewidth = 1, alpha=0.5)
    mean_color = '0.2'
    ax.plot(ts_rate, rates_aligned_mean, color=mean_color)
    ax.plot(ts_rate, rates_aligned_mean+rates_aligned_std, '--', color=mean_color)
    ax.plot(ts_rate, rates_aligned_mean-rates_aligned_std, '--', color=mean_color)

    ax.text(0.02, 0.98, f'bandwidth: {bw_ms} ms', horizontalalignment='left',
         verticalalignment='top', transform=ax.transAxes)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('rate (Hz)')
    ax.set_title('Unit: ' + unit_id_s)
    #ax.set_title('aligned at reward start')

    return fig


def train_all(bin_window, bin_width, num_bins, cells_data_d, split=0):
    # Range of indices for the binned data for the given bin_window
    idx_range_before, idx_range_after = get_window_idx(bin_window, bin_width, num_bins)
    idx_window = list(idx_range_before) + list(idx_range_after)
    # rate_counts_train = rate_counts_aligned[:, list(idx_range_before) + list(idx_range_after)]
    
    for cell_id_s, data_d in cells_data_d.items():
        #print(cell_id_s)

        num_trials = data_d['num_reward_start']
        if split == None:
            train_inds = np.arange(num_trials) #range(num_trials)
            test_inds = np.arange(num_trials) #range(num_trials)
        else:
            inds_split = split_even_odd(num_trials)
            # select train and test trials:
            train_inds = inds_split[split] 
            test_inds = inds_split[(split+1) % 2]

        # Restrict the data arrays for the decoding analysis to the shorter bin_window:
        #spike_counts_aligned = data_d['spike_counts_list'][repeat][:, idx_window]
        rate_counts_train = data_d['rate_counts_aligned'][:, idx_window]
        
        #rate_counts_train = spike_counts_aligned[train_inds, :]
        #rate_counts_train = spike_counts_aligned[:, idx_window]
        #rate_counts_train = data_d['rate_counts_aligned'][:, idx_window]
        labels_train = data_d['true_labels'] # [train_inds] # use full labels array
        
        # Corresponding restricted data arrays to be used for testing
        spike_counts_test = data_d['spike_counts_aligned'][:, idx_window]
        #spike_counts_test = spike_counts_aligned[test_inds, :]
        #spike_counts_test = spike_counts_aligned[:, idx_window]
        #spike_counts_test = data_d['spike_counts_aligned'][:, idx_window]
        labels_temp = np.reshape(data_d['true_labels'][test_inds], (-1,), order='C')
        
        #num_trials = data_d['num_reward_start']
        #inds_split = split_even_odd(num_trials)
        
        #train_inds = np.arange(rate_counts_train.shape[0]) #range(num_trials) # inds_split[0]
        # Training
        expect_before, expect_after = train_decoder(rate_counts_train, labels_train, train_inds)
    
        # Check training, i.e. computation of expected counts from rate counts:
        # fig, ax = plt.subplots(1,1,dpi=100)
        # ax.plot(bin_width*np.array(idx_window), rate_counts_train.mean(axis=0)/bin_width)
        # ax.plot(bin_width*np.array(list(idx_range_before)), expect_before/bin_width)
        # ax.plot(bin_width*np.array(list(idx_range_after)), expect_after/bin_width)
    
        # expect_before, expect_after = train_decoder(rate_counts_train, data_d['true_labels'], inds_split[1])
        # ax.plot(bin_width*np.array(list(idx_range_before)), expect_before/bin_width)
        # ax.plot(bin_width*np.array(list(idx_range_after)), expect_after/bin_width)
    
        # ax.set_title(cell_id_s)
        # plt.show()
        # plt.close()
        # seems to work
    
        # At this point, we can already compute the log-probs
        # for each spike count pattern
        #test_inds = np.arange(spike_counts_test.shape[0]) #range(num_trials) # inds_split[1]
        spike_counts_temp = np.reshape(spike_counts_test[test_inds],
                                       (-1, spike_counts_test.shape[1]//2), order='C')
        # spike_counts_temp = np.reshape(spike_counts_test[test_inds],
        #                                (-1, spike_counts_test.shape[1]//2), order='C')

        # to be used for evaluation later, or put into an array
        #labels_temp = np.reshape(data_d['true_labels'][test_inds], (-1,), order='C')
        decoded_labels, log_probs = get_decoded_labels(spike_counts_temp,
                                                       expect_before.mean(), expect_after)
        # log-probabilities of before and after
        # see: log_probs = np.array([log_prob_before, log_prob_after]) 
        
        # store log-probs and corresponding true labels for testing: labels_temp
        # ignore index splits for now
        #data_d['repeat'] = repeat
        data_d['split'] = split
        data_d['window'] = bin_window # the selected window length
        data_d['labels_decode'] = labels_temp
        data_d['log_probs'] = log_probs
        
        # compute accuracy
        frac_correct, num_correct = get_accuracy(decoded_labels, labels_temp)
        #print('Accuracy:', frac_correct)
        data_d['frac_correct'] = frac_correct
    return


def test_decoding_all(keys_sorted, cells_data_d, rng_test, num_test_decoding = 10_000):
# Put computed log-probs and labels for testing of decoding in lists:
    labels_decode_l = []
    log_probs_l = []
    num_test_trials_l = []
    for key in keys_sorted:
        #print(key)
        labels_decode_l.append(cells_data_d[key]['labels_decode'])
        log_probs_l.append(cells_data_d[key]['log_probs'])
        num_test_trials_l.append(cells_data_d[key]['labels_decode'].shape[0]//2)
    
    # Actual testing:
    labels_trial = np.array([0,1])
    num_correct = np.zeros((len(keys_sorted),2), dtype=int)
    for r in range(num_test_decoding):
        log_probs_cells = np.zeros((len(keys_sorted),2,2))
    
        for i, key in enumerate(keys_sorted):
            num_test_trials = labels_decode_l[i].shape[0]//2
            trial_ind = rng_test.choice(num_test_trials)
            #labels_trial = labels_decode_l[i][2*trial_ind:2*(trial_ind+1)] # equals array([0,1])
            log_probs_temp = log_probs_l[i][2*trial_ind:2*(trial_ind+1),:]
            log_probs_cells[i] = log_probs_temp
            
        log_probs_csum = np.cumsum(log_probs_cells, axis=0)
        decoded_labels = np.argmax(log_probs_csum, axis=2)
        num_correct += (decoded_labels == labels_trial)

    frac_correct = num_correct/num_test_decoding
    return frac_correct

###############################################################