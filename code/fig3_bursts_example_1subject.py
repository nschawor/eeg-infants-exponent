""" This script reproduces the content of Fig. 3 in the manuscript."""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helper

plt.ion()
mne.set_log_level(verbose=False)
sessions = pd.read_csv('../csv/sessions.csv')

fmin = 3
fmax = 7

# load data and apply spatial filters
subject = 'sub-05_ses-03'
raw = mne.io.read_raw_fif('../working/ica/%s_raw.fif' % subject)
raw.load_data()
raw = helper.remove_bad_chans(subject, raw, interpolate=True)
raw.set_eeg_reference('average')

laplacians = helper.get_laplacian_chanlist()
raw_lap = helper.create_laplacian(laplacians, raw)


fig = plt.figure()
gs = gridspec.GridSpec(3, 4, width_ratios=[1, 2, .7, .7])

# plot spatial patterns
raw.filter(fmin, fmax)
cov_raw = np.cov(raw.get_data())

for i, deriv in enumerate(laplacians):
    sensors = laplacians[deriv]
    picks = mne.pick_channels(raw.ch_names, sensors, ordered=True)
    nr_channels = len(raw.ch_names)
    W = np.zeros((nr_channels))
    W[picks[0]] = 1
    W[picks[1:]] = -.25

    spatial_pattern = cov_raw @ W
    ax1 = plt.subplot(gs[i, 0])
    mne.viz.plot_topomap(spatial_pattern, raw.info, axes=ax1, cmap='PiYG')
    label = 'channel:\nLaplacian %s' % deriv.split('-')[0]
    ax1.set_ylabel(label, labelpad=20.5)
    ax1.yaxis.set_label_position('right')
    ax1.set_xlim(-.09, .09)


# load burst detection
mean_color = 'orange'
channels = ['C3-lap', 'C4-lap', 'Pz-lap']
results_dir = '../results/bursts/%s/' % (subject)

dfs = []
for channel in channels:
    print(channel)
    df_file_name = '%s/bursts_%s.csv' % (results_dir, channel)
    df = pd.read_csv(df_file_name)
    dfs.append(df)


raw_lap.filter(1, 45)
color = 'k'
idx1 = [184.25, 184.25, 61.75]
window = int(1.75*raw.info['sfreq'])
for i in range(3):
    ax = plt.subplot(gs[i, 1])
    idx = np.argmin(np.abs(raw_lap.times-idx1[i]))

    df_sel = dfs[i]
    df_sel = df_sel[df_sel.is_burst]
    df = df_sel[(df_sel.sample_peak > idx) & (df_sel.sample_peak < idx+window)]

    data = raw_lap.get_data()
    signal = data[i, idx:idx+window]
    normalize = np.ptp(signal)
    signal = signal/normalize
    time = raw_lap.times[idx:idx+window]
    ax.plot(time, signal+i, color=color, lw=1)

    for ii in range(len(df)):
        idx_l = df.iloc[ii].sample_last_trough
        idx_n = df.iloc[ii].sample_next_trough
        idx_p = df.iloc[ii].sample_peak
        begin = raw_lap.times[idx_l]
        end = raw_lap.times[idx_n]
        ax.axvspan(begin, end, alpha=0.35, facecolor='orange', edgecolor='r')
        ax.plot(raw_lap.times[idx_p], data[i, idx_p]/normalize+i, 'r.')
        ax.plot(raw_lap.times[idx_n], data[i, idx_n]/normalize+i, 'b.')

    ax.set_yticks([])
    ax.set_xticks(np.arange(time[0], time[-1]+0.1, .5))

    # plot histograms
    ax1 = plt.subplot(gs[i, 2])
    ax1.hist(df_sel.frequency, color='k')
    ax1.axvline(np.mean(df_sel.frequency), color=mean_color)


    ax2 = plt.subplot(gs[i, 3])
    ax2.hist(df_sel.volt_amp, color='k')
    ax2.axvline(np.mean(df_sel.volt_amp), color=mean_color)


ax1.set(xlabel='oscillation\nfrequency [Hz]')
ax2.set(xlabel='cycle \t\namplitude')
ax2.set_xticks([2.e-05, 3.e-05, 4.e-05])
ax2.set_xticklabels([2, 3, 4])
ax.set(yticks=[], xlabel='time [s]')

fig.set_size_inches(7.86, 4)
fig.tight_layout()
fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig.savefig('../results/fig3_burst_example.png', dpi=300)

fig.show()
