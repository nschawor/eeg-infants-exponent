""" This script reproduces the content of Fig. 1 in the manuscript."""

import mne
from mne.time_frequency import psd_multitaper
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import fooof
import helper

raw_folder = '../working/raw/'
exp_folder = '../results/exponents/'

plt.ion()

# fit one segment
session = 'sub-02_ses-02'
raw = mne.io.read_raw_fif('%s/%s_raw.fif' % (raw_folder, session))
raw.load_data()
raw = helper.remove_bad_chans(session, raw)
raw.pick_channels(['PO3'])

# cut in epochs
duration_epochs = 10
events = mne.make_fixed_length_events(raw, start=0, stop=raw.times[-1],
                                      duration=duration_epochs)
epochs = mne.Epochs(raw, events, tmin=0, tmax=duration_epochs,
                    baseline=(None, None), preload=True)

# select one segment for plotting
psd, freq = psd_multitaper(epochs, fmin=1, fmax=30.01, bandwidth=1)
fm = fooof.FOOOF()
which_segment = 0
fm.fit(freq, psd[which_segment, 0])


fig = plt.figure()
gs = gridspec.GridSpec(2, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.2])

# example trace plot
session = 'sub-02_ses-02'
raw = mne.io.read_raw_fif('%s/%s_raw.fif' % (raw_folder, session))
assert(len(raw.ch_names) == 32)

# example trace plot
raw.load_data()
raw.filter(1, 45)
raw.crop(308, None)
sensors = ['PO3']
raw.pick(sensors)
window = int(duration_epochs*raw.info['sfreq'])
nr_segments_to_plot = 7
colors = ['gray']*nr_segments_to_plot
colors[which_segment] = 'k'
ax = plt.subplot(gs[0, :3])
for i in range(nr_segments_to_plot):
    signal = raw._data[0, i*window:(i+1)*window]
    signal = signal/np.ptp(signal)
    ax.plot(raw.times[:window], signal+i, color=colors[i], lw=1)
ax.set(yticks=[], xlabel='time [s]')
ax.set_title(label='EEG activity for a single subject\nchannel PO3',
             ha='center')

# PSD example
ax = plt.subplot(gs[0, 3])

# plot psd
ax.semilogx(fm.freqs, fm.power_spectrum, color='k', lw=0.8)
freqs = range(1, 10)

# plot aperiodic fit
ap = fooof.sim.gen.gen_aperiodic(freqs, fm.aperiodic_params_)
ax.semilogx(freqs, ap, color='r', lw=1.5)
label = 'exp=%.3f' % fm.aperiodic_params_[1]
ax.text(0.3, 0.8, label, transform=ax.transAxes, color='k')

ax.set(xlabel='log frequency [Hz]', ylabel='log PSD', yticks=[],
       xticks=[1, 10, 25], xticklabels=[1, 10, 25], xlim=(1, 30.1))

# example histogram
mean_color = 'orange'
session = 'sub-02_ses-02'
df_file_name = '%s/%s_exponents.csv' % (exp_folder, session)
df_exp = pd.read_csv(df_file_name)

ax1 = plt.subplot(gs[0, 4])
ax1.hist(df_exp.PO3, color='k')
ax1.axvline(np.mean(df_exp.PO3), color=mean_color, alpha=0.8)
ax1.set(xlabel='exponent\nper segment', xticks=[2, 3, 4])
ax1.set_yticks([])

# time course of 1/f exponents
df = pd.read_csv('../csv/sessions.csv')
df = df[df.subject == 2]
nr_sessions = len(df)

# load exponents
exp_sessions = np.zeros((nr_sessions, 2))
for i_ses, session in enumerate(df.subject_id):
    df_file_name = '%s/%s_exponents.csv' % (exp_folder, session)
    df_exp = pd.read_csv(df_file_name)
    exp_sessions[i_ses, 0] = df_exp.mean().PO3
    exp_sessions[i_ses, 1] = df_exp.std().PO3


ax1 = plt.subplot(gs[0, 5])
ax1.errorbar(range(1, nr_sessions+1), exp_sessions[:, 0], color='k',
             yerr=exp_sessions[:, 1], marker='.', markerfacecolor=mean_color,
             markeredgecolor='k', lw=1, markersize=10)
ax1.set_xlabel('session')
ax1.set_xticks(range(1, nr_sessions+1))
ax1.set_ylabel('mean exponent\nacross segments')
ax1.yaxis.set_label_position('right')
ax1.yaxis.set_ticks_position('right')

# 1/f topo
vmin = 1.8
vmax = 2.8

for i_session in range(6):
    session = 'sub-02_ses-%02i' % (i_session+1)
    age = df[df.subject_id == session].age
    df_exp = pd.read_csv('%s/%s_exponents.csv' % (exp_folder, session))

    raw = mne.io.read_raw_fif('../working/raw/%s_raw.fif' % session)
    raw.load_data()
    raw = helper.remove_bad_chans(session, raw)
    df_exp = df_exp[raw.ch_names]

    exponents = df_exp.mean(skipna=True).to_numpy()
    idx = np.isnan(exponents)
    exponents = exponents[~idx]
    raw.pick_channels(np.array(raw.ch_names)[~idx])
    ax1 = plt.subplot(gs[1, i_session])

    im = mne.viz.plot_topomap(exponents, raw.info,
                              vmin=vmin, vmax=vmax, axes=ax1)
    ax1.set_xlabel('session %i\n%i days' % (i_session+1, age))


ax_cb = plt.subplot(gs[1, 6])
cb = plt.colorbar(im[0], orientation='vertical', cax=ax_cb)
cb.set_label('aperiodic exponent')

fig.set_size_inches(7.86, 4)
fig.savefig('../results/fig1_exponent_example.png', dpi=300)

fig.show()
