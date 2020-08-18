"""Computes spatial patterns for Laplacian filters.

In order to examine the spatial extent of the used Laplacian filter,
spatial patterns are computed with aid of the covariance matrix.
"""

import mne
import numpy as np
import pandas as pd
import helper

mne.set_log_level(verbose=False)
sessions = pd.read_csv('../csv/sessions.csv')

fmin = 3
fmax = 7

laplacians = helper.get_laplacian_chanlist()
nr_laplacians = len(laplacians)
nr_sessions = len(sessions)
nr_channels = 32

spatial_patterns = np.zeros((nr_laplacians, nr_sessions, nr_channels))

for i_ses, subject in enumerate(sessions.subject_id):
    print(subject)
    results_dir = '../results/bursts/%s/' % (subject)

    # load and filter data in frequency band of interest
    raw = mne.io.read_raw_fif('../working/ica/%s_raw.fif' % subject)
    raw.load_data()
    raw.set_eeg_reference('average')
    raw = helper.remove_bad_segments(subject, raw)
    raw.filter(fmin, fmax)

    # compute covariance matrix
    cov_raw = np.cov(raw.get_data())

    # compute spatial pattern
    for i, deriv in enumerate(laplacians):
        sensors = laplacians[deriv]
        picks = mne.pick_channels(raw.ch_names, sensors, ordered=True)
        nr_channels = len(raw.ch_names)
        W = np.zeros((nr_channels))
        W[picks[0]] = 1
        W[picks[1:]] = -.25
        spatial_patterns[i, i_ses] = cov_raw @ W

mean_patterns = np.mean(spatial_patterns, axis=1)
lap_names = list(laplacians.keys())
df = pd.DataFrame(mean_patterns, columns=raw.ch_names, index=lap_names)
df.to_csv('../results/mean_laplacian_patterns.csv', index=False)
