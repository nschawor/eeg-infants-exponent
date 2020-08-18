"""Computes 1/f-exponents for input segments of small length.

The EEG data is cut into segments of 10 s length, the power spectrum is
computed, power spectra are parametrized and exponent is saved in csv-files.
"""

import os
import mne
import numpy as np
import pandas as pd
import fooof

# parameters
df = pd.read_csv('../csv/sessions.csv')

duration_epochs = 10
rsquare_threshold = 0.95
bandwidth = 1
fmin = 1
fmax = 10

exp_folder = '../results/exponents/'
raw_folder = '../working/ica/'
os.makedirs(exp_folder, exist_ok=True)

# compute for all subjects
for subject in df.subject_id:

    df_file_name = '%s/%s_exponents.csv' % (exp_folder, subject)
    ica_file_name = '%s/%s_raw.fif' % (raw_folder, subject)

    print(subject)
    raw = mne.io.read_raw_fif(ica_file_name)
    raw.load_data()
    raw.set_eeg_reference('average')
    raw._data = raw.get_data(reject_by_annotation='NaN')

    # create segments
    events = mne.make_fixed_length_events(raw, start=0, stop=raw.times[-1],
                                          duration=duration_epochs)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration_epochs,
                        baseline=(None, None), preload=True)

    # check segments for NaNs, should be automatically rejected by mne.Epochs
    assert(np.sum(np.isnan(epochs.get_data())) == 0)

    nr_segments = len(epochs)
    nr_channels = len(raw.ch_names)

    # compute PSD
    psd, freq = mne.time_frequency.psd_multitaper(epochs, fmin=fmin, fmax=fmax,
                                                  bandwidth=bandwidth)

    # compute 1/f-fit
    exponent = np.zeros((nr_segments, nr_channels)) + np.nan
    rsquare = np.zeros((nr_segments, nr_channels)) + np.nan

    for i_chan in range(nr_channels):

        fm = fooof.FOOOFGroup(max_n_peaks=5)
        fm.fit(freq, psd[:, i_chan])

        # extract exponent
        results = fm.get_results()
        exponent[:, i_chan] = fm.get_params('aperiodic_params', col='exponent')
        rsquare[:, i_chan] = fm.get_params('r_squared')

    # save as data_frame
    df_exp = pd.DataFrame(exponent, columns=epochs.ch_names)
    df_exp.to_csv(df_file_name, index=False)

    df_r = pd.DataFrame(rsquare, columns=epochs.ch_names)
    df_r_file_name = '%s/%s_rsquare.csv' % (exp_folder, subject)
    df_r.to_csv(df_r_file_name, index=False)
