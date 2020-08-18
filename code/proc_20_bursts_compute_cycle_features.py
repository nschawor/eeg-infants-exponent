"""Computes waveform features of Laplacian channels."""

import os
import mne
import pandas as pd
from bycycle.features import compute_features
import helper

sessions = pd.read_csv('../csv/sessions.csv')
laplacians = helper.get_laplacian_chanlist()
fmin = 1
fmax = 45

# burst parameters
bandwidth = (3, 7)
osc_param = {'amplitude_fraction_threshold': 0.5,
             'amplitude_consistency_threshold': 0.5,
             'period_consistency_threshold': 0.5,
             'monotonicity_threshold': 0.5,
             'N_cycles_min': 3}

for i_id, subject in enumerate(sessions.subject_id):
    print(subject)
    results_dir = '../results/bursts/%s/' % (subject)
    os.makedirs(results_dir, exist_ok=True)
    ica_file_name = '../working/ica/%s_raw.fif' % subject

    raw = mne.io.read_raw_fif(ica_file_name)
    raw.load_data()
    raw.set_eeg_reference('average')
    fs = int(raw.info['sfreq'])

    # make Laplacian derivation
    raw = helper.create_laplacian(laplacians, raw)

    # apply broadband filter
    raw.filter(fmin, fmax)
    is_good = helper.return_good_segments(subject, raw)

    for j, channel in enumerate(raw.ch_names):
        print(subject, channel)
        df_file_name = '%s/bursts_%s.csv' % (results_dir, channel)
        data = raw.get_data()[j]

        Fs = raw.info['sfreq']
        df = compute_features(data, Fs, f_range=bandwidth,
                              burst_detection_kwargs=osc_param)

        # check if sample_peak in artefactual segment and exclude
        for i_burst in range(len(df)):
            if df.iloc[i_burst].is_burst:
                sample = df.iloc[i_burst].sample_peak
                if not(is_good[sample]):
                    df.at[i_burst, 'is_burst'] = False

        # transform period in burst frequency
        df = df.rename(columns={'period': 'frequency'})
        df['frequency'] = df.frequency.apply(lambda x: fs/x)

        # save dataframe to file
        df.to_csv(df_file_name, index=False)
