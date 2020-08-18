"""Computes mean waveform features for each session."""

import os
import pandas as pd
import numpy as np

sessions = pd.read_csv('../csv/sessions.csv')
sessions = sessions[['age', 'subject_id', 'subject']]
channels = ['Pz-lap', 'C3-lap', 'C4-lap']
feature_list = ['frequency', 'time_ptsym', 'time_rdsym', 'volt_amp']

results_folder = '../results/model_bursts/'
os.makedirs(results_folder, exist_ok=True)

nr_bursts = np.zeros((len(sessions), len(channels)))

for i_chan, channel in enumerate(channels):
    print('processing channel: %s' % channel)
    dfs = []
    for i_ses, session in enumerate(sessions.subject_id):
        results_dir = '../results/bursts/%s/' % session

        df_file_name = '%s/bursts_%s.csv' % (results_dir, channel)
        df = pd.read_csv(df_file_name)
        df = df[df.is_burst]
        df = df[feature_list]
        nr_bursts[i_ses, i_chan] = len(df)

        df1 = df.mean()
        dfs.append(df1)

    df_sel = pd.concat(dfs, axis=1).T
    df_sel.insert(0, 'age', sessions.age)
    df_sel.insert(0, 'subject', sessions.subject)
    df_file = '%s/burst_features_%s.csv' % (results_folder, channel)
    df_sel.to_csv(df_file, index=False)

print('number of analyzed bursts')
print('mean: %.2f, std: %.2f' % (np.mean(nr_bursts), np.std(nr_bursts)))
