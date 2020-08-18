""" This script unifies sampling frequency and channel names across datasets
    and saves all provided bdf-files into mne fif-file format.

"""

import os
import mne
import numpy as np
import pandas as pd

data_dir = '../data/'
raw_dir = '../working/raw/'
os.makedirs(raw_dir, exist_ok=True)

files = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.bdf')]
files = np.sort(files)

sessions = pd.read_csv('../csv/subject_list.csv')
sessions = sessions.set_index('old_id')

df = pd.read_csv('../csv/channel_names.csv')
df.set_index('index')

for name in sessions.index:
    print(name)
    raw_bdf_file = '%s/%s.bdf' % (data_dir, name)

    subject_id = sessions.loc[name].subject_id
    raw_fif_file = '%s/%s_raw.fif' % (raw_dir, subject_id)

    raw = mne.io.read_raw_bdf(raw_bdf_file)
    raw.load_data()

    # take a subset of channels in the raw file, the others are empty
    raw.pick_channels(raw.ch_names[:32])
    if raw.ch_names[0] == 'A1':
        channels = df.channel
    elif raw.ch_names[0] == 'Fp1':
        channels = raw.ch_names
    else:
        channels = [ch.split('-')[1] for ch in raw.ch_names]

    rename = dict(zip(raw.ch_names, channels))
    raw.rename_channels(rename)

    # create channel positions
    montage = mne.channels.make_standard_montage('biosemi32')
    raw.set_montage(montage)

    # one file has a higher sampling frequency
    if raw.info['sfreq'] > 512:
        new_sfreq = 512
        raw.resample(new_sfreq)

    # one file is much longer than the others
    if name == 'TD12v1':
        raw.crop(0, 412)

    raw.save(raw_fif_file, overwrite=True)
