""" Computes independent component analysis.

Computes independent component analysis in order to project out strong
muscle and noise components.
"""

import os
import ast
import pandas as pd
import mne
from mne.preprocessing import ICA
import helper

# define folders
raw_dir = '../working/raw/'
save_dir = '../working/ica/'
os.makedirs(save_dir, exist_ok=True)
mne.set_log_level(verbose=False)

# do it for all subjects
df1 = pd.read_csv('../csv/sessions.csv')
dfs = []
for session in df1.subject_id:

    print('processing subject: %s' % session)

    # load raw data
    raw = mne.io.read_raw_fif('%s/%s_raw.fif' % (raw_dir, session))
    raw.load_data()
    raw = helper.remove_bad_chans(session, raw, interpolate=True)
    raw = helper.set_annotations(session, raw)

    raw_ica = raw.copy().filter(2, 40)

    # ICA settings
    method = 'fastica'
    n_components = .95
    if session == 'sub-25_ses-03':
        n_components = .99
    decim = 1
    random_state = 23

    # compute ICA
    df_file = '../csv/ica_components.csv'
    df = pd.read_csv(df_file, index_col=False)
    df = df[df.subject_id == session]
    components = df['components'].apply(ast.literal_eval).values[0]

    ica = ICA(n_components=n_components, method=method, max_iter=100,
              random_state=random_state)
    ica.fit(raw_ica, decim=decim, reject_by_annotation=True)
    ica.exclude.extend(components)
    ica.apply(raw)

    file_name = '%s/%s_raw.fif' % (save_dir, session)
    raw.save(file_name, overwrite=True)
