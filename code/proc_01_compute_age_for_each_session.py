"""Computes participant ages from provided data.

Because the provided datasheet only provides the participant ages for two
sessions, the ages at the remaining sessions are calculated with aid of the
provided recording time in the EEG-file.

"""

import pandas as pd
import datetime
import mne

df = pd.read_csv('../csv/subject_list.csv')
df = df.set_index('subject_id')

timestamps = []
infants = []
for name in df.index:
    raw_file_name = '../working/raw/%s_raw.fif' % name
    raw = mne.io.read_raw_fif(raw_file_name, verbose=False)
    timestamp = raw.info['meas_date']  # readout measurement date from raw file
    timestamps.append(timestamp)
    infants.append(df.loc[name].old_id.split('v')[0])

df['recording'] = timestamps
df['infant'] = infants


# compute birthdays
session = pd.read_excel('../data/SourceDataToPublishWithRelPowEEG.xlsx')
session = session[session.Group == 'TD']
session = session[session.Visit == 1]

dates = session[['Infant', 'Visit', 'Chronological Age (days)']].dropna()
dates = dates.reset_index()

birthdays = []
for i in dates.index:
    t = df[df.infant == session.iloc[i].Infant]
    t = t.loc[t.session.idxmin()]
    age = int(session['Chronological Age (days)'].iloc[i])
    birthday = t.recording-datetime.timedelta(age)
    birthdays.append(birthday)

dates['birthday'] = birthdays

# compute ages for each infant at each session
ages = []
for name in df.index:
    birthday = dates[dates.Infant == df.loc[name].infant].birthday
    age = df.loc[name].recording-birthday
    age = age.dt.days.values[0]
    ages.append(age)

df['age'] = ages

# exclude dataset which does not contain enough data
rejected_datasets = ['TD25v4']
df = df[~df.old_id.isin(rejected_datasets)]

# save to dataframe
df = df.reset_index()
df.to_csv('../csv/sessions.csv', index=False)
