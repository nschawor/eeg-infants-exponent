"""Compile 1/f-exponents across sessions for further computation in R."""

import os
import pandas as pd
import numpy as np

df = pd.read_csv('../csv/sessions.csv')
fmin = 1
fmax = 10
rsquare_threshold = 0.95
exp_folder = '../results/exponents/'

dfs = []
nr_segments = []
for subject in df.subject_id:

    df_file_name = '%s/%s_exponents.csv' % (exp_folder, subject)
    df_exp = pd.read_csv(df_file_name)
    nr_segments.append(len(df_exp))

    df_file_name = '%s/%s_rsquare.csv' % (exp_folder, subject)
    df_r = pd.read_csv(df_file_name)

    # exclude all segments with a model fit worse than r_square threshold
    df_exp = df_exp.mask(df_r < rsquare_threshold)
    df_exp = df_exp.mean()
    df_exp['subject_id'] = subject

    dfs.append(df_exp)

print('number of analyzed segments')
print('mean: %.2f, std: %.2f' % (np.mean(nr_segments), np.std(nr_segments)))


df_all = pd.concat(dfs, axis=1, sort=False).T
df_file = '%s/exponents_segments.csv' % exp_folder
df_all.to_csv(df_file, index=False)

# resave with ages for R
df = pd.read_csv('../csv/sessions.csv')
df = df[['age', 'subject_id', 'subject']]
df = df.merge(df_all)
df = df.drop(labels=['subject_id'], axis=1)

results_folder = '../results/model_exponents/'
os.makedirs(results_folder, exist_ok=True)
df_file = '%s/exponents_for_linear_model.csv' % results_folder
df.to_csv(df_file, index=False)
