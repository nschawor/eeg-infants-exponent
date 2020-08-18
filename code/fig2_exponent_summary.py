""" This script reproduces the content of Fig. 2 in the manuscript."""

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import helper

plt.ion()

# merge exponents and subject characteristics
df = pd.read_csv('../csv/sessions.csv')
df = df[['age', 'subject_id', 'subject']]

fmin = 1
fmax = 10
exp_folder = '../results/exponents/'
model_folder = '../results/model_exponents/'
df_exp = pd.read_csv('%s/exponents_segments.csv' % exp_folder)
df = df.merge(df_exp)

# load montage for plotting
session = 'sub-02_ses-01'
raw = mne.io.read_raw_fif('../working/raw/%s_raw.fif' % session)
raw.crop(0, 1)
raw.load_data()

df_exp1 = df_exp.drop(['subject_id'], axis=1)
raw.reorder_channels(list(df_exp1.columns))

fig = plt.figure()
gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 1.5, 3])

# plot grand average exponent
width = 0.45
yoffset = 0.33
ax1 = ax1 = plt.subplot(gs[0, 0])

mean_exp = df_exp.mean()
im = mne.viz.plot_topomap(mean_exp, raw.info, axes=ax1,
                          vmin=mean_exp.min()-.05, vmax=mean_exp.max())
ax1.set_xlim(-0.095, 0.095)

cb = plt.colorbar(im[0], ax=ax1, orientation='horizontal', pad=0.05,
                  label='mean aperiodic exponent\nacross participants',
                  fraction=0.05)
cb.set_ticks([2.15, 2.25, 2.35])

df_file = '%s/exponent_model_results.csv' % model_folder
df_t = pd.read_csv(df_file)
nr_channels = 32
alpha_level = 0.01
h, new_pval = mne.stats.bonferroni_correction(df_t.p_estimate, alpha_level)


# plot exponent over time
ax1 = plt.subplot(gs[0, 2])
chan = 'PO3'

nr_subjects = 22
cmap = [plt.cm.tab20b(i) for i in np.linspace(0, 1, nr_subjects)]
ax1.set_prop_cycle('color', cmap)

kids = np.unique(df.subject)
for kid in kids:
    df_sel = df[df.subject == kid]
    df_exp1 = df_exp[chan][df.subject == kid]
    m = df_exp1.to_numpy()
    ax1.plot(df_sel.age, df_exp1, '.-', lw=1, markersize=5, alpha=1)

df_file = '%s/exponent_model_predictions.csv' % model_folder
df_pred = pd.read_csv(df_file)

ages = df_pred['age']
color = 'tab:green'
line = ax1.plot(ages, df_pred['pred'], '-', color=color, lw=1.5, zorder=-30)

ax1.fill_between(ages, df_pred['LCB0.025'], df_pred['UCB0.975'],
                 alpha=0.4, facecolor=color, zorder=-35)

for i in range(1, 7):
    ax1.axvline(i*30, color='gray', alpha=0.5)

ax1.set_xticks(range(30, 211, 30))
ax1.set_xlim(30, 210)
ax1.set_aspect(100)
ax1.set_xlabel('participant age [days]')
ax1.set_ylabel('aperiodic exponent for channel: %s' % chan)


# plot t-values over channels
mask = np.zeros((nr_channels,), dtype='bool')
mask[new_pval < alpha_level] = True
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                   linewidth=0, markersize=4)

ax1 = plt.subplot(gs[0, 1])

im = mne.viz.plot_topomap(df_t.t, raw.info, mask=mask, axes=ax1,
                          mask_params=mask_params)

ax1.set_xlim(-0.095, 0.095)

plt.colorbar(im[0], ax=ax1, orientation='horizontal', pad=0.05,
             label='t-value \nfor fixed effect age\non aperiodic exponent',
             fraction=0.05)

fig.set_size_inches(7.86, 3.6)
fig.tight_layout()
fig.savefig('../results/fig2_exponent_summary.png', dpi=300)
fig.show()
