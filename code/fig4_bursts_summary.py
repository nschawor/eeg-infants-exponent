""" This script reproduces content of Fig. 4 & Table 1 in the manuscript. """

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import helper

plt.ion()

results_dir = '../results/model_bursts/'
channels = ['C3-lap', 'C4-lap', 'Pz-lap']
labels = ['Laplacian %s' % (channel.split('-')[0]) for channel in channels]
fs = 512

mask_params = dict(marker='.', markerfacecolor='w', markeredgecolor='k',
                   linewidth=0, markersize=4, alpha=0.9)

patterns = pd.read_csv('../results/mean_laplacian_patterns.csv', index_col=0)
nr_patterns = len(patterns)
laplacians = helper.get_laplacian_chanlist()

subject = 'sub-01_ses-02'
raw = mne.io.read_raw_fif('../working/ica/%s_raw.fif' % subject)
raw.crop(0, 1)
raw.load_data()
raw.reorder_channels(list(patterns.columns))
assert(len(raw.ch_names) == 32)

colors = ('#3182BD', '#6BAED6', '#AD494A')
feature = 'frequency'

fig, ax = plt.subplots(1, 3, sharey=True)
cc = 0

channels = ['C3-lap', 'Pz-lap']
labels = ['Laplacian %s' % (channel.split('-')[0]) for channel in channels]

for i_chan, channel in enumerate(channels):
    print(channel)
    df = pd.read_csv('%s/burst_features_%s.csv' % (results_dir, channel))
    kids = np.unique(df.subject)
    nr_subjects = len(kids)

    df_file = '%s/model_bursts_%s_predictions.csv' \
              % (results_dir, channel)
    df_pred = pd.read_csv(df_file)
    age = df_pred['age']

    ax1 = ax[i_chan]

    # individual lines
    cmap = [plt.cm.tab20b(i) for i in np.linspace(0, 1, nr_subjects)]
    ax1.set_prop_cycle('color', cmap)

    for kid in kids:
        kid_age = df[df.subject == kid].age
        df_feat = df[feature][df.subject == kid]
        ax1.plot(kid_age, df_feat, '.-', markersize=5, alpha=1, lw=0.9)

    # plot laplacian pattern as inset
    pattern = patterns.loc[channel]
    lap_chans = laplacians[channel]
    print(lap_chans)
    picks = mne.pick_channels(raw.ch_names, lap_chans)
    mask = np.zeros((len(raw.ch_names),), dtype='bool')
    mask[picks] = True

    topo_size = '40%'
    ax_ins = inset_axes(ax1, width=topo_size, height=topo_size,
                        bbox_to_anchor=(.05, -0.55, 1, 1),
                        bbox_transform=ax1.transAxes)

    mne.viz.plot_topomap(pattern, raw.info, axes=ax_ins, mask=mask,
                         mask_params=mask_params, cmap='PiYG')

    # plot model fit
    color = 'tab:green'
    line = ax1.plot(age, df_pred['pred'], '-', color=color,
                    lw=1.5, zorder=-30)
    ax1.fill_between(age, df_pred['LCB0.025'], df_pred['UCB0.975'],
                     alpha=0.4, color=color, zorder=-35)

    # some figure cosmetics
    for i in range(1, 7):
        ax1.axvline(i*30, color='gray', alpha=0.4)
    ax1.set_xticks(range(30, 211, 30))
    ax1.set_xlim(30, 210)

    ax1.set_title('channel: Laplacian %s' % (channel.split('-')[0]))
    if i_chan == 0:
        ax1.set_ylabel('oscillation frequency [Hz]')
    ax1.set_xlabel('participant age [days]')

# plot population prediction
ax2 = ax[2]
channels = ['C3-lap', 'C4-lap', 'Pz-lap']
labels = ['Laplacian %s' % (channel.split('-')[0]) for channel in channels]

for i_chan, channel in enumerate(channels):
    df = pd.read_csv('%s/burst_features_%s.csv' % (results_dir, channel))
    kids = np.unique(df.subject)
    nr_subjects = len(kids)

    df_file = '%s/model_bursts_%s_predictions.csv' \
              % (results_dir, channel)
    df_pred = pd.read_csv(df_file)
    age = df_pred['age']
    line = ax2.plot(age, df_pred['pred'], '-', color=colors[i_chan], lw=1.5)
    ax2.fill_between(age, df_pred['LCB0.025'], df_pred['UCB0.975'],
                     alpha=0.4, facecolor=colors[i_chan])

ax2.set_xlabel('participant age [days]')
ax2.set_xlim(30, 210)
ax2.legend(labels, loc='lower right')
ax2.set_title('model fit')

ax2.set_xticks(range(30, 211, 30))
for i in range(1, 7):
    ax2.axvline(i*30, color='gray', alpha=0.4)

fig.set_size_inches(9, 3)
fig.tight_layout()
fig.savefig('../results/fig4_burst_summary.png', dpi=300)
fig.show()


# overview table for manuscript
dfs = []
for i_chan, channel in enumerate(channels):
    df = pd.read_csv('%s/model_bursts_%s.csv' % (results_dir, channel))
    df = df[['feature', 't', 'p']]
    df.columns = ['feature', 't_%s' % channel, 'p_%s' % channel]
    df = df.sort_values('feature')
    dfs.append(df)

df_all = pd.merge(dfs[0], dfs[1], on='feature')
df_all = pd.merge(df_all, dfs[2], on='feature')

df_all = df_all.set_index('feature')

columns = ['p_C3-lap', 'p_C4-lap', 'p_Pz-lap']
all_p_vals = df_all[columns]
h, new_p_val = mne.stats.bonferroni_correction(all_p_vals, alpha=0.05)

for i_col, c in enumerate(columns):
    df_all[c] = new_p_val[:, i_col]

print(df_all)
df_all.to_latex()
