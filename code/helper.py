import scipy.stats
import numpy as np
import os
import pandas as pd
import ast
import mne


def set_annotations(id, raw):
    annotation_file = '../csv/bad_segments/%s_annotations.csv' % id
    if os.path.exists(annotation_file):
        annot = mne.read_annotations(annotation_file)
        raw.set_annotations(annot)
    return raw


def remove_bad_chans(id, raw, interpolate=False):

    df_chan = pd.read_csv('../csv/bad_channels.csv')
    # remove bad channels
    df1 = df_chan[df_chan.subject_id == id]
    bad_chans = df1['bad_channels'].values[0].strip('[]').split(',')
    if len(bad_chans[0]) == 0:
        bad_chans = []

    # print(bad_chans)
    raw.info['bads'] = bad_chans

    if interpolate:
        raw.interpolate_bads()
    else:
        raw.drop_channels(raw.info['bads'])
    return raw


def return_good_segments(subject, raw):

    annotation_file = '../csv/bad_segments/%s_annotations.csv' % subject
    annotations = mne.read_annotations(annotation_file)
    raw.set_annotations(annotations)

    # index good segments
    is_good = np.ones_like(raw.times)
    for annotation in annotations:
        idx = np.argmin(np.abs(raw.times-annotation['onset']))
        dur = int(annotation['duration']*raw.info['sfreq'])
        is_good[idx:idx+dur] = False

    return is_good


def remove_bad_segments(subject, raw):

    annotation_file = '../csv/bad_segments/%s_annotations.csv' % subject
    if os.path.exists(annotation_file):
        annot = mne.read_annotations(annotation_file)
        raw.set_annotations(annot)

    data, times = raw.get_data(reject_by_annotation='omit', return_times=True)
    info = mne.create_info(raw.ch_names, raw.info['sfreq'], ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    montage = mne.channels.make_standard_montage('biosemi32')
    raw.set_montage(montage, raise_if_subset=False)

    return raw


def get_laplacian_chanlist():

    laplacians = {'C3-lap': ['C3', 'FC5', 'FC1', 'CP5', 'CP1'],
                  'C4-lap': ['C4', 'FC2', 'FC6', 'CP6', 'CP2'],
                  'Pz-lap': ['Pz', 'CP1', 'CP2', 'PO3', 'PO4']}

    return laplacians


def create_laplacian(laplacians, raw):

    raws = []
    for deriv in laplacians:
        sensors = laplacians[deriv]
        raw2 = raw.copy().pick_channels(sensors)
        raw2.reorder_channels(sensors)
        W = np.array([[1, -.25, -.25, -.25, -.25]])
        raw2._data[0] = W @ raw2._data
        raw2.drop_channels(raw2.ch_names[1:])
        raw2.rename_channels({sensors[0]: deriv})
        raws.append(raw2)

    raw_lap = raws[0].add_channels(raws[1:])

    return raw_lap
