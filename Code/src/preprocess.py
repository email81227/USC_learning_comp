from Code.src.utils import *
import glob
import numpy as np
import os
import pandas as pd


TR_DIR = r'Data/raw/train'
TS_DIR = r'Data/raw/test'


def get_training_data():
    tr = pd.read_csv(r'Data/raw/train/train.csv')

    samples = []
    for i, row in tr.iterrows():
        samples.append(RawSample(row.ID, row.Class, os.path.join(TR_DIR, 'Train', str(row.ID)+'.wav')))

    return samples


def get_test_data():
    tr = pd.read_csv(r'Data/raw/test/test.csv')

    samples = []
    for i, row in tr.iterrows():
        samples.append(RawSample(row.ID, None, os.path.join(TR_DIR, 'Test', str(row.ID) + '.wav')))

    return samples


def padding(seq, max_len=None):
    if max_len < seq.shape[0]:
        raise ValueError('The length of seq {} is shorter than max_len = {}'.format(seq.shape[0], max_len))

    return np.concatenate((seq, np.zeros(max_len - len(seq))))


def preprocess():
    return


if __name__ == '__main__':
    tr = get_training_data()
    ts = get_test_data()