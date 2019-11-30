import librosa
import os
import pandas as pd

TR_DIR = r'Data/raw/train'
TS_DIR = r'Data/raw/test'


class RawSample:
    def __init__(self, id, label, data_path):
        self.id = id
        self.label = label
        self.sample = None
        self.sample_path = data_path
        self.sample_rate = None

    def load_sample(self, **kwargs):
        self.sample, self.sample_rate = librosa.load(self.sample_path, **kwargs)


class Features:
    def __init__(self, id, features, label):
        self.id = id
        self.ft = features
        self.label = label


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


if __name__ == '__main__':
    tr = get_training_data()
    ts = get_test_data()