
import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb

from librosa.display import specshow
from os.path import join
from tempfile import TemporaryFile

data_path = r'D:\DataSet\UrbanSoundChallenge\train\Train'
test_path = r'D:\DataSet\UrbanSoundChallenge\test\Test'
sub_path = r'D:\DataSet\UrbanSoundChallenge\submission'
doc_path = r'D:\DataSet\UrbanSoundChallenge\train'

num_mfcc = 256
# Step 1 and  2 combined: Load audio files and extract Modeling
def parser(row, num_mfcc=num_mfcc):
    # function to load files and extract Modeling
    if hasattr(row, 'Class'):
        file_name = join(data_path, str(row.ID) + '.wav')
    else:
        file_name = join(test_path, str(row.ID) + '.wav')

    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # we extract mfcc feature from data
        # pdb.set_trace()
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc)  # .T
        # mfccs_mean = np.mean(mfccs, axis=0)
        # mfccs_min = np.min(mfccs, axis=0)
        # mfccs_max = np.max(mfccs, axis=0)
        # mfccs_median = np.median(mfccs, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None, None

    # feature = np.array([mfccs_mean, mfccs_min, mfccs_max, mfccs_median]).T
    feature = mfccs
    if hasattr(row, 'Class'):
        label = row.Class
        return [feature, label, feature.shape[1]]
    else:
        return [feature, feature.shape[1]]


def transformation(row, num_mfcc=num_mfcc):
    # function to load files and extract Modeling
    if hasattr(row, 'Class'):
        file_name = join(data_path, str(row.ID) + '.wav')
    else:
        file_name = join(test_path, str(row.ID) + '.wav')

    # handle exception to check if there isn't a file which is corrupted
    try:

        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # we extract mfcc feature from data
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc)

        plt.figure(figsize=(3, 2))
        librosa.display.specshow(mfccs, x_axis='time')

        if hasattr(row, 'Class'):
            plt.savefig(join(data_path, str(row.ID) + '.png'), dpi=80)
        else:
            plt.savefig(join(test_path, str(row.ID) + '.png'), dpi=80)

        plt.close()

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)


def saver(data_array, path, name, fmt='%.18e'):
    with open(join(path, name), 'wb') as f:
        np.savetxt(f, data_array, fmt)

def saver_hd(data_array, path, name):
    np.save(join(path, name), data_array)

train = pd.read_csv(join(doc_path, 'train.csv'))
test = pd.read_csv(join(r'D:\DataSet\UrbanSoundChallenge\test', 'test.csv'))


# temp = train.apply(transformation, axis=1)
# temp = test.apply(transformation, axis=1)

# pdb.set_trace()
train['length'] = 0
temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label', 'length']
max_len = max(temp.length)


def train_len_adjustment(row, length=max_len):
    if row.length < max_len:
        feature = np.concatenate((row['feature'], np.zeros((num_mfcc, max_len - row['length']))), axis=1)
        return [feature, row.label, row.length]
    else:
        return [row.feature, row.label, row.length]


# pdb.set_trace()
temp = temp.apply(train_len_adjustment, axis=1)

X = np.rollaxis(np.dstack(temp.feature.tolist()), -1)
y = np.array(temp.label.tolist())
try:
    saver_hd(X, data_path, r'mfccs_train_X_all_' + str(num_mfcc) + '.npy')
    saver_hd(y, data_path, r'mfccs_train_y_all_' + str(num_mfcc) + '.npy')
except:
    pdb.set_trace()

# pdb.set_trace()
test['length'] = 0
temp = test.apply(parser, axis=1)
temp.columns = ['feature', 'length']
# max_len = max(temp.length)


def test_len_adjustment(row, length=max_len):
    if row.length < max_len:
        feature = np.concatenate((row.feature, np.zeros((num_mfcc, max_len - row.length))), axis=1)
        return [feature, row.length]
    elif row.length > max_len:
        feature = row.feature
        for i in range(row.length - max_len):
            feature = np.delete(feature, -1, 1)
        return [feature, row.length]
    else:
        return [row.feature, row.length]


# pdb.set_trace()
temp = temp.apply(test_len_adjustment, axis=1)

try:
    X = np.rollaxis(np.dstack(temp.feature.tolist()), -1)
    saver_hd(X, test_path, name=r'mfccs_test_X_all_' + str(num_mfcc) + '.npy')
except:
    pdb.set_trace()