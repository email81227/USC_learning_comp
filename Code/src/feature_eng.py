from Code.src.utils import Features
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

import os
import pickle
import librosa
import numpy as np
import time


SAMPLE_RATE = 200
LENGTH = 7
MAX_LEN = SAMPLE_RATE * LENGTH
N_JOBS = 15


def get_mfcc(obj):
    mfcc = librosa.feature.mfcc(obj.sample, obj.sample_rate, n_mfcc=20)
    dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
    ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
    return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                           np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                           np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1)), axis=0)


def mfcc_extraction(objs):
    # Initial pool
    pool = Pool(N_JOBS)

    # Map mfcc to every object
    start = time.time()
    feats = pool.map(get_mfcc, objs)
    print('MFCC is done! in {:6.4f} sec'.format(time.time() - start))

    pool.close()
    pool.join()
    #
    # feats = np.array(feats)

    # return [Features(obj.id, feat, obj.label) for obj, feat in zip(objs, feats)]
    return feats


# for mfcc
def features_engineering(objs, training=True):
    # Get id
    sample_id = [obj.id for obj in objs]

    # Feature generated
    # X = [obj.ft for obj in objs]
    # X = np.array(X)
    mfcc = mfcc_extraction(objs)

    #

    # Merge the features


    # Label encoding
    if training:
        y = label_encoding(objs, training)
        y = np.array(y)

        return sample_id, X, y
    else:
        return sample_id, X, None


def label_encoding(objs, training=True):
    labels = [obj.label for obj in objs]

    if training:
        lbl = LabelEncoder()
        lbl.fit(labels)

        pickle.dump(lbl, open(os.path.join(r'Encoders/label', 'label_encoder.pkl'), 'wb'))
    else:
        lbl = pickle.load(open(os.path.join(r'Encoders/label', 'label_encoder.pkl'), 'rb'))

    return lbl.transform(labels)


if __name__ == '__main__':
    '''
    Sorting out the preprocessed data into modeling folder

    (preprocessed data == [feature_eng.py] ==> modeling)
    
    '''
    SAVE_DIR = r'Data/Modeling'

    tr = pickle.load(open(r'Data/preprocessed/train.pkl', 'rb'))
    ts = pickle.load(open(r'Data/preprocessed/test.pkl', 'rb'))

    tr = features_engineering(tr, True)
    ts = features_engineering(ts, False)

    pickle.dump(tr, open(os.path.join(SAVE_DIR, 'train.pkl'), 'wb'))
    pickle.dump(ts, open(os.path.join(SAVE_DIR, 'test.pkl'), 'wb'))
