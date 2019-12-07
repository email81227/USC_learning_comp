from Code.src.utils import Features
from multiprocessing import Pool

import librosa
import numpy as np
import time


SAMPLE_RATE = 200
LENGTH = 7
MAX_LEN = SAMPLE_RATE * LENGTH
N_JOBS = 15


def padding(obj, max_len=MAX_LEN):
    '''

    :param obj: (Array) 1 dimension
    :param max_len: (Int)
    :return: Array 1 dimension
    '''
    if obj.sample.shape[0] < max_len:
        src = np.zeros(max_len)
        src[:obj.sample.shape[0]] = obj.sample
    else:
        src = obj.sample[:max_len]

    obj.sample = src
    return obj


def preprocess(objs):
    # Initial pool
    pool = Pool(N_JOBS)

    # Map mfcc to every object
    start = time.time()
    feats = pool.map(get_mfcc, objs)
    print('MFCC is done! in {:6.4f} sec'.format(time.time() - start))

    pool.close()
    pool.join()

    feats = np.array(feats)

    return [Features(obj.id, feat, obj.label) for obj, feat in zip(objs, feats)]


def get_mfcc(src):
    # sample, rate = src.load_sample()
    sample, rate = librosa.load(src.sample_path)

    mfcc = librosa.feature.mfcc(sample, rate, n_mfcc=20)
    dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
    ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
    return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                           np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                           np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1))
                          , axis=0)


if __name__ == '__main__':
    '''
    Sorting out the raw data into preprocessed
    
    (raw (wav) data == [preprocess.py] ==> preprocessed)
    '''
    from Code.src.utils import get_test_data, get_training_data

    import os
    import pickle

    SAVE_DIR = r'Data/preprocessed'

    tr = get_training_data()
    ts = get_test_data()

    tr = preprocess(tr)
    ts = preprocess(ts)

    pickle.dump(tr, open(os.path.join(SAVE_DIR, 'train.pkl'), 'wb'))
    pickle.dump(ts, open(os.path.join(SAVE_DIR, 'test.pkl'), 'wb'))
