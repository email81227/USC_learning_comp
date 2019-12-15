from multiprocessing import Pool

import librosa
import numpy as np
import time


SAMPLE_RATE = 22500
LENGTH = 4
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
    objs = pool.map(load_audio, objs)
    print('MFCC is done! in {:6.4f} sec'.format(time.time() - start))

    objs = pool.map(padding, objs)
    print('MFCC is done! in {:6.4f} sec'.format(time.time() - start))

    pool.close()
    pool.join()

    return objs


def load_audio(obj):
    obj.sample, obj.sample_rate = librosa.load(obj.sample_path)
    return obj


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
