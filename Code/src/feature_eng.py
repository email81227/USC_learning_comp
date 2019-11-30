from Code.src.utils import *
from multiprocessing import Pool

import librosa
import numpy as np
import time

N_JOBS = 15


# for mfcc
def apply_mfcc_to_sources(objs):
    # Initial pool
    pool = Pool()

    # Map mfcc to every object
    start = time.time()
    feats = pool.map(get_mfcc, objs)
    print('MFCC is done! in {:6.4f} sec'.format(time.time() - start))

    pool.close()
    pool.join()

    feats = np.array(feats)

    for i, (obj, feat) in enumerate(zip(objs, feats)):
        feats[i] = Features(obj.id, feat, obj.label)

    return feats


def features_engineering(objs):

    return


def get_mfcc(src):
    sample, rate = src.load_sample()

    mfcc = librosa.feature.mfcc(sample, rate, n_mfcc=20)
    dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
    ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
    return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                           np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                           np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1))
                          , axis=0)