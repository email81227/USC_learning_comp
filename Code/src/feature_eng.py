from multiprocessing import Pool

import librosa
import numpy as np
import time

N_JOBS = 15


# for mfcc
def apply_mfcc_to_sources(seq_src):
    start = time.time()

    pool = Pool()
    feats = pool.map(get_mfcc, seq_src)
    feats = np.array(feats)

    np.save(r'PATH_TO_SAVE', feats)

    print('MFCC is done! in {:6.4f} sec'.format(time.time() - start))

    pool.close()
    pool.join()


def get_mfcc(src, sr):
    mfcc = librosa.feature.mfcc(src, sr, n_mfcc=20)
    dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
    ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
    return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                           np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                           np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1))
                          , axis=0)