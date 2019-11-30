from multiprocessing import Pool

import numpy as np


SAMPLE_RATE = 22500
LENGTH = 7
MAX_LEN = SAMPLE_RATE * LENGTH


def padding(obj, max_len=MAX_LEN):
    '''

    :param obj: (Array) 1 dimension
    :param max_len: (Int)
    :return: Array 1 dimension
    '''
    if obj.shape[0] < max_len:
        src = np.zeros(max_len)
        src[:obj.shape[0]] = obj
    else:
        src = obj[:max_len]

    return src


def preprocess(objs):
    # Initial pool
    pool = Pool()

    #
    objs = pool.map(padding, objs)

    pool.close()
    pool.join()
    return objs


if __name__ == '__main__':
   from Code.src.utils import *

