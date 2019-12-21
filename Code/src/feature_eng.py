from Code.src.auto_encoders import AutoEncoder
from Code.src.preprocess import SAMPLE_RATE, LENGTH
from Code.src.utils import Features
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
import pickle
import librosa
import numpy as np
import tensorflow as tf
import time


LATENT_DIM = 32
MAX_LEN = SAMPLE_RATE * LENGTH
N_JOBS = 15


# for mfcc
def features_engineering(objs, training=True):
    # Get id
    sample_id = [obj.id for obj in objs]

    # MFCC feature generated
    mfcc_feats = mfcc_extraction(objs)

    # AutoEncoder
    ae_feats = feature_extraction_by_autoencoder(objs)

    # Label encoding
    if training:
        y = label_encoding(objs, training)
        y = np.array(y)

    # Merge the features
    features = []
    for mfcc, ae, obj, lbl in zip(mfcc_feats, ae_feats, objs, y):
        features.append(Features(obj.id,
                                 {'mfcc': mfcc,
                                  'autoencode': ae},
                                 lbl if training else None))
    return features


def feature_extraction_by_autoencoder(objs):
    tr_X, ts_X = train_test_split(objs, test_size=.2, random_state=42, stratify=[obj.label for obj in objs])

    tr_X = [obj.sample for obj in tr_X]
    ts_X = [obj.sample for obj in ts_X]

    tr_X = tf.convert_to_tensor(tr_X)
    ts_X = tf.convert_to_tensor(ts_X)

    ae = AutoEncoder(LATENT_DIM, MAX_LEN)
    ae.compile(optimizer=tf.keras.optimizers.Adadelta(1.0, .95),
               loss=tf.keras.losses.BinaryCrossentropy())
    ae.fit(tr_X, tr_X,
           epochs=10,
           batch_size=32,
           shuffle=True,
           validation_data=(ts_X, ts_X))

    ae.save(r'Model/formal/dnn_autoencoder.h5')

    del tr_X, ts_X

    tmp = [obj.sample for obj in objs]
    tmp = tf.convert_to_tensor(tmp)
    tmp = ae.encode(tmp)

    return tmp


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
