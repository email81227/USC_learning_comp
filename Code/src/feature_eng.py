from Code.src.tf_autoencoders import AutoEncoder
from Code.src.preprocess import SAMPLE_RATE, LENGTH
from Code.src.utils import Features
from collections import defaultdict
from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
import pickle
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import time


LATENT_DIM = 32
MAX_LEN = SAMPLE_RATE * LENGTH
N_JOBS = 15


# for mfcc
def features_engineering(objs, training=True):
    # Label encoding
    if training:
        ys = label_encoding(objs, training)
        ys = np.array(ys)

        # List of feature objects
        features = []
        for obj, y in zip(objs, ys):
            features.append(Features(obj.id, {}, y))
    else:
        features = []
        for obj in objs:
            features.append(Features(obj.id, {}, None))

    # Feature extracted by multiprocess
    feats = distributed_extraction(objs)

    # AutoEncoder
    ae_feats = feature_extraction_by_autoencoder(objs, training=training)

    # Feature merge
    for ae_feat, obj in zip(ae_feats, features):
        obj.ft['AE'] = ae_feat
        obj.ft.update(feats[obj.id])

    return features


def feature_extraction_by_autoencoder(objs, training):
    WEIGHT_DIR = r'Model/formal/dnn_autoencoder.h5'
    if training:
        tr_X, ts_X = train_test_split(objs, test_size=.2, random_state=42, stratify=[obj.label for obj in objs])

        tr_X = [obj.sample for obj in tr_X]
        ts_X = [obj.sample for obj in ts_X]

        tr_X = tf.convert_to_tensor(tr_X)
        ts_X = tf.convert_to_tensor(ts_X)

        ae = AutoEncoder(LATENT_DIM, MAX_LEN)
        ae.compile(optimizer=tf.keras.optimizers.Adadelta(1.0, .95),
                   loss=tf.keras.losses.BinaryCrossentropy())
        ae.fit(tr_X, tr_X,
               epochs=50,
               batch_size=32,
               shuffle=True,
               validation_data=(ts_X, ts_X))

        ae.save_weights(WEIGHT_DIR)

        del tr_X, ts_X

    else:
        ae = AutoEncoder(LATENT_DIM, MAX_LEN)
        ae.compile(optimizer=tf.keras.optimizers.Adadelta(1.0, .95),
                   loss = tf.keras.losses.BinaryCrossentropy())
        ae.generative_net.build(input_shape=(1, LATENT_DIM))
        ae.load_weights(WEIGHT_DIR)

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


def label_encoding(objs, training=True):
    labels = [obj.label for obj in objs]

    if training:
        lbl = LabelEncoder()
        lbl.fit(labels)

        pickle.dump(lbl, open(os.path.join(r'Encoders/label', 'label_encoder.pkl'), 'wb'))
    else:
        lbl = pickle.load(open(os.path.join(r'Encoders/label', 'label_encoder.pkl'), 'rb'))

    return lbl.transform(labels)


def distributed_extraction(objs):
    # Initial pool
    pool = Pool(N_JOBS)

    # Map mfcc to every object
    start = time.time()
    mfcc_feats = pool.map(get_mfcc, objs)
    print('MFCC is done! in {:6.4f} sec'.format(time.time() - start))

    mel_spec_feats = pool.map(plot_melspectrogram, objs)
    print('MelSpectogram ploting is done! in {:6.4f} sec'.format(time.time() - start))

    pool.close()
    pool.join()
    #
    feats = defaultdict(dict)

    for mfcc, mel_spec, obj in zip(mfcc_feats, mel_spec_feats, objs):
        feats[obj.id].update({'MFCC': mfcc,
                              'Mel_Spectogram': mel_spec})

    return feats


def plot_melspectrogram(obj):
    mel_spec = librosa.feature.melspectrogram(y=obj.sample, sr=obj.sample_rate, n_mels=128, fmax=8000)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    fig_name = str(obj.id) + '.png'
    fig_dir = os.path.join(r'Data/Modeling/melspectograms', fig_name)

    plt.figure(figsize=(4, 2))

    librosa.display.specshow(mel_spec, sr=obj.sample_rate, x_axis=None, y_axis=None)

    plt.tight_layout()
    plt.savefig(fig_dir, dpi=50)
    plt.close()

    return fig_dir


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
