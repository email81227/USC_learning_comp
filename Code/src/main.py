# For reproducible results
import numpy as np
np.random.seed(123)

from Code.src.models import *
from Code.src.preprocess import N_JOBS
from multiprocessing import Pool
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import cv2
import datetime
import os
import pandas as pd
import pickle
import time

DATA_DIR = r'Data/Modeling'


def concatenate(obj):
    return np.concatenate((np.array(obj.ft['AE']), obj.ft['MFCC']), axis=0)


def early_stop():
    return EarlyStopping(monitor='val_loss', patience=5, verbose=1)


def feature_concatenate(objs):
    # Initial pool
    pool = Pool(N_JOBS)

    # Map mfcc to every object
    start = time.time()

    tmp = pool.map(concatenate, objs)
    print('Concatenate is done! in {:6.4f} sec'.format(time.time() - start))

    pool.close()
    pool.join()
    #
    return np.array(tmp)


def img_loader(obj):
    return cv2.imread(obj.ft['Mel_Spectogram'])


def learning_rate_schedule():
    return LearningRateScheduler(lambda x: 1. / (1. + x))


def load_imgs(objs):
    # Initial pool
    pool = Pool(N_JOBS)

    # Map mfcc to every object
    start = time.time()

    tmp = pool.map(img_loader, objs)
    print('Imgs loaded! in {:6.4f} sec'.format(time.time() - start))

    pool.close()
    pool.join()
    #
    return np.array(tmp)


def model_keeper():
    return ModelCheckpoint('Model/tmp/_tmp_best.hdf5', save_best_only=True, monitor='val_loss', mode='min')


def train_dnn():
    # Get data
    samples = pickle.load(open(os.path.join(DATA_DIR, 'train.pkl'), 'rb'))

    # One Hot encode label for deep-learning model
    tr_y = to_categorical([sample.label for sample in samples])

    # Concatenate the numerical arrays
    tr_X = feature_concatenate(samples)

    # Initial model
    dnn = dense_model(shape=(tr_X.shape[1], ))

    # Training
    dnn.fit(tr_X, tr_y, **training_config())

    # Get best model
    dnn = load_model(r'Model/tmp/_tmp_best.hdf5')
    dnn.save('Model/formal/dnn.h5')


def train_cust_resnet():
    # Get data
    samples = pickle.load(open(os.path.join(DATA_DIR, 'train.pkl'), 'rb'))

    # One Hot encode label for deep-learning model
    tr_y = to_categorical([sample.label for sample in samples])

    # Get image inputs
    tr_X = load_imgs(samples)

    # Initial model
    resnet = customized_resnet(shape=tr_X.shape[1:])

    # Training
    resnet.fit(tr_X, tr_y, **training_config())

    # Get best model
    resnet = load_model(r'Model/tmp/_tmp_best.hdf5')
    resnet.save('Model/formal/cust_resnet.h5')


def training_config():
    cfg = {'epochs': 30,
           'batch_size': 256,
           'validation_split': .10,
           'validation_data': None,
           'class_weight': None,
           'sample_weight': None,
           'callbacks': [early_stop(),
                         # learning_rate_schedule(),
                         model_keeper()
                         ],
           }

    return cfg


def predict_cust_resnet():
    #
    stamp = datetime.datetime.now()
    stamp = stamp.strftime(format='%Y%m%d_%H%M%S')

    # Get test data
    samples = pickle.load(open(os.path.join(DATA_DIR, 'test.pkl'), 'rb'))

    # Concatenate the numerical arrays
    ts_X = load_imgs(samples)

    # Load model
    resnet = load_model(r'Model/formal/cust_resnet.h5')

    # Predict
    pred_y = resnet.predict(ts_X.astype(np.float16))
    pred_y = np.argmax(pred_y, axis=1)

    # Get label encoder
    lbl = pickle.load(open(os.path.join(r'Encoders/label', 'label_encoder.pkl'), 'rb'))

    # Create submission csv
    subm = pd.DataFrame({'Class': lbl.inverse_transform(pred_y),
                         'ID': [sample.id for sample in samples]})

    subm.to_csv(r'Output/submission_{}.csv'.format(stamp), index=False)


def predict_dnn():
    #
    stamp = datetime.datetime.now()
    stamp = stamp.strftime(format='%Y%m%d_%H%M%S')

    # Get test data
    samples = pickle.load(open(os.path.join(DATA_DIR, 'test.pkl'), 'rb'))

    # Concatenate the numerical arrays
    ts_X = feature_concatenate(samples)

    # Load model
    dnn = load_model(r'Model/formal/dnn.h5')

    # Predict
    pred_y = dnn.predict_classes(ts_X)

    # Get label encoder
    lbl = pickle.load(open(os.path.join(r'Encoders/label', 'label_encoder.pkl'), 'rb'))

    # Create submission csv
    subm = pd.DataFrame({'Class': lbl.inverse_transform(pred_y),
                         'ID': [sample.id for sample in samples]})

    subm.to_csv(r'Output/submission_{}.csv'.format(stamp), index=False)


if __name__ == '__main__':
    # train_dnn()
    # predict_dnn()

    train_cust_resnet()
    predict_cust_resnet()