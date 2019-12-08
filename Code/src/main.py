# For reproducible results
import numpy as np
np.random.seed(123)

from Code.src.models import *
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import datetime
import os
import pandas as pd
import pickle

DATA_DIR = r'Data/Modeling'


def training_config():
    cfg = {'epochs': 30,
           'batch_size': 256,
           'validation_split': .2,
           'validation_data': None,
           'class_weight': None,
           'sample_weight': None,
           'callbacks': [early_stop(),
                         # learning_rate_schedule(),
                         model_keeper()
                         ],
           }

    return cfg


def early_stop():
    return EarlyStopping(monitor='val_loss', patience=5, verbose=1)


def learning_rate_schedule():
    return LearningRateScheduler(lambda x: 1. / (1. + x))


def model_keeper():
    return ModelCheckpoint('Model/tmp/_tmp_best.hdf5', save_best_only=True, monitor='val_loss', mode='min')


def train():
    # Get data
    sample_id, tr_X, tr_y = pickle.load(open(os.path.join(DATA_DIR, 'train.pkl'), 'rb'))

    # One Hot encode label for deep-learning model
    tr_y = to_categorical(tr_y)

    # Initial model
    dnn = dense_model()

    # Training
    dnn.fit(tr_X, tr_y, **training_config())

    # Get best model
    dnn = load_model(r'Model/tmp/_tmp_best.hdf5')
    dnn.save('Model/formal/dnn.h5')


def predict_submit():
    #
    stamp = datetime.datetime.now()
    stamp = stamp.strftime(format='%Y%m%d_%H%M%S')

    # Get test data
    sample_id, ts_X, pred_y = pickle.load(open(os.path.join(DATA_DIR, 'test.pkl'), 'rb'))

    # Load model
    dnn = load_model(r'Model/formal/dnn.h5')

    # Predict
    pred_y = dnn.predict_classes(ts_X)

    # Get label encoder
    lbl = pickle.load(open(os.path.join(r'Encoders/label', 'label_encoder.pkl'), 'rb'))

    # Create submission csv
    subm = pd.DataFrame({'Class': lbl.inverse_transform(pred_y),
                         'ID': sample_id})

    subm.to_csv(r'Output/submission_{}.csv'.format(stamp), index=False)


if __name__ == '__main__':
    train()
    predict_submit()