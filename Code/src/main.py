# For reproducible results
from Code.src.config import *
np.random.seed(123)

from Code.src.tf_models import *
from Code.src.preprocess import N_JOBS
from multiprocessing import Pool
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import collections
import cv2
import datetime
import os
import pandas as pd

DATA_DIR = r'Data/Modeling'


def feature_combine(obj):
    return np.concatenate((np.array(obj.ft['AE']), obj.ft['MFCC']), axis=0)


def early_stop():
    return EarlyStopping(monitor='val_loss', patience=20, verbose=1)


def feature_concatenate(objs):
    # Initial pool
    pool = Pool(N_JOBS)

    # Map mfcc to every object
    start = time.time()

    tmp = pool.map(feature_combine, objs)
    print('Concatenate is done! in {:6.4f} sec'.format(time.time() - start))

    pool.close()
    pool.join()
    #
    return np.array(tmp)


def mel_spec_img_loader(obj):
    return cv2.imread(obj.ft['Mel_Spectogram'])


def wave_img_loader(obj):
    return cv2.imread(obj.ft['Wave_Plot'])


def learning_rate_schedule():
    return LearningRateScheduler(lambda x: 1. / (1. + x))


def load_imgs(objs):
    # Initial pool
    pool = Pool(N_JOBS)

    # Map mfcc to every object
    start = time.time()

    tmp1 = pool.map(mel_spec_img_loader, objs)
    print('Imgs loaded! in {:6.4f} sec'.format(time.time() - start))

    tmp2 = pool.map(wave_img_loader, objs)
    print('Imgs loaded! in {:6.4f} sec'.format(time.time() - start))

    pool.close()
    pool.join()
    #
    return np.array(tmp1), np.array(tmp2)


def model_keeper():
    return ModelCheckpoint('Model/tmp/_tmp_best.hdf5', save_best_only=True, monitor='val_loss', mode='min')


def train_dnn():
    # Get data
    samples = pickle.load(open(os.path.join(DATA_DIR, 'train.pkl'), 'rb'))

    # One Hot encode label for deep-learning model
    tr_y = to_categorical([sample.label for sample in samples])

    # Concatenate the numerical arrays
    tr_X = feature_concatenate(samples)

    # Get class weights
    cw = train_class_weight(samples)

    # Initial model
    dnn = dense_model(shape=(tr_X.shape[1], ))

    # Training
    dnn.fit(tr_X, tr_y, **training_config(cw))

    # Get best model
    dnn = load_model(r'Model/tmp/_tmp_best.hdf5')
    dnn.save('Model/formal/dnn.h5')


def train_multiple_inputs_model():
    # Get data
    samples = pickle.load(open(os.path.join(DATA_DIR, 'train.pkl'), 'rb'))

    # One Hot encode label for deep-learning model
    tr_y = to_categorical([sample.label for sample in samples])

    # Prepare inputs
    input_enco = np.array([sample.ft['AE'] for sample in samples])
    input_mfcc = np.array([sample.ft['MFCC'] for sample in samples])
    mel_spec_imgs, wavec_imgs = load_imgs(samples)

    # Get class weights
    cw = train_class_weight(samples)

    # Info for model creating
    inputs = {'AE'  : {'type': 'dense', 'shape': (input_enco.shape[1],)},
              'MFCC': {'type': 'dense', 'shape': (input_mfcc.shape[1],)},
              'Spec': {'type': 'resnet', 'shape': (150, 300, 3)},
              'Wave': {'type': 'resnet', 'shape': (150, 300, 3)}}

    # Init model
    model = ComplexInput(inputs, 10)
    model.model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                        loss='categorical_crossentropy',
                        metrics=['acc'])

    # Training
    model.model.fit([input_enco,
                     input_mfcc,
                     mel_spec_imgs,
                     wavec_imgs], tr_y, **training_config(cw))

    # Get best model
    resnet = load_model(r'Model/tmp/_tmp_best.hdf5')
    resnet.save('Model/formal/multiple_inputs.h5')


def train_cust_resnet():
    # Get data
    samples = pickle.load(open(os.path.join(DATA_DIR, 'train.pkl'), 'rb'))

    # One Hot encode label for deep-learning model
    tr_y = to_categorical([sample.label for sample in samples])

    # Get image inputs
    tr_X = load_imgs(samples)

    # Get class weights
    cw = train_class_weight(samples)

    # Initial model
    resnet = customized_resnet(shape=tr_X.shape[1:])

    # Training
    resnet.fit(tr_X, tr_y, **training_config(cw))

    # Get best model
    resnet = load_model(r'Model/tmp/_tmp_best.hdf5')
    resnet.save('Model/formal/cust_resnet.h5')


def train_class_weight(samples):
    # Dictionary of class and number of them
    weights = collections.Counter([sample.label for sample in samples])
    tot_wgt = sum(weights.values())

    weights = {k: tot_wgt/v for k, v in weights.items()}
    max_wgt = max(weights.values())

    return {k: v/max_wgt for k, v in weights.items()}


def training_config(class_weight=None):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8)

    cfg = {'epochs': 50,
           'batch_size': 32,
           'validation_split': .10,
           'validation_data': None,
           'class_weight': class_weight,
           'sample_weight': None,
           'callbacks': [early_stop(),
                         reduce_lr,
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


def predict_multiple_inputs_model():
    #
    stamp = datetime.datetime.now()
    stamp = stamp.strftime(format='%Y%m%d_%H%M%S')

    # Get test data
    samples = pickle.load(open(os.path.join(DATA_DIR, 'test.pkl'), 'rb'))

    # Concatenate the numerical arrays
    input_enco = np.array([sample.ft['AE'] for sample in samples])
    input_mfcc = np.array([sample.ft['MFCC'] for sample in samples])
    mel_spec_imgs, wavec_imgs = load_imgs(samples)

    # Load model
    resnet = load_model(r'Model/formal/multiple_inputs.h5')

    # Predict
    pred_y = resnet.predict([input_enco.astype(np.float16),
                             input_mfcc.astype(np.float16),
                             mel_spec_imgs.astype(np.float16),
                             wavec_imgs.astype(np.float16)])
    pred_y = np.argmax(pred_y, axis=1)

    # Get label encoder
    lbl = pickle.load(open(os.path.join(r'Encoders/label', 'label_encoder.pkl'), 'rb'))

    # Create submission csv
    subm = pd.DataFrame({'Class': lbl.inverse_transform(pred_y),
                         'ID': [sample.id for sample in samples]})

    subm.to_csv(r'Output/submission_{}.csv'.format(stamp), index=False)


if __name__ == '__main__':
    # train_dnn()
    # predict_dnn()

    train_multiple_inputs_model()
    predict_multiple_inputs_model()