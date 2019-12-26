# For reproducible results
import numpy as np
np.random.seed(123)

from Code.src.models import *
from Code.src.preprocess import N_JOBS
from multiprocessing import Pool
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

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


# TODO:
def img_feature_load(objs):
    # initialize our images array (i.e., the house images themselves)
    images = []

    # loop over the indexes of the houses
    for obj in objs:
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        housePaths = sorted(list(glob.glob(basePath)))

        # initialize our list of input images along with the output image
        # after *combining* the four input images
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype="uint8")

        # loop over the input house paths
        for housePath in housePaths:
            # load the input image, resize it to be 32 32, and then
            # update the list of input images
            image = cv2.imread(housePath)
            image = cv2.resize(image, (32, 32))
            inputImages.append(image)

        # tile the four input images in the output image such the first
        # image goes in the top-right corner, the second image in the
        # top-left corner, the third image in the bottom-right corner,
        # and the final image in the bottom-left corner
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, 0:32] = inputImages[3]

        # add the tiled image to our set of images the network will be
        # trained on
        images.append(outputImage)

    # return our set of images
    return np.array(images)


def learning_rate_schedule():
    return LearningRateScheduler(lambda x: 1. / (1. + x))


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




def training_config():
    cfg = {'epochs': 50,
           'batch_size': 64,
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


def predict():
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
    train_dnn()
    predict()