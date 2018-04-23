import tensorflow as tf

from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import GRU, LSTM
from keras.layers import Conv1D, MaxPool1D, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

'''
    Reference:
        Keras installation
            https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-2
            Up to step 4.
        TensorFlow installation
            https://www.tensorflow.org/install/
    Dependencies:
        Python 3.5
        SciPy/ NumPy
        Matplotlib
        TensorFlow (Require CUDA/cuDNN and a listed graphic card if chosen the version GPU support)
'''

# from keras.datasets import mnist
# # Load pre-shuffled MNIST data into train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# print(X_train.shape)
# plt.imshow(X_train[0])
# plt.show()

'''
    Reference:
        https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/
        
    Improved reference:
        https://blog.manash.me/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b
'''
import glob
import IPython.display as ipd
import librosa
import librosa.display as display
import numpy as np
import os
import pandas as pd
import pdb
import random

from matplotlib import pyplot as plt
from os.path import join
from sklearn.preprocessing import LabelEncoder

from keras.backend.tensorflow_backend import set_session


# Step 1 and  2 combined: Load audio files and extract features (done in *_preprocessing)
def loader(path, name, dtype='float'):
    with open(join(path, name), 'r') as f:
        data_array = np.loadtxt(f, dtype=dtype)

    return data_array


def loader_hd(path, name):
    data_array = np.load(join(path, name))

    return data_array

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

doc_path = r'D:\DataSet\UrbanSoundChallenge\train'

data_path = r'D:\DataSet\UrbanSoundChallenge\train\Train'
test_path = r'D:\DataSet\UrbanSoundChallenge\test\Test'
sub_path = r'D:\DataSet\UrbanSoundChallenge\submission'

train = pd.read_csv(join(doc_path, 'train.csv'))
test = pd.read_csv(join(r'D:\DataSet\UrbanSoundChallenge\test', 'test.csv'))

# pdb.set_trace()
X = loader_hd(data_path, r'mfccs_train_X_all_256.npy')
y = loader_hd(data_path, r'mfccs_train_y_all_256.npy')

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y))


# Step 4: Run a deep learning model and get results
num_labels = y.shape[1]

# The model needs to know what input shape it should expect.
# For this reason, the first layer in a Sequential model (and only the first,
# because following layers can do automatic shape inference) needs to receive
# information about its input shape.
# From first ref:
#   build model
#   The Sequential model is a linear stack of layers.
# model = Sequential()
# #
# model.add(Dense(16, input_shape=(40,)))  # first layer
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(num_labels))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#
# # Epochs: One epoch consists of one full training cycle on the training set.
# # Once every sample in the set is seen, you start again - marking the beginning
# # of the 2nd epoch.
# model.fit(X, y, batch_size=8, epochs=12, validation_data=(X, y))
#

# X = X.reshape(X.shape[0], X.shape[1], 1)
# y = y.reshape(y.shape[0], y.shape[1], 1)


# From second ref (CNN):
def cnn1D(X_tr, y_tr):

    model = Sequential()
    model.add(Conv1D(256, 3, activation='relu', input_shape=(X_tr.shape[1], X_tr.shape[2]),
                     W_constraint=maxnorm(3), name='C1'))

    model.add(Conv1D(256, 3, activation='relu', name='C2'))
    model.add(MaxPool1D(pool_size=3))
    # model.add(Dropout(0.1))

    model.add(Conv1D(512, 2, activation='relu', name='C3'))
    model.add(MaxPool1D(pool_size=2))
    # model.add(Dropout(0.1))

    model.add(Flatten(name='F1'))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3), name='FD1'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax', name='FD2'))
    # https://keras.io/losses/
    # https://keras.io/optimizers/
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # No progress bar displayed: verbose=0
    model.fit(X_tr, y_tr, batch_size=200, epochs=50, validation_split=0.1, verbose=0)

    print(model.summary())
    return model


def cnn2D(X_tr, y_tr):
    model = Sequential()

    model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', padding='same',
                     input_shape=(X_tr.shape[1], X_tr.shape[2], 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(12, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(24, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # model.fit(X_tr, y_tr, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)
    # model.train_on_batch(X_tr, y_tr)
    model.fit(X_tr, y_tr, batch_size=200, epochs=50, validation_split=0.1, verbose=0)

    print(model.summary())

    return model


def rnn(X_tr, y_tr):
    model = Sequential()

    model.add(LSTM(1024, input_shape=(X_tr.shape[1], X_tr.shape[2])))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax', name='FD2'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.fit(X_tr, y_tr, batch_size=200, epochs=25, validation_split=0.1,verbose=0) # validation_split=0.1,
    print(model.summary())

    return model


def crnn(X_tr, y_tr):

    model= Sequential()

    model.add(Conv1D(256, 2, activation='relu', input_shape=(X_tr.shape[1], X_tr.shape[2]),
                     W_constraint=maxnorm(2), name='C1'))
    model.add(Dropout(0.2))

    model.add(Conv1D(256, kernel_size=2, padding='same', activation='relu', name='CD2'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.4))

    # model.add(Conv1D(512, kernel_size=2, padding='same', activation='relu', name='CD3'))
    # model.add(MaxPool1D(pool_size=2))
    # model.add(Dropout(0.8))

    model.add(GRU(128, return_sequences=True, name='GRU1'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='FD1'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # pdb.set_trace()
    model.fit(X_tr, y_tr, batch_size=128, epochs=50, validation_split=0.1,verbose=0)
    print(model.summary())

    return model


#
# pdb.set_trace()
try:

    # model = cnn1D(X, y)
    # model = cnn2D(X, y)
    # model = rnn(X, y)
    model = crnn(X, y)
except:
    pdb.set_trace()

# Predictive
# pdb.set_trace()
X = loader_hd(test_path, r'mfccs_test_X_all_256.npy')
# X = X.reshape(X.shape[0], X.shape[1], 1)
try:
    prediction = model.predict(X)
except:
    pdb.set_trace()

classes = prediction.argmax(axis=-1)
test['Class'] = list(lb.inverse_transform(classes))
test.to_csv(join(sub_path, 'sub_CRNN_all.csv'), index=False)
