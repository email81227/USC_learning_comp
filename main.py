import numpy as np
import os
import pandas as pd
import pdb

from DeepModels import *
from keras.utils import np_utils

from os.path import join
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Customized functions
from PublicFunctions import *

train_path = r'D:\DataSet\UrbanSoundChallenge\train'
test_path = r'D:\DataSet\UrbanSoundChallenge\test'


train = pd.read_csv(join(train_path, 'train.csv'))
test = pd.read_csv(join(test_path, 'test.csv'))


X_tr = loader_hd(train_path, r'Train\mfccs_train_X_all_256.npy')
y_tr = loader_hd(train_path, r'Train\mfccs_train_y_all_256.npy')

X_ts = loader_hd(test_path, r'Test\mfccs_test_X_all_256.npy')
# y_ts = loader_hd(test_path, r'test_y_256.npy')

lb = LabelEncoder()

y_tr = np_utils.to_categorical(lb.fit_transform(y_tr))

# pdb.set_trace()
# Training
epochs = 50
b_size = 10
vaalid = 0.2

# Convolution 1D
cnn = cnn1D(X_tr, y_tr)

cnn.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
cnn.fit(X_tr, y_tr, batch_size=b_size, epochs=epochs, validation_split=vaalid, verbose=0)

# Convolution 2D
# X_tr = np.reshape(X_tr, (X_tr.shape[0], X_tr.shape[1], X_tr.shape[2], 1))
# X_ts = np.reshape(X_ts, (X_ts.shape[0], X_ts.shape[1], X_ts.shape[2], 1))
# cnn = cnn2D(X_tr, y_tr)
#
# cnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# cnn.fit(X_tr, y_tr, batch_size=b_size, epochs=epochs, verbose=0, validation_split=vaalid)
# model.train_on_batch(X_tr, y_tr)
# model.fit(X_tr, y_tr, batch_size=200, epochs=50, validation_split=0.1, verbose=0)

# pdb.set_trace()
prediction = cnn.predict(X_ts)

# classes = prediction.argmax(axis=-1)
# test['Prediction_Class'] = list(lb.inverse_transform(classes))

# print('Accuracy:' + str(sum(test['Category']==test['Prediction_Class'])/len(test)))
# pdb.set_trace()

classes = prediction[:len(test)].argmax(axis=-1)
test['Class'] = list(lb.inverse_transform(classes))
test.to_csv(join(r'D:\DataSet\UrbanSoundChallenge\submission', 'sub_CNN_2D_1.csv'), index=False)