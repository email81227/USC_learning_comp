
import keras
import numpy as np
import pandas as pd
import pdb
import tensorflow as tf

from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils

from math import ceil
from numpy import random
from os.path import join
from PIL import Image
from sklearn.preprocessing import LabelEncoder


def loader(path, name, dtype='float'):
    with open(join(path, name), 'r') as f:
        data_array = np.loadtxt(f, dtype=dtype)

    return data_array


def model_saver(model, path=r'.'):
    # serialize model to JSON
    model_json = model.to_json()
    with open(join(path, "model.json"), "w") as json_file:
        json_file.write(model_json)

# For images
def parser(row, num_mfcc=128):
    # function to load files and extract features
    if hasattr(row, 'Class'):
        file_name = join(data_path, str(row.ID) + '.png')
    else:
        file_name = join(test_path, str(row.ID) + '.png')

    # handle exception to check if there isn't a file which is corrupted
    try:
        feature = np.array(Image.open(file_name).convert('RGB'), 'f')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None, None

    if hasattr(row, 'Class'):
        label = row.Class
        return [feature, label]
    else:
        return [feature]


def datagenerater(df, batch_size=4, mod='train'):
    batch_features = np.zeros((batch_size, 160, 240, 3))
    batch_labels = np.zeros((batch_size, 10))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = random.choice(len(df), 1)

            temp = df.iloc[index].apply(parser, axis=1)
            # pdb.set_trace()
            batch_features[i] = np.array(temp.ID.tolist())
            if mod=='train':
                batch_labels[i] = np_utils.to_categorical(lb.transform(temp.Class), 10)

        if mod == 'train':
            yield batch_features, batch_labels
        else:
            yield batch_features


def cnn2D(train):
    model = Sequential()

    model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(160, 240, 3)))
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

    # validation_data=datagenerater(train, 4), validation_steps=len(train)/12/4
    print(model.summary())

    return model


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

doc_path = r'D:\DataSet\UrbanSoundChallenge\train'

data_path = r'D:\DataSet\UrbanSoundChallenge\train\Train'
test_path = r'D:\DataSet\UrbanSoundChallenge\test\Test'
sub_path = r'D:\DataSet\UrbanSoundChallenge\submission'

train = pd.read_csv(join(doc_path, 'train.csv'))
test = pd.read_csv(join(r'D:\DataSet\UrbanSoundChallenge\test', 'test.csv'))

# Create a label encoder for all the labels found
lb = LabelEncoder()
lb.fit(train.Class.unique().ravel())

# pdb.set_trace()
try:
    model = cnn2D(train)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # model.fit(X_tr, y_tr, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)
    # model.train_on_batch(X_tr, y_tr)
    model.fit_generator(datagenerater(train), epochs=25, verbose=0,
                        steps_per_epoch=ceil(len(train)/4))

    model_saver(model, r'D:\DataSet\UrbanSoundChallenge\model')
except:
    pdb.set_trace()

# Predictive
# temp = test.apply(parser, axis=1)
# temp.columns = ['feature']
# X = np.array(temp.feature.tolist())
# pdb.set_trace()
try:
    # Steps = number of sample / batch size of generator
    prediction = model.predict_generator(datagenerater(test, 12, mod='test'),
                                         ceil(len(test)/12), verbose=0)
except:
    pdb.set_trace()

classes = prediction[:len(test)].argmax(axis=-1)
test['Class'] = list(lb.inverse_transform(classes))
test.to_csv(join(sub_path, 'sub_CNN_2D.csv'), index=False)