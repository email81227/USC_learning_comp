import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import GRU, LSTM
from keras.layers import Conv1D, MaxPool1D, Conv2D, MaxPooling2D, SimpleRNNCell
from keras.models import Sequential
from keras.optimizers import Adam

'''
Reference:
    https://keras.io/getting-started/sequential-model-guide/
'''

def stocked_LSTM(X_tr, y_tr):
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True, input_shape=(X_tr.shape[1], X_tr.shape[2])))

    # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))

    # return a single vector of dimension 32
    model.add(LSTM(32))


    model.add(Dense(10, activation='softmax'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model