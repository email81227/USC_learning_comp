import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import GRU, LSTM
from keras.layers import Conv1D, MaxPool1D, Conv2D, MaxPooling2D, SimpleRNNCell
from keras.models import Sequential
from keras.optimizers import Adam


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
    model.add(Conv1D(64, kernel_size=16, activation='relu', input_shape=(X_tr.shape[1], X_tr.shape[2]),
                     W_constraint=maxnorm(16), name='C1'))

    model.add(Conv1D(64, kernel_size=3, activation='relu', name='C2'))
    model.add(MaxPool1D(pool_size=3))
    model.add(Dropout(0.1))

    model.add(Conv1D(64, 2, activation='relu', name='C3'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Flatten(name='F1'))
    model.add(Dense(256, activation='relu', name='FD1'))
    model.add(Dropout(0.2))

    model.add(Dense(y_tr.shape[1], activation='softmax', name='FD2'))
    # https://keras.io/losses/
    # https://keras.io/optimizers/
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # No progress bar displayed: verbose=0
    # model.fit(X_tr, y_tr, batch_size=200, epochs=50, validation_split=0.1, verbose=0)

    print(model.summary())
    return model


def cnn2D(X_tr, y_tr):
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(16, 8), activation='relu', padding='same',
                     input_shape=(X_tr.shape[1], X_tr.shape[2], 1)))
    model.add(MaxPooling2D(pool_size=(8, 4), strides=(4, 4)))
#    model.add(Dropout(0.1))

    model.add(Conv2D(16, (4, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(4, 2)))
#    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
#    model.add(Dropout(0.5))

    model.add(Dense(y_tr.shape[1], activation='softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # model.fit(X_tr, y_tr, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)
    # model.train_on_batch(X_tr, y_tr)
    # model.fit(X_tr, y_tr, batch_size=200, epochs=50, validation_split=0.1, verbose=0)

    print(model.summary())

    return model


def LSTM(X_tr, y_tr):
    model = Sequential()

    model.add(LSTM(1024, input_shape=(X_tr.shape[1], X_tr.shape[2])))
    model.add(Dropout(0.2))

    model.add(Dense(y_tr.shape[1], activation='softmax', name='FD2'))

    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # model.fit(X_tr, y_tr, batch_size=200, epochs=25, validation_split=0.1,verbose=0) # validation_split=0.1,
    print(model.summary())

    return model

def GRU(X_tr, y_tr):
    model = Sequential()

    model.add(SimpleRNNCell(1024, input_shape=(X_tr.shape[1], X_tr.shape[2])))
    model.add(Dropout(0.2))

    model.add(Dense(y_tr.shape[1], activation='softmax', name='FD2'))

    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # model.fit(X_tr, y_tr, batch_size=200, epochs=25, validation_split=0.1,verbose=0) # validation_split=0.1,
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

    model.add(Dense(y_tr.shape[1], activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # pdb.set_trace()
    # model.fit(X_tr, y_tr, batch_size=128, epochs=50, validation_split=0.1,verbose=0)
    print(model.summary())

    return model