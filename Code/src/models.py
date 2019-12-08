from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# from tensorflow.keras import Model


# class DeepModel(Model):
#     def __init__(self, config, *input, **kwargs):
#         super(DeepModel, self).__init__()
#         # Save config in model
#         self.config = config


def dense_model(shape=(120,)):
    model = Sequential()
    model.add(Dense(1024, input_shape=shape, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    return model


class DNN:
    def __init__(self):
        model = Sequential()
        model.add(Dense(1024, input_shape=(120,), activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

        print(model.summary())

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)