from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model


class ComplexInput(keras.Model):
    def __init__(self, input_shapes, num_class):
        '''

        :param input_shapes: A dictionary. The keys are name and the values are also dictionaries with shape and type info.
        '''
        super(ComplexInput, self).__init__()

        self.num_class = num_class
        self.input_seq = []

        for k, v in input_shapes.items():
            if v['type'].startswith('dense'):
                self.input_seq.append(self.dense_inputs(v['shape'], k))
            elif v['type'].startswith('resnet'):
                self.input_seq.append(self.resnet_inputs(v['shape'], k))

        feat_inputs = concatenate([model.output for model in self.input_seq])

        x = Dense(1024, activation='relu')(feat_inputs)
        outputs = Dense(self.num_class, activation='softmax')(x)

        self.model = Model(inputs=[model.input for model in self.input_seq], outputs=outputs)

    def dense_inputs(self, shape, name):
        inputs = Input(shape=shape, name=name)
        x = Dense(1024, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        # x = Dense(self.num_class, activation='softmax')(x)
        return Model(inputs, x)

    def resnet_inputs(self, shape, name):
        inputs = Input(shape=shape, name=name)
        x = Conv2D(32, 3, activation='relu')(inputs)
        x = Conv2D(64, 3, activation='relu')(x)
        block_1_output = MaxPooling2D(3)(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        block_2_output = keras.layers.add([x, block_1_output])

        x = Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        block_3_output = keras.layers.add([x, block_2_output])

        x = Conv2D(64, 3, activation='relu')(block_3_output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        # x = Dense(self.num_class, activation='softmax')(x)
        return Model(inputs, x)

    # Parameters for input size = 50,999
    def proposed_cnn(self, shape, name):
        inputs = Input(shape=shape, name=name)
        x = Conv1D(16, 64, 2, activation='relu')(inputs)
        block_1_output = MaxPooling1D(8, 8)(x)

        x = Conv1D(32, 32, 2, activation='relu')(block_1_output)
        block_2_output = MaxPooling1D(8, 8)(x)

        x = Conv1D(64, 16, 2, activation='relu')(block_2_output)
        x = Conv1D(128, 8, 2, activation='relu')(x)
        x = Conv1D(256, 4, 2, activation='relu')(x)
        block_3_output = MaxPooling1D(4, 4)(x)

        x = Dense(128, activation='relu')(block_3_output)
        x = Dense(64, activation='relu')(x)
        return Model(inputs, x)

    def call(self, samples, training=None, mask=None):
        return self.model(samples)


def dense_model(shape=(120,)):
    model = Sequential()
    model.add(Dense(1024, input_shape=shape, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # print(model.summary())
    return model


def customized_resnet(shape=(300, 150, 3)):
    inputs = Input(shape=shape, name='img')
    x = Conv2D(32, 3, activation='relu')(inputs)
    x = Conv2D(64, 3, activation='relu')(x)
    block_1_output = MaxPooling2D(3)(x)

    x = Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = keras.layers.add([x, block_1_output])

    x = Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = keras.layers.add([x, block_2_output])

    x = Conv2D(64, 3, activation='relu')(block_3_output)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs, name='cust_resnet')
    # model.summary()

    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


if __name__ == '__main__':
    '''
    Sorting out the preprocessed data into modeling folder

    (preprocessed data == [feature_eng.py] ==> modeling)

    '''

    import pickle

    SAVE_DIR = r'Data/Modeling'

    tr = pickle.load(open(r'Data/Modeling/train.pkl', 'rb'))
    ts = pickle.load(open(r'Data/Modeling/test.pkl', 'rb'))
