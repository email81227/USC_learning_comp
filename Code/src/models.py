from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
# from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model


def dense_model(shape=(120,)):
    model = Sequential()
    model.add(Dense(1024, input_shape=shape, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
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
    model.summary()

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
