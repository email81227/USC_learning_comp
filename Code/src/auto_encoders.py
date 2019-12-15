import glob
import numpy as np
import os
import tensorflow as tf
import time


LATENT_SHAPE = 32


class Encoder(tf.keras.layers.Layer):
    '''
    As part of auto-encoder, inherit from Layer makes sense
    '''
    def __init__(self, target_dim):
        super(Encoder, self).__init__()
        self.output_layer = tf.keras.layers.Dense(
            units=target_dim,
            activation='relu'
        )

    def call(self, samples):
        # activation = self.hidden_layer(input_features)
        return self.output_layer(samples)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, target_dim):
        super(Decoder, self).__init__()
        self.output_layer = tf.keras.layers.Dense(
            units=input_dim,
            activation='sigmoid'
        )

    def call(self, encode):
        return self.output_layer(encode)


class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape, target_dim=LATENT_SHAPE):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(target_dim=target_dim)
        self.decoder = Decoder(input_shape, target_dim=target_dim)

        self.model = tf.keras.models.Sequential()
        self.model.add(self.encoder)
        self.model.add(self.decoder)

    def call(self, samples):
        encoded = self.encoder(samples)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt
    import pickle
    import PIL

    '''
    Sorting out the preprocessed data into modeling folder

    (preprocessed data == [feature_eng.py] ==> modeling)

    '''
    (tr_X, _), (ts_X, _) = tf.keras.datasets.mnist.load_data()

    tr_X = tr_X.astype('float32') / 255.
    ts_X = ts_X.astype('float32') / 255.
    tr_X = tr_X.reshape((len(tr_X), np.prod(tr_X.shape[1:])))
    ts_X = ts_X.reshape((len(ts_X), np.prod(ts_X.shape[1:])))
    print(tr_X.shape)
    print(ts_X.shape)
    
    model = Autoencoder(784)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.fit(tr_X, tr_X,
              epochs=50,
              batch_size=256,
              shuffle=True,
              validation_data=(ts_X, ts_X))

    encoded_imgs = model.encoder(ts_X)
    decoded_imgs = model.decoder(encoded_imgs)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(ts_X[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(tf.reshape(decoded_imgs[i], (28, 28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()