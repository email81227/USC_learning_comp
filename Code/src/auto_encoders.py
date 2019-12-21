import glob
import numpy as np
import os
import tensorflow as tf
import time


LATENT_SHAPE = 32


# TODO: Check why performance differnet between Keras + tensorflow and tf.keras
# >>> The optimizer Ada-delta has different default learning rate which is 1e-3 and 1.0 respectively
class AutoEncoder(tf.keras.Model):
    def __init__(self, latent_dim, original_size):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.inference_net = tf.keras.Sequential()
        self.inference_net.add(tf.keras.layers.InputLayer(input_shape=original_size))
        self.inference_net.add(tf.keras.layers.Dense(self.latent_dim, activation='relu'))

        self.generative_net = tf.keras.Sequential()
        self.generative_net.add(tf.keras.layers.Dense(original_size, activation='sigmoid'))

    def encode(self, samples):
        return self.inference_net(samples)

    def decode(self, encoded):
        return self.generative_net(encoded)

    def call(self, samples, training=None, mask=None):
        encoded = self.inference_net(samples)
        decoded = self.generative_net(encoded)
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

    # If using tf.keras
    tr_X = tf.convert_to_tensor(tr_X)
    ts_X = tf.convert_to_tensor(ts_X)
    
    model = AutoEncoder(32, 784)
    model.compile(optimizer=tf.keras.optimizers.Adadelta(1.0, .95),
                  loss=tf.keras.losses.BinaryCrossentropy())
    model.fit(tr_X, tr_X,
              epochs=50,
              batch_size=256,
              shuffle=True,
              validation_data=(ts_X, ts_X))

    encoded_imgs = model.encode(ts_X)
    decoded_imgs = model.decode(encoded_imgs)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(tf.reshape(ts_X[i], (28, 28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()