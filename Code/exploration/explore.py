import collections
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

samples = pickle.load(open(r'Data/preprocessed/train.pkl', 'rb'))
samples_dict = collections.defaultdict(list)

for i, sample in enumerate(samples):
    samples_dict[sample.label].append(i)

# 5 wave plots
plt.figure(figsize=(20, 10))
for i, cate in enumerate(samples_dict.keys()):
    indexes = random.sample(samples_dict[cate], 5)

    for j, idx in enumerate(indexes):
        ax = plt.subplot(5, 10, j*10+i+1)

        librosa.display.waveplot(samples[idx].sample,
                                 sr=samples[idx].sample_rate,
                                 max_points=None)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
plt.show()

# 5 Spectrograms
plt.figure(figsize=(20, 10))
for i, cate in enumerate(samples_dict.keys()):
    indexes = random.sample(samples_dict[cate], 5)

    for j, idx in enumerate(indexes):
        ax = plt.subplot(5, 10, j*10+i+1)

        mel_spec = librosa.feature.melspectrogram(y=samples[idx].sample, sr=samples[idx].sample_rate, n_mels=128, fmax=8000)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec, sr=samples[idx].sample_rate, x_axis=None, y_axis=None)

        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
plt.show()