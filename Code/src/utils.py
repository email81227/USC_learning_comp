import librosa


class RawSample:
    def __init__(self, id, label, data_path):
        self.id = id
        self.label = label
        self.sample_path = data_path

    def load_sample(self):
        return librosa.load(self.sample_path)


class Features:
    def __init__(self, id):
        self.id = id
