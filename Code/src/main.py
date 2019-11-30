from Code.src.feature_eng import *
from Code.src.preprocess import *
from Code.src.utils import *


def main():
    # Get data
    objs = get_training_data()

    # Get samples
    for obj in objs:
        obj.load_sample(**{'sr': SAMPLE_RATE})

    # Pre-processing
    objs = preprocess(objs)

    # Get features
    objs = features_engineering(objs)

    # Training

    return objs


if __name__ == '__main__':
    objs = main()
