from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle

N_JOBS = 15


# for mfcc
def features_engineering(objs, training=True):
    # Get id
    sample_id = [obj.id for obj in objs]

    # Feature generated
    X = [obj.ft for obj in objs]
    X = np.array(X)
    #

    #

    # Label encoding
    if training:
        y = label_encoding(objs, training)
        y = np.array(y)

        return sample_id, X, y
    else:
        return sample_id, X, None


def label_encoding(objs, training=True):
    labels = [obj.label for obj in objs]

    if training:
        lbl = LabelEncoder()
        lbl.fit(labels)

        pickle.dump(lbl, open(os.path.join(r'Encoders/label', 'label_encoder.pkl'), 'wb'))
    else:
        lbl = pickle.load(open(os.path.join(r'Encoders/label', 'label_encoder.pkl'), 'rb'))

    return lbl.transform(labels)


if __name__ == '__main__':
    '''
    Sorting out the preprocessed data into modeling folder

    (preprocessed data == [feature_eng.py] ==> modeling)
    
    '''
    SAVE_DIR = r'Data/Modeling'

    tr = pickle.load(open(r'Data/preprocessed/train.pkl', 'rb'))
    ts = pickle.load(open(r'Data/preprocessed/test.pkl', 'rb'))

    tr = features_engineering(tr, True)
    ts = features_engineering(ts, False)

    pickle.dump(tr, open(os.path.join(SAVE_DIR, 'train.pkl'), 'wb'))
    pickle.dump(ts, open(os.path.join(SAVE_DIR, 'test.pkl'), 'wb'))
