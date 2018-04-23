import numpy as np
import pandas as pd
import pdb

from os.path import join
from sklearn import svm, model_selection
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline


def loader(path, name, dtype='float'):
    with open(join(path, name), 'r') as f:
        data_array = np.loadtxt(f, dtype=dtype)

    return data_array


def loader_hd(path, name):
    data_array = np.load(join(path, name))

    return data_array


def svc_param_selection(model, Xtr, ytr, Xts, sample_weight=None, nfolds=3):
    Cs = list(np.logspace(8, 13, 11))        # list(np.logspace(5, 15, 11))
    gammas = list(np.linspace(0.002, 0.003, 10))   # list(np.logspace(-9, 1, 11))
    # degrees = [3, 4, 5]
    param_grid = {'svc__C': Cs, 'svc__gamma': gammas}  # , 'degree': degrees}
    grid_search = model_selection.GridSearchCV(model, param_grid, cv=nfolds, n_jobs=1)
    grid_search.fit(Xtr, ytr, sample_weight)

    print(grid_search.best_params_)
    return grid_search.predict(Xts), grid_search.cv_results_


doc_path = r'D:\DataSet\UrbanSoundChallenge\train'

data_path = r'D:\DataSet\UrbanSoundChallenge\train\Train'
test_path = r'D:\DataSet\UrbanSoundChallenge\test\Test'
sub_path = r'D:\DataSet\UrbanSoundChallenge\submission'

train = pd.read_csv(join(doc_path, 'train.csv'))
test = pd.read_csv(join(r'D:\DataSet\UrbanSoundChallenge\test', 'test.csv'))

# pdb.set_trace()
# Simple svm with mean mfcc
'''
    References:
        https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
        https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
'''

# Xtr = loader(data_path, r'mfcc_train_X_128')   # , mfcc_train_X_full_256.npy
# ytr = loader(data_path, r'mfcc_train_y_128', 'str')   # , mfcc_train_y_full_256.npy
#
# class_weight = (sum(train.Class.value_counts())/train.Class.value_counts()).to_dict()
# sample_weight = np.ones(len(ytr))
# for idy, y in enumerate(ytr):
#     sample_weight[idy] *= class_weight[y]
#
# # pdb.set_trace()
# # Create SVM classification object
# # model = svm.SVC(kernel='rbf', class_weight='balanced')
# #
# # model.fit(X, y)
# # # model.score(Xtr, ytr)
# # #Predict Output
# Xts = loader(test_path, r'mfcc_test_X_128')   # mfcc_test_X_full_256.npy
# # prediction = model.predict(Xts)
#
# prediction, results = svc_param_selection(Xtr, ytr, Xts, class_weight, sample_weight)
#
# print(results)
# pdb.set_trace()

# Original mfccs with pca-svm pipeline:
'''
    Ref:
        https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
'''


def pca_svc_pipe(Xtr, ytr, Xts, sample_weight=None, class_weight='balanced'):
    pca = RandomizedPCA(n_components=256, whiten=True, random_state=None)
    svc = svm.SVC(kernel='rbf', class_weight=class_weight)
    model = make_pipeline(pca, svc)

    prediction, results = svc_param_selection(model, Xtr, ytr, Xts, sample_weight)
    results = pd.DataFrame(results)
    results.to_csv(join(sub_path, 'cv_results_3.csv'), index=False)
    return prediction


# pdb.set_trace()
Xtr = loader_hd(data_path, r'mfcc_train_X_full_256.npy')
ytr = loader_hd(data_path, r'mfcc_train_y_full_256.npy')

Xts = loader_hd(test_path, r'mfcc_test_X_full_256.npy')

class_weight = (sum(train.Class.value_counts())/train.Class.value_counts()).to_dict()
sample_weight = np.ones(len(ytr))
for idy, y in enumerate(ytr):
    sample_weight[idy] *= class_weight[y]

# Expected 2D array and need to be flattened.
nsamples, nx, ny = Xtr.shape
Xtr = Xtr.reshape(nsamples, nx * ny)

nsamples, nx, ny = Xts.shape
Xts = Xts.reshape(nsamples, nx * ny)

prediction = pca_svc_pipe(Xtr, ytr, Xts, sample_weight)

test['Class'] = prediction
test.to_csv(join(sub_path, 'sub_PCA_SVM_rbf_full_3.csv'), index=False)