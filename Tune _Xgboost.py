import pandas as pd
import pdb
import xgboost as xgb

from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from xgboost.sklearn import XGBClassifier

# Customized functions
from PublicFunctions import *


train_path = r'D:\DataSet\UrbanSoundChallenge\train'
test_path = r'D:\DataSet\UrbanSoundChallenge\test'


train = pd.read_csv(join(train_path, 'train.csv'))
test = pd.read_csv(join(test_path, 'test.csv'))


X_tr = loader_hd(train_path, r'Train\mfccs_train_X_all_64.npy')
y_tr = loader_hd(train_path, r'Train\mfccs_train_y_all_64.npy')

X_ts = loader_hd(test_path, r'Test\mfccs_test_X_all_64.npy')

X_tr = X_tr.reshape((X_tr.shape[0], X_tr.shape[1] * X_tr.shape[2]))
X_ts = X_ts.reshape((X_ts.shape[0], X_ts.shape[1] * X_ts.shape[2]))
# pdb.set_trace()


lb = LabelEncoder()
y_tr = lb.fit_transform(y_tr)


def xgboost_param_selection(param_grid, Xtr, ytr, Xts, sample_weight=None, nfolds=5):
    model = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5, eta=2, silent=1,
                          min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                          objective='multi:softprob', nthread=4, scale_pos_weight=1, num_class=10)
    gr_search = GridSearchCV(model, param_grid, cv=nfolds, n_jobs=1, iid=False)
    gr_search.fit(Xtr, ytr, sample_weight)

    print(gr_search.best_params_)
    return gr_search.predict(Xts), gr_search.cv_results_


dtrain = xgb.DMatrix(X_tr, label=y_tr)
dtest = xgb.DMatrix(X_ts)

# xgboost
max_depth = range(6, 10, 2)         # list(np.logspace(5, 15, 11))
min_child_weight = range(1, 3, 2)   # list(np.logspace(-9, 1, 11))
# degrees = [3, 4, 5]
param_grid = {'max_depth': max_depth, 'min_child_weight': min_child_weight}  # , 'degree': degrees}

# param = {
#     'max_depth': 25,  # the maximum depth of each tree
#     'eta': 0.2,  # the training step for each iteration
#     'silent': 1,  # logging mode - quiet
#     'objective': 'multi:softprob',  # error evaluation for multiclass training
#     'num_class': 10}  # the number of classes that exist in this datset
# num_round = 200  # the number of training iterations
#
# bst = xgb.train(param, dtrain, num_round)
# # bst.dump_model('dump.raw.txt')
#
#
# preds = bst.predict(dtest)
# prediction = np.asarray([np.argmax(line) for line in preds])
# pdb.set_trace()

prediction, tune_result = xgboost_param_selection(param_grid, X_tr, y_tr, X_ts)
tune_result = pd.DataFrame(tune_result)
tune_result.to_csv(join(r'D:\DataSet\UrbanSoundChallenge\submission', 'xgb_tune_results.csv'), index=False)

test['Class'] = list(lb.inverse_transform(prediction))
test.to_csv(join(r'D:\DataSet\UrbanSoundChallenge\submission', 'sub_xgb_r200.csv'), index=False)
