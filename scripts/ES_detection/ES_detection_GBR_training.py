#!/usr/bin/env python

################################################
# LOAD DATA
################################################

import pickle
import numpy as np
from sklearn.externals import joblib


(study_indices,
 X_train, y_train,
 X_test, y_test) = joblib.load('../../data_proc/ES_detection_training.pkl')


################################################
# DEFINE MODEL
################################################

from sklearn.ensemble import GradientBoostingRegressor

clf_params = {
    'n_estimators': 3000,
    'max_depth': 4,
    'min_samples_split': 1,
    'learning_rate': 0.02,
    'loss': 'huber',
    'verbose': 1
}
clf = GradientBoostingRegressor(**clf_params)


################################################
# TRAIN MODEL
################################################

clf.fit(X_train, y_train)


################################################
# EVALUATE MODEL
################################################

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, clf.predict(X_test))
mse = mean_squared_error(y_test, clf.predict(X_test))

print('MAE:', mae)
print('MSE:', mse)


################################################
# SAVE MODEL
################################################

joblib.dump(clf, '../../model_weights/ES_detection_GBR.pkl') 
