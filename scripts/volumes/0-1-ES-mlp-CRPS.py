#!/usr/bin/env python

# run script with THEANO_FLAGS='floatX=float32,device=gpu2,nvcc.fastmath=True'
# also run without buffering: 'python -u ...'
# for example:
# - THEANO_FLAGS='floatX=float32,device=gpu2,nvcc.fastmath=True' python -u sunnybrook_LV.py >sunnybrook_LV.log 2>&1 &

from keras.models import Sequential, Graph
from keras.layers.core import Activation, Dense, Dropout, Flatten, Merge, Reshape, Lambda
from keras.layers.core import TimeDistributedDense, TimeDistributedMerge
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ParametricSoftplus, ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import initializations
from keras.layers.core import Layer
from keras import backend as K
from theano import tensor as T

# for preventing python max recursion limit error
import sys
sys.setrecursionlimit(50000)

################################################
# LOAD AND SHUFFLE DATA
################################################

import pickle
import numpy as np
from sklearn.externals import joblib
import random

(pts_train, pts_train_val, 
 data_ED_train, data_ES_train, 
 labels_value_ED_train, labels_value_ES_train, labels_func_ED_train, labels_func_ES_train, 
 data_ED_train_val, data_ES_train_val, 
 labels_value_ED_train_val, labels_value_ES_train_val, labels_func_ED_train_val, labels_func_ES_train_val, 
 data_ED_val, data_ES_val, data_val_pt_index) = joblib.load('../../data_proc/data_with_base_masks.pkl')
 
shuffle_index = list(range(data_ES_train[0].shape[0]))
random.shuffle(shuffle_index)
for i in range(len(data_ES_train)):
    data_ES_train[i] = data_ES_train[i][shuffle_index]
labels_value_ES_train = labels_value_ES_train[shuffle_index]
labels_func_ES_train = labels_func_ES_train[shuffle_index]

shuffle_index = list(range(data_ES_train_val[0].shape[0]))
random.shuffle(shuffle_index)
for i in range(len(data_ES_train_val)):
   data_ES_train_val[i] = data_ES_train_val[i][shuffle_index]
labels_value_ES_train_val = labels_value_ES_train_val[shuffle_index]
labels_func_ES_train_val = labels_func_ES_train_val[shuffle_index]

################################################
# DEFINE LOSS FUNCTIONS
################################################

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# currently for theano, would need modifications for tensorflow
def CRPS(y_true, y_pred):
    return K.mean(K.square(T.cumsum(y_pred, axis=-1) - y_true), axis=-1)

################################################
# DEFINE NEURAL NETWORKS
################################################

img_size = 256
nb_slices_z = 12

model_ES_func_regression = Graph()

for i in range(nb_slices_z):
    model_ES_func_regression.add_input(name='input_mask_{}'.format(i), input_shape=(1, img_size, img_size))

for i in range(nb_slices_z):
    model_ES_func_regression.add_node(Flatten(), 
                                       name='apply_mask_{}'.format(i), 
                                       input='input_mask_{}'.format(i))

model_ES_func_regression.add_node(Dense(600, activation='softmax'), 
                                   name='pre-output', 
                                   inputs=['apply_mask_{}'.format(i) for i in range(nb_slices_z)],
                                   merge_mode='concat',
                                   concat_axis=-1)
model_ES_func_regression.add_output(name='output', input='pre-output')

model_ES_func_regression.compile('adam', {'output': CRPS})

################################################
# TRAIN NEURAL NETWORKS
################################################

batch_size = 16
nb_epoch = 200

checkpointer = ModelCheckpoint(filepath='../../model_weights/0-model_ES_mlp_regression.hdf5',
                               verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

train_obj = {}
for i in range(nb_slices_z):
    train_obj['input_img_{}'.format(i)] = data_ES_train[2*i]
    train_obj['input_mask_{}'.format(i)] = data_ES_train[2*i+1]
train_obj['output'] = labels_func_ES_train

train_val_obj = {}
for i in range(nb_slices_z):
    train_val_obj['input_img_{}'.format(i)] = data_ES_train_val[2*i]
    train_val_obj['input_mask_{}'.format(i)] = data_ES_train_val[2*i+1]
train_val_obj['output'] = labels_func_ES_train_val

model_ES_func_regression.fit(train_obj,
                             batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, shuffle=True,
                             validation_data=train_val_obj,
                             callbacks=[checkpointer, earlystopping])

