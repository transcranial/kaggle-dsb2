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
# NORMALIZER
################################################

def apply_per_slice_norm(arr):
    mean = np.mean(arr.ravel())
    std = np.std(arr.ravel())
    if std == 0:
        return np.zeros(arr.shape)
    return (arr - mean) / std

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
 
masked_data_train = []
masked_data_train_val = []
 
shuffle_index = list(range(data_ED_train[0].shape[0]))
random.shuffle(shuffle_index)
for i in range(len(data_ED_train) // 2):
    masked_data_train.append(apply_per_slice_norm(data_ED_train[i*2][shuffle_index] * data_ED_train[i*2+1][shuffle_index]))
labels_value_ED_train = labels_value_ED_train[shuffle_index]
labels_func_ED_train = labels_func_ED_train[shuffle_index]

shuffle_index = list(range(data_ED_train_val[0].shape[0]))
random.shuffle(shuffle_index)
for i in range(len(data_ED_train_val) // 2):
    masked_data_train_val.append(apply_per_slice_norm(data_ED_train_val[i*2][shuffle_index] * data_ED_train_val[i*2+1][shuffle_index]))
labels_value_ED_train_val = labels_value_ED_train_val[shuffle_index]
labels_func_ED_train_val = labels_func_ED_train_val[shuffle_index]

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

model_ED = Graph()

for i in range(nb_slices_z):
    model_ED.add_input(name='input_masked_{}'.format(i), input_shape=(1, img_size, img_size))

model_ED.add_shared_node(Convolution2D(16, 7, 7, init='he_uniform', border_mode='same', dim_ordering='th'),
                         name='shared-conv-1-1', 
                         inputs=['input_masked_{}'.format(i) for i in range(nb_slices_z)])
model_ED.add_shared_node(ELU(), name='shared-conv-1-1-activ', inputs=['shared-conv-1-1'])
model_ED.add_shared_node(MaxPooling2D(pool_size=(4,4), strides=None, border_mode='valid', dim_ordering='th'),
                         name='shared-pool-1', inputs=['shared-conv-1-1-activ'])
model_ED.add_shared_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                         name='shared-conv-2-1', 
                         inputs=['shared-pool-1'])
model_ED.add_shared_node(ELU(), name='shared-conv-2-1-activ', inputs=['shared-conv-2-1'])
model_ED.add_shared_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                         name='shared-pool-2', inputs=['shared-conv-2-1-activ'])
model_ED.add_shared_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                         name='shared-conv-3-1', 
                         inputs=['shared-pool-2'])
model_ED.add_shared_node(ELU(), name='shared-conv-3-1-activ', inputs=['shared-conv-3-1'])
model_ED.add_shared_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                         name='shared-pool-3', inputs=['shared-conv-3-1-activ'])
model_ED.add_shared_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                         name='shared-conv-4-1', 
                         inputs=['shared-pool-3'])
model_ED.add_shared_node(ELU(), name='shared-conv-4-1-activ', inputs=['shared-conv-4-1'])
model_ED.add_shared_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                         name='shared-pool-4', inputs=['shared-conv-4-1-activ'])
model_ED.add_shared_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                         name='shared-conv-5-1', 
                         inputs=['shared-pool-4'])
model_ED.add_shared_node(ELU(), name='shared-conv-5-1-activ', inputs=['shared-conv-5-1'])
model_ED.add_shared_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                         name='shared-pool-5', inputs=['shared-conv-5-1-activ'])
                         
model_ED.add_shared_node(Flatten(), name='flatten', inputs=['shared-pool-5'], merge_mode='concat')

model_ED.add_node(Dense(600, activation='softmax'), 
                  name='pre-output', 
                  input='flatten', 
                  merge_mode='concat')
model_ED.add_output(name='output', input='pre-output')

model_ED.compile('adam', {'output': CRPS})

################################################
# TRAIN NEURAL NETWORKS
################################################

batch_size = 16
nb_epoch = 200

checkpointer = ModelCheckpoint(filepath='../../model_weights/0-2-model_ED_convnet_CRPS.hdf5',
                               verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

train_obj = {}
for i in range(nb_slices_z):
    train_obj['input_masked_{}'.format(i)] = masked_data_train[i]
train_obj['output'] = labels_func_ED_train

train_val_obj = {}
for i in range(nb_slices_z):
    train_val_obj['input_masked_{}'.format(i)] = masked_data_train_val[i]
train_val_obj['output'] = labels_func_ED_train_val

model_ED.fit(train_obj,
             batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, shuffle=True,
             validation_data=train_val_obj,
             callbacks=[checkpointer, earlystopping])

