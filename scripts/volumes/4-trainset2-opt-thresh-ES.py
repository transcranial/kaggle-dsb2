#!/usr/bin/env python

# run script with THEANO_FLAGS='floatX=float32,device=gpu2,nvcc.fastmath=True'
# also run without buffering: 'python -u ...'
# for example:
# - THEANO_FLAGS='floatX=float32,device=gpu2,nvcc.fastmath=True' python -u sunnybrook_LV.py >sunnybrook_LV.log 2>&1 &

# trained with keras installed from commit 02d5f72

import pickle
import numpy as np
import h5py
from sklearn.externals import joblib
import random
from keras.models import Sequential, Graph
from keras.layers.core import Layer, Activation, Dense, Dropout, Flatten, Merge, Reshape, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ParametricSoftplus, ELU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

# for preventing python max recursion limit error
import sys
sys.setrecursionlimit(50000)


################################################
# LOAD DATA
################################################

(_, _, data_ES_train, data_ES_train_val) = joblib.load('../../data_proc/trainset2_data_for_optimal_threshold_net.pkl')
    

################################################
# DEFINE NEURAL NETWORKS
################################################

opt_thresholds_ES = Graph()

opt_thresholds_ES.add_input(name='imgs_3d', input_shape=(1, 24, 96, 96))

opt_thresholds_ES.add_node(Convolution3D(32, 3, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                           name='conv-1-1', input='imgs_3d')
opt_thresholds_ES.add_node(BatchNormalization(), name='conv-1-1-bn', input='conv-1-1')
opt_thresholds_ES.add_node(ELU(), name='conv-1-1-activ', input='conv-1-1-bn')
opt_thresholds_ES.add_node(Convolution3D(32, 3, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                           name='conv-1-2', input='conv-1-1-activ')
opt_thresholds_ES.add_node(BatchNormalization(), name='conv-1-2-bn', input='conv-1-2')
opt_thresholds_ES.add_node(ELU(), name='conv-1-2-activ', input='conv-1-2-bn')
opt_thresholds_ES.add_node(MaxPooling3D(pool_size=(2, 2, 2), border_mode='valid', dim_ordering='th'),
                           name='pool-1', input='conv-1-2-activ')

opt_thresholds_ES.add_node(Convolution3D(64, 3, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                           name='conv-2-1', input='pool-1')
opt_thresholds_ES.add_node(BatchNormalization(), name='conv-2-1-bn', input='conv-2-1')
opt_thresholds_ES.add_node(ELU(), name='conv-2-1-activ', input='conv-2-1-bn')
opt_thresholds_ES.add_node(Convolution3D(64, 3, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                           name='conv-2-2', input='conv-2-1-activ')
opt_thresholds_ES.add_node(BatchNormalization(), name='conv-2-2-bn', input='conv-2-2')
opt_thresholds_ES.add_node(ELU(), name='conv-2-2-activ', input='conv-2-2-bn')
opt_thresholds_ES.add_node(MaxPooling3D(pool_size=(2, 2, 2), border_mode='valid', dim_ordering='th'),
                           name='pool-2', input='conv-2-2-activ')

opt_thresholds_ES.add_node(Convolution3D(128, 3, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                           name='conv-3-1', input='pool-2')
opt_thresholds_ES.add_node(BatchNormalization(), name='conv-3-1-bn', input='conv-3-1')
opt_thresholds_ES.add_node(ELU(), name='conv-3-1-activ', input='conv-3-1-bn')
opt_thresholds_ES.add_node(Convolution3D(128, 3, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                           name='conv-3-2', input='conv-3-1-activ')
opt_thresholds_ES.add_node(BatchNormalization(), name='conv-3-2-bn', input='conv-3-2')
opt_thresholds_ES.add_node(ELU(), name='conv-3-2-activ', input='conv-3-2-bn')
opt_thresholds_ES.add_node(Convolution3D(128, 3, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                           name='conv-3-3', input='conv-3-2-activ')
opt_thresholds_ES.add_node(BatchNormalization(), name='conv-3-3-bn', input='conv-3-3')
opt_thresholds_ES.add_node(ELU(), name='conv-3-3-activ', input='conv-3-3-bn')
opt_thresholds_ES.add_node(MaxPooling3D(pool_size=(2, 2, 2), border_mode='valid', dim_ordering='th'),
                           name='pool-3', input='conv-3-3-activ')

opt_thresholds_ES.add_node(Flatten(), name='flatten', input='pool-3')
opt_thresholds_ES.add_node(Dense(1024, activation='relu'), name='fc-1', input='flatten')
opt_thresholds_ES.add_node(Dropout(0.5), name='dropout', input='fc-1')
opt_thresholds_ES.add_node(Dense(1, activation='linear'), name='fc-2', input='dropout')

opt_thresholds_ES.add_output(name='opt_intensity_threshold', input='fc-2')

opt_thresholds_ES.compile('adam', {'opt_intensity_threshold': 'mse'})


################################################
# TRAIN NEURAL NETWORK
################################################

batch_size = 12
nb_epoch = 50

checkpointer = ModelCheckpoint(filepath='../../model_weights/weights_trainset2_thresh_optimizer_ES.hdf5',
                               verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

opt_thresholds_ES.fit(data_ES_train,
                      batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, shuffle=True,
                      validation_data=data_ES_train_val,
                      callbacks=[checkpointer, earlystopping])