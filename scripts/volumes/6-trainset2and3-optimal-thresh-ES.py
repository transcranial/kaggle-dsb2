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

# for preventing python max recursion limit error
import sys
sys.setrecursionlimit(50000)


################################################
# LOAD DATA
################################################

(_, _, _, _, 
 data_optimal_thresh_ES_train, labels_optimal_thresh_ES_train, 
 data_optimal_thresh_ES_train_val, labels_optimal_thresh_ES_train_val) = joblib.load('../../data_proc/trainset2and3_optimal_thresh.pkl')
    

################################################
# DEFINE NEURAL NETWORKS
################################################

thresh_optimizer_ES = Sequential()

thresh_optimizer_ES.add(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th', 
                                      input_shape=(1, 96, 96)))
thresh_optimizer_ES.add(BatchNormalization())
thresh_optimizer_ES.add(ELU())
thresh_optimizer_ES.add(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ES.add(BatchNormalization())
thresh_optimizer_ES.add(ELU())
thresh_optimizer_ES.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid', dim_ordering='th'))

thresh_optimizer_ES.add(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ES.add(BatchNormalization())
thresh_optimizer_ES.add(ELU())
thresh_optimizer_ES.add(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ES.add(BatchNormalization())
thresh_optimizer_ES.add(ELU())
thresh_optimizer_ES.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid', dim_ordering='th'))

thresh_optimizer_ES.add(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ES.add(BatchNormalization())
thresh_optimizer_ES.add(ELU())
thresh_optimizer_ES.add(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ES.add(BatchNormalization())
thresh_optimizer_ES.add(ELU())
thresh_optimizer_ES.add(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ES.add(BatchNormalization())
thresh_optimizer_ES.add(ELU())
thresh_optimizer_ES.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid', dim_ordering='th'))

thresh_optimizer_ES.add(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ES.add(BatchNormalization())
thresh_optimizer_ES.add(ELU())
thresh_optimizer_ES.add(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ES.add(BatchNormalization())
thresh_optimizer_ES.add(ELU())
thresh_optimizer_ES.add(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ES.add(BatchNormalization())
thresh_optimizer_ES.add(ELU())
thresh_optimizer_ES.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid', dim_ordering='th'))

thresh_optimizer_ES.add(Flatten())
thresh_optimizer_ES.add(Dense(1024, activation='relu'))
thresh_optimizer_ES.add(Dropout(0.5))
thresh_optimizer_ES.add(Dense(1, activation='linear'))

thresh_optimizer_ES.compile(optimizer='adam', loss='mse')


################################################
# TRAIN NEURAL NETWORK
################################################

batch_size = 64
nb_epoch = 100

checkpointer = ModelCheckpoint(filepath='../../model_weights/weights_trainset2and3_thresh_optimizer_ES.hdf5',
                               verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

thresh_optimizer_ES.fit(data_optimal_thresh_ES_train, labels_optimal_thresh_ES_train,
                        batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, shuffle=True, show_accuracy=False,
                        validation_data=(data_optimal_thresh_ES_train_val, labels_optimal_thresh_ES_train_val),
                        callbacks=[checkpointer, earlystopping])