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

(data_ED_train, data_ED_train_val, _, _) = joblib.load('../../data_proc/trainset2_data_for_optimal_threshold_net.pkl')
    

################################################
# DEFINE NEURAL NETWORKS
################################################

thresh_optimizer_ED = Sequential()

thresh_optimizer_ED.add(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th', 
                                      input_shape=(1, 96, 96)))
thresh_optimizer_ED.add(BatchNormalization())
thresh_optimizer_ED.add(ELU())
thresh_optimizer_ED.add(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ED.add(BatchNormalization())
thresh_optimizer_ED.add(ELU())
thresh_optimizer_ED.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid', dim_ordering='th'))

thresh_optimizer_ED.add(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ED.add(BatchNormalization())
thresh_optimizer_ED.add(ELU())
thresh_optimizer_ED.add(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ED.add(BatchNormalization())
thresh_optimizer_ED.add(ELU())
thresh_optimizer_ED.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid', dim_ordering='th'))

thresh_optimizer_ED.add(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ED.add(BatchNormalization())
thresh_optimizer_ED.add(ELU())
thresh_optimizer_ED.add(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ED.add(BatchNormalization())
thresh_optimizer_ED.add(ELU())
thresh_optimizer_ED.add(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ED.add(BatchNormalization())
thresh_optimizer_ED.add(ELU())
thresh_optimizer_ED.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid', dim_ordering='th'))

thresh_optimizer_ED.add(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ED.add(BatchNormalization())
thresh_optimizer_ED.add(ELU())
thresh_optimizer_ED.add(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ED.add(BatchNormalization())
thresh_optimizer_ED.add(ELU())
thresh_optimizer_ED.add(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'))
thresh_optimizer_ED.add(BatchNormalization())
thresh_optimizer_ED.add(ELU())
thresh_optimizer_ED.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid', dim_ordering='th'))

thresh_optimizer_ED.add(Flatten())
thresh_optimizer_ED.add(Dense(1024, activation='relu'))
thresh_optimizer_ED.add(Dropout(0.5))
thresh_optimizer_ED.add(Dense(1, activation='linear'))

thresh_optimizer_ED.compile(optimizer='adam', loss='mae')


################################################
# TRAIN NEURAL NETWORK
################################################

batch_size = 64
nb_epoch = 100

checkpointer = ModelCheckpoint(filepath='../../model_weights/weights_trainset2_thresh_optimizer_ED.hdf5',
                               verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

thresh_optimizer_ED.fit(data_ED_train['imgs'], data_ED_train['opt_intensity_threshold'],
                        batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, shuffle=True,
                        validation_data=(data_ED_train_val['imgs'], data_ED_train_val['opt_intensity_threshold']),
                        callbacks=[checkpointer, earlystopping])