#!/usr/bin/env python

# run script with THEANO_FLAGS='floatX=float32,device=gpu1,nvcc.fastmath=True'
# also run without buffering: 'python -u ...'
# for example:
# - THEANO_FLAGS='floatX=float32,device=gpu1,nvcc.fastmath=True' python -u ES_detection_training.py >ES_detection_training.log 2>&1 &

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
from keras import backend as K


# for preventing python max recursion limit error
import sys
sys.setrecursionlimit(50000)


################################################
# IMPORT 3D LAYERS
# modified from https://github.com/fchollet/keras/pull/718 by @MinhazPalasara
################################################

from convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D


################################################
# LOAD DATA
################################################

import pickle
import numpy as np
from sklearn.externals import joblib

(study_numbers_train, study_numbers_train_val,
 data_train, labels_train,
 data_train_val, labels_train_val) = joblib.load('../../data_proc/ES_detection_training.pkl')


################################################
# DEFINE LOSS FUNCTION
################################################

def L2norm(y_true, y_pred):
    return K.square(y_pred * 32 - y_true)


################################################
# DEFINE NEURAL NETWORK
################################################

es_frame_model = Sequential()
es_frame_model.add(Convolution3D(16, 3, 3, 3, init='he_uniform', border_mode='same', input_shape=(1, 32, 64, 64)))
es_frame_model.add(BatchNormalization())
es_frame_model.add(ELU())
es_frame_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, ignore_border=False))
es_frame_model.add(Convolution3D(32, 3, 3, 3, init='he_uniform', border_mode='same'))
es_frame_model.add(BatchNormalization())
es_frame_model.add(ELU())
es_frame_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, ignore_border=False))
es_frame_model.add(Convolution3D(64, 3, 3, 3, init='he_uniform', border_mode='same'))
es_frame_model.add(BatchNormalization())
es_frame_model.add(ELU())
es_frame_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, ignore_border=False))
es_frame_model.add(Convolution3D(128, 3, 3, 3, init='he_uniform', border_mode='same'))
es_frame_model.add(BatchNormalization())
es_frame_model.add(ELU())
es_frame_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, ignore_border=False))
es_frame_model.add(Flatten())
es_frame_model.add(Dense(4096, activation='relu'))
es_frame_model.add(Dense(1))
es_frame_model.add(Activation('sigmoid'))

es_frame_model.compile(loss=L2norm, optimizer='adam')


################################################
# TRAIN NEURAL NETWORK
################################################

batch_size = 32
nb_epoch = 100

checkpointer = ModelCheckpoint(filepath='../../model_weights/ES_frame_detection.hdf5',
                               verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

es_frame_model.fit(data_train, labels_train,
                   batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=2, shuffle=True,
                   validation_data=(data_train_val, labels_train_val),
                   callbacks=[checkpointer, earlystopping])
