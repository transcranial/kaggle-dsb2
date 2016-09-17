#!/usr/bin/env python

# run script with THEANO_FLAGS='floatX=float32,device=gpu2,nvcc.fastmath=True'
# also run without buffering: 'python -u ...'
# for example:
# - THEANO_FLAGS='floatX=float32,device=gpu2,nvcc.fastmath=True' python -u sunnybrook_LV_outer.py >sunnybrook_LV_outer.log 2>&1 &

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

import pickle
import numpy as np
from sklearn.externals import joblib

(validation_studies, training_studies,
 data_full_training, labels_full_training,
 data_full_validation, labels_full_validation) = joblib.load('../../data_proc/sunnybrook_ocontour_segmentation_training.pkl')


################################################
# DEFINE 2D LOSS FUNCTIONS
################################################

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=None, keepdims=False))

def binaryCE(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=None, keepdims=False)


################################################
# DEFINE ROTATE90 LAYER
################################################

class Rotate90(Layer):
    def __init__(self, direction='clockwise', **kwargs):
        super(Rotate90, self).__init__(**kwargs)
        self.direction = direction

    def get_output(self, train):
        X = self.get_input(train)
        if self.direction == 'clockwise':
            return X.transpose((0, 2, 1))[:, :, ::-1]
        elif self.direction == 'counterclockwise':
            return X.transpose((0, 2, 1))[:, ::-1, :]
        else:
            raise

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(Rotate90, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


################################################
# DEFINE NEURAL NETWORK
################################################

model = Graph()

model.add_input(name='input', input_shape=(1, 256, 256))

model.add_node(Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-1-1', input='input')
model.add_node(BatchNormalization(), name='conv-1-1-bn', input='conv-1-1')
model.add_node(ELU(), name='conv-1-1-activ', input='conv-1-1-bn')
model.add_node(Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-1-2', input='conv-1-1-activ')
model.add_node(BatchNormalization(), name='conv-1-2-bn', input='conv-1-2')
model.add_node(ELU(), name='conv-1-2-activ', input='conv-1-2-bn')
model.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
               name='pool-1', input='conv-1-2-activ')

model.add_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-2-1', input='pool-1')
model.add_node(BatchNormalization(), name='conv-2-1-bn', input='conv-2-1')
model.add_node(ELU(), name='conv-2-1-activ', input='conv-2-1-bn')
model.add_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-2-2', input='conv-2-1-activ')
model.add_node(BatchNormalization(), name='conv-2-2-bn', input='conv-2-2')
model.add_node(ELU(), name='conv-2-2-activ', input='conv-2-2-bn')
model.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
               name='pool-2', input='conv-2-2-activ')

model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-3-1', input='pool-2')
model.add_node(BatchNormalization(), name='conv-3-1-bn', input='conv-3-1')
model.add_node(ELU(), name='conv-3-1-activ', input='conv-3-1-bn')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-3-2', input='conv-3-1-activ')
model.add_node(BatchNormalization(), name='conv-3-2-bn', input='conv-3-2')
model.add_node(ELU(), name='conv-3-2-activ', input='conv-3-2-bn')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-3-3', input='conv-3-2-activ')
model.add_node(BatchNormalization(), name='conv-3-3-bn', input='conv-3-3')
model.add_node(ELU(), name='conv-3-3-activ', input='conv-3-3-bn')
model.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
               name='pool-3', input='conv-3-3-activ')

model.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-4-1', input='pool-3')
model.add_node(BatchNormalization(), name='conv-4-1-bn', input='conv-4-1')
model.add_node(ELU(), name='conv-4-1-activ', input='conv-4-1-bn')
model.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-4-2', input='conv-4-1-activ')
model.add_node(BatchNormalization(), name='conv-4-2-bn', input='conv-4-2')
model.add_node(ELU(), name='conv-4-2-activ', input='conv-4-2-bn')
model.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-4-3', input='conv-4-2-activ')
model.add_node(BatchNormalization(), name='conv-4-3-bn', input='conv-4-3')
model.add_node(ELU(), name='conv-4-3-activ', input='conv-4-3-bn')
model.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
               name='pool-4', input='conv-4-3-activ')

model.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-5-1', input='pool-4')
model.add_node(BatchNormalization(), name='conv-5-1-bn', input='conv-5-1')
model.add_node(ELU(), name='conv-5-1-activ', input='conv-5-1-bn')
model.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-5-2', input='conv-5-1-activ')
model.add_node(BatchNormalization(), name='conv-5-2-bn', input='conv-5-2')
model.add_node(ELU(), name='conv-5-2-activ', input='conv-5-2-bn')
model.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='conv-5-3', input='conv-5-2-activ')
model.add_node(BatchNormalization(), name='conv-5-3-bn', input='conv-5-3')
model.add_node(ELU(), name='conv-5-3-activ', input='conv-5-3-bn')
model.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
               name='pool-5', input='conv-5-3-activ')

model.add_node(Flatten(), name='flatten', input='pool-5')
model.add_node(Dense(4096, activation='relu'), name='fc-1', input='flatten')
model.add_node(Dropout(0.5), name='dropout-1', input='fc-1')
model.add_node(Dense(4096, activation='relu'), name='fc-2', input='dropout-1')
model.add_node(Dropout(0.5), name='dropout-2', input='fc-2')
model.add_node(Reshape((64, 8, 8)), name='reshape', input='dropout-2')

model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-1', input='reshape')
model.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-1-1', input='unpool-1')
model.add_node(BatchNormalization(), name='deconv-1-1-bn', input='deconv-1-1')
model.add_node(ELU(), name='deconv-1-1-activ', input='deconv-1-1-bn')
model.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-1-2', input='deconv-1-1-activ')
model.add_node(BatchNormalization(), name='deconv-1-2-bn', input='deconv-1-2')
model.add_node(ELU(), name='deconv-1-2-activ', input='deconv-1-2-bn')
model.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-1-3', input='deconv-1-2-activ')
model.add_node(BatchNormalization(), name='deconv-1-3-bn', input='deconv-1-3')
model.add_node(ELU(), name='deconv-1-3-activ', input='deconv-1-3-bn')

model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-2', input='deconv-1-3-activ')
model.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-2-1', input='unpool-2')
model.add_node(BatchNormalization(), name='deconv-2-1-bn', input='deconv-2-1')
model.add_node(ELU(), name='deconv-2-1-activ', input='deconv-2-1-bn')
model.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-2-2', input='deconv-2-1-activ')
model.add_node(BatchNormalization(), name='deconv-2-2-bn', input='deconv-2-2')
model.add_node(ELU(), name='deconv-2-2-activ', input='deconv-2-2-bn')
model.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-2-3', input='deconv-2-2-activ')
model.add_node(BatchNormalization(), name='deconv-2-3-bn', input='deconv-2-3')
model.add_node(ELU(), name='deconv-2-3-activ', input='deconv-2-3-bn')

model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-3', input='deconv-2-3-activ')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-3-1', input='unpool-3')
model.add_node(BatchNormalization(), name='deconv-3-1-bn', input='deconv-3-1')
model.add_node(ELU(), name='deconv-3-1-activ', input='deconv-3-1-bn')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-3-2', input='deconv-3-1-activ')
model.add_node(BatchNormalization(), name='deconv-3-2-bn', input='deconv-3-2')
model.add_node(ELU(), name='deconv-3-2-activ', input='deconv-3-2-bn')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-3-3', input='deconv-3-2-activ')
model.add_node(BatchNormalization(), name='deconv-3-3-bn', input='deconv-3-3')
model.add_node(ELU(), name='deconv-3-3-activ', input='deconv-3-3-bn')

model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-4', input='deconv-3-3-activ')
model.add_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-4-1', input='unpool-4')
model.add_node(BatchNormalization(), name='deconv-4-1-bn', input='deconv-4-1')
model.add_node(ELU(), name='deconv-4-1-activ', input='deconv-4-1-bn')
model.add_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-4-2', input='deconv-4-1-activ')
model.add_node(BatchNormalization(), name='deconv-4-2-bn', input='deconv-4-2')
model.add_node(ELU(), name='deconv-4-2-activ', input='deconv-4-2-bn')

model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-5', input='deconv-4-2-activ')
model.add_node(Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-5-1', input='unpool-5')
model.add_node(BatchNormalization(), name='deconv-5-1-bn', input='deconv-5-1')
model.add_node(ELU(), name='deconv-5-1-activ', input='deconv-5-1-bn')
model.add_node(Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='deconv-5-2', input='deconv-5-1-activ')
model.add_node(BatchNormalization(), name='deconv-5-2-bn', input='deconv-5-2')
model.add_node(ELU(), name='deconv-5-2-activ', input='deconv-5-2-bn')

model.add_node(Convolution2D(1, 1, 1, activation='sigmoid', init='uniform', border_mode='same', dim_ordering='th'),
               name='prob-map', input='deconv-5-2-activ')
model.add_node(Reshape((256, 256)), name='prob-map-reshape', input='prob-map')
model.add_node(Dropout(0.5), name='prob-map-dropout', input='prob-map-reshape')

model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True),
               name='rnn-we', input='prob-map-dropout')
model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True),
               name='rnn-ew', input='prob-map-dropout')
model.add_node(TimeDistributedDense(256, init='uniform', activation='sigmoid'),
               name='rnn-1', inputs=['rnn-we', 'rnn-ew'], merge_mode='concat', concat_axis=-1)

model.add_node(Rotate90(direction='counterclockwise'), name='rotate', input='prob-map-dropout')
model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True),
               name='rnn-ns', input='rotate')
model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True),
               name='rnn-sn', input='rotate')
model.add_node(TimeDistributedDense(256, init='uniform', activation='sigmoid'),
               name='rnn-2-rotated', inputs=['rnn-ns', 'rnn-sn'], merge_mode='concat', concat_axis=-1)
model.add_node(Rotate90(direction='clockwise'), name='rnn-2', input='rnn-2-rotated')

model.add_node(Activation('linear'), name='pre-output', inputs=['rnn-1', 'rnn-2'], merge_mode='ave')
model.add_output(name='output', input='pre-output')

model.compile('adam', {'output': binaryCE})


################################################
# TRAIN NEURAL NETWORK
################################################

batch_size = 32
nb_epoch = 100

checkpointer = ModelCheckpoint(filepath='../../model_weights/sunnybrook_ocontour_segmentation.hdf5',
                               verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)

model.fit({'input': data_full_training, 'output': labels_full_training},
          batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, shuffle=True,
          validation_data={'input': data_full_validation, 'output': labels_full_validation},
          callbacks=[checkpointer, earlystopping])
