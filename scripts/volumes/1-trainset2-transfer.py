#!/usr/bin/env python

# run script with THEANO_FLAGS='floatX=float32,device=gpu2,nvcc.fastmath=True'
# also run without buffering: 'python -u ...'
# for example:
# - THEANO_FLAGS='floatX=float32,device=gpu2,nvcc.fastmath=True' python -u sunnybrook_LV.py >sunnybrook_LV.log 2>&1 &

# trained with keras installed from commit 5bcac37

import pickle
import numpy as np
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
# DEFINE 2D LOSS FUNCTIONS
################################################

def MAE(y_true, y_pred):
    return K.mean(K.abs(K.sum(y_pred, axis=0, keepdims=True) - y_true), axis=None)
    
    
################################################
# DEFINE LAMBDA LAYER FUNCTIONS
################################################

def to_area(X):
    from theano import tensor as T
    return T.sum(T.switch(T.ge(T.flatten(X, outdim=2), 0.5), 1, 0), axis=1, keepdims=True)


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
# DEFINE NEURAL NETWORKS
################################################

# ED

vol_model_ED = Graph()

vol_model_ED.add_input(name='input', input_shape=(1, 96, 96))

vol_model_ED.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-1-1', input='input')
vol_model_ED.add_node(BatchNormalization(), name='conv-1-1-bn', input='conv-1-1')
vol_model_ED.add_node(ELU(), name='conv-1-1-activ', input='conv-1-1-bn')
vol_model_ED.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-1-2', input='conv-1-1-activ')
vol_model_ED.add_node(BatchNormalization(), name='conv-1-2-bn', input='conv-1-2')
vol_model_ED.add_node(ELU(), name='conv-1-2-activ', input='conv-1-2-bn')
vol_model_ED.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                      name='pool-1', input='conv-1-2-activ')

vol_model_ED.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-2-1', input='pool-1')
vol_model_ED.add_node(BatchNormalization(), name='conv-2-1-bn', input='conv-2-1')
vol_model_ED.add_node(ELU(), name='conv-2-1-activ', input='conv-2-1-bn')
vol_model_ED.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-2-2', input='conv-2-1-activ')
vol_model_ED.add_node(BatchNormalization(), name='conv-2-2-bn', input='conv-2-2')
vol_model_ED.add_node(ELU(), name='conv-2-2-activ', input='conv-2-2-bn')
vol_model_ED.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                      name='pool-2', input='conv-2-2-activ')

vol_model_ED.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-3-1', input='pool-2')
vol_model_ED.add_node(BatchNormalization(), name='conv-3-1-bn', input='conv-3-1')
vol_model_ED.add_node(ELU(), name='conv-3-1-activ', input='conv-3-1-bn')
vol_model_ED.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-3-2', input='conv-3-1-activ')
vol_model_ED.add_node(BatchNormalization(), name='conv-3-2-bn', input='conv-3-2')
vol_model_ED.add_node(ELU(), name='conv-3-2-activ', input='conv-3-2-bn')
vol_model_ED.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-3-3', input='conv-3-2-activ')
vol_model_ED.add_node(BatchNormalization(), name='conv-3-3-bn', input='conv-3-3')
vol_model_ED.add_node(ELU(), name='conv-3-3-activ', input='conv-3-3-bn')
vol_model_ED.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                      name='pool-3', input='conv-3-3-activ')

vol_model_ED.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-4-1', input='pool-3')
vol_model_ED.add_node(BatchNormalization(), name='conv-4-1-bn', input='conv-4-1')
vol_model_ED.add_node(ELU(), name='conv-4-1-activ', input='conv-4-1-bn')
vol_model_ED.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-4-2', input='conv-4-1-activ')
vol_model_ED.add_node(BatchNormalization(), name='conv-4-2-bn', input='conv-4-2')
vol_model_ED.add_node(ELU(), name='conv-4-2-activ', input='conv-4-2-bn')
vol_model_ED.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-4-3', input='conv-4-2-activ')
vol_model_ED.add_node(BatchNormalization(), name='conv-4-3-bn', input='conv-4-3')
vol_model_ED.add_node(ELU(), name='conv-4-3-activ', input='conv-4-3-bn')
vol_model_ED.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                      name='pool-4', input='conv-4-3-activ')

vol_model_ED.add_node(Flatten(), name='flatten', input='pool-4')
vol_model_ED.add_node(Dense(2304, activation='relu'), name='fc-1', input='flatten')
vol_model_ED.add_node(Dropout(0.5), name='dropout-1', input='fc-1')
vol_model_ED.add_node(Dense(2304, activation='relu'), name='fc-2', input='dropout-1')
vol_model_ED.add_node(Dropout(0.5), name='dropout-2', input='fc-2')
vol_model_ED.add_node(Reshape((64, 6, 6)), name='reshape', input='dropout-2')

vol_model_ED.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-1', input='reshape')
vol_model_ED.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-1-1', input='unpool-1')
vol_model_ED.add_node(BatchNormalization(), name='deconv-1-1-bn', input='deconv-1-1')
vol_model_ED.add_node(ELU(), name='deconv-1-1-activ', input='deconv-1-1-bn')
vol_model_ED.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-1-2', input='deconv-1-1-activ')
vol_model_ED.add_node(BatchNormalization(), name='deconv-1-2-bn', input='deconv-1-2')
vol_model_ED.add_node(ELU(), name='deconv-1-2-activ', input='deconv-1-2-bn')
vol_model_ED.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-1-3', input='deconv-1-2-activ')
vol_model_ED.add_node(BatchNormalization(), name='deconv-1-3-bn', input='deconv-1-3')
vol_model_ED.add_node(ELU(), name='deconv-1-3-activ', input='deconv-1-3-bn')

vol_model_ED.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-2', input='deconv-1-3-activ')
vol_model_ED.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-2-1', input='unpool-2')
vol_model_ED.add_node(BatchNormalization(), name='deconv-2-1-bn', input='deconv-2-1')
vol_model_ED.add_node(ELU(), name='deconv-2-1-activ', input='deconv-2-1-bn')
vol_model_ED.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-2-2', input='deconv-2-1-activ')
vol_model_ED.add_node(BatchNormalization(), name='deconv-2-2-bn', input='deconv-2-2')
vol_model_ED.add_node(ELU(), name='deconv-2-2-activ', input='deconv-2-2-bn')
vol_model_ED.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-2-3', input='deconv-2-2-activ')
vol_model_ED.add_node(BatchNormalization(), name='deconv-2-3-bn', input='deconv-2-3')
vol_model_ED.add_node(ELU(), name='deconv-2-3-activ', input='deconv-2-3-bn')

vol_model_ED.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-3', input='deconv-2-3-activ')
vol_model_ED.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-3-1', input='unpool-3')
vol_model_ED.add_node(BatchNormalization(), name='deconv-3-1-bn', input='deconv-3-1')
vol_model_ED.add_node(ELU(), name='deconv-3-1-activ', input='deconv-3-1-bn')
vol_model_ED.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-3-2', input='deconv-3-1-activ')
vol_model_ED.add_node(BatchNormalization(), name='deconv-3-2-bn', input='deconv-3-2')
vol_model_ED.add_node(ELU(), name='deconv-3-2-activ', input='deconv-3-2-bn')

vol_model_ED.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-4', input='deconv-3-2-activ')
vol_model_ED.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-4-1', input='unpool-4')
vol_model_ED.add_node(BatchNormalization(), name='deconv-4-1-bn', input='deconv-4-1')
vol_model_ED.add_node(ELU(), name='deconv-4-1-activ', input='deconv-4-1-bn')
vol_model_ED.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-4-2', input='deconv-4-1-activ')
vol_model_ED.add_node(BatchNormalization(), name='deconv-4-2-bn', input='deconv-4-2')
vol_model_ED.add_node(ELU(), name='deconv-4-2-activ', input='deconv-4-2-bn')

vol_model_ED.add_node(Convolution2D(1, 1, 1, activation='sigmoid', init='uniform', border_mode='same', dim_ordering='th'),
                      name='prob-map', input='deconv-4-2-activ')
vol_model_ED.add_node(Reshape((96, 96)), name='prob-map-reshape', input='prob-map')
vol_model_ED.add_node(Dropout(0.5), name='prob-map-dropout', input='prob-map-reshape')

vol_model_ED.add_node(GRU(96, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True),
                      name='rnn-we', input='prob-map-dropout')
vol_model_ED.add_node(GRU(96, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True),
                      name='rnn-ew', input='prob-map-dropout')
vol_model_ED.add_node(TimeDistributedDense(96, init='uniform', activation='sigmoid'),
                      name='rnn-1', inputs=['rnn-we', 'rnn-ew'], merge_mode='concat', concat_axis=-1)

vol_model_ED.add_node(Rotate90(direction='counterclockwise'), name='rotate', input='prob-map-dropout')
vol_model_ED.add_node(GRU(96, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True),
                      name='rnn-ns', input='rotate')
vol_model_ED.add_node(GRU(96, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True),
                      name='rnn-sn', input='rotate')
vol_model_ED.add_node(TimeDistributedDense(96, init='uniform', activation='sigmoid'),
                      name='rnn-2-rotated', inputs=['rnn-ns', 'rnn-sn'], merge_mode='concat', concat_axis=-1)
vol_model_ED.add_node(Rotate90(direction='clockwise'), name='rnn-2', input='rnn-2-rotated')

vol_model_ED.add_node(Activation('linear'), name='pre-output', inputs=['rnn-1', 'rnn-2'], merge_mode='mul')

vol_model_ED.load_weights('../../model_weights/weights_trainset2_local.hdf5')

# freeze early layers
#for name, layer in vol_model_ED.nodes.items():
#    if name not in ['prob-map', 'rnn-we', 'rnn-ew', 'rnn-1', 'rnn-ns', 'rnn-sn', 'rnn-2-rotated']:
#        layer.trainable = False
        
# additional layers on top of segmentation layers

vol_model_ED.add_input(name='scaling', input_shape=(1,))
vol_model_ED.add_node(Lambda(to_area, output_shape=(1,)),
                      name='area', input='pre-output')
vol_model_ED.add_node(Activation('linear'),
                      name='area-scaled', inputs=['area', 'scaling'], merge_mode='mul')
vol_model_ED.add_output(name='volume', input='area-scaled')

vol_model_ED.compile('adam', {'volume': MAE})


# ES

vol_model_ES = Graph()

vol_model_ES.add_input(name='input', input_shape=(1, 96, 96))

vol_model_ES.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-1-1', input='input')
vol_model_ES.add_node(BatchNormalization(), name='conv-1-1-bn', input='conv-1-1')
vol_model_ES.add_node(ELU(), name='conv-1-1-activ', input='conv-1-1-bn')
vol_model_ES.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-1-2', input='conv-1-1-activ')
vol_model_ES.add_node(BatchNormalization(), name='conv-1-2-bn', input='conv-1-2')
vol_model_ES.add_node(ELU(), name='conv-1-2-activ', input='conv-1-2-bn')
vol_model_ES.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                      name='pool-1', input='conv-1-2-activ')

vol_model_ES.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-2-1', input='pool-1')
vol_model_ES.add_node(BatchNormalization(), name='conv-2-1-bn', input='conv-2-1')
vol_model_ES.add_node(ELU(), name='conv-2-1-activ', input='conv-2-1-bn')
vol_model_ES.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-2-2', input='conv-2-1-activ')
vol_model_ES.add_node(BatchNormalization(), name='conv-2-2-bn', input='conv-2-2')
vol_model_ES.add_node(ELU(), name='conv-2-2-activ', input='conv-2-2-bn')
vol_model_ES.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                      name='pool-2', input='conv-2-2-activ')

vol_model_ES.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-3-1', input='pool-2')
vol_model_ES.add_node(BatchNormalization(), name='conv-3-1-bn', input='conv-3-1')
vol_model_ES.add_node(ELU(), name='conv-3-1-activ', input='conv-3-1-bn')
vol_model_ES.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-3-2', input='conv-3-1-activ')
vol_model_ES.add_node(BatchNormalization(), name='conv-3-2-bn', input='conv-3-2')
vol_model_ES.add_node(ELU(), name='conv-3-2-activ', input='conv-3-2-bn')
vol_model_ES.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-3-3', input='conv-3-2-activ')
vol_model_ES.add_node(BatchNormalization(), name='conv-3-3-bn', input='conv-3-3')
vol_model_ES.add_node(ELU(), name='conv-3-3-activ', input='conv-3-3-bn')
vol_model_ES.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                      name='pool-3', input='conv-3-3-activ')

vol_model_ES.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-4-1', input='pool-3')
vol_model_ES.add_node(BatchNormalization(), name='conv-4-1-bn', input='conv-4-1')
vol_model_ES.add_node(ELU(), name='conv-4-1-activ', input='conv-4-1-bn')
vol_model_ES.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-4-2', input='conv-4-1-activ')
vol_model_ES.add_node(BatchNormalization(), name='conv-4-2-bn', input='conv-4-2')
vol_model_ES.add_node(ELU(), name='conv-4-2-activ', input='conv-4-2-bn')
vol_model_ES.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='conv-4-3', input='conv-4-2-activ')
vol_model_ES.add_node(BatchNormalization(), name='conv-4-3-bn', input='conv-4-3')
vol_model_ES.add_node(ELU(), name='conv-4-3-activ', input='conv-4-3-bn')
vol_model_ES.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
                      name='pool-4', input='conv-4-3-activ')

vol_model_ES.add_node(Flatten(), name='flatten', input='pool-4')
vol_model_ES.add_node(Dense(2304, activation='relu'), name='fc-1', input='flatten')
vol_model_ES.add_node(Dropout(0.5), name='dropout-1', input='fc-1')
vol_model_ES.add_node(Dense(2304, activation='relu'), name='fc-2', input='dropout-1')
vol_model_ES.add_node(Dropout(0.5), name='dropout-2', input='fc-2')
vol_model_ES.add_node(Reshape((64, 6, 6)), name='reshape', input='dropout-2')

vol_model_ES.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-1', input='reshape')
vol_model_ES.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-1-1', input='unpool-1')
vol_model_ES.add_node(BatchNormalization(), name='deconv-1-1-bn', input='deconv-1-1')
vol_model_ES.add_node(ELU(), name='deconv-1-1-activ', input='deconv-1-1-bn')
vol_model_ES.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-1-2', input='deconv-1-1-activ')
vol_model_ES.add_node(BatchNormalization(), name='deconv-1-2-bn', input='deconv-1-2')
vol_model_ES.add_node(ELU(), name='deconv-1-2-activ', input='deconv-1-2-bn')
vol_model_ES.add_node(Convolution2D(512, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-1-3', input='deconv-1-2-activ')
vol_model_ES.add_node(BatchNormalization(), name='deconv-1-3-bn', input='deconv-1-3')
vol_model_ES.add_node(ELU(), name='deconv-1-3-activ', input='deconv-1-3-bn')

vol_model_ES.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-2', input='deconv-1-3-activ')
vol_model_ES.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-2-1', input='unpool-2')
vol_model_ES.add_node(BatchNormalization(), name='deconv-2-1-bn', input='deconv-2-1')
vol_model_ES.add_node(ELU(), name='deconv-2-1-activ', input='deconv-2-1-bn')
vol_model_ES.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-2-2', input='deconv-2-1-activ')
vol_model_ES.add_node(BatchNormalization(), name='deconv-2-2-bn', input='deconv-2-2')
vol_model_ES.add_node(ELU(), name='deconv-2-2-activ', input='deconv-2-2-bn')
vol_model_ES.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-2-3', input='deconv-2-2-activ')
vol_model_ES.add_node(BatchNormalization(), name='deconv-2-3-bn', input='deconv-2-3')
vol_model_ES.add_node(ELU(), name='deconv-2-3-activ', input='deconv-2-3-bn')

vol_model_ES.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-3', input='deconv-2-3-activ')
vol_model_ES.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-3-1', input='unpool-3')
vol_model_ES.add_node(BatchNormalization(), name='deconv-3-1-bn', input='deconv-3-1')
vol_model_ES.add_node(ELU(), name='deconv-3-1-activ', input='deconv-3-1-bn')
vol_model_ES.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-3-2', input='deconv-3-1-activ')
vol_model_ES.add_node(BatchNormalization(), name='deconv-3-2-bn', input='deconv-3-2')
vol_model_ES.add_node(ELU(), name='deconv-3-2-activ', input='deconv-3-2-bn')

vol_model_ES.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='unpool-4', input='deconv-3-2-activ')
vol_model_ES.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-4-1', input='unpool-4')
vol_model_ES.add_node(BatchNormalization(), name='deconv-4-1-bn', input='deconv-4-1')
vol_model_ES.add_node(ELU(), name='deconv-4-1-activ', input='deconv-4-1-bn')
vol_model_ES.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
                      name='deconv-4-2', input='deconv-4-1-activ')
vol_model_ES.add_node(BatchNormalization(), name='deconv-4-2-bn', input='deconv-4-2')
vol_model_ES.add_node(ELU(), name='deconv-4-2-activ', input='deconv-4-2-bn')

vol_model_ES.add_node(Convolution2D(1, 1, 1, activation='sigmoid', init='uniform', border_mode='same', dim_ordering='th'),
                      name='prob-map', input='deconv-4-2-activ')
vol_model_ES.add_node(Reshape((96, 96)), name='prob-map-reshape', input='prob-map')
vol_model_ES.add_node(Dropout(0.5), name='prob-map-dropout', input='prob-map-reshape')

vol_model_ES.add_node(GRU(96, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True),
                      name='rnn-we', input='prob-map-dropout')
vol_model_ES.add_node(GRU(96, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True),
                      name='rnn-ew', input='prob-map-dropout')
vol_model_ES.add_node(TimeDistributedDense(96, init='uniform', activation='sigmoid'),
                      name='rnn-1', inputs=['rnn-we', 'rnn-ew'], merge_mode='concat', concat_axis=-1)

vol_model_ES.add_node(Rotate90(direction='counterclockwise'), name='rotate', input='prob-map-dropout')
vol_model_ES.add_node(GRU(96, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True),
                      name='rnn-ns', input='rotate')
vol_model_ES.add_node(GRU(96, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True),
                      name='rnn-sn', input='rotate')
vol_model_ES.add_node(TimeDistributedDense(96, init='uniform', activation='sigmoid'),
                      name='rnn-2-rotated', inputs=['rnn-ns', 'rnn-sn'], merge_mode='concat', concat_axis=-1)
vol_model_ES.add_node(Rotate90(direction='clockwise'), name='rnn-2', input='rnn-2-rotated')

vol_model_ES.add_node(Activation('linear'), name='pre-output', inputs=['rnn-1', 'rnn-2'], merge_mode='mul')

vol_model_ES.load_weights('../../model_weights/weights_trainset2_local.hdf5')

# freeze early layers
#for name, layer in vol_model_ES.nodes.items():
#    if name not in ['prob-map', 'rnn-we', 'rnn-ew', 'rnn-1', 'rnn-ns', 'rnn-sn', 'rnn-2-rotated']:
#        layer.trainable = False
        
# additional layers on top of segmentation layers

vol_model_ES.add_input(name='scaling', input_shape=(1,))
vol_model_ES.add_node(Lambda(to_area, output_shape=(1,)),
                      name='area', input='pre-output')
vol_model_ES.add_node(Activation('linear'),
                      name='area-scaled', inputs=['area', 'scaling'], merge_mode='mul')
vol_model_ES.add_output(name='volume', input='area-scaled')

vol_model_ES.compile('adam', {'volume': MAE})


################################################
# TRAIN NEURAL NETWORK
################################################

# load data
with open('../../data_proc/data_localized_transfer_learning.pkl', 'rb') as f:
    (pts_train, pts_train_val, 
     data_ED_train_batches, data_ED_train_val_batches, 
     data_ES_train_batches, data_ES_train_val_batches,
     data_ED_val_batches, data_ES_val_batches, pt_indices_val) = pickle.load(f)

for batch in data_ED_train_batches:
    batch['volume'] = np.tile(np.array([batch['volume']]), (batch['input'].shape[0], 1))
for batch in data_ED_train_val_batches:
    batch['volume'] = np.tile(np.array([batch['volume']]), (batch['input'].shape[0], 1))
for batch in data_ES_train_batches:
    batch['volume'] = np.tile(np.array([batch['volume']]), (batch['input'].shape[0], 1))
for batch in data_ES_train_val_batches:
    batch['volume'] = np.tile(np.array([batch['volume']]), (batch['input'].shape[0], 1))

# train ED

nb_epochs = 100
train_rand_index = list(range(475))
train_val_rand_index = list(range(25))

loss_best = 1e6
val_loss_best = 1e6
patience = 0

for epoch in range(nb_epochs):
    random.shuffle(train_rand_index)
    random.shuffle(train_val_rand_index)
    loss_tot = 0
    val_loss_tot = 0
    loss_avg = 1e6
    val_loss_avg = 1e6
    
    print('[ED] epoch {}...'.format(epoch))
    for m, idx in enumerate(train_rand_index):
        batch = data_ED_train_batches[idx]
        outs = vol_model_ED.train_on_batch(batch)
        loss_tot += outs[0]
        loss_avg = loss_tot / (m+1)
    print('    training loss: {}'.format(loss_avg))
    for m, idx in enumerate(train_val_rand_index):
        batch = data_ED_train_val_batches[idx]
        outs = vol_model_ED.test_on_batch(batch)
        val_loss_tot += outs[0]
        val_loss_avg = val_loss_tot / (m+1)
    print('    validation loss: {}'.format(val_loss_avg))
    
    if val_loss_avg < val_loss_best:
        print('  ~~~ saving weights to ../../model_weights/weights_trainset2_local_ED_mae_transfer.hdf5 ~~~')
        vol_model_ED.save_weights('../../model_weights/weights_trainset2_local_ED_mae_transfer.hdf5', 
                                  overwrite=True)
        val_loss_best = val_loss_avg
        patience = 0
    else:
        patience += 1
    
    if patience > 10:
        print('~~~~~~ EARLY STOPPING ~~~~~~')
        break
        
# train ES

nb_epochs = 100
train_rand_index = list(range(475))
train_val_rand_index = list(range(25))

loss_best = 1e6
val_loss_best = 1e6
patience = 0

for epoch in range(nb_epochs):
    random.shuffle(train_rand_index)
    random.shuffle(train_val_rand_index)
    loss_tot = 0
    val_loss_tot = 0
    loss_avg = 1e6
    val_loss_avg = 1e6
    
    print('[ES] epoch {}...'.format(epoch))
    for m, idx in enumerate(train_rand_index):
        batch = data_ES_train_batches[idx]
        outs = vol_model_ES.train_on_batch(batch)
        loss_tot += outs[0]
        loss_avg = loss_tot / (m+1)
    print('    training loss: {}'.format(loss_avg))
    for m, idx in enumerate(train_val_rand_index):
        batch = data_ES_train_val_batches[idx]
        outs = vol_model_ES.test_on_batch(batch)
        val_loss_tot += outs[0]
        val_loss_avg = val_loss_tot / (m+1)
    print('    validation loss: {}'.format(val_loss_avg))
    
    if val_loss_avg < val_loss_best:
        print('  ~~~ saving weights to ../../model_weights/weights_trainset2_local_ES_mae_transfer.hdf5 ~~~')
        vol_model_ES.save_weights('../../model_weights/weights_trainset2_local_ES_mae_transfer.hdf5', 
                                  overwrite=True)
        val_loss_best = val_loss_avg
        patience = 0
    else:
        patience += 1
    
    if patience > 10:
        print('~~~~~~ EARLY STOPPING ~~~~~~')
        break
