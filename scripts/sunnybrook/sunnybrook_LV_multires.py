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

################################################
# LOAD DATA
################################################

import pickle
import numpy as np
from sklearn.externals import joblib

(validation_studies, training_studies,
 data_full_training, data_half_training, data_quarter_training, labels_full_training,
 data_full_validation, data_half_validation, data_quarter_validation, labels_full_validation) = \
    joblib.load('../../data_proc/sunnybrook_segmentation_training_multires.pkl')

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

model.add_input(name='input_full', input_shape=(1, 256, 256))
model.add_input(name='input_half', input_shape=(1, 128, 128))
model.add_input(name='input_quarter', input_shape=(1, 64, 64))

model.add_node(Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='full-conv-1', input='input_full')
model.add_node(BatchNormalization(), name='full-bn-1', input='full-conv-1')
model.add_node(ELU(), name='full-activ-1', input='full-bn-1')
model.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
               name='full-pool-1', input='full-activ-1')
model.add_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='full-conv-2', input='full-pool-1')
model.add_node(BatchNormalization(), name='full-bn-2', input='full-conv-2')
model.add_node(ELU(), name='full-activ-2', input='full-bn-2')
model.add_node(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'),
               name='full-pool-2', input='full-activ-2')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='full-conv-3', input='full-pool-2')
model.add_node(BatchNormalization(), name='full-bn-3', input='full-conv-3')
model.add_node(ELU(), name='full-activ-3', input='full-bn-3')
model.add_node(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'),
               name='full-pool-3', input='full-activ-3')
model.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='full-conv-4', input='full-pool-3')
model.add_node(BatchNormalization(), name='full-bn-4', input='full-conv-4')
model.add_node(ELU(), name='full-activ-4', input='full-bn-4')
model.add_node(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'),
               name='full-pool-4', input='full-activ-4')
model.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='full-conv-5', input='full-pool-4')
model.add_node(BatchNormalization(), name='full-bn-5', input='full-conv-5')
model.add_node(ELU(), name='full-activ-5', input='full-bn-5')
model.add_node(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'),
               name='full-pool-5', input='full-activ-5')
model.add_node(Flatten(), name='full-flatten', input='full-pool-5')
model.add_node(Dense(4096, activation='relu'), name='full-fc-1', input='full-flatten')
model.add_node(Dropout(0.5), name='full-dropout-1', input='full-fc-1')
model.add_node(Reshape((64, 8, 8)), name='full-reshape', input='full-dropout-1')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='full-unpool-1', input='full-reshape')
model.add_node(Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='full-deconv-1', input='full-unpool-1')
model.add_node(BatchNormalization(), name='full-bn-6', input='full-deconv-1')
model.add_node(ELU(), name='full-activ-6', input='full-bn-6')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='full-unpool-2', input='full-activ-6')
model.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='full-deconv-2', input='full-unpool-2')
model.add_node(BatchNormalization(), name='full-bn-7', input='full-deconv-2')
model.add_node(ELU(), name='full-activ-7', input='full-bn-7')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='full-unpool-3', input='full-activ-7')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='full-deconv-3', input='full-unpool-3')
model.add_node(BatchNormalization(), name='full-bn-8', input='full-deconv-3')
model.add_node(ELU(), name='full-activ-8', input='full-bn-8')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='full-unpool-4', input='full-activ-8')
model.add_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='full-deconv-4', input='full-unpool-4')
model.add_node(BatchNormalization(), name='full-bn-9', input='full-deconv-4')
model.add_node(ELU(), name='full-activ-9', input='full-bn-9')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='full-unpool-5', input='full-activ-9')
model.add_node(Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='full-deconv-5', input='full-unpool-5')
model.add_node(BatchNormalization(), name='full-bn-10', input='full-deconv-5')
model.add_node(ELU(), name='full-activ-10', input='full-bn-10')
model.add_node(Convolution2D(1, 1, 1, activation='sigmoid', init='uniform', border_mode='same', dim_ordering='th'),
               name='full-prob-map', input='full-activ-10')
model.add_node(Reshape((256, 256)), name='full-prob-map-reshape', input='full-prob-map')
model.add_node(Dropout(0.5), name='full-prob-map-dropout', input='full-prob-map-reshape')

model.add_node(Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='half-conv-1', input='input_half')
model.add_node(BatchNormalization(), name='half-bn-1', input='half-conv-1')
model.add_node(ELU(), name='half-activ-1', input='half-bn-1')
model.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
               name='half-pool-1', input='half-activ-1')
model.add_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='half-conv-2', input='half-pool-1')
model.add_node(BatchNormalization(), name='half-bn-2', input='half-conv-2')
model.add_node(ELU(), name='half-activ-2', input='half-bn-2')
model.add_node(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'),
               name='half-pool-2', input='half-activ-2')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='half-conv-3', input='half-pool-2')
model.add_node(BatchNormalization(), name='half-bn-3', input='half-conv-3')
model.add_node(ELU(), name='half-activ-3', input='half-bn-3')
model.add_node(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'),
               name='half-pool-3', input='half-activ-3')
model.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='half-conv-4', input='half-pool-3')
model.add_node(BatchNormalization(), name='half-bn-4', input='half-conv-4')
model.add_node(ELU(), name='half-activ-4', input='half-bn-4')
model.add_node(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'),
               name='half-pool-4', input='half-activ-4')
model.add_node(Flatten(), name='half-flatten', input='half-pool-4')
model.add_node(Dense(4096, activation='relu'), name='half-fc-1', input='half-flatten')
model.add_node(Dropout(0.5), name='half-dropout-1', input='half-fc-1')
model.add_node(Reshape((64, 8, 8)), name='half-reshape', input='half-dropout-1')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='half-unpool-1', input='half-reshape')
model.add_node(Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='half-deconv-1', input='half-unpool-1')
model.add_node(BatchNormalization(), name='half-bn-5', input='half-deconv-1')
model.add_node(ELU(), name='half-activ-5', input='half-bn-5')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='half-unpool-2', input='half-activ-5')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='half-deconv-2', input='half-unpool-2')
model.add_node(BatchNormalization(), name='half-bn-6', input='half-deconv-2')
model.add_node(ELU(), name='half-activ-6', input='half-bn-6')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='half-unpool-3', input='half-activ-6')
model.add_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='half-deconv-3', input='half-unpool-3')
model.add_node(BatchNormalization(), name='half-bn-7', input='half-deconv-3')
model.add_node(ELU(), name='half-activ-7', input='half-bn-7')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='half-unpool-4', input='half-activ-7')
model.add_node(Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='half-deconv-4', input='half-unpool-4')
model.add_node(BatchNormalization(), name='half-bn-8', input='half-deconv-4')
model.add_node(ELU(), name='half-activ-8', input='half-bn-8')
model.add_node(Convolution2D(1, 1, 1, activation='sigmoid', init='uniform', border_mode='same', dim_ordering='th'),
               name='half-prob-map', input='half-activ-8')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='half-upsample', input='half-prob-map')
model.add_node(Reshape((256, 256)), name='half-prob-map-reshape', input='half-upsample')
model.add_node(Dropout(0.5), name='half-prob-map-dropout', input='half-prob-map-reshape')

model.add_node(Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='quarter-conv-1', input='input_quarter')
model.add_node(BatchNormalization(), name='quarter-bn-1', input='quarter-conv-1')
model.add_node(ELU(), name='quarter-activ-1', input='quarter-bn-1')
model.add_node(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th'),
               name='quarter-pool-1', input='quarter-activ-1')
model.add_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='quarter-conv-2', input='quarter-pool-1')
model.add_node(BatchNormalization(), name='quarter-bn-2', input='quarter-conv-2')
model.add_node(ELU(), name='quarter-activ-2', input='quarter-bn-2')
model.add_node(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'),
               name='quarter-pool-2', input='quarter-activ-2')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='quarter-conv-3', input='quarter-pool-2')
model.add_node(BatchNormalization(), name='quarter-bn-3', input='quarter-conv-3')
model.add_node(ELU(), name='quarter-activ-3', input='quarter-bn-3')
model.add_node(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'),
               name='quarter-pool-3', input='quarter-activ-3')
model.add_node(Flatten(), name='quarter-flatten', input='quarter-pool-3')
model.add_node(Dense(4096, activation='relu'), name='quarter-fc-1', input='quarter-flatten')
model.add_node(Dropout(0.5), name='quarter-dropout-1', input='quarter-fc-1')
model.add_node(Reshape((64, 8, 8)), name='quarter-reshape', input='quarter-dropout-1')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='quarter-unpool-1', input='quarter-reshape')
model.add_node(Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='quarter-deconv-1', input='quarter-unpool-1')
model.add_node(BatchNormalization(), name='quarter-bn-4', input='quarter-deconv-1')
model.add_node(ELU(), name='quarter-activ-4', input='quarter-bn-4')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='quarter-unpool-2', input='quarter-activ-4')
model.add_node(Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='quarter-deconv-2', input='quarter-unpool-2')
model.add_node(BatchNormalization(), name='quarter-bn-5', input='quarter-deconv-2')
model.add_node(ELU(), name='quarter-activ-5', input='quarter-bn-5')
model.add_node(UpSampling2D(size=(2, 2), dim_ordering='th'), name='quarter-unpool-3', input='quarter-activ-5')
model.add_node(Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th'),
               name='quarter-deconv-3', input='quarter-unpool-3')
model.add_node(BatchNormalization(), name='quarter-bn-6', input='quarter-deconv-3')
model.add_node(ELU(), name='quarter-activ-6', input='quarter-bn-6')
model.add_node(Convolution2D(1, 1, 1, activation='sigmoid', init='uniform', border_mode='same', dim_ordering='th'),
               name='quarter-prob-map', input='quarter-activ-6')
model.add_node(UpSampling2D(size=(4, 4), dim_ordering='th'), name='quarter-upsample', input='quarter-prob-map')
model.add_node(Reshape((256, 256)), name='quarter-prob-map-reshape', input='quarter-upsample')
model.add_node(Dropout(0.5), name='quarter-prob-map-dropout', input='quarter-prob-map-reshape')

model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True),
               name='rnn-we-1', inputs=['full-prob-map-dropout', 'half-prob-map-dropout', 'quarter-prob-map-dropout'],
               merge_mode='sum')
model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True),
               name='rnn-we-2', input='rnn-we-1')
model.add_node(Reshape((1, 256, 256)), name='rnn-we-reshaped', input='rnn-we-2')
model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True),
               name='rnn-ew-1', inputs=['full-prob-map-dropout', 'half-prob-map-dropout', 'quarter-prob-map-dropout'],
               merge_mode='sum')
model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True),
               name='rnn-ew-2', input='rnn-ew-1')
model.add_node(Reshape((1, 256, 256)), name='rnn-ew-reshaped', input='rnn-ew-2')

model.add_node(Rotate90(direction='counterclockwise'),
               name='rotate', inputs=['full-prob-map-dropout', 'half-prob-map-dropout', 'quarter-prob-map-dropout'],
               merge_mode='sum')
model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True),
               name='rnn-ns-1', input='rotate')
model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True),
               name='rnn-ns-2-rotated', input='rnn-ns-1')
model.add_node(Rotate90(direction='clockwise'), name='rnn-ns-2', input='rnn-ns-2-rotated')
model.add_node(Reshape((1, 256, 256)), name='rnn-ns-reshaped', input='rnn-ns-2')
model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True),
               name='rnn-sn-1', input='rotate')
model.add_node(GRU(256, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True),
               name='rnn-sn-2-rotated', input='rnn-sn-1')
model.add_node(Rotate90(direction='clockwise'), name='rnn-sn-2', input='rnn-sn-2-rotated')
model.add_node(Reshape((1, 256, 256)), name='rnn-sn-reshaped', input='rnn-sn-2')

model.add_node(Activation('linear'),
               name='rnn-concat', inputs=['rnn-we-reshaped', 'rnn-ew-reshaped', 'rnn-ns-reshaped', 'rnn-sn-reshaped'],
               merge_mode='concat', concat_axis=-3)
model.add_node(Convolution2D(1, 1, 1, activation='sigmoid', init='uniform', border_mode='same', dim_ordering='th'),
               name='rnn-reduce', input='rnn-concat')
model.add_node(Reshape((256, 256)), name='pre-output', input='rnn-reduce')

model.add_output(name='output', input='pre-output')

model.compile('adam', {'output': binaryCE})


################################################
# TRAIN NEURAL NETWORK
################################################

batch_size = 64
nb_epoch = 10

checkpointer = ModelCheckpoint(filepath='../../model_weights/sunnybrook_segmentation_multires.hdf5',
                               verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

model.fit({'input_full': data_full_training,
           'input_half': data_half_training,
           'input_quarter': data_quarter_training,
           'output': labels_full_training},
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          verbose=2,
          shuffle=True,
          class_weight={'output': {0: 1, 1: 10}},
          validation_data={'input_full': data_full_validation,
                           'input_half': data_half_validation,
                           'input_quarter': data_quarter_validation,
                           'output': labels_full_validation},
          callbacks=[checkpointer, earlystopping])
