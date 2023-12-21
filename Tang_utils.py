import numpy as np
import tensorflow as tf
import keras
import os
import keras.backend as K
from keras.layers import *
from keras import Input
from keras import initializers, regularizers
from keras.models import Model
from config_parser import Parser


def get_temporal_encoder(input_shapes):
    initializer = initializers.he_uniform
    regularizer = regularizers.l2(0.001)

    channel_inputs = []
    layer_poolings = [[] for _ in range(4)]
    for channel, input_shape in enumerate(input_shapes):
        channel_inputs.append(Input(shape=input_shape))

        filters = [32, 64, 64, 128, 128]
        kernel_size = [15, 10, 10, 5, 5]

        X = channel_inputs[channel]
        pool_num = 0
        for k, layer in enumerate(range(1, 6)):
            norm = BatchNormalization(name='channel_'+str(channel+1)+'->batch_norm_' + str(layer))
            cnn = Conv1D(
                filters=filters[k],
                kernel_size=kernel_size[k],
                strides=1,
                padding='same',
                kernel_initializer=initializer(),
                name='channel_'+str(channel+1)+'->motion_conv_' + str(layer))
            activation = ReLU(name='channel_'+str(channel+1)+'->ReLU_' + str(layer))
            pooling = MaxPooling1D(4, strides=2, padding='same',
                                   name='channel_'+str(channel+1)+'->MaxPooling_' + str(layer))

            if layer != 2:
                X = norm(X)
            X = cnn(X)
            X = activation(X)
            X = pooling(X)

            if layer != 4:
                layer_poolings[pool_num].append(X)
                pool_num += 1

    layer_encodings = []
    for k, channel_poolings in enumerate(layer_poolings):
        layer_output = concatenate(channel_poolings, axis=-1, name='concat_'+str(k+1))
        lstm = Bidirectional(LSTM(units=128), name='BiLSTM_'+str(k+1))
        layer_encoding = lstm(layer_output)
        layer_encodings.append(layer_encoding)

    encoding = concatenate(layer_encodings, name='final_concat')

    flatten = Flatten()
    encoding = flatten(encoding)

    dense = Dense(units=128,
                  kernel_initializer=initializer(),
                  name='motion_dense',
                  kernel_regularizer=regularizer)
    activation = ReLU()

    output = dense(encoding)
    output = activation(output)

    return Model(inputs=channel_inputs,
                 outputs=output,
                 name='temporal_encoder')


def get_classifier(n_units=8):
    input = keras.Input(shape=128)

    X = input

    dense = Dense(units=n_units,
                  activation='softmax',
                  kernel_initializer=initializers.glorot_uniform(),
                  name='class_layer')

    y_pred = dense(X)

    return Model(inputs=input,
                 outputs=y_pred,
                 name='classifier')


def get_motion_model(input_shapes, n_classes=8):
    conf = Parser()
    conf.get_args()

    motion_encoder = get_temporal_encoder(input_shapes)
    classifier = get_classifier(n_classes)

    motion_inputs = []
    for input_shape in input_shapes:
        motion_inputs.append(Input(input_shape))

    motion_encodings = motion_encoder(motion_inputs)
    y_pred = classifier(motion_encodings)

    return Model(motion_inputs, y_pred)






