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
    initializer = initializers.he_uniform()
    regularizer = regularizers.l2(0.001)

    input = Input(input_shapes)
    X = input

    _, channels = input_shapes

    filters = [32, 64, 64, 64, 64, 64]
    kernel_size = [15, 10, 10, 5, 5, 5]
    for k, layer in enumerate(range(1, 7)):
        cnn = Conv1D(
            filters=filters[k],
            kernel_size=kernel_size[k],
            strides=1,
            padding='same',
            kernel_initializer=initializer,
            name='motion_conv_' + str(layer),
            kernel_regularizer=regularizer)
        activation = LeakyReLU(0.01)
        pooling = MaxPooling1D(4, strides=2, padding='same')

        X = cnn(X)
        X = activation(X)
        X = pooling(X)

    flatten = Flatten()
    X = flatten(X)

    dense = Dense(units=200,
                  kernel_initializer=initializer,
                  name='motion_dense_1',
                  kernel_regularizer=regularizer)
    activation = LeakyReLU(0.01)

    X = dense(X)
    X = activation(X)

    output = X

    return Model(inputs=input,
                 outputs=output,
                 name='temporal_encoder')


def get_classifier(n_units=8):
    input = keras.Input(shape=200)

    X = input

    dense = Dense(units=n_units,
                  activation='softmax',
                  kernel_initializer=initializers.glorot_uniform(),
                  kernel_regularizer=regularizers.l2(0.001),
                  name='class_layer')

    y_pred = dense(X)

    return Model(inputs=input,
                 outputs=y_pred,
                 name='classifier')


def get_motion_model(input_shapes, n_classes):
    conf = Parser()
    conf.get_args()

    motion_shape = input_shapes[0]
    motion_encoder = get_temporal_encoder(motion_shape)
    classifier = get_classifier(n_classes)

    motion_input = Input(motion_shape)
    motion_encodings = motion_encoder(motion_input)
    y_pred = classifier(motion_encodings)

    return Model(motion_input, y_pred)






