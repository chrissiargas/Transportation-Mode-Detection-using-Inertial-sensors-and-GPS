import tensorflow as tf
import keras
from keras.layers import *
from keras import Input
from keras import initializers
from keras.models import Model


def get_spectrogram_encoder(input_shapes):
    initializer = initializers.he_uniform()

    input = Input(input_shapes)
    X = input

    _, _, channels = input_shapes

    padding = ZeroPadding2D(padding=(1, 1))  # same padding
    cnn = Conv2D(
        filters=16,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer=initializer,
        name='motion_conv_1'
    )
    activation = ReLU()
    pooling = MaxPooling2D((2, 2), strides=2)

    X = padding(X)
    X = cnn(X)
    X = activation(X)
    X = pooling(X)

    padding = ZeroPadding2D(padding=(1, 1))  # same padding
    cnn = Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer=initializer,
        name='motion_conv_2'
    )
    activation = ReLU()
    pooling = MaxPooling2D((2, 2), strides=2)

    X = padding(X)
    X = cnn(X)
    X = activation(X)
    X = pooling(X)

    cnn = Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer=initializer,
        name='motion_conv_3'
    )
    activation = ReLU()
    pooling = MaxPooling2D((2, 2), strides=2)

    X = cnn(X)
    X = activation(X)
    X = pooling(X)

    flatten = Flatten()
    dropout = Dropout(rate=0.25)

    X = flatten(X)
    X = dropout(X)

    dense = Dense(units=128,
                  kernel_initializer=initializer,
                  name='motion_dense_1')
    activation = ReLU()
    dropout = Dropout(rate=0.5)

    X = dense(X)
    X = activation(X)
    X = dropout(X)

    output = X

    return Model(inputs=input,
                 outputs=output,
                 name='spectrogram_encoder')


def get_classifier():
    input = keras.Input(shape=128)

    dense = Dense(units=8,
                  kernel_initializer=initializers.he_uniform(),
                  name='final_dense')

    X = dense(input)

    activation = tf.keras.layers.Activation(activation='softmax', name='class_activation')
    y_pred = activation(X)

    return Model(inputs=input,
                 outputs=y_pred,
                 name='classifier')


def get_motion_model(input_shapes):
    spectro_encoder = get_spectrogram_encoder(input_shapes[0])
    classifier = get_classifier()

    inputs = Input(input_shapes[0])
    X = inputs

    X = spectro_encoder(X)
    X = classifier(X)

    outputs = X

    return Model(inputs=inputs, outputs=outputs)

