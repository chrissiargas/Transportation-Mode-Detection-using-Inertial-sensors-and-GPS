import tensorflow as tf
from keras.layers import *
from keras import Input
from keras.models import Model

resnet_args = {'f': 3,
               'filters': [64, 128, 128, 128],
               'kernel_sizes': [3, 2, 2, 4],
               's': 4,
               'p': 2}


def attention_layer(lstm):
    dense1 = Dense(128, activation="softmax")(lstm)
    dense2 = Dense(128, activation="softmax")(dense1)
    lstm = multiply([lstm, dense2])
    return lstm


def mlp_layer(lstm):
    fc = Dense(128, activation='relu', kernel_initializer='truncated_normal')(lstm)
    fc = Dropout(0.2)(fc)
    fc = Dense(256, activation='relu', kernel_initializer='truncated_normal')(fc)
    fc = Dropout(0.2)(fc)
    fc = Dense(512, activation='relu', kernel_initializer='truncated_normal')(fc)
    fc = Dropout(0.2)(fc)
    fc = Dense(1024, activation='relu', kernel_initializer='truncated_normal')(fc)
    fc = Dropout(0.2)(fc)
    output = Dense(8, activation='softmax', name='output')(fc)
    return output


def lstm_layer(all_resnet):
    lstm = LSTM(128, input_shape=(36, 128), activation='tanh',
                dropout=0.2)(all_resnet)
    return lstm


def simple_cnn(X_input, net_id):
    X = Conv1D(filters=32, kernel_size=3,
               kernel_initializer="glorot_uniform", name="conv1_%s_" % net_id)(X_input)
    X = Activation("relu")(X)
    return X


def convolutional_block(X, f, filters, kernel_sizes, stride, net_id):
    conv_name_base = 'resnet_' + net_id

    F1, F2, F3, F4 = filters
    K1, K2, K3, K4 = kernel_sizes

    X_shortcut = X

    X = Conv1D(F1, K1, name=conv_name_base + '_A',
               padding='same')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(K2, padding='same')(X)

    X = Conv1D(F2, f, name=conv_name_base + '_B',
               padding='same')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(K3, padding='same')(X)

    X_shortcut = Conv1D(F4, kernel_size=K4, strides=stride, padding='same', name=conv_name_base + '_shortcut')(X_shortcut)
    X = tf.keras.layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def res_net(X_input, net_id, args):
    X = convolutional_block(X_input, f=args['f'], filters=args['filters'], kernel_sizes=args['kernel_sizes'],
                            stride=args['s'], net_id=net_id)
    X = MaxPooling1D(pool_size=args['p'], padding="same")(X)
    return X


def get_motion_model(input_shapes):

    sensor_inputs = []
    for channel, input_shape in enumerate(input_shapes):
        sensor_inputs.append(Input(shape=input_shape))

    sensor_outputs = []
    for s, sensor_input in enumerate(sensor_inputs):
        channels = tf.split(sensor_input, sensor_input.shape[-1], axis=-1)
        channel_outputs = []
        for c, channel in enumerate(channels):
            channel_output = res_net(channel, 'sensor->' + str(s) + '_channel->' + str(c), resnet_args)
            channel_outputs.append(channel_output)

        sensor_output = concatenate(channel_outputs)
        sensor_outputs.append(sensor_output)

    concat_inputs = sensor_outputs
    cnn_outputs = []
    for s, concat_input in enumerate(concat_inputs):
        cnn_output = simple_cnn(concat_input, 'sensor->' + str(s))
        cnn_outputs.append(cnn_output)

    X = concatenate(cnn_outputs)
    X = lstm_layer(X)
    X = attention_layer(X)
    output = mlp_layer(X)

    return Model(inputs=sensor_inputs,
                 outputs=output,
                 name='temporal_encoder')
