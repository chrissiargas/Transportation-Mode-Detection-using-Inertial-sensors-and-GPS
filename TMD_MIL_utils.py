import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import *
from keras import Input
from keras import initializers, regularizers
from keras.models import Model
from config_parser import Parser

CLASS_ACTIVATION = 'sigmoid'


def get_spectrogram_encoder(input_shapes, L, use_dropout=False, last_dropout=True):
    initializer = initializers.he_uniform()

    input = Input(input_shapes)
    X = input

    _, _, channels = input_shapes

    norm = BatchNormalization(name='motion_norm_1')
    X = norm(X)

    padding = ZeroPadding2D(padding=(1, 1))  # same padding
    cnn = Conv2D(
        filters=16,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer=initializer,
        name='motion_conv_1'
    )
    norm = BatchNormalization(name='motion_norm_2')
    activation = ReLU()
    pooling = MaxPooling2D((2, 2), strides=2)

    X = padding(X)
    X = cnn(X)
    X = norm(X)
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
    norm = BatchNormalization(name='motion_norm_3')
    activation = ReLU()
    pooling = MaxPooling2D((2, 2), strides=2)

    X = padding(X)
    X = cnn(X)
    X = norm(X)
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
    norm = BatchNormalization(name='motion_norm_4')
    activation = ReLU()
    pooling = MaxPooling2D((2, 2), strides=2)

    X = cnn(X)
    X = norm(X)
    X = activation(X)
    X = pooling(X)

    flatten = Flatten()
    dropout = Dropout(rate=0.3)

    X = flatten(X)
    if use_dropout:
        X = dropout(X)

    dense = Dense(units=128,
                  kernel_initializer=initializer,
                  name='motion_dense_1')
    norm = BatchNormalization(name='motion_norm_5')
    activation = ReLU()
    dropout = Dropout(rate=0.25)

    X = dense(X)
    X = norm(X)
    X = activation(X)
    if use_dropout:
        X = dropout(X)

    dense = Dense(units=L,
                  kernel_initializer=initializer,
                  name='motion_dense_2')
    norm = BatchNormalization(name='motion_norm_6')
    activation = ReLU()
    dropout = Dropout(rate=0.25)

    X = dense(X)
    X = norm(X)
    X = activation(X)
    if use_dropout and last_dropout:
        X = dropout(X)

    output = X

    return Model(inputs=input,
                 outputs=output,
                 name='spectrogram_encoder')


def get_MIL_attention(L, D):
    initializer = initializers.glorot_uniform()
    regularizer = regularizers.l2(0.01)

    encodings = Input(shape=L)

    D_layer = Dense(units=D,
                    activation='tanh',
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    name='D_layer')
    G_layer = Dense(units=D,
                    activation='sigmoid',
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    name='G_layer')
    K_layer = Dense(units=1,
                    kernel_initializer=initializer,
                    name='K_layer')

    attention_ws = D_layer(encodings)
    attention_ws = attention_ws * G_layer(encodings)
    attention_ws = K_layer(attention_ws)

    return Model(inputs=encodings,
                 outputs=attention_ws,
                 name='MIL_attention')


def get_location_encoder(input_shapes, L):
    initializer = initializers.he_uniform()
    window_shape = input_shapes[0]
    features_shape = input_shapes[1]

    window = Input(shape=window_shape)
    features = Input(shape=features_shape)

    X = window

    norm = BatchNormalization(name='location_norm_1')
    X = norm(X)

    lstm = Bidirectional(LSTM(units=128, name='location_BiLSTM'))
    X = lstm(X)

    X = tf.concat([X, features], axis=1)

    dense = Dense(
        units=128,
        kernel_initializer=initializer,
        name='location_dense_1'
    )
    norm = BatchNormalization(name='location_norm_2')
    activation = ReLU()

    X = dense(X)
    X = norm(X)
    X = activation(X)

    dense = Dense(
        units=64,
        kernel_initializer=initializer,
        name='location_dense_2'
    )
    norm = BatchNormalization(name='location_norm_3')
    activation = ReLU()

    X = dense(X)
    X = norm(X)
    X = activation(X)

    dense = Dense(
        units=L,
        kernel_initializer=initializer,
        name='location_dense_3'
    )
    norm = BatchNormalization(name='location_norm_4')
    activation = ReLU()

    X = dense(X)
    X = norm(X)
    X = activation(X)

    output = X

    return Model(inputs=[window, features],
                 outputs=output,
                 name='location_encoder')


def get_classifier(L, n_units=8, has_head=False):
    input = keras.Input(shape=L)

    X = input
    if has_head:
        dense = Dense(
            units=L // 2,
            kernel_initializer=initializers.he_uniform(),
            name='head_dense'
        )
        norm = BatchNormalization(name='head_norm')
        activation = ReLU()

        X = dense(X)
        X = norm(X)
        X = activation(X)

    dense = Dense(units=n_units,
                  kernel_initializer=initializers.glorot_uniform(),
                  name='final_dense')

    X = dense(X)

    activation = tf.keras.layers.Activation(activation=CLASS_ACTIVATION, name='class_activation')
    y_pred = activation(X)

    return Model(inputs=input,
                 outputs=y_pred,
                 name='classifier')


def get_MIL_model(input_shapes, dpd_motion_encoder=None, dpd_location_encoder=None):
    conf = Parser()
    conf.get_args()

    motion_shape = list(input_shapes[0])
    loc_w_shape = list(input_shapes[1])
    loc_fts_shape = list(input_shapes[2])
    pos_size = input_shapes[3]

    mot_bag_size = motion_shape[0]
    loc_bag_size = loc_w_shape[0]
    motion_size = motion_shape[1:]

    motion_transfer = True if dpd_motion_encoder else False
    motion_encoder = get_spectrogram_encoder(motion_size, conf.L,
                                             use_dropout=not motion_transfer,
                                             last_dropout=False)

    if motion_transfer:
        motion_encoder.set_weights(dpd_motion_encoder.get_layer('spectrogram_encoder').get_weights())
        motion_encoder.trainable = False

    location_transfer = True if dpd_location_encoder else False
    location_encoder = get_location_encoder([loc_w_shape[1:], loc_fts_shape[1:]], conf.L)

    if location_transfer:
        location_encoder.set_weights(dpd_location_encoder.get_layer('location_encoder').get_weights())
        location_encoder.trainable = False

    if conf.motion_MIL:
        mot_MIL_attention = get_MIL_attention(conf.L, conf.D)
        mot_att_softmax = Softmax(name='motion_attention_softmax')

    if conf.fusion == 'MIL':
        MIL_attention = get_MIL_attention(conf.L, conf.D)
        att_softmax = Softmax(name='attention_softmax')

    n_classes = 5 if conf.motorized else 8

    L = 2 * conf.L if conf.fusion == 'concat' else conf.L
    classifier = get_classifier(L, n_classes, has_head=True)

    motion_bags = Input(motion_shape)
    loc_w_bags = Input(loc_w_shape)
    loc_fts_bags = Input(loc_fts_shape)
    motion_positions = Input(pos_size, name='positional')

    batch_size = tf.shape(motion_bags)[0]
    motion_instances = tf.reshape(motion_bags, (batch_size * mot_bag_size, *motion_size))
    loc_w_instances = tf.reshape(loc_w_bags, (batch_size * loc_bag_size, *loc_w_shape[1:]))
    loc_fts_instances = tf.reshape(loc_fts_bags, (batch_size * loc_bag_size, *loc_fts_shape[1:]))

    motion_encodings = motion_encoder(motion_instances)
    location_encodings = location_encoder([loc_w_instances, loc_fts_instances])

    if conf.motion_MIL:
        mot_attention_ws = mot_MIL_attention(motion_encodings)
        mot_attention_ws = tf.reshape(mot_attention_ws, [batch_size, mot_bag_size])

        attention_ws = tf.expand_dims(mot_att_softmax(mot_attention_ws), -2)
        motion_encodings = tf.reshape(motion_encodings, [batch_size, mot_bag_size, conf.L])

        motion_encodings = tf.matmul(attention_ws, motion_encodings)
        mot_bag_size = 1
    else:
        motion_encodings = tf.reshape(motion_encodings, (batch_size, mot_bag_size, conf.L))

    location_encodings = tf.reshape(location_encodings, (batch_size, loc_bag_size, conf.L))
    encodings = concatenate([motion_encodings, location_encodings], axis=-2)

    if conf.fusion == 'MIL':
        encodings = tf.reshape(encodings, (batch_size * (mot_bag_size + loc_bag_size), conf.L))
        attention_ws = MIL_attention(encodings)
        attention_ws = tf.reshape(attention_ws, (batch_size, mot_bag_size + loc_bag_size))

        attention_ws = tf.expand_dims(att_softmax(attention_ws), -2)
        encodings = tf.reshape(encodings, (batch_size, mot_bag_size + loc_bag_size, conf.L))
        flatten = Flatten()
        pooling = flatten(tf.matmul(attention_ws, encodings))

    elif conf.fusion == 'concat':
        pooling = tf.reshape(encodings, (batch_size, conf.L * (mot_bag_size + loc_bag_size)))

    y_pred = classifier(pooling)

    return Model([motion_bags, loc_w_bags, loc_fts_bags, motion_positions], y_pred)


def get_motion_model(input_shapes):
    conf = Parser()
    conf.get_args()

    motion_shape = list(input_shapes[0])
    pos_size = input_shapes[1]

    if conf.motion_MIL:
        mot_bag_size = motion_shape[0]
        motion_size = motion_shape[1:]
    else:
        motion_size = motion_shape

    motion_encoder = get_spectrogram_encoder(motion_size, conf.L, use_dropout=True)

    if conf.motion_MIL:
        MIL_attention = get_MIL_attention(conf.L, conf.D)

    n_classes = 5 if conf.motorized else 8
    classifier = get_classifier(conf.L, n_classes, has_head=False)

    motion_input = Input(motion_shape)
    motion_positions = Input(pos_size, name='positional')

    if conf.motion_MIL:
        motion_bags = motion_input
        batch_size = tf.shape(motion_bags)[0]
        motion_instances = tf.reshape(motion_bags, (batch_size * mot_bag_size, *motion_size))

        motion_encodings = motion_encoder(motion_instances)
        attention_ws = MIL_attention(motion_encodings)
        attention_ws = tf.reshape(attention_ws, [batch_size, mot_bag_size])

        softmax = Softmax(name='motion_attention_softmax')
        attention_ws = tf.expand_dims(softmax(attention_ws), -2)
        motion_encodings = tf.reshape(motion_encodings, [batch_size, mot_bag_size, conf.L])

        motion_pooling = tf.squeeze(tf.matmul(attention_ws, motion_encodings), axis=-2)

    else:
        motion_pooling = motion_encoder(motion_input)

    y_pred = classifier(motion_pooling)

    return Model([motion_input, motion_positions], y_pred)


def get_location_model(input_shapes):
    conf = Parser()
    conf.get_args()

    loc_win_shape = input_shapes[0]
    loc_fts_shape = input_shapes[1]

    location_encoder = get_location_encoder([loc_win_shape, loc_fts_shape], conf.L)

    n_classes = 5 if conf.motorized else 8
    classifier = get_classifier(conf.L, n_classes, has_head=False)

    loc_window = Input(loc_win_shape)
    loc_features = Input(loc_fts_shape)

    location_encodings = location_encoder([loc_window, loc_features])

    y_pred = classifier(location_encodings)

    return Model([loc_window, loc_features], y_pred)
