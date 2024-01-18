import tensorflow as tf
from keras.layers import *
from keras import Input
from keras.models import Model


def GFE_Block(x):
    x = tf.keras.layers.Dense(64)(x)  # Project Layer
    x = tf.keras.layers.MultiHeadAttention(8, 64)(x, x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x


hidden_dim = 64
num_patches = 500
dropout_rate = 0.5
token_mlp_dim = 128  # Ds
channel_mlp_dim = 256  # Dc
num_mixer_layers = 8

# Mixer
cross_time_mixer = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=token_mlp_dim, activation="gelu"),  # Mixer (F, T)(T, Ds) => (F, Ds)
        tf.keras.layers.Dense(units=num_patches),  # (F, Ds)(Ds, T) => (F, T)
        tf.keras.layers.Dropout(0.5)
    ], name="cross_time_mixer"
)
cross_senor_mixer = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=channel_mlp_dim, activation="gelu"),  # Mixer (T, F)(F, Dc) => (T, Dc)
        tf.keras.layers.Dense(units=hidden_dim),  # (T, Dc)(Dc, F) => (T, F)
        tf.keras.layers.Dropout(0.5)
    ],
    name="cross_sensor_mixer"
)


def LFE_Block(inputs):
    inputs = tf.keras.layers.Dense(hidden_dim)(inputs)  # Project Layer
    x = inputs
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Permute([2, 1])(x)
    x = cross_time_mixer(x)
    x = tf.keras.layers.Permute([2, 1])(x)
    x_shortcut_2 = tf.keras.layers.add([x, inputs])
    x = tf.keras.layers.LayerNormalization()(x_shortcut_2)  # skip connection
    x = cross_senor_mixer(x)
    x = tf.keras.layers.add([x, x_shortcut_2])  # skip connection
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x


# Fusion Layer
def Fusion_Layer(x1, x2):
    x1 = tf.keras.layers.Reshape(target_shape=(64, 1))(x1)
    x2 = tf.keras.layers.Reshape(target_shape=(64, 1))(x2)
    x = tf.keras.layers.Concatenate(axis=2)([x1, x2])
    x = tf.keras.layers.Dense(1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    return x


# MLP Layer
def MLP_Layer(inputs):
    x = inputs
    x = tf.keras.layers.Dense(units=64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=64, activation="relu")(x)
    x = tf.keras.layers.add([x, inputs])

    output = tf.keras.layers.Dense(8, activation="softmax")(x)
    return output


def get_motion_model(input_shapes):

    inputs = Input(input_shapes)
    X = inputs

    X1 = GFE_Block(X)
    X2 = LFE_Block(X)
    X = Fusion_Layer(X1, X2)
    outputs = MLP_Layer(X)

    return Model(inputs=inputs,
                 outputs=outputs,
                 name='temporal_encoder')