import numpy as np
from scipy.signal import spectrogram
from scipy.interpolate import interp2d
from math import floor, ceil
import matplotlib.pyplot as plt
from config_parser import Parser
from typing import *
import pickle
from augmentations import get_mask
from geopy.distance import great_circle

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class one_hot_transformer:
    def __init__(self, mode_names, mode_ids, motorized=False):
        self.mode_names = mode_names
        self.mode_ids = mode_ids
        self.motorized = motorized
        self.motorized_modes = ['car', 'bus', 'train', 'subway']
        self.n_modes = max(self.mode_ids.values()) + 1
        self.diag = np.eye(self.n_modes)

    def __call__(self, info):
        if self.mode_names[info[-1]] not in self.mode_ids.keys():
            return None

        id = self.mode_ids[self.mode_names[info[-1]]]
        return self.diag[id]


class temporal_transformer:

    def __init__(self):
        self.channel_shapes = None
        self.channel_signals = None
        config = Parser()
        config.get_args()
        self.conf = config

        with open('info/segmented_features.pickle', 'rb') as handle:
            features = pickle.load(handle)
            self.mot_features = features['mot_features']

        self.pivot = self.conf.mot_bag_size // 2
        self.bag_size = self.conf.mot_bag_size
        self.length = self.conf.mot_length
        self.channels = None

        self.shape = self.get_shape()

    def get_shape(self):
        if self.conf.combine_sensors == 'concat':
            in_features = [feature for feature in self.conf.motion_features if feature in self.mot_features]
            self.channels = len(in_features)
            if self.conf.in_bags:
                return [(self.bag_size, self.length, self.channels)]
            else:
                return [(self.length, self.channels)]

        if self.conf.combine_sensors == 'separate':
            self.n_channels = len(self.conf.separated_channels)
            self.channel_signals = [0 for _ in self.conf.separated_channels]
            self.channel_shapes = [None for _ in self.conf.separated_channels]

            for id, channel_features in enumerate(self.conf.separated_channels):
                for feature in channel_features:
                    if feature in self.mot_features and self.conf.motion_features:
                        self.channel_signals[id] += 1

            if self.conf.in_bags:
                for k, n_signals in enumerate(self.channel_signals):
                    self.channel_shapes[k] = (self.bag_size, self.length, n_signals)
            else:
                for k, n_signals in enumerate(self.channel_signals):
                    self.channel_shapes[k] = (self.length, n_signals)

            return self.channel_shapes

    def __call__(self, instances, training: bool = False, preprocessing: bool = False):

        if not self.conf.in_bags:
            instances = instances[[self.pivot]]

        if preprocessing:
            signals = {}

        elif self.conf.combine_sensors == 'concat':
            outputs = np.zeros((len(instances), self.length, self.channels))
            channel = 0

        elif self.conf.combine_sensors == 'separate':
            channel_outputs = [np.zeros((len(instances), self.length, self.channel_signals[k]))
                               for k in range(self.n_channels)]

        if preprocessing or self.conf.combine_sensors == 'concat':

            for feature in self.conf.motion_features:
                if feature in self.mot_features:
                    key = self.mot_features[feature]
                    signal = instances[:, :, key - 1]

                    if preprocessing:
                        signals[feature] = signal

                    elif self.conf.combine_sensors == 'concat':
                        outputs[..., channel] = signal
                        channel += 1

        elif self.conf.combine_sensors == 'separate':
            for k, channel_features in enumerate(self.conf.separated_channels):
                for f, feature in enumerate(channel_features):
                    if feature in self.mot_features and feature in self.conf.motion_features:
                        key = self.mot_features[feature]
                        signal = instances[:, :, key - 1]
                        channel_outputs[k][..., f] = signal

        if preprocessing:
            outputs = signals

        elif self.conf.combine_sensors == 'concat':
            if not self.conf.in_bags:
                outputs = outputs[0]

            outputs = [outputs]

        elif self.conf.combine_sensors == 'separate':
            if not self.conf.in_bags:
                outputs = [bag[0] for bag in channel_outputs]

            else:
                outputs = channel_outputs

        return outputs


class spectro_transformer:

    def __init__(self):
        config = Parser()
        config.get_args()
        self.conf = config

        with open('info/segmented_features.pickle', 'rb') as handle:
            features = pickle.load(handle)
            self.mot_features = features['mot_features']

        self.n_channels = None
        self.channel_shapes = None
        self.channel_signals = None

        self.pivot = self.conf.mot_bag_size // 2
        self.bag_size = self.conf.mot_bag_size
        self.height, self.width = 48, 48
        self.out_size = (self.height, self.width)
        self.channels = None

        self.nperseg = self.conf.spectro_window * self.conf.motion_fs
        self.noverlap = self.conf.spectro_overlap * self.conf.motion_fs

        self.shape = self.get_shape()
        self.temporal_transformer = temporal_transformer()

    def get_shape(self):
        if self.conf.combine_sensors == 'concat':
            in_features = [feature for feature in self.conf.motion_features if feature in self.mot_features]
            self.channels = len(in_features)
            if self.conf.in_bags:
                return [(self.bag_size, self.height, self.width, self.channels)]
            else:
                return [(self.height, self.width, self.channels)]

        if self.conf.combine_sensors == 'separate':
            self.n_channels = len(self.conf.separated_channels)
            self.channel_signals = [0 for _ in self.conf.separated_channels]
            self.channel_shapes = [None for _ in self.conf.separated_channels]

            for id, channel_features in enumerate(self.conf.separated_channels):
                for feature in channel_features:
                    if feature in self.mot_features and self.conf.motion_features:
                        self.channel_signals[id] += 1

            if self.conf.in_bags:
                for k, n_signals in enumerate(self.channel_signals):
                    self.channel_shapes[k] = (self.bag_size, self.height, self.width, n_signals)
            else:
                for k, n_signals in enumerate(self.channel_signals):
                    self.channel_shapes[k] = (self.height, self.width, n_signals)

            return self.channel_shapes

    def resize(self, spectros, freq, time):
        n_instances = spectros.shape[0]
        out_f, out_t = self.height, self.width
        out_spectrograms = np.zeros((n_instances, out_f, out_t), dtype=np.float32)

        if self.conf.f_interp == 'log':
            log_f = np.log(freq + freq[1])
            log_f_normalized = (log_f - log_f[0]) / (log_f[-1] - log_f[0])
            f = out_f * log_f_normalized

        else:
            f_normalized = (freq - freq[0]) / (freq[-1] - freq[0])
            f = out_f * f_normalized

        t_normalized = (time - time[0]) / (time[-1] - time[0])
        t = out_t * t_normalized

        f_i = np.arange(out_f)
        t_i = np.arange(out_t)

        for i, spectro in enumerate(spectros):
            spectrogram_fn = interp2d(t, f, spectro, copy=False)
            out_spectrograms[i, :, :] = spectrogram_fn(f_i, t_i)

        return out_spectrograms

    def get_spectrogram(self, signal, mask):
        f, t, spectro = spectrogram(signal, fs=self.conf.motion_fs,
                                         nperseg=self.nperseg,
                                         noverlap=self.noverlap)

        spectro = self.resize(spectro, f, t)

        if mask:
            spectro = mask(spectro)

        if self.conf.log_power:
            np.log(spectro + 1e-10, dtype=np.float32, out=spectro)

        return spectro

    def __call__(self, instances, training: bool = False):
        if self.conf.combine_sensors == 'concat':
            outputs = np.zeros((len(instances), self.height, self.width, self.channels))
            channel = 0

        elif self.conf.combine_sensors == 'separate':
            channel_outputs = [np.zeros((len(instances), self.width, self.height, self.channel_signals[k]))
                               for k in range(self.n_channels)]

        if training:
            mask = get_mask(self.conf.spectro_augmentations, self.out_size)
        else:
            mask = None

        instances = self.temporal_transformer(instances, training, preprocessing=True)

        if self.conf.combine_sensors == 'concat':
            for feature in instances.keys():
                signal = instances[feature]
                spectro = self.get_spectrogram(signal, mask)

                outputs[..., channel] = spectro
                channel += 1

        if self.conf.combine_sensors == 'separate':
            for k, channel_features in enumerate(self.conf.separated_channels):
                for f, feature in enumerate(channel_features):
                    if feature in self.mot_features and feature in self.conf.motion_features:
                        signal = instances[feature]
                        spectro = self.get_spectrogram(signal, mask)

                        channel_outputs[k][..., f] = spectro

        if self.conf.combine_sensors == 'concat':
            if not self.conf.in_bags:
                outputs = outputs[0]

            outputs = [outputs]

        elif self.conf.combine_sensors == 'separate':
            if not self.conf.in_bags:
                outputs = [bag[0] for bag in channel_outputs]

            else:
                outputs = channel_outputs

        return outputs


class series_transformer:
    def __init__(self):
        config = Parser()
        config.get_args()
        self.conf = config

        with open('info/segmented_features.pickle', 'rb') as handle:
            features = pickle.load(handle)
            self.time_features = features['loc_features']

        self.stat_features = ['Min', 'Max', 'Mean', 'Std']
        self.arithmetic_features = ['Distance', 'Velocity', 'Movability']

        self.length = self.conf.loc_length
        self.channels = 0
        self.n_features = 0
        self.shape = self.get_shape()

    def get_shape(self):
        in_features = [feature for feature in self.conf.time_features if feature in self.time_features]
        self.channels = len(in_features)

        if self.conf.in_bags:
            window_shape = 1, self.length, self.channels
        else:
            window_shape = self.length, self.channels

        in_stat_features = [feature for feature in self.conf.window_features if feature in self.stat_features]
        self.n_features = self.channels * len(in_stat_features)

        in_arithmetic_features = [feature for feature in self.conf.window_features if
                                  feature in self.arithmetic_features]
        self.n_features += len(in_arithmetic_features)

        if self.conf.in_bags:
            features_shape = 1, self.n_features
        else:
            features_shape = self.n_features

        return window_shape, features_shape

    def get_distance(self, lat, long, i):
        if np.isnan([lat[i], long[i], lat[i - 1], long[i - 1]]).any():
            return np.nan

        point1 = (lat[i - 1], long[i - 1])
        point2 = (lat[i], long[i])
        distance = great_circle(point1, point2).m
        return distance

    def __call__(self, input, training: bool = False):

        window = np.zeros((self.length, self.channels))
        channel = 0
        for feature in self.conf.time_features:
            if feature in self.time_features:
                key = self.time_features[feature]
                timeseq = input[:, key - 1]

                window[:, channel] = timeseq
                channel += 1

        features = np.zeros(self.n_features)
        n = 0
        for feature in self.conf.window_features:
            if feature in self.stat_features:
                if feature == 'Mean':
                    values = [np.nanmean(window[:, i]) for i in range(window.shape[1])]

                elif feature == 'Std':
                    values = [np.nanstd(window[:, i]) for i in range(window.shape[1])]

                elif feature == 'Min':
                    values = [np.nanmin(window[:, i]) for i in range(window.shape[1])]

                elif feature == 'Max':
                    values = [np.nanmax(window[:, i]) for i in range(window.shape[1])]

                features[n: n + len(values)] = values
                n += len(values)

            if feature in self.arithmetic_features:
                if feature == 'Movability':
                    time = input[:, self.time_features['Time']]
                    lat = input[:, self.time_features['Lat']]
                    long = input[:, self.time_features['Long']]

                    start_point = (lat[0], long[0])
                    end_point = (lat[-1], long[-1])
                    total_distance = great_circle(start_point, end_point).m

                    distances = [self.get_distance(lat, long, time, i) for i in range(1, input.shape[0])]
                    sum_distance = sum(distances)

                    value = total_distance / (sum_distance + 1e-10)

                features[n] = value
                n += 1

        if self.conf.in_bags:
            window = window[np.newaxis, ...]
            features = features[np.newaxis, ...]

        return window, features
