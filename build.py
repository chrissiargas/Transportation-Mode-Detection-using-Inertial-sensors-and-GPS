import matplotlib.pyplot as plt

from config_parser import Parser
import os
import numpy as np
import pickle
from segment import Segmenter
import tensorflow as tf
from transformers import spectro_transformer, temporal_transformer, one_hot_transformer, series_transformer
import random
from split import lopo_split, lopo_split_old
import contextlib
from sklearn.model_selection import train_test_split


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class Builder:
    def __init__(self, regenerate: bool = False):
        self.output_type = None
        self.output_shape = None
        self.test_size = None
        self.test_indices = None
        self.val_size = None
        self.val_indices = None
        self.train_size = None
        self.train_indices = None
        self.input_type = None
        self.input_shape = None
        self.labels_transformer = None
        self.loc_features_shape = None
        self.loc_window_shape = None
        self.location_transformer = None
        self.motion_shape = None
        self.motion_transformer = None

        config = Parser()
        config.get_args()
        self.conf = config

        self.load_path = os.path.join(os.path.expanduser('~'), self.conf.path, self.conf.dataset + '-segmented')

        segmenter = Segmenter(delete_src=regenerate, delete_dst=regenerate)
        can_load = len(os.listdir(self.load_path)) > 0
        self.motion, self.motion_info, self.location, self.loc_info = segmenter(load=can_load, verbose=not can_load)

        with open('info/' + self.conf.dataset + '/segmented_sizes.pickle', 'rb') as handle:
            sizes = pickle.load(handle)
            self.n_mot = sizes['n_mot']
            self.n_loc = sizes['n_loc']

        with open('info/' + self.conf.dataset + '/segmented_features.pickle', 'rb') as handle:
            features = pickle.load(handle)
            self.mot_features = features['mot_features']
            self.loc_features = features['loc_features']

        self.location = self.location[self.conf.gps_position]
        self.loc_info = self.loc_info[self.conf.gps_position]

        motion = {}
        for position in self.motion.keys():
            if position in self.conf.inertial_positions:
                motion[position] = self.motion[position]

        self.motion = motion

        self.total_len = len(self.motion_info)
        self.threshold = self.conf.diff_threshold * 1000.
        self.sync_pairs = [-1 for _ in range(self.total_len)]
        self.n_modes = 8

        self.mode_ids = {
            'undefined': 0,
            'still': 1,
            'walk': 2,
            'run': 3,
            'bike': 4,
            'car': 5,
            'bus': 6,
            'train': 7,
            'subway': 8
        }
        self.mode_names = {v: k for k, v in self.mode_ids.items()}
        self.modes = ['undefined', 'still', 'walk',
                      'run', 'bike', 'car',
                      'bus', 'train', 'subway']

    def init_config(self):
        config = Parser()
        config.get_args()
        self.conf = config

    def combine_to_bags(self):
        self.sync_pairs = [-1 for _ in range(self.total_len)]

        mot_offset = 0
        loc_offset = 0
        subjects = np.unique(self.motion_info[:, 2])

        for subject in subjects:
            sub_mot_info = self.motion_info[self.motion_info[:, 2] == subject]
            sub_loc_info = self.loc_info[self.loc_info[:, 2] == subject]
            dates = np.unique(sub_mot_info[:, 3])

            for date in dates:
                date_mot_info = sub_mot_info[sub_mot_info[:, 3] == date]
                date_loc_info = sub_loc_info[sub_loc_info[:, 3] == date]

                mot_times = (date_mot_info[:, 0] + date_mot_info[:, 1]) / 2.
                loc_times = (date_loc_info[:, 0] + date_loc_info[:, 1]) / 2.
                for i, mot_time in enumerate(mot_times):
                    diffs = np.abs(mot_time - loc_times)
                    j = np.argmin(diffs)

                    if diffs[j] < self.threshold:
                        self.sync_pairs[mot_offset + i] = loc_offset + j

                mot_offset += len(date_mot_info)
                loc_offset += len(date_loc_info)

    def assign_one(self, indices, position):
        indices = [[index, [position
                            for _ in range(self.conf.mot_bag_size)]]
                   for index in indices]

        return indices

    def assign_same(self, indices, oversampling):

        if oversampling:
            indices = [[index, [position
                                for _ in range(self.conf.mot_bag_size)]]
                       for index in indices
                       for position in self.motion.keys()]
        else:

            indices = [[index, [position
                                for _ in range(self.conf.mot_bag_size)]]
                       for index in indices
                       for position in random.sample(self.motion.keys(), 1)]

        return indices

    def assign_random(self, indices, oversampling):
        if oversampling:
            test_indices = []
            for index in indices:
                positions = [random.sample(self.motion.keys(), len(self.motion.keys())) for _ in
                             range(self.conf.mot_bag_size)]
                position_lists = list(map(list, zip(*positions)))

                if not len(test_indices):
                    test_indices = [[index, [position for position in position_list]] for position_list in
                                    position_lists]

                else:
                    test_indices.extend([[index, position_list] for position_list in position_lists])

            indices = test_indices

        else:
            indices = [[index, [random.choice(list(self.motion.keys()))
                                for _ in range(self.conf.mot_bag_size)]]
                       for index in indices]

        return indices

    def assign_position_(self, indices, bag_position, oversampling):

        if bag_position in self.motion.keys():
            indices = self.assign_one(indices, bag_position)

        elif bag_position == 'same':
            indices = self.assign_same(indices, oversampling)

        elif bag_position == 'random':
            indices = self.assign_random(indices, oversampling)

        return indices

    def assign_position(self):
        self.train_indices = self.assign_position_(self.train_indices,
                                                   self.conf.train_bag_position,
                                                   self.conf.train_oversampling)
        self.test_indices = self.assign_position_(self.test_indices,
                                                  self.conf.test_bag_position,
                                                  self.conf.test_oversampling)
        self.val_indices = self.assign_position_(self.val_indices,
                                                 self.conf.val_bag_position,
                                                 self.conf.val_oversampling)

        self.train_size = len(self.train_indices)
        self.val_size = len(self.val_indices)
        self.test_size = len(self.test_indices)

        if self.conf.random:
            random.shuffle(self.test_indices)
            random.shuffle(self.val_indices)
            random.shuffle(self.train_indices)

    def split(self):
        train_indices = []
        test_indices = []

        if self.conf.train_test_split == 'loso':
            test_user = self.conf.train_test_hold_out
            subjects = np.unique(self.motion_info[:, 2])

            if test_user in subjects:
                train_indices = np.argwhere(self.motion_info[:, 2] != test_user).flatten()
                test_indices = np.argwhere(self.motion_info[:, 2] == test_user).flatten()
            else:
                print('Terminated: No test user')
                exit()

        elif self.conf.train_test_split == 'random':
            labels = self.motion_info[:, 4]
            indices = np.arange(labels.shape[0])

            train_indices, test_indices, _, _ = train_test_split(indices,
                                                                 labels,
                                                                 stratify=labels,
                                                                 test_size=self.conf.train_test_hold_out,
                                                                 random_state=48)

        elif self.conf.train_test_split in ['ldo_start', 'ldo_end', 'ldo_random']:
            days = np.unique(self.motion_info[:, 3])

            if self.conf.train_test_split == 'ldo_start':
                test_days = days[:self.conf.train_test_hold_out]
            elif self.conf.train_test_split == 'ldo_end':
                test_days = days[-self.conf.train_test_hold_out:]
            elif self.conf.train_test_split == 'ldo_random':
                with temp_seed(48):
                    test_days = np.random.choice(days, size=self.conf.train_test_hold_out, replace=False)

            train_indices = np.argwhere(~np.in1d(self.motion_info[:, 3], test_days)).flatten()
            test_indices = np.argwhere(np.in1d(self.motion_info[:, 3], test_days)).flatten()

        if self.conf.train_val_split == 'lopo_stratified':
            split_ratio = self.conf.train_val_hold_out
            if isinstance(split_ratio, float):
                how = 'old'
                if how == 'old':
                    train_val_indices = []
                    for index, (label, user) in enumerate(zip(self.motion_info[:, 4],
                                                              self.motion_info[:, 2])):

                        if user == self.conf.train_test_hold_out:
                            continue

                        else:
                            train_val_indices.append([index, user * 10 + label])

                    train_indices, val_indices = lopo_split_old(train_val_indices,
                                                                self.conf.train_val_hold_out)

                elif how == 'new':
                    train_indices, val_indices = lopo_split(self.motion_info[train_indices],
                                                            train_indices,
                                                            self.conf.train_val_hold_out)
            else:
                print('Terminated: Wrong Hold-Out Type')
                exit()

        elif self.conf.train_val_split == 'random':
            train_labels = self.motion_info[train_indices, -1]

            train_indices, val_indices, _, _ = train_test_split(train_indices,
                                                                train_labels,
                                                                stratify=train_labels,
                                                                test_size=self.conf.train_val_hold_out)

        elif self.conf.train_val_split in ['ldo_start', 'ldo_end', 'ldo_random']:
            days = np.unique(self.motion_info[train_indices, 3])

            if self.conf.train_test_split == 'ldo_start':
                val_days = days[:self.conf.train_val_hold_out]
            elif self.conf.train_test_split == 'ldo_end':
                val_days = days[-self.conf.train_val_hold_out:]
            elif self.conf.train_test_split == 'ldo_random':
                with temp_seed(48):
                    val_days = np.random.choice(days, size=self.conf.train_val_hold_out, replace=False)

            train_indices = np.argwhere(~np.in1d(self.motion_info[:, 3], val_days)).flatten()
            val_indices = np.argwhere(np.in1d(self.motion_info[:, 3], val_days)).flatten()

        self.train_indices = train_indices
        self.train_size = len(train_indices)
        self.val_indices = val_indices
        self.val_size = len(val_indices)
        self.test_indices = test_indices
        self.test_size = len(test_indices)

    def drop_nans(self, after_split: bool = False):
        Out = []
        In = []
        positions = list(self.motion.keys())

        for i in range(len(self.motion_info)):
            has_nan = False
            for position in positions:
                segment = self.motion[position][i]
                if np.isnan(segment).any():
                    has_nan = True
                    break

            if has_nan:
                Out.append(i)
            else:
                In.append(i)

        if after_split:
            self.train_indices = [train_index for train_index in self.train_indices if train_index not in Out]
            self.val_indices = [val_index for val_index in self.val_indices if val_index not in Out]
            self.test_indices = [test_index for test_index in self.test_indices if test_index not in Out]

            self.train_size = len(self.train_indices)
            self.val_size = len(self.val_indices)
            self.test_size = len(self.test_indices)

        else:
            self.motion_info = self.motion_info[In]
            for position in self.motion.keys():
                self.motion[position] = self.motion[position][In]

    def drop_labels(self, after_split: bool = False):
        Out = []
        for target in self.conf.exclude_modes:
            target_id = self.mode_ids[target]

            tg_out = np.argwhere(self.motion_info[:, -1] == target_id).flatten().tolist()
            Out.extend(tg_out)

            self.modes.remove(target)

        self.mode_ids = {mode: id for id, mode in enumerate(self.modes)}

        if after_split:
            self.train_indices = [train_index for train_index in self.train_indices if train_index not in Out]
            self.val_indices = [val_index for val_index in self.val_indices if val_index not in Out]
            self.test_indices = [test_index for test_index in self.test_indices if test_index not in Out]

            self.train_size = len(self.train_indices)
            self.val_size = len(self.val_indices)
            self.test_size = len(self.test_indices)

        else:
            self.motion_info = np.delete(self.motion_info, Out, axis=0)
            for position in self.motion.keys():
                self.motion[position] = np.delete(self.motion[position], Out, axis=0)

        if self.conf.motorized:
            motorized_modes = ['car', 'bus', 'train', 'subway']

            for motorized_mode in motorized_modes:
                self.modes.remove(motorized_mode)
            self.modes.append('motorized')

            self.mode_ids = {}
            counter = 0
            for id, mode in self.mode_names.items():
                if mode in self.modes:
                    self.mode_ids[mode] = counter
                    counter += 1
                elif mode in motorized_modes:
                    self.mode_ids[mode] = len(self.modes) - 1

        self.n_modes = len(self.modes)

    def drop_GPS_loss(self, thres: int = 1):
        dropped = []
        for i, loc_index in enumerate(self.sync_pairs):
            if loc_index == -1:
                if thres != 0:
                    dropped.append(i)
                continue

            if sum(~np.isnan(self.location[loc_index, :, 0])) < thres:
                dropped.append(i)

        self.train_indices = [train_index for train_index in self.train_indices if train_index not in dropped]
        self.val_indices = [val_index for val_index in self.val_indices if val_index not in dropped]
        self.test_indices = [test_index for test_index in self.test_indices if test_index not in dropped]

        self.train_size = len(self.train_indices)
        self.val_size = len(self.val_indices)
        self.test_size = len(self.test_indices)

    def get_transformers(self, motion_transfer=False, location_transfer=False):
        if self.conf.motion_form == 'spectrogram':
            self.motion_transformer = spectro_transformer()
        elif self.conf.motion_form == 'temporal':
            self.motion_transformer = temporal_transformer()

        self.motion_shape = self.motion_transformer.shape

        self.location_transformer = series_transformer()
        self.loc_window_shape, self.loc_features_shape = self.location_transformer.shape

        self.labels_transformer = one_hot_transformer(self.mode_names,
                                                      self.mode_ids,
                                                      self.conf.motorized)

        if motion_transfer:
            if self.conf.get_position:
                self.input_shape = (*self.motion_shape, self.conf.mot_bag_size)
                self.input_type = (*[tf.float32 for _ in self.motion_shape], tf.string)
            else:
                self.input_shape = (*self.motion_shape,)
                self.input_type = (*[tf.float32 for _ in self.motion_shape],)

        elif location_transfer:
            self.input_shape = (self.loc_window_shape, self.loc_features_shape)
            self.input_type = (tf.float32, tf.float32)

        else:
            if self.conf.get_position:
                self.input_shape = (
                    *self.motion_shape, self.loc_window_shape, self.loc_features_shape, self.conf.mot_bag_size)
                self.input_type = (*[tf.float32 for _ in self.motion_shape], tf.float32, tf.float32, tf.string)
            else:
                self.input_shape = (*self.motion_shape, self.loc_window_shape, self.loc_features_shape)
                self.input_type = (*[tf.float32 for _ in self.motion_shape], tf.float32, tf.float32)

        self.output_shape = self.labels_transformer.n_modes
        self.output_type = tf.float32

    def create_instances(self, index, positions):
        instances = []
        for i, position in enumerate(positions):
            bag = self.motion[position][index]
            instances.append(bag[i * self.conf.mot_stride: i * self.conf.mot_stride + self.conf.mot_length])

        return np.array(instances)

    def get_Xy(self, index, positions,
               motion_only=False, location_only=False,
               is_training=False, one_instance=False):
        if not location_only:
            mot_instances = self.create_instances(index, positions)
            mot_instances = self.motion_transformer(mot_instances, is_training)

            if one_instance:
                mot_instances = [mot_instance[np.newaxis, ...] for mot_instance in mot_instances]

        if not motion_only:
            loc_index = self.sync_pairs[index]

            if loc_index == -1:
                loc_window = np.zeros((self.conf.loc_length, len(self.loc_features)))
                loc_window[...] = np.nan
            else:
                loc_window = self.location[loc_index]

            loc_window, loc_features = self.location_transformer(loc_window, is_training)

            if one_instance:
                loc_window = loc_window[np.newaxis, ...]
                loc_features = loc_features[np.newaxis, ...]

        y = self.labels_transformer(self.motion_info[index])

        if motion_only:
            if self.conf.get_position:
                return (*mot_instances, positions), y
            else:
                return (*mot_instances,), y

        elif location_only:
            return (loc_window, loc_features), y

        else:
            if self.conf.get_position:
                return (*mot_instances, loc_window, loc_features, positions), y
            else:
                return (*mot_instances, loc_window, loc_features), y

    def to_generator(self, motion_only=False, location_only=False,
                     is_training=False, is_validation=False, is_test=False):
        if is_training:
            indices = self.train_indices

        elif is_validation:
            indices = self.val_indices

        elif is_test:
            indices = self.test_indices

        def gen():
            for index in indices:
                bag_index = index[0]
                bag_positions = index[1]

                X, y = self.get_Xy(bag_index, bag_positions,
                                   motion_only, location_only,
                                   is_training)
                yield X, y

        return tf.data.Dataset.from_generator(
            gen,
            output_types=(self.input_type, self.output_type),
            output_shapes=(self.input_shape, self.output_shape)
        )

    def generate(self, motion_only=False, location_only=False):
        train = self.to_generator(motion_only, location_only, is_training=True)
        val = self.to_generator(motion_only, location_only, is_validation=True)
        test = self.to_generator(motion_only, location_only, is_test=True)

        return train, val, test

    def batch_and_prefetch(self, train, val, test):

        train = train.cache().shuffle(1000).repeat().batch(batch_size=self.conf.batch_size).prefetch(tf.data.AUTOTUNE)
        val = val.cache().batch(batch_size=self.conf.batch_size).prefetch(tf.data.AUTOTUNE)
        test = test.batch(batch_size=self.conf.batch_size).prefetch(tf.data.AUTOTUNE)

        return train, val, test

    def __call__(self, motion_transfer=False, location_transfer=False, bagging=True, batch_prefetch=True, init=False):
        if init:
            self.__init__()

        self.init_config()

        self.drop_labels(False)
        self.drop_nans(False)

        if bagging and not motion_transfer:
            self.combine_to_bags()

        self.split()

        if not motion_transfer:
            self.drop_GPS_loss(thres=0)

        self.assign_position()

        self.get_transformers(motion_transfer, location_transfer)

        train, val, test = self.generate(motion_transfer, location_transfer)

        if batch_prefetch:
            train, val, test = self.batch_and_prefetch(train, val, test)

        return train, val, test


if __name__ == '__main__':
    A = Builder(regenerate=False)
    train, val, test = A(motion_transfer=False, batch_prefetch=False)
    for i, instance in test.take(100):
        pass