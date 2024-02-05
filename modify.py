from config_parser import Parser
from resample import Resampler
import os
import shutil
import json
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle
import pickle


class Modifier:

    def __init__(self, delete_src: bool = False, delete_dst: bool = False):
        config = Parser()
        config.get_args()
        self.conf = config

        self.from_path = os.path.join(os.path.expanduser('~'), self.conf.path, self.conf.dataset + '-resampled')

        self.resampler = Resampler(delete_src=delete_src, delete_dst=delete_src)
        src_exists = len(os.listdir(self.from_path)) > 0
        self.motion, self.location, self.labels = self.resampler(load=src_exists, verbose=not src_exists)

        self.n_mot = self.resampler.n_mot
        self.n_loc = self.resampler.n_loc
        self.n_lbs = self.resampler.n_lbs

        self.modified_path = os.path.join(os.path.expanduser('~'), self.conf.path, self.conf.dataset + '-modified')

        if delete_dst:
            try:
                shutil.rmtree(self.modified_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if not os.path.exists(self.modified_path):
            os.makedirs(self.modified_path)

        self.avail_mot_sensors = ['acc', 'gyr', 'mag', 'lacc', 'bar']
        self.avail_mot_features = ['normXYZ', 'jerk', 'jerkX', 'jerkY', 'jerkZ']

        with open('info/' + self.conf.dataset + '/initial_features.pickle', 'rb') as handle:
            features = pickle.load(handle)
            self.mot_features = features['mot_features']
            self.loc_features = features['loc_features']
            self.lbs_features = features['lbs_features']

        self.n_mot_features = len(self.mot_features)
        for v, virtual in enumerate(self.conf.mot_virtuals):
            sensor, feature = virtual.split('_')
            if sensor not in self.avail_mot_sensors:
                print("Terminated: Unknown sensor: %s" % sensor)
                quit()
            if feature not in self.avail_mot_features:
                print("Terminated: Unknown feature: %s" % feature)
                quit()
            self.mot_features[virtual] = self.n_mot_features + v
        self.n_mot_features += len(self.conf.mot_virtuals)

        self.n_lbs_features = len(self.lbs_features)

        self.avail_loc_features = ['velocity', 'acceleration']

        self.n_loc_features = len(self.loc_features)
        for f, feature in enumerate(self.conf.loc_virtuals):
            if feature not in self.avail_loc_features:
                print("Terminated: Unknown feature: %s" % virtual)
                quit()

            self.loc_features[feature] = self.n_loc_features + f
        self.n_loc_features += len(self.conf.loc_virtuals)

        features = {
            'mot_features': self.mot_features,
            'loc_features': self.loc_features,
            'lbs_features': self.lbs_features,
        }
        with open('info/' + self.conf.dataset + '/modified_features.pickle', 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def show_sizes(self):
        print("Motion sizes:")
        print(json.dumps(self.n_mot, indent=4))
        print("Location sizes:")
        print(json.dumps(self.n_loc, indent=4))

    def __call__(self, verbose: bool = True, load: bool = False, delete_data: bool = True):
        self.verbose = verbose
        motion, location, labels = self.modify(load)
        if delete_data:
            del self.motion
            del self.location
            del self.labels

        return motion, location, labels

    def modify(self, load: bool = False):
        load = os.path.exists(self.modified_path) and load

        motion, labels = self.modify_motion(load)
        location = self.modify_location(load)

        return motion, location, labels

    def lowpass(self, x: np.ndarray, alpha=0.8) -> np.ndarray:
        gravity_estimate = np.zeros_like(x)

        for k in range(1, len(x)):
            if np.isnan(x[k]):
                gravity_estimate[k] = 0
            else:
                gravity_estimate[k] = alpha * gravity_estimate[k - 1] + (1 - alpha) * x[k]

        linear_acceleration = x - gravity_estimate

        return linear_acceleration

    def filter_signals(self, orig_signals: np.ndarray) -> np.ndarray:
        fil_signals = orig_signals.copy()

        if self.conf.mot_filter is None:
            pass
        elif self.conf.mot_filter == 'lowpass':
            fil_signals[:, 1:] = np.apply_along_axis(
                lambda x: self.lowpass(x),
                axis=0, arr=orig_signals[:, 1:])

        return fil_signals

    def smooth_signals(self, orig_signals: np.ndarray) -> np.ndarray:
        sm_signals = orig_signals.copy()

        if self.conf.mot_smoother is None:
            pass
        elif self.conf.mot_smoother == 'moving_average':
            w = self.conf.mot_sm_w
            sm_signals[:, 1:] = np.apply_along_axis(
                lambda x: np.convolve(x, np.ones(w), 'same') / w,
                axis=0, arr=orig_signals[:, 1:])

        return sm_signals

    def expand_signals(self, orig_signals: np.ndarray) -> np.ndarray:
        orient_ids = {'X': 0, 'Y': 1, 'Z': 2}
        virt_signals = np.zeros((orig_signals.shape[0], len(self.conf.mot_virtuals)))

        for s, signal in enumerate(self.conf.mot_virtuals):
            sensor = signal.split('_')[0]
            feature = signal.split('_')[1]

            if sensor in self.avail_mot_sensors:
                if sensor == 'acc':
                    sensor_features = [self.mot_features[feature] for feature in ['Acc_x', 'Acc_y', 'Acc_z']]
                elif sensor == 'gyr':
                    sensor_features = [self.mot_features[feature] for feature in ['Gyr_x', 'Gyr_y', 'Gyr_z']]
                elif sensor == 'mag':
                    sensor_features = [self.mot_features[feature] for feature in ['Mag_x', 'Mag_y', 'Mag_z']]
                elif sensor == 'lacc':
                    sensor_features = [self.mot_features[feature] for feature in ['Lacc_x', 'Lacc_y', 'Lacc_z']]
                sensor_signals = orig_signals[:, sensor_features]

            else:
                sensor_signals = None
                print('Unknown sensor {}'.format(sensor))

            if feature in self.avail_mot_features:
                if feature == 'normXYZ':
                    virt_signal = np.sqrt(np.sum(np.square(sensor_signals), axis=1))

                elif feature in ['jerkX', 'jerkY', 'jerkZ']:
                    orientation = feature[-1]
                    id = orient_ids[orientation]
                    jerk = np.diff(sensor_signals[:, id], axis=0) * self.conf.motion_fs
                    virt_signal = np.concatenate(([np.nan], jerk))

                elif feature == 'jerk':
                    ds_dt = np.diff(sensor_signals, axis=0) * self.conf.motion_fs
                    jerk = np.sqrt(np.sum(np.square(ds_dt), axis=1))
                    virt_signal = np.concatenate(([np.nan], jerk))


            else:
                virt_signal = None
                print('Unknown feature {}'.format(feature))

            virt_signals[:, s] = virt_signal

        exp_signals = np.concatenate((orig_signals, virt_signals), axis=1)
        return exp_signals

    def rescale_signals(self, orig_signals: np.ndarray, mean: float=None, std:float=None) -> np.ndarray:
        sc_signals = orig_signals.copy()

        if self.conf.mot_rescaler == 'standard':

            if mean is not None and std is not None:
                rescaler = StandardScaler(with_mean=True, with_std=True)
                rescaler.mean_ = mean
                rescaler.scale_ = std
                sc_signals[:, 1:] = rescaler.transform(orig_signals[:, 1:])

            else:
                rescaler = StandardScaler()
                sc_signals[:, 1:] = rescaler.fit_transform(orig_signals[:, 1:])

        return sc_signals

    def modify_motion(self, load: bool = False) -> Tuple[Dict, Dict]:
        print("Modifying Motion Data...")

        modified_motion = {}

        for sub_id in self.motion.keys():
            modified_motion[sub_id] = {}
            for date in self.motion[sub_id].keys():
                modified_motion[sub_id][date] = {}
                for position in self.motion[sub_id][date].keys():
                    filename = sub_id + '_' + date + '_' + position + '_' + 'motion' + '.mmap'
                    dst = os.path.join(self.modified_path, filename)
                    exists = os.path.exists(dst)

                    if load and exists:
                        modified_motion[sub_id][date][position] = None
                    else:
                        orig_signals = self.motion[sub_id][date][position]
                        modified_signals = self.expand_signals(orig_signals)
                        modified_signals = self.filter_signals(modified_signals)
                        modified_signals = self.smooth_signals(modified_signals)
                        modified_motion[sub_id][date][position] = modified_signals

                filename = sub_id + '_' + date + '_' + 'labels' + '.mmap'
                dst = os.path.join(self.modified_path, filename)
                exists = os.path.exists(dst)

                n_lines = self.n_lbs[sub_id][date]
                shape = (n_lines, self.n_lbs_features)

                if load and exists:
                    pass
                else:
                    mmap = np.memmap(dst, mode='w+', dtype=np.int64, shape=shape)
                    mmap[...] = self.labels[sub_id][date]

        total_mean = None
        total_std = None
        if self.conf.mot_rescaler == 'standard' and not load:
            all_signals = None
            for sub_id in modified_motion.keys():
                for date in modified_motion[sub_id].keys():
                    for position in modified_motion[sub_id][date].keys():
                        modified_signals = modified_motion[sub_id][date][position]
                        if all_signals is None:
                            all_signals = modified_signals[:, 1:]
                        else:
                            all_signals = np.concatenate((all_signals, modified_signals[:, 1:]), axis=0)

            total_mean = np.nanmean(all_signals, axis=0)
            total_std = np.nanstd(all_signals, axis=0)
            del all_signals

        print(total_mean)
        print(total_std)
        for sub_id in modified_motion.keys():
            for date in modified_motion[sub_id].keys():
                for position in modified_motion[sub_id][date].keys():
                    filename = sub_id + '_' + date + '_' + position + '_' + 'motion' + '.mmap'
                    dst = os.path.join(self.modified_path, filename)
                    exists = os.path.exists(dst)

                    n_lines = self.n_mot[sub_id][date][position]
                    shape = (n_lines, self.n_mot_features)

                    if load and exists:
                        mmap = np.memmap(dst, mode='r+', dtype=np.float64, shape=shape)
                    else:
                        mmap = np.memmap(dst, mode='w+', dtype=np.float64, shape=shape)

                        modified_signals = modified_motion[sub_id][date][position]
                        modified_signals = self.rescale_signals(modified_signals, total_mean, total_std)

                        mmap[...] = modified_signals

                    modified_motion[sub_id][date][position] = mmap

        return modified_motion, self.labels

    def get_velocity(self, lat, long, time, i):
        if np.isnan([lat[i], long[i], lat[i-1], long[i-1]]).any():
            return np.nan

        point1 = (lat[i - 1], long[i - 1])
        point2 = (lat[i], long[i])
        distance = great_circle(point1, point2).m
        velocity = 1000. * distance / (time[i] - time[i-1])
        return velocity

    def get_acceleration(self, lat, long, time, i):
        vel1 = self.get_velocity(lat, long, time, i-1)
        vel2 = self.get_velocity(lat, long, time, i)
        if np.isnan([vel1, vel2]).any():
            return np.nan

        dvel = vel2 - vel1
        acceleration = 1000. * dvel / (time[i] - time[i-1])

        return acceleration

    def expand_features(self, orig_features: np.ndarray) -> np.ndarray:
        virt_features = np.zeros((orig_features.shape[0], len(self.conf.loc_virtuals)))

        for f, feature in enumerate(self.conf.loc_virtuals):
            if feature in self.avail_loc_features:
                time = orig_features[:, self.loc_features['Time']]
                lat = orig_features[:, self.loc_features['Lat']]
                long = orig_features[:, self.loc_features['Long']]

                if feature == 'velocity':
                    velocity = np.array([self.get_velocity(lat, long, time, i) for
                                     i in range(1, orig_features.shape[0])])
                    virt_feature = np.concatenate(([np.nan], velocity))

                elif feature == 'acceleration':
                    acceleration = np.array([self.get_acceleration(lat, long, time, i) for
                                     i in range(2, orig_features.shape[0])])
                    virt_feature = np.concatenate(([np.nan, np.nan], acceleration))

            else:
                virt_feature = None
                print('Unknown feature {}'.format(feature))

            virt_features[:, f] = virt_feature

        exp_features = np.concatenate((orig_features, virt_features), axis=1)

        return exp_features

    def rescale_features(self, orig_signals: np.ndarray, mean: float = None, std: float = None) -> np.ndarray:
        sc_signals = orig_signals.copy()

        if self.conf.mot_rescaler == 'standard':
            if mean is not None and std is not None:
                rescaler = StandardScaler(with_mean=True, with_std=True)
                rescaler.mean_ = mean
                rescaler.scale_ = std
                sc_signals[:, 5:] = rescaler.fit_transform(orig_signals[:, 5:])

            else:
                rescaler = StandardScaler()
                sc_signals[:, 5:] = rescaler.fit_transform(orig_signals[:, 5:])

        return sc_signals

    def modify_location(self, load: bool = False, n_exists: bool = False) -> Dict:
        print("Modifying Location Data...")

        modified_location = {}
        for sub_id in self.location.keys():
            modified_location[sub_id] = {}
            for date in self.location[sub_id].keys():
                modified_location[sub_id][date] = {}
                for position in self.location[sub_id][date].keys():
                    filename = sub_id + '_' + date + '_' + position + '_' + 'location' + '.mmap'
                    dst = os.path.join(self.modified_path, filename)
                    exists = os.path.exists(dst)

                    if load and exists:
                        modified_location[sub_id][date][position] = None
                    else:
                        orig_features = self.location[sub_id][date][position]
                        modified_features = self.expand_features(orig_features)
                        modified_features = self.rescale_features(modified_features)
                        modified_location[sub_id][date][position] = modified_features

        total_mean = None
        total_std = None
        if self.conf.loc_rescaler == 'standard' and not load:
            all_features = None
            for sub_id in self.location.keys():
                for date in self.location[sub_id].keys():
                    for position in self.location[sub_id][date].keys():
                        modified_features = modified_location[sub_id][date][position]
                        if all_features is None:
                            all_features = modified_features[:, 5:]
                        else:
                            all_features = np.concatenate((all_features, modified_features[:, 5:]), axis=0)

            total_mean = np.nanmean(all_features, axis=0)
            total_std = np.nanstd(all_features, axis=0)
            del all_features

        for sub_id in self.location.keys():
            for date in self.location[sub_id].keys():
                for position in self.location[sub_id][date].keys():
                    filename = sub_id + '_' + date + '_' + position + '_' + 'location' + '.mmap'
                    dst = os.path.join(self.modified_path, filename)
                    exists = os.path.exists(dst)

                    n_lines = self.n_loc[sub_id][date][position]
                    shape = (n_lines, self.n_loc_features)

                    if load and exists:
                        mmap = np.memmap(dst, mode='r+', dtype=np.float64, shape=shape)
                    else:
                        mmap = np.memmap(dst, mode='w+', dtype=np.float64, shape=shape)

                        modified_features = modified_location[sub_id][date][position]
                        modified_features = self.rescale_features(modified_features, total_mean, total_std)

                        mmap[...] = modified_features

                    modified_location[sub_id][date][position] = mmap

        return modified_location


if __name__ == '__main__':
    A = Modifier(delete_src=False, delete_dst=True)
    A(load=False)