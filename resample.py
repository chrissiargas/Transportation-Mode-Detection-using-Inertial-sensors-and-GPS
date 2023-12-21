from config_parser import Parser
from extract import Extractor
import os
import shutil
import json
import math
import numpy as np
import scipy.signal as ssn
from typing import Tuple, Dict
from scipy.interpolate import interp1d
import pickle


class Resampler:

    def __init__(self, delete_src: bool = False, delete_dst: bool = False):
        config = Parser()
        config.get_args()
        self.conf = config

        self.from_path = os.path.join(os.path.expanduser('~'), self.conf.path, self.conf.dataset + '-extracted')

        self.extractor = Extractor(delete_src)
        src_exists = len(os.listdir(self.from_path)) > 0
        self.motion, self.location, self.labels = self.extractor(load=src_exists, verbose=not src_exists)

        self.n_mot = self.extractor.n_mot
        self.n_loc = self.extractor.n_loc
        self.n_lbs = self.extractor.n_lbs

        self.resampled_path = os.path.join(os.path.expanduser('~'), self.conf.path, self.conf.dataset + '-resampled')

        if delete_dst:
            try:
                shutil.rmtree(self.resampled_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if not os.path.exists(self.resampled_path):
            os.makedirs(self.resampled_path)

        self.threshold = self.conf.max_distance * 1000.
        self.new_location_T = self.conf.location_T * 1000.

    def __call__(self, verbose: bool = True, load: bool = False, delete_data: bool = True):
        self.verbose = verbose
        motion, location, labels = self.resample(load)
        if delete_data:
            del self.motion
            del self.location
            del self.labels

        return motion, location, labels

    def show_sizes(self):
        print("Motion sizes:")
        print(json.dumps(self.n_mot, indent=4))
        print("Location sizes:")
        print(json.dumps(self.n_loc, indent=4))

    def resample(self, load: bool = False):
        load = os.path.exists(self.resampled_path) and load

        n_exists = False
        if load and os.path.exists('info/' + self.conf.dataset + '/resampled_sizes.pickle'):
            with open('info/' + self.conf.dataset + '/resampled_sizes.pickle', 'rb') as handle:
                sizes = pickle.load(handle)
                self.n_mot = sizes['n_mot']
                self.n_loc = sizes['n_loc']
                self.n_lbs = sizes['n_lbs']
                n_exists = True

        location = self.resample_location(load, n_exists)
        motion, labels = self.resample_motion(load, n_exists)

        if not load:
            sizes = {
                'n_mot': self.n_mot,
                'n_loc': self.n_loc,
                'n_lbs': self.n_lbs,
            }
            with open('info/' + self.conf.dataset + '/resampled_sizes.pickle', 'wb') as handle:
                pickle.dump(sizes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return motion, location, labels

    def resample_motion(self, load: bool = False, n_exists: bool = False) -> Tuple[Dict, Dict]:
        print("Resampling Motion Data...")
        step = int(self.conf.old_motion_fs / self.conf.motion_fs)

        resampled_motion = {}
        resampled_labels = {}

        for sub_id in self.motion.keys():
            resampled_motion[sub_id] = {}
            resampled_labels[sub_id] = {}
            for date in self.motion[sub_id].keys():
                resampled_motion[sub_id][date] = {}
                resampled_labels[sub_id][date] = {}

                if n_exists:
                    n_lines = self.n_lbs[sub_id][date]
                else:
                    n_lines = math.ceil(self.n_lbs[sub_id][date] / step)

                if self.verbose:
                    print(f'User: {sub_id}, date: {date}, feature: Motion')
                    print('Number of Lines After Resampling: {}'.format(n_lines))
                    print()

                for position in self.motion[sub_id][date].keys():
                    filename = sub_id + '_' + date + '_' + position + '_' + 'motion' + '.mmap'
                    dst = os.path.join(self.resampled_path, filename)
                    exists = os.path.exists(dst)

                    shape = (n_lines, self.extractor.n_mot_features)

                    if load and exists:
                        mmap = np.memmap(dst, mode='r+', dtype=np.float64, shape=shape)
                    else:
                        mmap = np.memmap(dst, mode='w+', dtype=np.float64, shape=shape)

                        mmap[:, 0] = self.motion[sub_id][date][position][::step, 0]
                        mmap[:, 1:] = ssn.decimate(
                            x=self.motion[sub_id][date][position][:, 1:],
                            q=step,
                            ftype='fir',
                            axis=0
                        )

                    self.n_mot[sub_id][date][position] = n_lines
                    resampled_motion[sub_id][date][position] = mmap

                filename = sub_id + '_' + date + '_' + 'labels' + '.mmap'
                dst = os.path.join(self.resampled_path, filename)
                exists = os.path.exists(dst)

                shape = (n_lines, self.extractor.n_lbs_features)

                if load and exists:
                    mmap = np.memmap(dst, mode='r+', dtype=np.int64, shape=shape)
                else:
                    mmap = np.memmap(dst, mode='w+', dtype=np.int64, shape=shape)
                    mmap[...] = self.labels[sub_id][date][::step]

                self.n_lbs[sub_id][date] = n_lines
                resampled_labels[sub_id][date] = mmap

        return resampled_motion, resampled_labels

    def get_valid_points(self, x: np.ndarray, n: int) -> np.ndarray:
        valid_points = []

        j = 0
        min_distance = np.inf
        timestamps = iter(np.arange(start=x[0, 0], stop=x[-1, 0] + 1e-4, step=self.new_location_T))
        timestamp = next(timestamps)

        try:
            while True:
                distance = np.abs((x[j, 0] - timestamp))

                if j == n-1:
                    if distance <= min_distance:
                        if distance <= self.threshold:
                            valid_points.append(j)
                        else:
                            valid_points.append(-1)

                    elif min_distance <= self.threshold:
                            valid_points.append(j-1)

                    return np.array(valid_points)

                if distance <= min_distance:
                    min_distance = distance
                    j += 1

                else:
                    if min_distance <= self.threshold:
                        valid_points.append(j-1)
                        timestamp = next(timestamps)

                    else:
                        valid_points.append(-1)
                        timestamp = next(timestamps)

                        for last in range(len(valid_points)):
                            if valid_points[-last-1] != -1:
                                j = valid_points[-last-1]
                                break

                    min_distance = np.inf

        except StopIteration:
            return np.array(valid_points)

    def resample_location(self, load: bool = False, n_exists: bool = False) -> Dict:
        print("Resampling Location Data...")

        resampled_location = {}
        for sub_id in self.location.keys():
            resampled_location[sub_id] = {}
            for date in self.location[sub_id].keys():
                resampled_location[sub_id][date] = {}
                for position in self.location[sub_id][date].keys():
                    filename = sub_id + '_' + date + '_' + position + '_' + 'location' + '.mmap'
                    dst = os.path.join(self.resampled_path, filename)
                    exists = os.path.exists(dst)

                    old_loc = self.location[sub_id][date][position]
                    old_n = self.n_loc[sub_id][date][position]

                    if load and exists:
                        new_n = self.n_loc[sub_id][date][position]
                    else:
                        valid_points = self.get_valid_points(x = old_loc, n = old_n)
                        new_n = valid_points.shape[0]
                        self.n_loc[sub_id][date][position] = new_n

                    if self.verbose:
                        print(f'User: {sub_id}, date: {date}, position: {position}, feature: Location')
                        print('Number of Lines After Resampling: {}'.format(new_n))
                        print()

                    shape = (new_n, self.extractor.n_loc_features)

                    if load and exists:
                        mmap = np.memmap(dst, mode='r+', dtype=np.float64, shape=shape)
                    else:
                        mmap = np.memmap(dst, mode='w+', dtype=np.float64, shape=shape)

                        old_t = old_loc[:, 0]
                        old_recordings = old_loc[:, 1:]
                        new_t = np.arange(old_t[0], old_t[-1] + 1e-4, step=self.new_location_T)
                        f = interp1d(old_t, old_recordings, kind=self.conf.sampling_method, axis=0, fill_value='extrapolate')
                        new_recordings = f(new_t)
                        new_recordings[valid_points == -1] = np.nan
                        mmap[...] = np.concatenate((new_t[:, np.newaxis], new_recordings), axis=1)

                    resampled_location[sub_id][date][position] = mmap

        return resampled_location

import pandas as pd
if __name__ == '__main__':
    A = Resampler(delete_src=False)
    A(load=False)


