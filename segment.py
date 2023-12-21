from config_parser import Parser
from modify import Modifier
import os
import shutil
import json
import math
import numpy as np
import scipy.signal as ssn
from typing import Tuple, Dict
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle


class Segmenter:
    def __init__(self, delete_src: bool = False, delete_dst: bool = False):
        config = Parser()
        config.get_args()
        self.conf = config

        self.from_path = os.path.join(os.path.expanduser('~'), self.conf.path, self.conf.dataset + '-modified')

        self.modifier = Modifier(delete_src, delete_src)
        src_exists = len(os.listdir(self.from_path)) > 0
        self.motion, self.location, self.labels = self.modifier(load=src_exists, verbose=not src_exists)

        self.n_mot = self.modifier.n_mot
        self.n_loc = self.modifier.n_loc
        self.n_lbs = self.modifier.n_lbs

        self.segmented_path = os.path.join(os.path.expanduser('~'), self.conf.path, self.conf.dataset + '-segmented')

        if delete_dst:
            try:
                shutil.rmtree(self.segmented_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if not os.path.exists(self.segmented_path):
            os.makedirs(self.segmented_path)

        with open('info/' + self.conf.dataset + '/modified_features.pickle', 'rb') as handle:
            features = pickle.load(handle)
            self.mot_features = features['mot_features']
            self.loc_features = features['loc_features']
            self.lbs_features = features['lbs_features']

        del self.mot_features['Time']
        self.mot_channels = len(self.mot_features)
        del self.loc_features['Time']
        self.loc_channels = len(self.loc_features)

        self.mot_length = self.conf.mot_length + (self.conf.mot_bag_size - 1) * self.conf.mot_bag_step

        features = {
            'mot_features': self.mot_features,
            'loc_features': self.loc_features,
        }
        with open('info/' + self.conf.dataset + '/segmented_features.pickle', 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def show_sizes(self):
        print("Motion sizes:")
        print(json.dumps(self.n_mot, indent=4))
        print("Location sizes:")
        print(json.dumps(self.n_loc, indent=4))

    def __call__(self, verbose: bool = True, load: bool = False, delete_data: bool = True):
        self.verbose = verbose
        motion, mot_info, location, loc_info = self.segment(load)
        if delete_data:
            del self.motion
            del self.location
            del self.labels

        return motion, mot_info, location, loc_info

    def segment(self, load: bool = False):
        load = os.path.exists(self.segmented_path) and load

        n_exists = False
        if load and os.path.exists('info/' + self.conf.dataset + '/segmented_sizes.pickle'):
            with open('info/' + self.conf.dataset + '/segmented_sizes.pickle', 'rb') as handle:
                sizes = pickle.load(handle)
                self.n_mot = sizes['n_mot']
                self.n_loc = sizes['n_loc']
                n_exists = True

        location, loc_info = self.segment_location(load, n_exists)
        motion, mot_info = self.segment_motion(load, n_exists)

        if not load:
            sizes = {
                'n_mot': self.n_mot,
                'n_loc': self.n_loc,
            }
            with open('info/' + self.conf.dataset + '/segmented_sizes.pickle', 'wb') as handle:
                pickle.dump(sizes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return motion, mot_info, location, loc_info

    def get_mot_shape(self, position: str):
        total_segments = 0

        for sub_id in self.n_mot.keys():
            for date in self.n_mot[sub_id].keys():
                n = self.n_mot[sub_id][date][position]
                n_segments = max(0, math.ceil((n - self.mot_length + 1) / self.conf.mot_stride))
                total_segments += n_segments

        return total_segments, self.mot_length, self.mot_channels

    def to_segments(self, X: np.ndarray, n_segments: int, length: int, stride: int) -> np.ndarray:
        X = X.copy()
        X = np.lib.stride_tricks.as_strided(
            X,
            shape=(n_segments, length, X.shape[1]),
            strides=(stride * X.strides[0], X.strides[0], X.strides[1]))

        return X

    def majority_voting(self, x: np.ndarray) -> int:
        stride = self.conf.mot_stride
        length = self.conf.mot_length
        pivot = self.conf.mot_bag_size // 2
        pivot_instance = x[pivot * stride: pivot * stride + length]

        counts = np.bincount(pivot_instance)
        return np.argmax(counts)

    def get_labels(self, X: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda x: self.majority_voting(x), axis=1, arr=X)

    def segment_motion(self, load: bool = False, n_exists: bool = False) -> Tuple[Dict, np.ndarray]:
        print("Segmenting Motion Data...")

        for sub_id in self.motion.keys():
            for date in self.motion[sub_id].keys():
                positions = list(self.motion[sub_id][date].keys())
                break

        segmented_motion = {}
        for position in positions:
            filename = position + '_' + 'motion' + '.mmap'
            dst = os.path.join(self.segmented_path, filename)
            exists = os.path.exists(dst)

            if load and exists:
                if n_exists:
                    total_segments, length, channels = self.n_mot
                    shape = (total_segments, length, channels)

                mmap = np.memmap(dst, mode='r+', dtype=np.float32, shape=shape)

            else:
                total_segments, length, channels = self.get_mot_shape(position)
                shape = (total_segments, length, channels)
                mmap = np.memmap(dst, mode='w+', dtype=np.float32, shape=shape)

                offset = 0
                for sub_id in self.motion.keys():
                    for date in self.motion[sub_id].keys():
                        n = self.n_mot[sub_id][date][position]
                        n_segments = max(0, math.ceil((n - length + 1) / self.conf.mot_stride))

                        if self.verbose:
                            print(f'User: {sub_id}, date: {date}, feature: Motion')
                            print('Number of Segments After Segmenting: {}'.format(n_segments))
                            print()

                        if n_segments > 0:
                            segments = self.to_segments(self.motion[sub_id][date][position],
                                                        n_segments, self.mot_length, self.conf.mot_stride)
                            mmap[offset: offset + n_segments] = segments[..., 1:]

                        offset += n_segments

            segmented_motion[position] = mmap

        filename = 'motion_info' + '.mmap'
        dst = os.path.join(self.segmented_path, filename)
        exists = os.path.exists(dst)

        if load and exists:
            if n_exists:
                total_segments, _, _ = self.n_mot
                shape = (total_segments, 5)

            mmap = np.memmap(dst, mode='r+', dtype=np.int64, shape=shape)

        else:
            total_segments, length, _ = self.get_mot_shape(positions[0])
            shape = (total_segments, 5)
            mmap = np.memmap(dst, mode='w+', dtype=np.int64, shape=shape)

            offset = 0
            for sub_id in self.labels.keys():
                for date_id, date in enumerate(self.labels[sub_id].keys()):
                    n = self.n_lbs[sub_id][date]
                    n_segments = max(0, math.ceil((n - length + 1) / self.conf.mot_stride))

                    if self.verbose:
                        print(f'User: {sub_id}, date: {date}, feature: Labels')
                        print('Number of Segments After Segmenting: {}'.format(n_segments))
                        print()

                    if n_segments > 0:
                        segments = self.to_segments(self.labels[sub_id][date],
                                                    n_segments, self.mot_length, self.conf.mot_stride)

                        mmap[offset: offset + n_segments, 0] = segments[:, 0, 0]
                        mmap[offset: offset + n_segments, 1] = segments[:, -1, 0]
                        mmap[offset: offset + n_segments, 2] = int(sub_id)
                        mmap[offset: offset + n_segments, 3] = date_id
                        mmap[offset: offset + n_segments, 4] = self.get_labels(segments[..., 1])

                    offset += n_segments

        segmented_info = mmap

        # start = 80
        # t = np.arange(segmented_info[start, 0], segmented_info[start, 1]+1,
        #               1000 / self.conf.new_motion_fs , dtype=np.int64)
        # plt.plot(t, segmented_motion[position][start, :, 0])
        # plt.plot(t, segmented_motion[position][start, :, 1])
        # plt.plot(t, segmented_motion[position][start, :, 2])
        # plt.plot(t, segmented_motion[position][start, :, -2])
        # plt.show()
        # t = np.arange(segmented_info[start+1, 0], segmented_info[start+1, 1] + 1,
        #               1000 / self.conf.new_motion_fs, dtype=np.int64)
        # plt.plot(t, segmented_motion[position][start+1, :, 0])
        # plt.plot(t, segmented_motion[position][start+1, :, 1])
        # plt.plot(t, segmented_motion[position][start+1, :, 2])
        # plt.plot(t, segmented_motion[position][start+1, :, -2])
        # plt.show()

        self.n_mot = [total_segments, length, channels]

        return segmented_motion, segmented_info

    def get_loc_shape(self, position: str):
        total_segments = 0

        print(self.n_loc)
        for sub_id in self.n_loc.keys():
            for date in self.n_loc[sub_id].keys():
                n = self.n_loc[sub_id][date][position]
                n_segments = max(0, math.ceil((n - self.conf.loc_length + 1) / self.conf.loc_stride))
                total_segments += n_segments

        return total_segments, self.conf.loc_length, self.loc_channels

    def segment_location(self, load: bool = False, n_exists: bool = False) -> Dict:
        print("Segmenting Motion Data...")

        for sub_id in self.location.keys():
            for date in self.location[sub_id].keys():
                positions = list(self.location[sub_id][date].keys())
                break

        segmented_location = {}
        segmented_info = {}

        for position in positions:
            loc_file = position + '_' + 'location' + '.mmap'
            info_file = position + '_' + 'loc_info' + '.mmap'
            loc_dst = os.path.join(self.segmented_path, loc_file)
            info_dst = os.path.join(self.segmented_path, info_file)
            exists = os.path.exists(loc_dst) and os.path.exists(info_dst)

            if load and exists:
                if n_exists:
                    total_segments, length, channels = self.n_loc
                    loc_shape = (total_segments, length, channels)
                    info_shape = (total_segments, 4)

                loc_mmap = np.memmap(loc_dst, mode='r+', dtype=np.float32, shape=loc_shape)
                info_mmap = np.memmap(info_dst, mode='r+', dtype=np.int64, shape=info_shape)

            else:
                total_segments, length, channels = self.get_loc_shape(position)
                loc_shape = (total_segments, length, channels)
                info_shape = (total_segments, 4)

                loc_mmap = np.memmap(loc_dst, mode='w+', dtype=np.float32, shape=loc_shape)
                info_mmap = np.memmap(info_dst, mode='w+', dtype=np.int64, shape=info_shape)

                offset = 0
                for sub_id in self.location.keys():
                    for date_id, date in enumerate(self.location[sub_id].keys()):
                        n = self.n_loc[sub_id][date][position]
                        n_segments = max(0, math.ceil((n - self.conf.loc_length + 1) / self.conf.loc_stride))

                        if self.verbose:
                            print(f'User: {sub_id}, date: {date}, feature: Location')
                            print('Number of Segments After Segmenting: {}'.format(n_segments))
                            print()

                        if n_segments > 0:
                            segments = self.to_segments(self.location[sub_id][date][position],
                                                        n_segments, self.conf.loc_length, self.conf.loc_stride)
                            loc_mmap[offset: offset + n_segments] = segments[..., 1:]

                            info_mmap[offset: offset + n_segments, 0] = segments[:, 0, 0]
                            info_mmap[offset: offset + n_segments, 1] = segments[:, -1, 0]
                            info_mmap[offset: offset + n_segments, 2] = int(sub_id)
                            info_mmap[offset: offset + n_segments, 3] = date_id

                        offset += n_segments

            segmented_location[position] = loc_mmap
            segmented_info[position] = info_mmap

            # start = 80
            # t = np.arange(segmented_info[position][start, 0], segmented_info[position][start, 1]+1,
            #               self.conf.loc_stride * 60000, dtype=np.int64)
            # plt.plot(t, segmented_location[position][start, :, 4])
            # plt.plot(t, segmented_location[position][start, :, 5])
            # plt.show()
            # t = np.arange(segmented_info[position][start+1, 0], segmented_info[position][start+1, 1] + 1,
            #               self.conf.loc_stride * 60000, dtype=np.int64)
            # plt.plot(t, segmented_location[position][start+1, :, 4])
            # plt.plot(t, segmented_location[position][start+1, :, 5])
            # plt.show()

        self.n_loc = [total_segments, length, channels]

        return segmented_location, segmented_info


if __name__ == '__main__':
    A = Segmenter(delete_src=True, delete_dst=False)
    A(load=False)

