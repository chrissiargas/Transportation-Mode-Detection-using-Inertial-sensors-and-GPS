import numpy as np

from config_parser import Parser
import os
import subprocess
from typing import Tuple, Dict
import pandas as pd
import pickle
import shutil


class Extractor:
    def __init__(self, delete_dst: bool = False):
        config = Parser()
        config.get_args()
        self.conf = config

        self.acc = {
            1: 'Acc_x',
            2: 'Acc_y',
            3: 'Acc_z'
        }

        self.gyr = {
            5: 'Gyr_x',
            6: 'Gyr_y',
            7: 'Gyr_z'
        }

        self.mag = {
            8: 'Mag_x',
            9: 'Mag_y',
            10: 'Mag_z',
        }

        self.n_mot_features = 1
        self.mot_features = ['Time']
        self.mot_signals = []
        for signal in self.conf.signals:
            if signal == 'Acc':
                self.n_mot_features += 3
                self.mot_features.extend(list(self.acc.values()))
                self.mot_signals.append(self.acc)
            elif signal == 'Gyr':
                self.n_mot_features += 3
                self.mot_features.extend(list(self.gyr.values()))
                self.mot_signals.append(self.gyr)
            elif signal == 'Mag':
                self.n_mot_features += 3
                self.mot_features.extend(list(self.mag.values()))
                self.mot_signals.append(self.mag)

        self.mot_features = {k:v for v, k in enumerate(self.mot_features)}

        self.loc = {
            3: 'Acc',
            4: 'Lat',
            5: 'Long',
            6: 'Alt'
        }
        self.n_loc_features = 5
        self.loc_features = ['Time', 'Acc', 'Lat', 'Long', 'Alt']
        self.loc_features = {k: v for v, k in enumerate(self.loc_features)}

        self.labels = {
            0: 'null',
            1: 'still',
            2: 'walk',
            3: 'run',
            4: 'bike',
            5: 'car',
            6: 'bus',
            7: 'train',
            8: 'subway',
        }
        self.n_lbs_features = 2
        self.lbs_features = ['Time', 'Label']
        self.lbs_features = {k: v for v, k in enumerate(self.lbs_features)}

        self.from_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.path,
            self.conf.dataset
        )

        if self.conf.dataset == 'SHL-complete':
            self.from_path = os.path.join(
                self.from_path,
                'release'
            )

        self.to_path = os.path.join(
            os.path.expanduser('~'),
            self.conf.path,
            self.conf.dataset + '-extracted'
        )

        if delete_dst:
            try:
                shutil.rmtree(self.to_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if not os.path.exists(self.to_path):
            os.makedirs(self.to_path)

        self.n_mot = {}
        self.n_loc = {}
        self.n_lbs = {}

        features = {
            'mot_features': self.mot_features,
            'loc_features': self.loc_features,
            'lbs_features': self.lbs_features,
        }
        with open('info/' + self.conf.dataset + '/initial_features.pickle', 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self, verbose=True, load=False, *args, **kwargs) -> Tuple[Dict, Dict, Dict]:
        self.verbose = verbose

        location = {}
        motion = {}
        labels = {}

        subject_dir = os.listdir(self.from_path)

        n_exists = False
        if load and os.path.exists('info/' + self.conf.dataset + '/initial_sizes.pickle'):
            with open('info/' + self.conf.dataset + '/initial_sizes.pickle', 'rb') as handle:
                sizes = pickle.load(handle)
                self.n_mot = sizes['n_mot']
                self.n_loc = sizes['n_loc']
                self.n_lbs = sizes['n_lab']
                n_exists = True

        for sub_fold in subject_dir:
            sub_id = sub_fold[-1]

            location[sub_id] = {}
            motion[sub_id] = {}
            labels[sub_id] = {}

            if not n_exists:
                self.n_mot[sub_id] = {}
                self.n_loc[sub_id] = {}
                self.n_lbs[sub_id] = {}

            if self.conf.dataset == 'SHL-preview':
                sub_path = os.path.join(self.from_path, sub_fold,
                                        'SHLDataset_preview_v1',
                                        'User' + sub_id)
                date_dir = [f for f in os.listdir(sub_path) if 'mat' not in f]

            elif self.conf.dataset == 'SHL-complete':
                sub_path = os.path.join(self.from_path, sub_fold)
                date_dir = [f for f in os.listdir(sub_path) if ('pdf' not in f)]

            for file_date in date_dir:
                date_path = os.path.join(sub_path, file_date)
                rec_dir = os.listdir(date_path)

                motion_files = [rec for rec in rec_dir if 'Motion' in rec]
                location_files = [rec for rec in rec_dir if 'Location' in rec]
                labels_file = [rec for rec in rec_dir if 'Label' in rec]

                if len(labels_file):
                    labels_file = labels_file[0]
                else:
                    continue

                if 'm' in file_date:
                    date = file_date.replace('m', '')
                    date = date + 'm'
                else:
                    date = file_date

                location[sub_id][date] = {}
                motion[sub_id][date] = {}

                if not n_exists:
                    self.n_mot[sub_id][date] = {}
                    self.n_loc[sub_id][date] = {}

                for motion_file in motion_files:
                    position, _ = motion_file.split('_')
                    motion_path = os.path.join(date_path, motion_file)
                    filename = sub_id + '_' + date + '_' + position + '_' + 'motion' + '.mmap'
                    dst = os.path.join(self.to_path, filename)
                    exists = os.path.exists(dst)

                    if load and exists:
                        n_lines = self.n_mot[sub_id][date][position]
                    else:
                        stdout = subprocess.Popen(
                            "wc -l < " + motion_path,
                            shell=True,
                            stdout=subprocess.PIPE)

                        n_lines, _ = stdout.communicate()
                        n_lines = int(n_lines)
                        self.n_mot[sub_id][date][position] = n_lines

                    if self.verbose:
                        print(f'User: {sub_id}, date: {date}, position: {position}, feature: Motion')
                        print('Number of Lines: {}'.format(n_lines))
                        print()

                    shape = (n_lines, self.n_mot_features)
                    if load and exists:
                        mmap = np.memmap(dst, mode='r+', dtype=np.float64, shape=shape)
                    else:
                        offset = 0
                        mmap = np.memmap(dst, mode='w+', dtype=np.float64, shape=shape)

                        for batch in pd.read_csv(motion_path, delimiter=' ',
                                                 chunksize=5000, header=None, nrows=None):
                            indices = [0]
                            for signal in self.mot_signals:
                                indices.extend(signal.keys())

                            mmap[offset: offset + batch.shape[0]] = batch.iloc[:, indices]
                            offset += batch.shape[0]

                    motion[sub_id][date][position] = mmap

                for location_file in location_files:
                    position, _ = location_file.split('_')
                    location_path = os.path.join(date_path, location_file)
                    filename = sub_id + '_' + date + '_' + position + '_' + 'location' + '.mmap'
                    dst = os.path.join(self.to_path, filename)
                    exists = os.path.exists(dst)

                    if load and exists:
                        n_lines = self.n_loc[sub_id][date][position]
                    else:
                        stdout = subprocess.Popen(
                            "wc -l < " + location_path,
                            shell=True,
                            stdout=subprocess.PIPE)



                        n_lines, _ = stdout.communicate()
                        n_lines = int(n_lines)
                        self.n_loc[sub_id][date][position] = n_lines

                    if self.verbose:
                        print(f'User: {sub_id}, date: {date}, position: {position}, feature: Location')
                        print('Number of Lines: {}'.format(n_lines))
                        print()

                    shape = (n_lines, self.n_loc_features)
                    if load and exists:
                        mmap = np.memmap(dst, mode='r+', dtype=np.float64, shape=shape)
                    if not exists:
                        offset = 0
                        mmap = np.memmap(dst, mode='w+', dtype=np.float64, shape=shape)

                        for batch in pd.read_csv(location_path, delimiter=' ',
                                                 chunksize=5000, header=None, nrows=None):
                            indices = [0, *self.loc.keys()]

                            mmap[offset: offset + batch.shape[0]] = batch.iloc[:, indices]
                            offset += batch.shape[0]

                    location[sub_id][date][position] = mmap

                labels_path = os.path.join(date_path, labels_file)
                filename = sub_id + '_' + date + '_' + 'labels' + '.mmap'
                dst = os.path.join(
                    self.to_path,
                    filename
                )
                exists = os.path.exists(dst)

                if load and exists:
                    n_lines = self.n_lbs[sub_id][date]
                else:
                    stdout = subprocess.Popen(
                        "wc -l < " + labels_path,
                        shell=True,
                        stdout=subprocess.PIPE)

                    n_lines, _ = stdout.communicate()
                    n_lines = int(n_lines)
                    self.n_lbs[sub_id][date] = n_lines

                if self.verbose:
                    print(f'User: {sub_id}, date: {date}, feature: Labels')
                    print('Number of Lines: {}'.format(n_lines))
                    print()

                shape = (n_lines, self.n_lbs_features)
                if load and exists:
                    mmap = np.memmap(dst, mode='r+', dtype=np.int64, shape=shape)
                else:
                    offset = 0
                    mmap = np.memmap(dst, mode='w+', dtype=np.int64, shape=shape)

                    for batch in pd.read_csv(labels_path, delimiter=' ',
                                             chunksize=5000, header=None, nrows=None):

                        mmap[offset: offset + batch.shape[0]] = batch.iloc[:, [0, 1]]
                        offset += batch.shape[0]

                labels[sub_id][date] = mmap

        if not load:
            sizes = {
                'n_mot': self.n_mot,
                'n_loc': self.n_loc,
                'n_lab': self.n_lbs,
            }
            with open('info/' + self.conf.dataset + '/initial_sizes.pickle', 'wb') as handle:
                pickle.dump(sizes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return motion, location, labels


if __name__ == '__main__':
    ext = Extractor(delete_dst=True)
    location, motion, labels = ext(load=False)
