import numpy as np
import pandas as pd
from hmm_filter.hmm_filter import HMMFilter
from build import Builder
from keras.models import Model
from typing import *
import random
import itertools
from sklearn.preprocessing import normalize

def get_YY_seq(data: Builder, model: Model, motion_only=False, train=False, test=False) \
        -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]]]:
    if data.conf.train_test_split == 'loso':
        test_user = data.conf.train_test_hold_out
        subjects = np.unique(data.motion_info[:, 2])

        if test_user in subjects:
            train_indices = np.argwhere(data.motion_info[:, 2] != test_user).flatten()
            test_indices = np.argwhere(data.motion_info[:, 2] == test_user).flatten()

    elif data.conf.train_test_split in ['ldo_start', 'ldo_end', 'ldo_random']:
        days = np.unique(data.motion_info[:, 3])

        if data.conf.train_test_split == 'ldo_start':
            test_days = days[:data.conf.train_test_hold_out]
        elif data.conf.train_test_split == 'ldo_end':
            test_days = days[-data.conf.train_test_hold_out:]
        elif data.conf.train_test_split == 'ldo_random':
            test_days = np.random.choice(days, size=data.conf.train_test_hold_out, replace=False)

        train_indices = np.argwhere(~np.in1d(data.motion_info[:, 3], test_days)).flatten()
        test_indices = np.argwhere(np.in1d(data.motion_info[:, 3], test_days)).flatten()

    if train:
        indices = train_indices
    elif test:
        indices = test_indices

    info = data.motion_info[indices]
    offset = 0
    threshold = data.conf.seq_thres * 1000.
    split_indices = []

    subjects = np.unique(info[:, 2])
    for subject in subjects:

        sub_info = info[info[:, 2] == subject]
        dates = np.unique(sub_info[:, 3])
        for date in dates:
            date_sub_info = sub_info[sub_info[:, 3] == date]

            T = date_sub_info[:, 0]
            N = len(T)

            split_indices_ = np.where(np.diff(T) > threshold)[0] + 1
            split_indices.extend(
                (np.concatenate(([0], split_indices_)) + offset).tolist()
            )
            offset += N

    split_indices.append(offset)

    splits = len(split_indices)
    idx_sequenced = [list(range(indices[split_indices[k]], indices[split_indices[k + 1] - 1] + 1)) for k in
                     range(splits - 1)]

    if data.conf.test_bag_position in data.motion.keys():
        n_scenarios = 1
        pos_sequenced = [[] for _ in range(n_scenarios)]

        for scenario in range(n_scenarios):
            for idx_sequence in idx_sequenced:
                N = len(idx_sequence)
                pos_N = N + data.conf.mot_bag_size - 1
                pos_sequence = [data.conf.test_bag_position for _ in range(pos_N)]
                pos_sequenced[scenario].append(pos_sequence)

    elif data.conf.test_bag_position == 'same':
        n_scenarios = len(data.motion.keys())
        pos_sequenced = [[] for _ in range(n_scenarios)]

        for scenario, position in enumerate(data.motion.keys()):
            for idx_sequence in idx_sequenced:
                N = len(idx_sequence)
                pos_N = N + data.conf.mot_bag_size - 1
                pos_sequence = [position for _ in range(pos_N)]
                pos_sequenced[scenario].append(pos_sequence)

    elif data.conf.test_bag_position == 'random':
        n_scenarios = len(data.motion.keys())
        pos_sequenced = [[] for _ in range(n_scenarios)]

        for idx_sequence in idx_sequenced:
            N = len(idx_sequence)
            pos_N = N + data.conf.mot_bag_size - 1
            positions = [random.sample(data.motion.keys(), n_scenarios) for _ in
                         range(pos_N)]
            pos_sequence = list(map(list, zip(*positions)))

            for scenario in range(n_scenarios):
                pos_sequenced[scenario].append(pos_sequence[scenario])

    true_sequenced = []
    pred_sequenced = []
    lens = []

    pos_sequenced = list(itertools.chain.from_iterable(pos_sequenced))
    idx_sequenced = idx_sequenced * n_scenarios

    for idx_sequence, pos_sequence in zip(idx_sequenced, pos_sequenced):
        true_sequence = []
        inputs_sequence = []
        for k, index in enumerate(idx_sequence):
            positions = [pos_sequence[j] for j in range(k, k + data.conf.mot_bag_size)]
            X, y = data.get_Xy(index, positions, motion_only, one_instance=False)
            y = np.argmax(y)

            true_sequence.append(y)
            inputs_sequence.append(X)

        true_sequenced.extend(true_sequence)
        lens.append(len(true_sequence))

        if test:
            inputs_batch = [np.array([input[i] for input in inputs_sequence]) for i in range(len(X))]
            pred_sequence = model.predict(inputs_batch, verbose=0)
            pred_sequence = normalize(pred_sequence, axis=1, norm='l1')
            pred_sequenced.extend(pred_sequence)

    return true_sequenced, pred_sequenced, lens


def get_YY_(data: Builder, model: Model, motion_only: bool = False, pred: bool = False):
    if pred:
        train = False
        test = True
    else:
        train = True
        test = False

    Y, Y_, lens = get_YY_seq(data, model, motion_only, train, test)

    YY_ = pd.DataFrame()
    begin = 0
    session_id = 0
    modes = list(range(len(data.modes)))

    for length in lens:

        if pred:
            session_ids = [session_id for _ in range(length)]
            cat_true_seq = Y[begin: begin + length]

            prob_pred_seq = [{k: v for k, v in zip(modes, probs) if v > 0} for probs in Y_[begin: begin + length]]
            cat_pred_seq = np.argmax(Y_[begin: begin + length], axis=1)

            seq = pd.DataFrame({'session_id': session_ids,
                                'true': cat_true_seq,
                                'pred': cat_pred_seq,
                                'prob': prob_pred_seq})

        else:
            session_ids = [session_id for _ in range(length)]
            cat_true_seq = Y[begin: begin + length]

            seq = pd.DataFrame({'session_id': session_ids,
                                'true': cat_true_seq})

        YY_ = pd.concat([YY_, seq], ignore_index=True)

        begin += length
        session_id += 1

    Y_ = np.argmax(Y_, axis=1) if pred else Y_

    return YY_, Y, Y_


class HMM_classifier:
    def __init__(self):
        self.classifier = HMMFilter()

    def fit(self, Y):
        self.classifier.fit(Y,
                            session_column="session_id",
                            prediction_column="true")

    def predict(self, YY_):
        post_Y_ = self.classifier.predict(YY_,
                                          session_column="session_id",
                                          probabs_column="prob",
                                          prediction_column="pred")

        return post_Y_['pred'].tolist()
