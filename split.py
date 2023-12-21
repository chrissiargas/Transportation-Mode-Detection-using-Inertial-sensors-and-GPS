import numpy as np
import pandas as pd


def lopo_split(train_info, train_indices, hold_out):
    offset = 0
    subjects = np.unique(train_info[:, 2])
    in_val = []
    for subject in subjects:
        period_indices = []
        period_lens = []
        period_modes = []

        sub_info = train_info[train_info[:, 2] == subject]
        dates = np.unique(sub_info[:, 3])
        for date in dates:
            date_sub_info = sub_info[sub_info[:, 3] == date]
            modes = date_sub_info[:, -1]
            N = len(modes)
            period_splits = np.where(np.diff(modes) != 0)[0] + 1
            mode_periods = np.split(modes, period_splits)
            period_indices.extend(
                (np.concatenate(([0], period_splits)) + offset).tolist()
            )
            period_lens.extend([mode_period.shape[0] for mode_period in mode_periods])
            period_modes.extend([mode_period[0] for mode_period in mode_periods])
            offset += N

        period_indices.append(offset)
        mode_count = {}
        val_mode_count = {}

        for period_mode, period_len in zip(period_modes, period_lens):
            if period_mode in mode_count.keys():
                mode_count[period_mode] += period_len
            else:
                mode_count[period_mode] = period_len

        for period_mode in mode_count.keys():
            val_mode_count[period_mode] = int(hold_out * mode_count[period_mode])

            min_diff = 10000000
            best_diff = 0
            for i, (this_period_mode, this_period_len) in enumerate(zip(period_modes, period_lens)):
                if this_period_mode == period_mode:
                    diff = val_mode_count[period_mode] - this_period_len
                    if abs(diff) < min_diff:
                        best_diff = diff
                        min_diff = abs(diff)
                        best_index = i

            in_val.extend(list(range(period_indices[best_index], period_indices[best_index + 1] + min(0, best_diff))))

    val_indices = [index for i, index in enumerate(train_indices) if i in in_val]
    train_indices = [index for i, index in enumerate(train_indices) if i not in in_val]

    return train_indices, val_indices


def lopo_split_old(indices, hold_out):
    seed = 1

    originalIndices = indices
    indices = pd.DataFrame(indices, columns=['index', 'user_label'])
    count = indices['user_label'].value_counts()
    val_count = count * hold_out
    val_count = val_count.astype('int32')
    val_indices = []

    for user_label, count in val_count.items():

        candidates = pd.DataFrame()
        tmp_count = count

        while candidates.empty:
            candidates = indices[indices['user_label'] == user_label].user_label.groupby(
                [indices.user_label, indices.user_label.diff().ne(0).cumsum()]).transform('size').ge(
                tmp_count).astype(int)
            candidates = pd.DataFrame(candidates)
            candidates = candidates[candidates['user_label'] == 1]
            tmp_count = int(tmp_count * 0.95)

        index = candidates.sample(random_state=seed).index[0]
        val_indices.append(index)
        n_indices = 1
        up = 1
        down = 1
        length = indices.shape[0]

        while n_indices < tmp_count - 1:

            if index + up < length and user_label == indices.iloc[index + up]['user_label']:
                val_indices.append(index + up)
                up += 1
                n_indices += 1

            if index - down >= 0 and user_label == indices.iloc[index - down]['user_label']:
                val_indices.append(index - down)
                down += 1
                n_indices += 1

    val_indices.sort()
    val_indices = [originalIndices.pop(i - shift)[0] for shift, i in enumerate(val_indices)]
    train_indices = [x[0] for x in originalIndices]

    return train_indices, val_indices