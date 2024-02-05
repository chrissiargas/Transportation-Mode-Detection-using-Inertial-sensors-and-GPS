import numpy as np
import pandas as pd
import ruamel.yaml
import os
import time
import gc

from parameters import (TMD_MIL_parameters, Liang_parameters,
                        Tang_parameters, SHL_complete_params,
                        SHL_preview_params, Ito_parameters)
from build import Builder
from config_parser import config_edit
import TMD_MIL_training
import Liang_training
import Tang_training
import Ito_training
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

REGENERATE = False
SPLIT = 'loso'
DATASET = 'SHL-preview'
EXP_TYPE = 'all_positions'
REPEATS = 3
scores_df = pd.DataFrame()
POSITIONS = ['same']
USERS = [1, 2, 3]


def config_save(file):
    yaml = ruamel.yaml.YAML()

    with open('config.yaml') as fp:
        parameters = yaml.load(fp)

    with open(file, 'w') as fb:
        yaml.dump(parameters, fb)


def save(path):
    try:
        os.makedirs(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    scores_file = os.path.join(path, "scores.csv")
    stats_file = os.path.join(path, "stats.csv")
    params_file = os.path.join(path, "parameters.yaml")

    scores_save(scores_file, stats_file)
    config_save(params_file)


def scores_save(scores_file, stats_file):
    global scores_df

    scores_df.to_csv(scores_file, index=False)

    mean_per_user = scores_df.groupby(['test_user']).mean()
    std_per_user = scores_df.groupby(['test_user']).std()

    mean_per_user.columns = [str(col) + '_mean' for col in mean_per_user.columns]
    std_per_user.columns = [str(col) + '_std' for col in std_per_user.columns]

    stats_df = pd.concat([mean_per_user, std_per_user], axis=1)
    stats_df.loc['All'] = stats_df.mean()
    stats_df['test_user'] = stats_df.index
    print(stats_df)

    stats_df.to_csv(stats_file, index=False)


def get_scores(scores, test_user=None, postprocessing=False):
    global scores_df

    if not postprocessing:
        acc, f1, pr, rec, _, _, _, _, cm = scores
        if test_user is None:
            score_dict = {'test_user': str(0),
                          'accuracy': acc,
                          'f1_score': f1,
                          'precision': pr,
                          'recall': rec}
        else:
            score_dict = {'test_user': str(test_user),
                          'accuracy': acc,
                          'f1_score': f1,
                          'precision': pr,
                          'recall': rec}

    else:
        acc, f1,  pr, rec, post_acc, post_f1, post_pr, post_rec, cm = scores
        if test_user is None:
            score_dict = {'test_user': str(0),
                          'accuracy': acc,
                          'f1_score': f1,
                          'precision': pr,
                          'recall': rec,
                          'post_accuracy': post_acc,
                          'post_f1_score': post_f1,
                          'post_precision': post_pr,
                          'post_recall': post_rec}

        else:
            score_dict = {'test_user': str(test_user),
                          'accuracy': acc,
                          'f1_score': f1,
                          'precision': pr,
                          'recall': rec,
                          'post_accuracy': post_acc,
                          'post_f1_score': post_f1,
                          'post_precision': post_pr,
                          'post_recall': post_rec}

    scores_df = pd.concat([scores_df, pd.DataFrame([score_dict])], ignore_index=True)
    print(scores_df)

    return scores_df


def reset_tensorflow_keras_backend():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    _ = gc.collect()


def TMD_MIL_experiment(path, regenerate=False):
    data = Builder(regenerate)
    _, _, scores, y, y_ = TMD_MIL_training.train(data, summary=True, verbose=True,
                                                 load=False, path=path,
                                                 eval=True, use_HMM=False)
    del data

    return scores, y, y_


def Liang_experiment(path, regenerate=False):
    data = Builder(regenerate)
    _, _, scores = Liang_training.train(data, summary=True, verbose=True, load=False, path=path, eval=True,
                                        use_HMM=True)
    del data

    return scores


def Tang_experiment(path, regenerate=False):
    data = Builder(regenerate)
    _, _, scores = Tang_training.train(data, summary=False, verbose=True, load=False, path=path, eval=True,
                                       use_HMM=True)
    del data

    return scores


def Ito_experiment(path, regenerate=False):
    data = Builder(regenerate)
    _, _, scores = Ito_training.train(data, summary=True, verbose=True, load=False, path=path, eval=True, use_HMM=True)
    del data

    return scores


def TMD_MIL(archive_path):
    global scores_df
    params = TMD_MIL_parameters

    for param_group, group_params in params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    if DATASET == 'SHL-preview':
        data_params = SHL_preview_params
    elif DATASET == 'SHL-complete':
        data_params = SHL_complete_params

    for param_group, group_params in data_params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))
    reset_tensorflow_keras_backend()

    repeats = REPEATS
    regenerate = REGENERATE
    split = SPLIT
    exp_type = EXP_TYPE
    users = USERS

    config_edit('build_args', 'train_test_split', split)

    Y = None
    Y_ = None

    if exp_type == 'all_positions':
        for position in POSITIONS:
            config_edit('build_args', 'test_bag_position', position)
            config_edit('build_args', 'val_bag_position', 'same')
            config_edit('build_args', 'train_bag_position', 'same')

            scores_df = pd.DataFrame()
            saves_path = os.path.join(archive, position)

            if split == 'loso':
                for test_user in users:
                    print('TEST USER: ' + str(test_user))
                    config_edit('build_args', 'train_test_hold_out', test_user)
                    for turn in range(repeats):
                        path = os.path.join(archive, "turn_" + str(turn), "test_user_" + str(test_user))
                        scores, y, y_ = TMD_MIL_experiment(path, regenerate)

                        # if Y is not None:
                        #     Y = np.concatenate((Y, y), axis=0)
                        #     Y_ = np.concatenate((Y_, y_), axis=0)
                        # else:
                        #     Y = y
                        #     Y_ = y_

                        get_scores(scores, test_user, True)
                        save(saves_path)
                        regenerate = False

            else:
                for turn in range(repeats):
                    path = os.path.join(archive, "turn_" + str(turn))

                    scores = TMD_MIL_experiment(path, regenerate)

                    get_scores(scores, postprocessing=True)
                    save(saves_path)
                    regenerate = False

    elif exp_type == 'one_position':
        for position in POSITIONS:
            config_edit('build_args', 'train_bag_position', position)
            config_edit('build_args', 'val_bag_position', position)
            config_edit('build_args', 'test_bag_position', position)

            scores_df = pd.DataFrame()
            saves_path = os.path.join(archive, position)

            if split == 'loso':
                for test_user in [1, 2, 3]:
                    print('TEST USER: ' + str(test_user))
                    config_edit('build_args', 'train_test_hold_out', test_user)
                    for turn in range(repeats):
                        path = os.path.join(archive, "turn_" + str(turn), "test_user_" + str(test_user))
                        scores = TMD_MIL_experiment(path, regenerate)

                        get_scores(scores, test_user, True)
                        save(saves_path)
                        regenerate = False

            else:
                for turn in range(repeats):
                    path = os.path.join(archive, "turn_" + str(turn))

                    scores = TMD_MIL_experiment(path, regenerate)

                    get_scores(scores, postprocessing=True)
                    save(saves_path)
                    regenerate = False

    # show_roc = False
    # if show_roc:
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     lw = 2
    #
    #     mode_names = ['still', 'walk',
    #                   'run', 'bike', 'car',
    #                   'bus', 'train', 'subway']
    #     for i in range(8):
    #         fpr[i], tpr[i], _ = roc_curve(Y[:, i], Y_[:, i])
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #
    #     fpr_grid = np.linspace(0.0, 1.0, 1000)
    #
    #     # Interpolate all ROC curves at these points
    #     mean_tpr = np.zeros_like(fpr_grid)
    #
    #     for i in range(8):
    #         mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    #
    #     # Average it and compute AUC
    #     mean_tpr /= 8
    #
    #     fpr["macro"] = fpr_grid
    #     tpr["macro"] = mean_tpr
    #     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    #     plt.plot(
    #         fpr["macro"],
    #         tpr["macro"],
    #         label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    #         color="navy",
    #         linestyle=":",
    #         linewidth=4,
    #     )
    #
    #     for i in range(8):
    #         plt.plot(fpr[i], tpr[i], lw=2,
    #                  label='ROC curve for {0} (area = {1:0.2f})'
    #                        ''.format(mode_names[i], roc_auc[i]))
    #
    #     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    #     plt.xlim([-0.05, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver operating characteristic for multi-class data')
    #     plt.legend(loc="lower right")
    #     plt.savefig("ROC_AUC.pdf", format="pdf", bbox_inches="tight")
    #     plt.show()

    return


def Liang(archive_path):
    global scores_df
    params = Liang_parameters

    for param_group, group_params in params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    if DATASET == 'SHL-preview':
        data_params = SHL_preview_params
    elif DATASET == 'SHL-complete':
        data_params = SHL_complete_params

    for param_group, group_params in data_params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))
    reset_tensorflow_keras_backend()

    repeats = REPEATS
    regenerate = REGENERATE
    split = SPLIT
    exp_type = EXP_TYPE

    config_edit('build_args', 'train_test_split', split)

    if exp_type == 'all_positions':
        for position in POSITIONS:
            config_edit('build_args', 'test_bag_position', position)
            config_edit('build_args', 'val_bag_position', 'same')
            config_edit('build_args', 'train_bag_position', 'same')

            scores_df = pd.DataFrame()
            saves_path = os.path.join(archive, position)

            if split == 'loso':
                for test_user in [1, 2, 3]:
                    print('TEST USER: ' + str(test_user))
                    config_edit('build_args', 'train_test_hold_out', test_user)
                    for turn in range(repeats):
                        path = os.path.join(archive, "turn_" + str(turn), "test_user_" + str(test_user))
                        scores = Liang_experiment(path, regenerate)

                        get_scores(scores, test_user, True)
                        save(saves_path)
                        regenerate = False

            elif split == 'random':
                config_edit('build_args', 'train_test_hold_out', 0.2)
                for turn in range(repeats):
                    path = os.path.join(archive, "turn_" + str(turn))

                    scores = Liang_experiment(path, regenerate)

                    get_scores(scores, postprocessing=True)
                    save(saves_path)
                    regenerate = False

            else:
                for turn in range(repeats):
                    path = os.path.join(archive, "turn_" + str(turn))

                    scores = Liang_experiment(path, regenerate)

                    get_scores(scores, postprocessing=True)
                    save(saves_path)
                    regenerate = False

    elif exp_type == 'one_position':
        for position in POSITIONS:
            config_edit('build_args', 'train_bag_position', position)
            config_edit('build_args', 'val_bag_position', position)
            config_edit('build_args', 'test_bag_position', position)

            scores_df = pd.DataFrame()
            saves_path = os.path.join(archive, position)

            if split == 'loso':
                for test_user in [1, 2, 3]:
                    print('TEST USER: ' + str(test_user))
                    config_edit('build_args', 'train_test_hold_out', test_user)
                    for turn in range(repeats):
                        path = os.path.join(archive, "turn_" + str(turn), "test_user_" + str(test_user))
                        scores = Liang_experiment(path, regenerate)

                        get_scores(scores, test_user, True)
                        save(saves_path)
                        regenerate = False

            elif split == 'random':
                config_edit('build_args', 'train_test_hold_out', 0.2)
                for turn in range(repeats):
                    path = os.path.join(archive, "turn_" + str(turn))

                    scores = Liang_experiment(path, regenerate)

                    get_scores(scores, postprocessing=True)
                    save(saves_path)
                    regenerate = False

            else:
                for turn in range(repeats):
                    path = os.path.join(archive, "turn_" + str(turn))

                    scores = Liang_experiment(path, regenerate)

                    get_scores(scores, postprocessing=True)
                    save(saves_path)
                    regenerate = False

    return


def Tang(archive_path):
    global scores_df
    params = Tang_parameters

    for param_group, group_params in params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    if DATASET == 'SHL-preview':
        data_params = SHL_preview_params
    elif DATASET == 'SHL-complete':
        data_params = SHL_complete_params

    for param_group, group_params in data_params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))
    reset_tensorflow_keras_backend()

    repeats = REPEATS
    regenerate = REGENERATE
    split = SPLIT
    exp_type = EXP_TYPE

    config_edit('build_args', 'train_test_split', split)

    if exp_type == 'all_positions':
        for position in POSITIONS:
            config_edit('build_args', 'test_bag_position', position)
            config_edit('build_args', 'val_bag_position', 'same')
            config_edit('build_args', 'train_bag_position', 'same')

            scores_df = pd.DataFrame()
            saves_path = os.path.join(archive, position)

            if split == 'loso':
                for test_user in [1, 2, 3]:
                    print('TEST USER: ' + str(test_user))
                    config_edit('build_args', 'train_test_hold_out', test_user)
                    for turn in range(repeats):
                        path = os.path.join(archive, "turn_" + str(turn), "test_user_" + str(test_user))
                        scores = Tang_experiment(path, regenerate)

                        get_scores(scores, test_user, True)
                        save(saves_path)
                        regenerate = False

            else:
                for turn in range(repeats):
                    path = os.path.join(archive, "turn_" + str(turn))

                    scores = Tang_experiment(path, regenerate)

                    get_scores(scores, postprocessing=True)
                    save(saves_path)
                    regenerate = False

    elif exp_type == 'one_position':
        for position in POSITIONS:
            config_edit('build_args', 'train_bag_position', position)
            config_edit('build_args', 'val_bag_position', position)
            config_edit('build_args', 'test_bag_position', position)

            scores_df = pd.DataFrame()
            saves_path = os.path.join(archive, position)

            if split == 'loso':
                for test_user in [1, 2, 3]:
                    print('TEST USER: ' + str(test_user))
                    config_edit('build_args', 'train_test_hold_out', test_user)
                    for turn in range(repeats):
                        path = os.path.join(archive, "turn_" + str(turn), "test_user_" + str(test_user))
                        scores = Tang_experiment(path, regenerate)

                        get_scores(scores, test_user, True)
                        save(saves_path)
                        regenerate = False

            else:
                for turn in range(repeats):
                    path = os.path.join(archive, "turn_" + str(turn))

                    scores = Tang_experiment(path, regenerate)

                    get_scores(scores, postprocessing=True)
                    save(saves_path)
                    regenerate = False

    return


def Ito(archive_path):
    global scores_df
    params = Ito_parameters

    for param_group, group_params in params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    if DATASET == 'SHL-preview':
        data_params = SHL_preview_params
    elif DATASET == 'SHL-complete':
        data_params = SHL_complete_params

    for param_group, group_params in data_params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))
    reset_tensorflow_keras_backend()

    repeats = REPEATS
    regenerate = REGENERATE
    split = SPLIT
    exp_type = EXP_TYPE

    config_edit('build_args', 'train_test_split', split)

    if exp_type == 'all_positions':
        for position in POSITIONS:
            config_edit('build_args', 'test_bag_position', position)
            config_edit('build_args', 'val_bag_position', 'same')
            config_edit('build_args', 'train_bag_position', 'same')

            scores_df = pd.DataFrame()
            saves_path = os.path.join(archive, position)

            if split == 'loso':
                for test_user in [1, 2, 3]:
                    print('TEST USER: ' + str(test_user))
                    config_edit('build_args', 'train_test_hold_out', test_user)
                    for turn in range(repeats):
                        path = os.path.join(archive, "turn_" + str(turn), "test_user_" + str(test_user))
                        scores = Ito_experiment(path, regenerate)

                        get_scores(scores, test_user, True)
                        save(saves_path)
                        regenerate = False

            else:
                for turn in range(repeats):
                    path = os.path.join(archive, "turn_" + str(turn))

                    scores = Ito_experiment(path, regenerate)

                    get_scores(scores, postprocessing=True)
                    save(saves_path)
                    regenerate = False

    elif exp_type == 'one_position':
        for position in POSITIONS:
            config_edit('build_args', 'train_bag_position', position)
            config_edit('build_args', 'val_bag_position', position)
            config_edit('build_args', 'test_bag_position', position)

            scores_df = pd.DataFrame()
            saves_path = os.path.join(archive, position)

            if split == 'loso':
                for test_user in [1, 2, 3]:
                    print('TEST USER: ' + str(test_user))
                    config_edit('build_args', 'train_test_hold_out', test_user)
                    for turn in range(repeats):
                        path = os.path.join(archive, "turn_" + str(turn), "test_user_" + str(test_user))
                        scores = Ito_experiment(path, regenerate)

                        get_scores(scores, test_user, True)
                        save(saves_path)
                        regenerate = False

            else:
                for turn in range(repeats):
                    path = os.path.join(archive, "turn_" + str(turn))

                    scores = Ito_experiment(path, regenerate)

                    get_scores(scores, postprocessing=True)
                    save(saves_path)
                    regenerate = False

    return
