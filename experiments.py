import copy
import pandas as pd
import ruamel.yaml
import os
import time
import gc

from parameters import TMD_MIL_parameters, Liang_parameters, Tang_parameters
from build import Builder
from config_parser import config_edit
import TMD_MIL_training
import Liang_training
import Tang_training

REGENERATE = False


def reset_tensorflow_keras_backend():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    _ = gc.collect()


def TMD_MIL_experiment(path, regenerate=False):
    data = Builder(regenerate)
    history = TMD_MIL_training.train(data, summary=True, verbose=True, load=False, path=path)
    del data

    return history


def Liang_experiment(path, regenerate=False):
    data = Builder(regenerate)
    history = Liang_training.train(data, summary=True, verbose=True, load=False, path=path)
    del data

    return history


def Tang_experiment(path, regenerate=False):
    data = Builder(regenerate)
    history = Tang_training.train(data, summary=True, verbose=True, load=False, path=path)
    del data

    return history


def TMD_MIL(archive_path):
    params = TMD_MIL_parameters

    for param_group, group_params in params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))
    reset_tensorflow_keras_backend()

    repeats = 1
    regenerate = REGENERATE
    for _ in range(repeats):
        for test_user in [1, 2, 3]:
            print('TEST USER: ' + str(test_user))
            path = os.path.join(archive, "test_user_" + str(test_user))
            config_edit('build_args', 'train_test_hold_out', test_user)

            TMD_MIL_experiment(path, regenerate)
            regenerate = False

    return


def Liang(archive_path):
    params = Liang_parameters

    for param_group, group_params in params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))
    reset_tensorflow_keras_backend()

    repeats = 1
    regenerate = REGENERATE
    for _ in range(repeats):
        for test_user in [1, 2, 3]:
            print('TEST USER: ' + str(test_user))
            path = os.path.join(archive, "test_user_" + str(test_user))
            config_edit('build_args', 'train_test_hold_out', test_user)

            Liang_experiment(path, regenerate)
            regenerate = False

    return


def Tang(archive_path):
    params = Tang_parameters

    for param_group, group_params in params.items():
        for param_name, param_value in group_params.items():
            config_edit(param_group, param_name, param_value)

    archive = os.path.join(archive_path, "save-" + time.strftime("%Y%m%d-%H%M%S"))
    reset_tensorflow_keras_backend()

    repeats = 1
    regenerate = REGENERATE
    for _ in range(repeats):
        for test_user in [1, 2, 3]:
            print('TEST USER: ' + str(test_user))
            path = os.path.join(archive, "test_user_" + str(test_user))
            config_edit('build_args', 'train_test_hold_out', test_user)

            Tang_experiment(path, regenerate)
            regenerate = False

    return