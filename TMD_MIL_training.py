import shutil
import os
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.losses import CategoricalCrossentropy
from keras.metrics import categorical_accuracy
from build import Builder
from typing import Optional
from config_parser import Parser, config_edit
from TMD_MIL_utils import get_motion_model, get_location_model, get_MIL_model
from metrics import Metrics
from typing import Tuple


def motion_train(data: Builder, summary=True, verbose=True, load=False, path=None, scores=False) ->\
        Tuple[Builder, str, float, float, float]:
    conf = Parser()
    conf.get_args()

    model_name = 'motion_encoder'
    if not path:
        log_path = os.path.join('logs', model_name + '_TB')
        model_dir = os.path.join('models', model_name)
        model_file = '%s.h5' % model_name
        model_path = os.path.join(model_dir, model_file)
    else:
        log_path = os.path.join(path, 'logs', model_name + '_TB')
        model_dir = os.path.join(path, 'models', model_name)
        model_file = '%s.h5' % model_name
        model_path = os.path.join(model_dir, model_file)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    try:
        shutil.rmtree(log_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    try:
        os.remove(model_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    train, val, test = data(motion_transfer=True)
    model = get_motion_model(input_shapes=data.input_shape)

    if summary and verbose:
        print(model.summary())
        print('__________________________________________________________________________________________________')
        print(model.get_layer('spectrogram_encoder').summary())
        print('__________________________________________________________________________________________________')
        print(model.get_layer('classifier').summary())

    optimizer = Adam(lr=float(conf.learning_rate))
    loss_function = CategoricalCrossentropy()

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=[categorical_accuracy])

    if load:
        if not os.path.isdir(model_dir):
            return None

        model.load_weights(model_dir)

    val_steps = data.val_size // conf.batch_size
    train_steps = data.train_size // conf.batch_size
    test_steps = data.test_size // conf.batch_size

    tensorboard_callback = TensorBoard(log_path, histogram_freq=1)

    save_model = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        verbose=verbose,
        save_best_only=True,
        mode='min',
        save_weights_only=True)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=30,
        mode='min',
        verbose=verbose)

    reduce_lr_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=10,
        verbose=verbose,
        mode='min'
    )

    val_metrics = Metrics(val, val_steps, 'val', verbose)

    callbacks = [
        tensorboard_callback,
        save_model,
        early_stopping,
        reduce_lr_plateau,
        val_metrics
    ]

    history = model.fit(
        train,
        epochs=conf.epochs,
        steps_per_epoch=train_steps,
        validation_data=val,
        validation_steps=val_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        verbose=verbose
    )

    model.load_weights(model_path)

    test_metrics = Metrics(test, test_steps, 'test', verbose)
    model.evaluate(test, steps=test_steps, callbacks=[test_metrics])

    if scores:
        pass
    else:
        accuracy, f1_score, conf = 1., 1., 1.

    del train
    del val
    del test

    return data, model_path, accuracy, f1_score, conf


def location_train(data: Builder, summary=True, verbose=True, load=False, path=None, scores=False) ->\
        Tuple[Builder, str, float, float, float]:
    conf = Parser()
    conf.get_args()

    model_name = 'location_encoder'

    if not path:
        log_path = os.path.join('logs', model_name + '_TB')
        model_dir = os.path.join('models', model_name)
        model_file = '%s.h5' % model_name
        model_path = os.path.join(model_dir, model_file)
    else:
        log_path = os.path.join(path, 'logs', model_name + '_TB')
        model_dir = os.path.join(path, 'models', model_name)
        model_file = '%s.h5' % model_name
        model_path = os.path.join(model_dir, model_file)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    try:
        shutil.rmtree(log_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    try:
        os.remove(model_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    train, val, test = data(location_transfer=True)
    model = get_location_model(input_shapes=data.input_shape)

    if summary and verbose:
        print(model.summary())

    optimizer = Adam(lr=conf.learning_rate)
    loss_function = CategoricalCrossentropy()

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=[categorical_accuracy])

    if load:
        if not os.path.isdir(model_dir):
            return None

        model.load_weights(model_dir)

    val_steps = data.val_size // conf.batch_size
    train_steps = data.train_size // conf.batch_size
    test_steps = data.test_size // conf.batch_size

    tensorboard_callback = TensorBoard(log_path, histogram_freq=1)

    save_model = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        verbose=verbose,
        save_best_only=True,
        mode='min',
        save_weights_only=True)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=30,
        mode='auto',
        verbose=verbose)

    reduce_lr_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=10,
        verbose=verbose,
        mode='min'
    )

    val_metrics = Metrics(val, val_steps, verbose)

    callbacks = [
        tensorboard_callback,
        save_model,
        early_stopping,
        reduce_lr_plateau,
        val_metrics
    ]

    history = model.fit(
        train,
        epochs=conf.epochs,
        steps_per_epoch=train_steps,
        validation_data=val,
        validation_steps=val_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        verbose=verbose
    )

    model.load_weights(model_path)

    test_metrics = Metrics(test, test_steps, verbose)
    model.evaluate(test, steps=test_steps, callbacks=[test_metrics])

    if scores:
        pass
    else:
        accuracy, f1_score, conf = 1., 1., 1.

    del train
    del val
    del test

    return data, model_path, accuracy, f1_score, conf


def train(data: Builder, summary=True, verbose=True, load=False, path=None) -> Optional[Model]:
    conf = Parser()
    conf.get_args()

    motion_encoder = None
    if conf.motion_transfer == 'train' or conf.motion_transfer == 'load':
        config_edit('build_args', 'in_bags', False)

        data, weights_file, accuracy, f1_score, conf = motion_train(data=data,
                                                                    summary=summary,
                                                                    verbose=verbose,
                                                                    load=(conf.motion_transfer == 'load'),
                                                                    path=path,
                                                                    scores=False)

        motion_encoder = get_motion_model(data.input_shape)
        motion_encoder.load_weights(weights_file)

        return None

    location_encoder = None
    if conf.location_transfer == 'train' or conf.motion_transfer == 'load':
        data, weights_file, accuracy, f1_score, conf = location_train(data=data,
                                                                      summary=summary,
                                                                      verbose=verbose,
                                                                      load=(conf.motion_transfer == 'load'),
                                                                      path=path,
                                                                      scores=False)

        location_encoder = get_location_model(data.input_shape)
        location_encoder.load_weights(weights_file)

    conf = Parser()
    conf.get_args()

    model_name = 'TMD_MIL_classifier'

    if not path:
        log_path = os.path.join('logs', model_name + '_TB')
        model_dir = os.path.join('models', model_name)
        model_file = '%s.h5' % model_name
        model_path = os.path.join(model_dir, model_file)
    else:
        log_path = os.path.join(path, 'logs', model_name + '_TB')
        model_dir = os.path.join(path, 'models', model_name)
        model_file = '%s.h5' % model_name
        model_path = os.path.join(model_dir, model_file)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    try:
        shutil.rmtree(log_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    try:
        os.remove(model_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    train, val, test = data()
    model = get_MIL_model(input_shapes=data.input_shape)

    if summary and verbose:
        print(model.summary())

    optimizer = Adam(lr=conf.learning_rate)
    loss_function = CategoricalCrossentropy()

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=[categorical_accuracy])

    if load:
        if not os.path.isdir(model_dir):
            return None

        model.load_weights(model_dir)

    val_steps = data.val_size // conf.batch_size
    train_steps = data.train_size // conf.batch_size
    test_steps = data.test_size // conf.batch_size

    tensorboard_callback = TensorBoard(log_path, histogram_freq=1)

    save_model = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        verbose=verbose,
        save_best_only=True,
        mode='min',
        save_weights_only=True)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=30,
        mode='auto',
        verbose=verbose)

    reduce_lr_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=10,
        verbose=verbose,
        mode='min'
    )

    val_metrics = Metrics(val, val_steps, verbose)

    callbacks = [
        tensorboard_callback,
        save_model,
        early_stopping,
        reduce_lr_plateau,
        val_metrics
    ]

    history = model.fit(
        train,
        epochs=conf.epochs,
        steps_per_epoch=train_steps,
        validation_data=val,
        validation_steps=val_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        verbose=verbose
    )

    model.load_weights(model_path)

    test_metrics = Metrics(test, test_steps, verbose)
    model.evaluate(test, steps=test_steps, callbacks=[test_metrics])

    return history
