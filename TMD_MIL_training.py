import shutil
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.losses import CategoricalCrossentropy
from keras.metrics import categorical_accuracy
from build import Builder
from config_parser import Parser, config_edit
from TMD_MIL_utils import get_motion_model, get_location_model, get_MIL_model
from metrics import Metrics
from evaluate import evaluate


def motion_train(data: Builder, summary=True, verbose=True, load=False, path=None, eval=False, use_HMM=False):
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
        if not load:
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

        model.load_weights(model_path)

    else:
        val_steps = data.val_size // conf.batch_size
        train_steps = data.train_size // conf.batch_size

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
            patience=15,
            mode='min',
            verbose=verbose)

        reduce_lr_plateau = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.4,
            patience=10,
            verbose=verbose,
            mode='min'
        )

        # val_metrics = Metrics(val, val_steps, 'val', verbose)

        callbacks = [
            tensorboard_callback,
            save_model,
            early_stopping,
            reduce_lr_plateau
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

    test_steps = data.test_size // conf.batch_size

    test_metrics = Metrics(test, test_steps, 'test', verbose)
    model.evaluate(test, steps=test_steps, callbacks=[test_metrics])

    if eval:
        (accuracy, f1, precision, recall,
         post_accuracy, post_f1, post_precision, post_recall,
         cm_df, y, y_) = evaluate(data, model, motion_only=True, use_HMM=use_HMM, softmax=True)

    else:
        (accuracy, f1, precision, recall,
         post_accuracy, post_f1, post_precision, post_recall,
         cm_df, y, y_) = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.

    scores = [accuracy, f1, precision, recall, post_accuracy, post_f1, post_precision, post_recall, cm_df]

    del train
    del val
    del test

    return data, model_path, scores, y, y_


def location_train(data: Builder, summary=True, verbose=True, load=False, path=None, eval=False, use_HMM=False):
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

        model.load_weights(model_path)

    else:
        val_steps = data.val_size // conf.batch_size
        train_steps = data.train_size // conf.batch_size

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

        # val_metrics = Metrics(val, val_steps, verbose)

        callbacks = [
            tensorboard_callback,
            save_model,
            early_stopping,
            reduce_lr_plateau
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

    test_steps = data.test_size // conf.batch_size
    test_metrics = Metrics(test, test_steps, verbose)
    model.evaluate(test, steps=test_steps, callbacks=[test_metrics])

    if eval:
        accuracy, f1, post_accuracy, post_f1, cm_df = evaluate(data, model, use_HMM)
    else:
        accuracy, f1, post_accuracy, post_f1, cm_df = 0., 0., 0., 0., 0.

    scores = [accuracy, f1, post_accuracy, post_f1, cm_df]

    del train
    del val
    del test

    return data, model_path, scores


def train(data: Builder, summary=True, verbose=True, load=False, path=None, eval=False, use_HMM=False):
    conf = Parser()
    conf.get_args()

    motion_encoder = None
    if conf.motion_encoder == 'transfer' or conf.motion_encoder == 'train':
        config_edit('build_args', 'train_oversampling', True)

        only_motion = (conf.motion_encoder == 'train')
        transfer_motion = (conf.motion_encoder == 'transfer')
        eval = eval and only_motion
        use_HMM = use_HMM and only_motion

        if transfer_motion:
            config_edit('build_args', 'in_bags', False)

        data, weights_file, scores, y, y_ = motion_train(data=data,
                                                         summary=summary,
                                                         verbose=verbose,
                                                         load=load,
                                                         path=path,
                                                         eval=eval,
                                                         use_HMM=use_HMM)

        if only_motion:
            return data, weights_file, scores, y, y_

        motion_encoder = get_motion_model(data.input_shape)
        motion_encoder.load_weights(weights_file)

        config_edit('build_args', 'in_bags', True)

    location_encoder = None
    if conf.location_encoder == 'train' or conf.motion_encoder == 'load':
        data, weights_file, accuracy, f1_score, conf = location_train(data=data,
                                                                      summary=summary,
                                                                      verbose=verbose,
                                                                      load=(conf.motion_encoder == 'load'),
                                                                      path=path,
                                                                      eval=False,
                                                                      use_HMM=False)

        location_encoder = get_location_model(data.input_shape)
        location_encoder.load_weights(weights_file)

    config_edit('build_args', 'train_oversampling', True)
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
        if not load:
            os.remove(model_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    train, val, test = data(init=True)
    model = get_MIL_model(data.input_shape, motion_encoder, location_encoder)

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

        model.load_weights(model_path)

    else:
        val_steps = data.val_size // conf.batch_size
        train_steps = data.train_size // conf.batch_size

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

        # val_metrics = Metrics(val, val_steps, verbose)

        callbacks = [
            tensorboard_callback,
            save_model,
            early_stopping,
            reduce_lr_plateau
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

    test_steps = data.test_size // conf.batch_size

    test_metrics = Metrics(test, test_steps, verbose)
    model.evaluate(test, steps=test_steps, callbacks=[test_metrics])

    if eval:
        (accuracy, f1, precision, recall,
         post_accuracy, post_f1, post_precision, post_recall,
         cm_df, y, y_) = evaluate(data, model, motion_only=False, use_HMM=use_HMM, softmax=True)

    else:
        (accuracy, f1, precision, recall,
         post_accuracy, post_f1, post_precision, post_recall,
         cm_df) = 0., 0., 0., 0., 0., 0., 0., 0., 0.

    scores = [accuracy, f1, precision, recall, post_accuracy, post_f1, post_precision, post_recall, cm_df]

    del train
    del val
    del test

    return data, model_path, scores, y, y_


if __name__ == '__main__':
    archive = os.path.join('archive', 'TMD_MIL', "save-" + '20240123-142732')
    turn = 3
    test_user = 1

    path = os.path.join(archive, "turn_" + str(turn), "test_user_" + str(test_user))
    config_edit('build_args', 'train_test_hold_out', test_user)

    data = Builder()
    history = train(data, summary=False, verbose=True, load=True, path=path, eval=True, use_HMM=True)
