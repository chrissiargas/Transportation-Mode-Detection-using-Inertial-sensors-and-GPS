import shutil
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.losses import CategoricalCrossentropy, MeanSquaredError
from keras.metrics import categorical_accuracy
from build import Builder
from config_parser import Parser
from Tang_utils import get_motion_model
from metrics import Metrics
from typing import Tuple
from evaluate import evaluate


def train(data: Builder, summary=True, verbose=True, load=False, path=None, eval=False, use_HMM=False) ->\
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
        if not load:
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
        print(model.get_layer('temporal_encoder').summary())
        print('__________________________________________________________________________________________________')
        print(model.get_layer('classifier').summary())

    optimizer = Adam(learning_rate=float(conf.learning_rate))
    loss_function = MeanSquaredError()

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
            patience=5,
            mode='min',
            verbose=verbose)

        # val_metrics = Metrics(val, val_steps, 'val', verbose)

        callbacks = [
            tensorboard_callback,
            save_model,
            early_stopping
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
        accuracy, f1, post_accuracy, post_f1, cm_df = evaluate(data, model, use_HMM)

    else:
        accuracy, f1, post_accuracy, post_f1, cm_df = 0., 0., 0., 0., 0.

    scores = [accuracy, f1, post_accuracy, post_f1, cm_df]

    del train
    del val
    del test

    return data, model_path, scores


from parameters import Tang_parameters
from config_parser import config_edit
import ruamel.yaml

if __name__ == '__main__':
    archive = os.path.join('archive', 'Tang', "save-" + '20231221-153032')

    test_user = 2

    path = os.path.join(archive, "test_user_" + str(test_user))
    config_edit('build_args', 'train_test_hold_out', test_user)

    data = Builder()
    history = train(data, summary=False, verbose=True, load=True, path=path, eval=True, use_HMM=True)