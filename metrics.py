
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import Callback
from config_parser import Parser


class Metrics(Callback):
    def __init__(self, data: tf.data.Dataset, steps, name, verbose=1):
        super(Metrics, self).__init__()

        self.conf = Parser()
        self.conf.get_args()

        self.data = data
        self.steps = steps
        self.score = 'macro'
        self.verbose = verbose
        self.name = name

    def on_epoch_end(self, epoch, logs={}):
        total_size = self.conf.batch_size * self.steps
        step = 0
        pred = np.zeros(total_size)
        true = np.zeros(total_size)

        for batch in self.data.take(self.steps):
            inputs = batch[0]
            outputs = batch[1]

            pred[step * self.conf.batch_size: (step+1) * self.conf.batch_size] = (
                np.argmax(np.asarray(self.model.predict(inputs, verbose=0)), axis=1))
            true[step * self.conf.batch_size: (step+1) * self.conf.batch_size] = (
                np.argmax(outputs, axis=1)
            )
            step+=1

        F1_score = f1_score(true, pred, average=self.score)
        Recall = recall_score(true, pred, average=self.score)
        Precision = precision_score(true, pred, average=self.score)

        del pred
        del true

        if self.verbose:
            print(f" - {self.name}_f1: %f - {self.name}_precision: %f - {self.name}_recall: %f"
                  % (F1_score, Precision, Recall))

        return

    def on_test_end(self, logs={}):
        total_size = self.conf.batch_size * self.steps
        step = 0
        pred = np.zeros(total_size)
        true = np.zeros(total_size)

        for batch in self.data.take(self.steps):
            inputs = batch[0]
            outputs = batch[1]

            pred[step * self.conf.batch_size: (step+1) * self.conf.batch_size] = (
                np.argmax(np.asarray(self.model.predict(inputs, verbose=0)), axis=1))
            true[step * self.conf.batch_size: (step+1) * self.conf.batch_size] = (
                np.argmax(outputs, axis=1))
            step+=1

        F1_score = f1_score(true, pred, average=self.score)
        Recall = recall_score(true, pred, average=self.score)
        Precision = precision_score(true, pred, average=self.score)

        del pred
        del true

        if self.verbose:
            print(f" - {self.name}_f1: %f - {self.name}_precision: %f - {self.name}_recall: %f"
                  % (F1_score, Precision, Recall))

        return

