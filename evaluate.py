from build import Builder
import postprocess
from postprocess import HMM_classifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.models import Model


def evaluate(data: Builder, model: Model, use_HMM: bool = False, motion_only=False):
    train_Y, _, _ = postprocess.get_YY_(data, model, motion_only, pred=False)
    test_YY_, y, y_ = postprocess.get_YY_(data, model, motion_only, pred=True)

    accuracy = accuracy_score(y, y_)
    f1 = f1_score(y, y_, average='macro')
    cm = confusion_matrix(y, y_)

    cm_df = pd.DataFrame(cm, index=['{:}'.format(x) for x in data.modes],
                         columns=['{:}'.format(x) for x in data.modes])
    cm_df = cm_df.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('Accuracy without post-processing: {}'.format(accuracy))
    print('F1-Score without post-processing: {}'.format(f1))

    post_accuracy = None
    post_f1 = None
    if use_HMM:
        HMM = HMM_classifier()
        HMM.fit(train_Y)
        post_y_ = HMM.predict(test_YY_)

        post_accuracy = accuracy_score(y, post_y_)
        post_f1 = f1_score(y, post_y_, average='macro')

        print('Accuracy with post-processing: {}'.format(post_accuracy))
        print('F1-Score with post-processing: {}'.format(post_f1))

    return accuracy, f1, post_accuracy, post_f1, cm_df