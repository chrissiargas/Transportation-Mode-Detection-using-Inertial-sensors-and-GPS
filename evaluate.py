from build import Builder
import postprocess
from postprocess import HMM_classifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def evaluate(data: Builder, model: Model, use_HMM: bool = False,
             motion_only=False, softmax=False):
    show_roc = False

    train_Y, _, _ = postprocess.get_YY_(data, model, motion_only, pred=False, softmax=softmax)
    test_YY_, y, y_ = postprocess.get_YY_(data, model, motion_only, pred=True, softmax=softmax)
    prob_y_ = np.array([list(probs.values()) for probs in test_YY_['prob'].values])

    accuracy = accuracy_score(y, y_)
    f1 = f1_score(y, y_, average='macro')
    recall = recall_score(y, y_, average='macro')
    precision = precision_score(y, y_, average='macro')
    cm = confusion_matrix(y, y_)

    cm_df = pd.DataFrame(cm, index=['{:}'.format(x) for x in data.modes],
                         columns=['{:}'.format(x) for x in data.modes])
    cm_df = cm_df.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('Accuracy without post-processing: {}'.format(accuracy))
    print('F1-Score without post-processing: {}'.format(f1))
    print('Precision without post-processing: {}'.format(precision))
    print('Recall without post-processing: {}'.format(recall))

    post_accuracy = None
    post_f1 = None
    post_recall = None
    post_precision = None

    if use_HMM:
        HMM = HMM_classifier()
        HMM.fit(train_Y)
        post_y_ = HMM.predict(test_YY_)

        post_accuracy = accuracy_score(y, post_y_)
        post_f1 = f1_score(y, post_y_, average='macro')
        post_recall = recall_score(y, post_y_, average='macro')
        post_precision = precision_score(y, post_y_, average='macro')

        print('Accuracy with post-processing: {}'.format(post_accuracy))
        print('F1-Score with post-processing: {}'.format(post_f1))
        print('Precision with post-processing: {}'.format(post_precision))
        print('Recall with post-processing: {}'.format(post_recall))

    diag = np.eye(data.n_modes)
    y = np.array([diag[true] for true in y])

    if show_roc:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        lw = 2

        for i in range(data.n_modes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], prob_y_[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr_grid = np.linspace(0.0, 1.0, 1000)

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(8):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

        # Average it and compute AUC
        mean_tpr /= 8

        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        for i in range(data.n_modes):
            plt.plot(fpr[i], tpr[i], lw=2,
                     label='ROC curve for {0} (area = {1:0.2f})'
                           ''.format(data.modes[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-class data')
        plt.legend(loc="lower right")
        plt.show()

    return accuracy, f1, precision, recall, post_accuracy, post_f1, post_precision, post_recall, cm_df, y, prob_y_