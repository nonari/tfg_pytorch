import numpy as np
from sklearn import metrics


def average_metrics(scores):
    f1 = list(map(lambda e: e['f1'], scores))
    recall = list(map(lambda e: e['recall'], scores))
    jaccard = list(map(lambda e: e['jaccard'], scores))
    confusion = list(map(lambda e: e['confusion'], scores))
    loss = list(map(lambda e: e['loss'], scores))

    f1_mean = np.mean(f1, axis=0)
    recall = np.mean(recall, axis=0)
    jaccard = np.mean(jaccard, axis=0)
    confusion = np.mean(np.dstack(tuple(confusion)), axis=2)
    loss = sum(loss) / len(loss)

    return {"loss": loss, "f1": f1_mean, "recall": recall, "jaccard": jaccard, "confusion": confusion}


def specificity(true, prediction, classes=10):
    l = []
    for i in range(1, classes+1):
        true_copy = true.copy()
        pred_copy = prediction.copy()
        true_copy[true_copy != i] = 0
        pred_copy[pred_copy != i] = 0
        stat = metrics.confusion_matrix(true_copy, pred_copy).ravel()
        if len(stat) > 1:
            tn, fp, fn, tp = stat
            sp = tn / (tn + fp)
        else:
            sp = 1
        l.append(sp)

    return l
