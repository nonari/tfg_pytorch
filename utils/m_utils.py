import numpy as np
from sklearn import metrics

from unet.loader import im_to_tensor


def accuracy(true, prediction, classes=10):
    l = []
    for i in range(1, classes+1):
        true_copy = true.copy()
        pred_copy = prediction.copy()
        true_copy[true_copy != i] = 0
        pred_copy[pred_copy != i] = 0
        acc = metrics.accuracy_score(true_copy, pred_copy)
        # stat = metrics.confusion_matrix(true_copy, pred_copy).ravel()
        # if len(stat) > 1:
        #     tn, fp, fn, tp = stat
        #     sp = (tp + tn) / (tp + tn + fp + fn)
        # else:
        #     sp = 1
        l.append(acc)

    return l


def summarize_metrics(patient):
    ground = list(map(lambda e: e['truth'], patient))
    pred = list(map(lambda e: e['prediction'], patient))
    loss = list(map(lambda e: e['loss'], patient))

    ground = tuple(ground)
    pred = tuple(pred)

    ground = np.dstack(ground).flatten()
    pred = np.dstack(pred).flatten()

    fshape = (1, ground.shape[0])
    ground_tensor = im_to_tensor(ground.reshape(fshape), shape=fshape).numpy()
    pred_tensor = im_to_tensor(pred.reshape(fshape), shape=fshape).numpy()

    jaccard = metrics.jaccard_score(ground, pred, average=None)
    recall = metrics.recall_score(ground, pred, average=None)
    f1 = metrics.f1_score(ground, pred, average=None)
    # prec = metrics.precision_score(ground, pred, average=None)
    prec = specificityB(ground_tensor, pred_tensor)
    confusion = metrics.confusion_matrix(ground, pred, normalize='true')
    acc = accuracy(ground, pred)
    loss = np.mean(loss, axis=0)
    return {"loss": loss, "f1": f1, "recall": recall, "accuracy": acc,
            "jaccard": jaccard, "confusion": confusion, "specificity": prec}


def average_metrics(scores):
    f1 = list(map(lambda e: e['f1'], scores))
    recall = list(map(lambda e: e['recall'], scores))
    specif = list(map(lambda e: e['specificity'], scores))
    acc = list(map(lambda e: e['accuracy'], scores))
    jaccard = list(map(lambda e: e['jaccard'], scores))
    confusion = list(map(lambda e: e['confusion'], scores))
    loss = list(map(lambda e: e['loss'], scores))

    f1_mean = np.mean(f1, axis=0)
    recall = np.mean(recall, axis=0)
    jaccard = np.mean(jaccard, axis=0)
    spe_mean = np.mean(specif, axis=0)
    acc_mean = np.mean(acc, axis=0)
    confusion = np.mean(np.dstack(tuple(confusion)), axis=2)
    loss = np.mean(loss, axis=0)


    return {"loss": loss, "f1": f1_mean, "recall": recall, "specificity": spe_mean, "accuracy": acc_mean,
            "jaccard": jaccard, "confusion": confusion}


def stdev_metrics(scores):
    f1 = list(map(lambda e: e['f1'], scores))
    recall = list(map(lambda e: e['recall'], scores))
    specif = list(map(lambda e: e['specificity'], scores))
    acc = list(map(lambda e: e['accuracy'], scores))
    jaccard = list(map(lambda e: e['jaccard'], scores))
    confusion = list(map(lambda e: e['confusion'], scores))
    loss = list(map(lambda e: e['loss'], scores))

    f1_mean = np.std(f1, axis=0)
    recall = np.std(recall, axis=0)
    jaccard = np.std(jaccard, axis=0)
    spe_mean = np.std(specif, axis=0)
    acc_mean = np.std(acc, axis=0)
    confusion = np.std(np.dstack(tuple(confusion)), axis=2)
    loss = np.std(loss, axis=0)

    return {"loss": loss, "f1": f1_mean, "recall": recall, "specificity": spe_mean, "accuracy": acc_mean,
            "jaccard": jaccard, "confusion": confusion}


def specificityB(true, prediction):
    vp = np.logical_and(true, prediction)

    tmp1 = np.logical_not(true)
    tmp2 = np.logical_not(prediction)
    vn = np.logical_and(tmp1, tmp2)

    fp = np.logical_xor(prediction, vp)

    tmp = np.logical_not(prediction)
    fn = np.logical_xor(tmp, vn)

    vp = np.sum(vp, axis=1)
    fp = np.sum(fp, axis=1)
    fn = np.sum(fn, axis=1)
    vn = np.sum(vn, axis=1)
    vp = np.sum(vp, axis=1)
    fp = np.sum(fp, axis=1)
    fn = np.sum(fn, axis=1)
    vn = np.sum(vn, axis=1)

    s = vn / (vn + fp)

    return s.tolist()


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



