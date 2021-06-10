import numpy as np


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
