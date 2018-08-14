import numpy as np
from sklearn.metrics import accuracy_score


def accuracy(preds, target, thres=0.5):
    if len(preds[0]) == 1:
        return accuracy_score(preds > thres, target)
    else:
        return accuracy_score(np.argmax(preds, 1),
                              np.argmax(target, 1).astype('int32'))
