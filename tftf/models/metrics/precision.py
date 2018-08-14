import numpy as np
from sklearn.metrics import precision_score


def precision(preds, target, thres=0.5):
    if len(preds[0]) == 1:
        return precision_score(preds > thres, target)
    else:
        return precision_score(np.argmax(preds, 1),
                               np.argmax(target, 1).astype('int32'),
                               average='macro')
