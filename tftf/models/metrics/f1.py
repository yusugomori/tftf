import numpy as np
from sklearn.metrics import f1_score


def f1(preds, target, thres=0.5):
    if len(preds[0]) == 1:
        return f1_score(preds > thres, target)
    else:
        return f1_score(np.argmax(preds, 1),
                        np.argmax(target, 1).astype('int32'),
                        average='macro')
