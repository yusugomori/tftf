import numpy as np
from sklearn.metrics import recall_score


def recall(preds, target, thres=0.5):
    if len(preds[0]) == 1:
        return recall_score(preds > thres, target)
    else:
        return recall_score(np.argmax(preds, 1),
                            np.argmax(target, 1).astype('int32'),
                            average='macro')
