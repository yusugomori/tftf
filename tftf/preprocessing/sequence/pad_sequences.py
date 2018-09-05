import numpy as np


def pad_sequences(data,
                  padding='post',
                  value=0):
    '''
    # Arguments
        data: list of list

    # Returns
        numpy.ndarray
    '''
    if type(data) is not list or len(data) > 1:
        raise AttributeError('`data` must be list of a list.')
    maxlen = len(max(data, key=len))
