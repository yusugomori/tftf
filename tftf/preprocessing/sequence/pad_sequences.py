import numpy as np


def pad_sequences(data,
                  padding='pre',
                  value=0):
    '''
    # Arguments
        data: list of lists / np.array of lists

    # Returns
        numpy.ndarray
    '''
    if type(data[0]) is not list:
        raise ValueError('`data` must be a list of lists')
    maxlen = len(max(data, key=len))

    if padding == 'pre':
        data = \
            [[value] * (maxlen - len(data[i])) + data[i]
             for i in range(len(data))]
    else:
        data = \
            [data[i] + [value] * (maxlen - len(data[i]))
             for i in range(len(data))]

    return np.array(data)
