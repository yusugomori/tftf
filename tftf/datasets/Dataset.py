class Dataset(object):
    def __init__(self, data=None, target=None):
        self._data = data
        self._target = target

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, val):
        self._target = val
