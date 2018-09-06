from .pad_sequences import pad_sequences


class Pad(object):
    def __init__(self, padding='pre', value=0):
        self.padding = padding
        self.value = value

    def __call__(self, data):
        return pad_sequences(data,
                             padding=self.padding,
                             value=self.value)
