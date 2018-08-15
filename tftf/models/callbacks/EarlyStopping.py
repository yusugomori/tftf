class EarlyStopping(object):
    def __init__(self, patience=10, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def on_epoch_end(self, epoch, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('Early stopping on epoch {}.'.format(epoch))
                return True
        else:
            self._step = 0
            self._loss = loss
        return False
