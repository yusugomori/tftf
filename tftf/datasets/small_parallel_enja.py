import os
from .Dataset import Dataset


'''
Download 50k En/Ja Parallel Corpus
from https://github.com/odashi/small_parallel_enja
'''


def load_small_parallel_enja(path=None):
    path = path or 'small_parallel_enja'
    dir_path = os.path.join(os.path.expanduser('~'),
                            '.tftf',
                            path)
    # train_ja = os.path.join(dir_path, 'train.ja')
    # train_en = os.path.join(dir_path, 'train.en')
    # test_ja = os.path.join(dir_path, 'test.ja')
    # test_en = os.path.join(dir_path, 'test.en')

    files = ['train.ja', 'train.en', 'test.ja', 'test.en']

    for f in files:
        f_path = os.path.join(dir_path, f)


def _download(path):
    pass
