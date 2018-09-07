import os
import subprocess
import numpy as np
from .Dataset import Dataset


'''
Download 50k En/Ja Parallel Corpus
from https://github.com/odashi/small_parallel_enja
and transform words to IDs.
'''


def load_small_parallel_enja(path=None,
                             to_ja=True,
                             start_char=1,
                             end_char=2,
                             oov_char=3,
                             index_from=4,
                             bos='<BOS>',
                             eos='<EOS>'):
    url_base = 'https://raw.githubusercontent.com/' \
               'odashi/small_parallel_enja/master/'

    path = path or 'small_parallel_enja'
    dir_path = os.path.join(os.path.expanduser('~'),
                            '.tftf', 'datasets', path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    f_ja = ['train.ja', 'test.ja']
    f_en = ['train.en', 'test.en']

    for f in (f_ja + f_en):
        f_path = os.path.join(dir_path, f)
        if not os.path.exists(f_path):
            url = url_base + f
            print('Downloading {}'.format(f))
            cmd = ['curl', '-o', f_path, url]
            subprocess.call(cmd)

    f_train_ja = os.path.join(dir_path, f_ja[0])
    f_test_ja = os.path.join(dir_path, f_ja[1])
    f_train_en = os.path.join(dir_path, f_en[0])
    f_test_en = os.path.join(dir_path, f_en[1])

    (train_ja, test_ja), num_words_ja = _build(f_train_ja, f_test_ja)
    (train_en, test_en), num_words_en = _build(f_train_en, f_test_en)

    if to_ja:
        train_X, test_X, num_X = train_en, test_en, num_words_ja
        train_y, test_y, num_y = train_ja, test_ja, num_words_en
    else:
        train_X, test_X, num_X = train_ja, test_ja, num_words_en
        train_y, test_y, num_y = train_en, test_en, num_words_ja

    train_X, test_X = np.array(train_X), np.array(test_X)
    train_y, test_y = np.array(train_y), np.array(test_y)

    return (train_X, train_y), (test_X, test_y), (num_X, num_y)


def _build(f_train, f_test,
           start_char=1,
           end_char=2,
           oov_char=3,
           index_from=4,
           bos='<BOS>',
           eos='<EOS>'):

    builder = _Builder(start_char=start_char,
                       end_char=end_char,
                       oov_char=oov_char,
                       index_from=index_from,
                       bos=bos,
                       eos=eos)
    builder.fit(f_train)
    train = builder.transform(f_train)
    test = builder.transform(f_test)

    return (train, test), builder.num_words


class _Builder(object):
    def __init__(self,
                 start_char=1,
                 end_char=2,
                 oov_char=3,
                 index_from=4,
                 bos='<BOS>',
                 eos='<EOS>'):
        self._vocab = None
        self._w2i = None
        self._i2w = None

        self.start_char = start_char
        self.end_char = end_char
        self.oov_char = oov_char
        self.index_from = index_from
        self.bos = bos
        self.eos = eos

    @property
    def num_words(self):
        return len(self._w2i)

    def fit(self, f_path):
        self._vocab = set()
        self._w2i = {}
        for line in open(f_path, encoding='utf-8'):
            _sentence = line.strip().split()
            self._vocab.update(_sentence)

        self._w2i = {w: (i + self.index_from)
                     for i, w in enumerate(self._vocab)}
        self._w2i[self.bos] = self.start_char
        self._w2i[self.eos] = self.end_char
        self._i2w = {i: w for w, i in self._w2i.items()}

    def transform(self, f_path):
        if self._vocab is None or self._w2i is None:
            raise AttributeError('`{}.fit` must be called before `transform`.'
                                 ''.format(self.__class__.__name__))
        sentences = []
        for line in open(f_path, encoding='utf-8'):
            _sentence = line.strip().split()
            _sentence = [self.bos] + _sentence + [self.eos]
            sentences.append(self._encode(_sentence))
        return sentences

    def _encode(self, sentence):
        encoded = []
        for w in sentence:
            if w not in self._w2i:
                id = self.oov_char
            else:
                id = self._w2i[w]
            encoded.append(id)

        return encoded
