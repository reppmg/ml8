from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

char_count = 57
sentece_length = 40

path = get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = sentece_length
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
print('chars', chars)

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

np.random.seed(322)
coef = [np.random.rand() + 0.01 for i in range(sentece_length)]
coef[1] = 0.00000001
# feed
symbols = x[1]
print(text[sentece_length:2*sentece_length], end="")
for i in range(100):
    preds = [(symbols[j] * coef[j]) for j in range(sentece_length)]
    preds = np.sum(preds, axis=0)
    preds = np.asarray(preds).astype('float64') + 0.001
    preds = np.log(preds) / 0.5
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    new_symbol_index = np.argmax(probas)
    # new_symbol_index = np.argmax(preds)
    print(indices_char[new_symbol_index], end="")
    new_symbol = np.zeros(len(char_indices))
    new_symbol[new_symbol_index] = 1
    for j in range(len(symbols)-1):
        symbols[j] = symbols[j + 1]
    symbols[sentece_length - 1] = new_symbol
