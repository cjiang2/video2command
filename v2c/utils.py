import os
import sys
from collections import Counter
import operator
if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

import numpy as np

# ------------------------------------------------------------
# Functions for NLP, vocabulary, word tokens processing
# ------------------------------------------------------------

class Vocabulary(object):
    """Simple vocabulary wrapper.
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_counts = {}

    def __call__(self, word):
        if not word in self.word2idx:
            return None   # Return None if unknown word
            #return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word, freq=None):
        """Add individual word to vocabulary.
        """
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        if freq is not None:
            self.word_counts[word] = freq
        else:
            self.word_counts[word] = 1

    def get_words(self):
        """Return vocabulary.
        """
        return sorted(self.word2idx.keys())

    def get_bias_init_vector(self):
        """Calculate bias vector from word frequency distribution.
        NOTE: From NeuralTalk.
        """
        words = sorted(self.word2idx.keys())
        bias_init_vector = np.array([1.0*self.word_counts[word] for word in words])
        bias_init_vector /= np.sum(bias_init_vector) # Normalize to frequencies
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector) # Shift to nice numeric range
        return bias_init_vector

def build_vocab(texts, 
                frequency=None, 
                filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',
                lower=True,
                split=" ", 
                return_counter=False,
                special_tokens=['<sos>', '<eos>', '<unk>']):
    """Build vocabulary over texts/captions from training set.
    """
    # Load annotations
    counter = Counter()
    for i, text in enumerate(texts):
        tokens = word_tokenize(text, filters, lower, split)
        #print(tokens)
        counter.update(tokens)
        if (i+1) % 5000 == 0:
            print('{} captions tokenized...'.format(i+1))
    print('Done.')

    # Filter out words lower than the defined frequency
    if frequency is not None:
        counter = {word: cnt for word, cnt in counter.items() if cnt >= frequency}
    else:
        counter = counter

    # Only return counter if specified
    if return_counter:
        return counter

    # Create a vocabulary warpper
    vocab = Vocabulary()
    #vocab.add_word('<pad>')     # 0 is reserved for padding
    if special_tokens:
        for token in special_tokens:
            vocab.add_word(token)

    words = sorted(counter.keys())
    for word in words:
        vocab.add_word(word, counter[word])
    return vocab

def get_maxlen(texts):
    """Calculate the maximum document length for a list of texts.
    """
    return max([len(x.split(" ")) for x in texts])

def word_tokenize(text,
                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',
                  lower=True, 
                  split=" "):
    """Converts a text to a sequence of words (or tokens).
    """
    if lower:
        text = text.lower()
    text = text.translate(maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [i for i in seq if i]

def text_to_sequence(text,
                     vocab,
                     filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',
                     lower=True, 
                     split=" "):
    """Convert a text to numerical sequence.
    """
    tokens = word_tokenize(text, filters, lower, split)
    seq = []
    for token in tokens:
        word_index = vocab(token)
        if word_index is not None:  # Filter out unknown words
            seq.extend([word_index])
    return seq

def sequence_to_text(seq, 
                     vocab, 
                     filter_specials=True, 
                     specials=['<pad>', '<sos>', '<eos>']):
    """Restore sequence back to text.
    """
    tokens = []
    for idx in seq:
        tokens.append(vocab.idx2word.get(idx))
    if filter_specials:
        tokens =  filter_tokens(tokens, specials)
    return ' '.join(tokens)

def texts_to_sequences(texts,
                       vocab,
                       filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',
                       lower=True, 
                       split=" "):
    """Wrapper to convert batch of texts to sequences.
    """
    seqs = []
    for text in texts:
        seqs.append(text_to_sequence(text, vocab, filters, lower, split))
    return np.array(seqs)

def filter_tokens(tokens, 
                  specials=['<pad>', '<sos>', '<eos>']):
    """Filter specified words.
    """
    filtered = []
    for token in tokens:
        if token not in specials:
            filtered.append(token)
    return filtered

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """(Same from Tensorflow) Pads sequences to the same length.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x