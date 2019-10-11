import os
import sys

import numpy as np
import torch
from torch.utils import data

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from v2c.model import *
from datasets import iit_v2c
from v2c import utils
from v2c.config import *

dataset_path = os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C')
backbone = 'resnet50'
annotation_file_train = 'train.txt'
annotation_file_test = 'test.txt'

# ------------------------------
# LV1: Individual function test
# ------------------------------

# UnitTest on IIT-V2C dataset parser
# Train annotation
annotations_train = iit_v2c.load_annotations(dataset_path, annotation_file_train)
iit_v2c.summary(annotations_train)

clips_name_train, captions_train = iit_v2c.clipsname_captions(annotations_train)
print(clips_name_train[0], captions_train[0])
print(len(clips_name_train), len(captions_train))
print()

# Test annotation
annotations_test = iit_v2c.load_annotations(dataset_path, annotation_file_test)
iit_v2c.summary(annotations_test)

clips_name_test, captions_test = iit_v2c.clipsname_captions(annotations_test)
print(clips_name_test[0], captions_test[0])
print(len(clips_name_test), len(captions_test))

# UnitTest on NLP funcs
maxlen = utils.get_maxlen(captions_train)
print('maximum sequence length:', maxlen)

# UnitTest on vocabulary
print()
vocab = utils.build_vocab(captions_train, special_tokens=['<sos>', '<eos>'])
print()

# Process text tokens
targets_train = utils.texts_to_sequences(captions_train, vocab)
targets_train = utils.pad_sequences(targets_train, maxlen, padding='post')
print('Padded targets(train):', targets_train.shape)
idx = 10
print('Target:', targets_train[idx])
print('Translated Text:', utils.sequence_to_text(targets_train[idx], vocab, filter_specials=False))
print('Original Text:', captions_train[idx])

print()
print('Vocabulary:')
print(vocab.word2idx)

# ------------------------------
# LV2: tf.dataset funcs and config test
# ------------------------------

# Configuration for hperparameters
class TrainConfig(Config):
    """Configuration for training with IIT-V2C.
    """
    NAME = 'v2c_IIT-V2C'
    MODE = 'train'
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C')
    MAXLEN = 10

# Test configuration
config = TrainConfig()
config.display()
print()

# Test parse_dataset
clips, targets, vocab, config = iit_v2c.parse_dataset(config)
config.display()

print('Vocabulary:')
print(vocab.word2idx)
print('length ("<pad>" included):', len(vocab))
print('dataset:', len(clips), len(targets))
print()

train_dataset = iit_v2c.FeatureDataset(clips, targets)
train_loader = data.DataLoader(train_dataset, 
                               batch_size=config.BATCH_SIZE, 
                               shuffle=True, 
                               num_workers=config.WORKERS)
# Test torch dataloader object
for i, (Xv, S) in enumerate(train_loader):
    print(Xv.shape, S.shape)
    break

# Test parse_dataset on test_dataset
config.MODE = 'test'
clips_test, targets_test, vocab_test, config = iit_v2c.parse_dataset(config, vocab)
print(vocab == vocab_test)
test_dataset = iit_v2c.FeatureDataset(clips_test, targets_test)
test_loader = data.DataLoader(test_dataset, 
                              batch_size=config.BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=config.WORKERS)

# Test tf.dataset object
for i, (Xv, S) in enumerate(test_loader):
    print(Xv.shape, S.shape)
    break
