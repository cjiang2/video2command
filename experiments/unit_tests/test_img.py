import os
import sys

import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from v2c.model import *
from datasets import iit_v2c
from v2c import utils
from v2c.config import *

# Configuration for hperparameters
class TrainConfig(Config):
    """Configuration for training with IIT-V2C.
    """
    NAME = 'v2c_IIT-V2C'
    MODE = 'train'
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C')
    MAXLEN = 10
    ROOT_DIR = ROOT_DIR
    
# Test configuration
config = TrainConfig()
config.display()
print()

# Test parse_dataset
annotation_file = config.MODE + '.txt'
clips, targets, vocab, config = iit_v2c.parse_dataset(config, annotation_file, numpy_features=False)
config.display()

print('Vocabulary:')
print(vocab.word2idx)
print('length ("<pad>" included):', len(vocab))
print('dataset:', len(clips), len(targets))
print()

transform = transforms.Compose([transforms.Resize(224), 
                                transforms.ToTensor()])

train_dataset = iit_v2c.FeatureDataset(clips, 
                                       targets, 
                                       numpy_features=False, 
                                       transform=transform)
# Test torch dataloader object
for i, (Xv, S, clip_name) in enumerate(train_dataset):
    print(Xv.shape, S.shape, clip_name)
    break
