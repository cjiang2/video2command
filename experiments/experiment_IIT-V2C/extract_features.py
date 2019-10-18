import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import datasets.iit_v2c as iit_v2c
from v2c.config import *
from v2c.model import *

# Configuration for hperparameters
class FEConfig(Config):
    """Configuration for feature extraction.
    """
    NAME = 'Feature_Extraction'
    MODE = 'test'
    ROOT_DIR = ROOT_DIR
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C')
    WINDOW_SIZE = 30
    BATCH_SIZE = 50

def extract(dataset_path,
            dataset,
            model_name):
    # Create output directory
    output_path = os.path.join(dataset_path, model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Prepare pre-trained model
    print('Loading pre-trained model...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNNWrapper(backbone=model_name,
                       checkpoint_path=os.path.join(ROOT_DIR, 'checkpoints', 'backbone', 'resnet50.pth'))
    model.eval()
    model.to(device)
    print('Done loading.')

    # Feature extraction
    for i, (Xv, S, clip_name) in enumerate(dataset):
        with torch.no_grad():
            Xv = Xv.to(device)
            print('-'*30)
            print('Processing clip {}...'.format(clip_name))
            #print(imgs_path, clip_name)
            #assert len(imgs_path) == 30
            outputs = model(Xv)
            outputs = outputs.view(outputs.shape[0], -1)

            # Save into clips
            outfile_path = os.path.join(output_path, clip_name+'.npy')
            np.save(outfile_path, outputs.cpu().numpy())
            print('{}: {}'.format(clip_name+'.npy', S))
            print('Shape: {}, saved to {}.'.format(outputs.shape, outfile_path))
    del model
    return

def main_iit_v2c():
    # Parameters
    config = FEConfig()
    model_names = ['resnet50']

    annotation_files = ['train.txt', 'test.txt']
    for annotation_file in annotation_files:
        annotations = iit_v2c.load_annotations(config.DATASET_PATH, annotation_file)

        # Get torch.dataset object
        clips, targets, vocab, config = iit_v2c.parse_dataset(config, 
                                                              annotation_file,
                                                              numpy_features=False)
        config.display()
        transform = transforms.Compose([transforms.Resize((224, 224)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image_dataset = iit_v2c.FeatureDataset(clips, 
                                               targets, 
                                               numpy_features=False, 
                                               transform=transform)

        for model_name in model_names:
            extract(config.DATASET_PATH, image_dataset, model_name)


if __name__ == '__main__':
    main_iit_v2c()
