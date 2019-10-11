import os
import sys
import glob
import pickle

import cv2
import numpy as np
import torch
from torch.utils import data

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import v2c.utils as utils

# ----------------------------------------
# Functions for IIT-V2C Database Integration
# ----------------------------------------

def load_annotations(dataset_path=os.path.join('datasets', 'IIT-V2C'),
                     annotation_file='train.txt'):
    """Helper function to parse IIT-V2C dataset.
    """
    def get_frames_no(init_frame_no, end_frame_no):
        frames = []
        for i in range(init_frame_no, end_frame_no+1, 1):
            frames.append(i)
        return frames

    # Read annotations
    annotations = {}
    with open(os.path.join(dataset_path, annotation_file), 'r') as f:
        i = 0
        annotation = []
        for line in f:
            line = line.strip()
            i += 1
            annotation.append(line)

            if i % 4 == 0:
                # Test cases
                #print(annotation)
                #assert annotation[-1] == ''
                #assert len(annotation[1].split(' ')) == 2
                # Collect Video Name, Annotated Sub-Video Clip id
                video_fname, video_id = '_'.join(annotation[0].split('_')[:-1]), annotation[0].split('_')[-1]

                # Collect init frame no ~ end frame no 
                # Populate frames and commands
                init_frame_no, end_frame_no = int(annotation[1].split(' ')[0]), int(annotation[1].split(' ')[1])
                frames = get_frames_no(init_frame_no, end_frame_no)
                command = annotation[2].strip().split(' ')

                if video_fname not in annotations:
                    annotations[video_fname] = [[video_id, frames, command]]
                else:
                    annotations[video_fname].append([video_id, frames, command])

                annotation = []

    return annotations

def summary(annotations):
    """Helper function for IIT-V2C dataset summary.
    """
    num_clips = 0
    num_frames = 0
    for video_path in annotations.keys():
        annotations_by_video = annotations[video_path]
        num_clips += len(annotations_by_video)
        for annotation in annotations_by_video:
            num_frames += len(annotation[1])
    print('# videos in total:', len(annotations))
    print('# sub-video clips in annotation file:', num_clips)
    print('# frames in total:', num_frames)

def clipsname_captions(annotations,
                       padding_words=True):
    """Get (clip_name, target) pairs from annotation.
    """
    # Parse all (inputs, captions) pair
    clips_name, captions = [], []
    for video_fname in annotations.keys():
        annotations_by_clip = annotations[video_fname]
        for annotation in annotations_by_clip:
            # Clip name
            clip_name = video_fname + '_' + annotation[0]

            # Get command caption
            if padding_words:
                target = '<sos> '+' '.join(annotation[2])+' <eos>'
            else:
                target = ' '.join(annotation[2])

            captions.append(target)
            clips_name.append(clip_name)

    return clips_name, captions

# For custom feature extraction
def avi2frames(dataset_path, 
               in_folder='avi_video',
               out_folder='images'):
    """Convert avi videos from IIT-V2C dataset into images.
    WARNING: SLOW + TIME CONSUMING process! Final images take LARGE DISK SPACE!
    """
    videos_path = glob.glob(os.path.join(dataset_path, in_folder, '*.avi'))
    for video_path in videos_path:
        video_fname = video_path.strip().split('/')[-1][:-4]
        save_path = os.path.join(dataset_path, out_folder, video_fname)
        print('Saving:', save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # OpenCV video feed
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        count = 0   # Start frame index as 0, as stated by the author
        cap.set(cv2.CAP_PROP_FPS, 15)   # Force fps into 15, as stated by the author
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        while success:
            cv2.imwrite(os.path.join(save_path, 'frame%d.png' % count), frame)  # Save into loseless *.png format
            count += 1
            success, frame = cap.read()
    return True

def imgspath_targets_v1(annotations, 
                        max_frames=30,
                        dataset_path=os.path.join('datasets', 'IIT-V2C'),
                        folder='images',
                        synthetic_frame_path=os.path.join('imagenet_frame.png'),
                        padding_words=True):
    """Get training/test image-command pairs.
    Version v2 strategy: Same as original IIT-V2C strategy, have a number 
    of max_frames images per sample. Cut images larger than max_frames, populate
    sample if no. images is smaller than max_frames.
    """
    def get_frames_path(frames_no,
                        video_fname,
                        max_frames=30,
                        dataset_path=os.path.join('datasets', 'IIT-V2C'),
                        folder='images',
                        synthetic_frame_path=os.path.join('imagenet_frame.png')):
        """Helper func to parse image path from numbers.
        """
        # Cut additional images by getting min loop factor
        num_frames = len(frames_no)
        loop_factor = min(num_frames, max_frames)

        imgs_path = []
        for i in range(loop_factor):
            img_path = os.path.join(dataset_path, folder, video_fname, 'frame{}.png'.format(frames_no[i]))
            if os.path.isfile(img_path):    # Check if frame exists
                imgs_path.append(img_path)

        # Add synthetically made imagenet frame
        while len(imgs_path) < max_frames:
            imgs_path.append(synthetic_frame_path)

        return imgs_path

    # Parse all (inputs, targets) pair
    inputs, targets, clips_name = [], [], []
    for video_fname in annotations.keys():
        annotations_by_clip = annotations[video_fname]
        for annotation in annotations_by_clip:
            # Clip name
            clip_name = video_fname + '_' + annotation[0]

            # Get all images of the current clip
            frames_path = get_frames_path(annotation[1], 
                                          video_fname,
                                          max_frames,
                                          dataset_path,
                                          folder,
                                          synthetic_frame_path)

            # Get command caption
            if padding_words:
                target = '<sos> '+' '.join(annotation[2])+' <eos>'
            else:
                target = ' '.join(annotation[2])
            
            inputs.append(frames_path)
            targets.append(target)
            clips_name.append(clip_name)

    return inputs, targets, clips_name

# ----------------------------------------
# Functions for torch.data.Dataset
# ----------------------------------------

def parse_dataset(config, 
                  vocab=None):
    """Parse IIT-V2C dataset and update configuration.
    """

    # Load annotation 1st
    annotation_file = config.MODE + '.txt'
    annotations = load_annotations(config.DATASET_PATH, annotation_file)
    clips, captions = clipsname_captions(annotations)
    clips = [os.path.join(config.DATASET_PATH, config.BACKBONE, x + '.npy') for x in clips]

    # Build vocabulary
    if vocab is None:
        vocab = utils.build_vocab(captions, 
                                  frequency=config.FREQUENCY,
                                  special_tokens=config.SPECIAL_TOKENS)
    # Reset vocab_size
    config.VOCAB_SIZE = len(vocab)

    if config.MAXLEN is None:
        config.MAXLEN = utils.get_maxlen(captions) 
            
    # Process text tokens
    targets = utils.texts_to_sequences(captions, vocab)
    targets = utils.pad_sequences(targets, config.MAXLEN, padding='post')

    return clips, targets, vocab, config

class FeatureDataset(data.Dataset):
    """Create an instance of IIT-V2C dataset with pre-extracted features.
    """
    def __init__(self, 
                 inputs,
                 targets):
        # Load annotations
        self.inputs, self.targets = inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        Xv = np.load(self.inputs[idx])
        S = self.targets[idx]
        return Xv, S