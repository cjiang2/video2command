import os
import sys

import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from datasets import iit_v2c

success = iit_v2c.avi2frames(os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C'))
print('Done.', success)