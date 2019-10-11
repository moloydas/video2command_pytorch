import os
import glob
import sys
import pickle

import numpy as np
import torch
from torch.utils import data

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from v2c.model import *
from v2c import utils
from v2c.config import *
from datasets import iit_v2c

# Configuration for hperparameters
class TestConfig(Config):
    """Configuration for training with IIT-V2C.
    """
    NAME = 'v2c_IIT-V2C'
    MODE = 'test'
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C')
    MAXLEN = 10

# Setup configuration class
config = TestConfig()
# Setup tf.dataset object
vocab = pickle.load(open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'rb'))
clips, targets, _, config = iit_v2c.parse_dataset(config, vocab)
test_dataset = iit_v2c.FeatureDataset(clips, targets)
test_loader = data.DataLoader(train_dataset, 
                              batch_size=config.BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=config.WORKERS)
config.display()

# Setup and build video2command training inference
v2c_model = Video2Command(config)
v2c_model.build()

# Start evaluating
v2c_model.load_weights(os.path.join(config.CHECKPOINT_PATH, 'saved', 'v2c_epoch_150.pth'))
y_pred, y_true = v2c_model.evaluate(test_loader, vocab)

# Save to evaluation file
f = open(os.path.join(config.CHECKPOINT_PATH, 'prediction.txt'), 'w')

for i in range(len(y_pred)):
    print(y_pred[i])
    pred_command = utils.sequence_to_text(y_pred[i], vocab)
    print(y_true[i])
    true_command = utils.sequence_to_text(y_true[i], vocab)
    f.write('------------------------------------------\n')
    f.write(str(i) + '\n')
    f.write(pred_command + '\n')
    f.write(true_command + '\n')

print('Ready for cococaption.')