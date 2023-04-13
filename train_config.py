'''
Created on 12-Apr-2023

@author: EZIGO
'''
# import the necessary packages
import os
# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "datasets/rgb_80x80"
# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "datasets"
# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
# define the amount of data that will be used training
TRAIN_SPLIT = 0.8
# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.1

BS=64
alpha=0.01
epochs=25