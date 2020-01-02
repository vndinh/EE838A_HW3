import tensorflow as tf
import os
import shutil
from train import training
from valid import validate
from config import config
from test import testing

# Directories
# Validation
valid_dir = config.VALID.datapath
valid_gen_dir = config.VALID.deblur_gen_path
logs_valid = config.VALID.logs_valid
# Test
test_dir = config.TEST.datapath
test_gen_dir = config.TEST.deblur_gen_path
logs_test = config.TEST.logs_test
# Model
model_dir = config.TRAIN.model_dir

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='train', help='Running process')
  args = parser.parse_args()
  if args.mode == 'train':
    training()
  elif args.mode == 'valid':
    validate(valid_dir, valid_gen_dir, logs_valid, model_dir)
  elif args.mode == 'test':
    testing(test_dir, test_gen_dir, logs_test, model_dir)
  else:
    raise Exception("Unknown mode")


