from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.VALID = edict()
config.IMG = edict()
config.TEST = edict()

# Image Parameters
config.IMG.scale = 0.5

# Hyper Parameters
config.TRAIN.num_epoches = 1000
config.TRAIN.patch_size = 256
config.TRAIN.batch_size = 2
config.TRAIN.learning_rate_init = 5*1e-5
config.TRAIN.learning_rate_decay = 0.5
config.TRAIN.decay_period = 100

config.TRAIN.model_dir = '..\\model\\model.ckpt'
config.TRAIN.logs_dir = '..\\logs'
config.TRAIN.logs_train = '..\\logs\\logs_train.txt'
config.TRAIN.datapath = '..\\data\\train'

config.VALID.datapath = '..\\data\\valid'
config.VALID.deblur_gen_path = '..\\report\\valid_deblur_gen'
config.VALID.logs_valid = '..\\logs\\logs_valid.txt'

config.TEST.datapath = '..\\data\\test'
config.TEST.logs_test = '..\\logs\\logs_test.txt'
config.TEST.result_path = '..\\report'
config.TEST.deblur_gen_path = '..\\report\\test_deblur_gen'
