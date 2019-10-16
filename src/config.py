#!/usr/bin/env python

from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()

# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 8
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 0
_C.SYSTEM.DEVICE = 'cuda:5'
_C.SYSTEM.WAIT_TIME = 10
_C.SYSTEM.CHKPT_FREQ = 5
_C.SYSTEM.LOG_FREQ = 10
_C.SYSTEM.EXP_NAME = 'default'

# Training setting
_C.TRAIN = CN()
_C.TRAIN.SEED = 7
_C.TRAIN.ROOT = '/data/workspaces/krish/sd/msrab_hkuis/'
_C.TRAIN.NOISE_ROOT = ('../noisy_gt/noisy_gt_1', '../noisy_gt/noisy_gt_2', '../noisy_gt/noisy_gt_3')
_C.TRAIN.LIST = '/data/workspaces/krish/sd/msrab_hkuis/small.lst'
_C.TRAIN.LR = 1e-3
_C.TRAIN.LAMBDA = 0.01
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.EPOCHS = 20
_C.TRAIN.OPTIMIZER = 'ADAM'
_C.TRAIN.BETAS = (0.9, 0.99)
_C.TRAIN.IMG_SIZE = (256, 256)
_C.TRAIN.ALPHA = 0.01
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.NUM_MAPS = 3
_C.TRAIN.NUM_IMG = 3000
_C.TRAIN.SAVE_ROOT = '/data/workspaces/krish/sd/experiments'

# Validation Setting
_C.VAL = CN()
_C.VAL.ROOT = '/data/workspaces/krish/sd/msrab_hkuis/'
_C.VAL.LIST = '/data/workspaces/krish/sd/msrab_hkuis/small_val.lst'
_C.VAL.BATCH_SIZE = 5
# TEST Setting
_C.TEST = CN()
_C.TEST.ROOT = '/data/workspaces/krish/sd/msrab_hkuis/'
_C.TEST.LIST = '/data/workspaces/krish/sd/msrab_hkuis/small_test.lst'
_C.TEST.BATCH_SIZE = 5

# Noise
_C.NOISE = CN()
_C.NOISE.ALPHA=0.01


cfg = _C  # users can `from config import cfg`
