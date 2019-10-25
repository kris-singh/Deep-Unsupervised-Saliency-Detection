#!/usr/bin/env python

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
_C.SYSTEM.SEED = 43

# Training setting
_C.DATA= CN()
_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.ROOT = '/data/workspaces/krish/sd/msrab_hkuis/'
_C.DATA.TRAIN.NOISE_ROOT = ('../noisy_gt/noisy_gt_1', '../noisy_gt/noisy_gt_2', '../noisy_gt/noisy_gt_3')
_C.DATA.TRAIN.LIST = '/data/workspaces/krish/sd/msrab_hkuis/small.lst'
_C.DATA.VAL = CN()
_C.DATA.VAL.ROOT = '/data/workspaces/krish/sd/msrab_hkuis/'
_C.DATA.VAL.LIST = '/data/workspaces/krish/sd/msrab_hkuis/small_val.lst'
_C.DATA.TEST = CN()
_C.DATA.TEST.ROOT = '/data/workspaces/krish/sd/msrab_hkuis/'
_C.DATA.TEST.LIST = '/data/workspaces/krish/sd/msrab_hkuis/small_test.lst'


# Optimizer
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = 'Adam'
_C.SOLVER.SCHEDULER = 'StepLR'
_C.SOLVER.LAMBDA = 0.01
_C.SOLVER.EPOCHS = 20
_C.SOLVER.IMG_SIZE = (256, 256)
_C.SOLVER.ALPHA = 0.01
_C.SOLVER.BETAS = (0.9, 0.99)
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.LR = 3e-4
_C.SOLVER.WEIGHT_DECAY = 0.9
_C.SOLVER.STEP_SIZE = 10
_C.SOLVER.FACTOR = 0.1
_C.SOLVER.MIN_LR = 1e-8
_C.SOLVER.PAITENCE = 10
_C.SOLVER.BATCH_SIZE = 2
_C.SOLVER.NUM_MAPS = 3
_C.SOLVER.NUM_IMG = 3000

# Validation Setting
_C.VAL = CN()
# TEST Setting
_C.TEST = CN()
# Noise
_C.NOISE = CN()
_C.NOISE.ALPHA=0.01


_C.SAVE_ROOT = '/data/workspaces/krish/sd/experiments'

cfg = _C  # users can `from config import cfg`
