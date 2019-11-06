#!/usr/bin/env bash
# python -m ipdb deep_unsup_sd.py --clean_run True SYSTEM.DEVICE 'cuda:1' TRAIN.EPOCHS 2000 TRAIN.IMG_SIZE '(128, 128)' TRAIN.BATCH_SIZE 5 VAL.BATCH_SIZE 5 SYSTEM.EXP_NAME 'full-update-pred-only' TRAIN.NUM_MAPS 3 TRAIN.LR 0.001 SYSTEM.WAIT_TIME 500
python -m ipdb ../src/deep_unsup_sd.py --config_file ../configs/sgd/noise.yaml --clean_run TRUE
