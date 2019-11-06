#!/usr/bin/env python

import numpy as np
import torch


def to_np(x):
    if not x.device == 'cpu':
        return x.cpu().detach().numpy()
    else:
        return x.detach().numpy()

def count_nonzero(x):
    """
    x: np.array of shape (1, ) or (2, ?)
    """
    return len(np.transpose(np.nonzero(x)))


def set_seeds(cfg):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(cfg.SYSTEM.SEED)
    np.random.seed(cfg.SYSTEM.SEED)


def custom_lr(base_lr, maxiter, power):
    fnc = lambda iter: base_lr * (1 - iter / maxiter) ** power
    return fnc
