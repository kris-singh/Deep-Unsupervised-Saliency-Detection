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
