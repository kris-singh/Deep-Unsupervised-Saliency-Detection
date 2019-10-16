import torch


def min2d(tensor, dim1=2, dim2=3):
    return torch.min(torch.min(tensor, dim1, keepdim=True)[0], dim2, keepdim=True)[0]


def max2d(tensor, dim1=2, dim2=3):
    return torch.max(torch.max(tensor, dim1, keepdim=True)[0], dim2, keepdim=True)[0]
