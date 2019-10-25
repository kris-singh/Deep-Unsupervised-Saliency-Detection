#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.models.segmentation import deeplabv3_resnet101

from collections import defaultdict, Iterable

class NoiseModule:
    def __init__(self, cfg):
        super(NoiseModule, self).__init__()
        self.num_imgs = cfg.SOLVER.NUM_IMG
        self.num_maps = cfg.SOLVER.NUM_MAPS
        self.h, self.w = cfg.SOLVER.IMG_SIZE
        self.num_pixels = self.h * self.w
        self.alpha = cfg.NOISE.ALPHA
        self.device = cfg.SYSTEM.DEVICE
        self.noise_variance = torch.zeros(self.num_imgs * self.num_pixels)
        self.mean = torch.zeros(self.noise_variance.shape[0])

    def get_index(self, arr=None, img_idx=None):
        arr = self.noise_variance
        idx = img_idx * self.num_pixels
        return arr[idx:idx+self.num_pixels], np.arange(idx, idx+self.num_pixels)

    def get_index_multiple(self, arr=None, img_idxs=None):
            arr = self.noise_variance
            noise = np.zeros((len(img_idxs), self.num_pixels), dtype=np.float)
            noise_idx = np.zeros((len(img_idxs), self.num_pixels), dtype=np.float)
            for key, img_idx in enumerate(img_idxs):
                idx = img_idx * self.num_pixels
                noise[key] = arr[idx:idx+self.num_pixels]
                noise_idx[key] = np.arange(idx, idx+self.num_pixels)
            return noise, noise_idx

    def loss(self, var1, var2):
        var1 += 1e-6
        var2 += 1e-6
        noise_loss = 0
        assert var1.dim() == 2
        assert var2.dim() == 2
        for idx in range(var1.shape[0]):
            covar1 = torch.diag(var1[idx]).cpu()
            covar2 = torch.diag(var2[idx]).cpu()
            mean = torch.zeros(covar1.shape[0]).cpu()
            dist1, dist2 = MultivariateNormal(mean, covar1), MultivariateNormal(mean, covar2)
            noise_loss += kl_divergence(dist1, dist2)
        return noise_loss.to(self.device)

    def sample_noise(self, idxs):
        samples = torch.zeros(len(idxs), self.num_maps, self.h, self.w)
        for idx, img_idx in enumerate(idxs):
            var, var_idx = self.get_index(self.noise_variance, img_idx)
            var += 1e-6
            var = torch.diag(var).to(self.device)
            mean = torch.zeros(var.shape[0]).to(self.device)
            dist = torch.distributions.MultivariateNormal(mean, var)
            for map_idx in range(self.num_maps):
                sample = dist.sample().view(self.h, self.w)
                samples[idx][map_idx] = sample
        return samples

    def update(self, idxs, pred_var):
        print("Updating Noise")
        print("----------------------------------------")
        pred_var = pred_var.detach().cpu()
        var, update_index = self.get_index_multiple(img_idxs=idxs)
        var = torch.tensor(var, dtype=torch.float).cpu()
        var = var
        var = torch.sqrt(var**2 + self.alpha * (pred_var - var)**2)
        for idx, u_idx in enumerate(update_index):
            u_idx = u_idx.astype(np.int)
            self.noise_variance[u_idx] = torch.sqrt(var[idx])
        print(self.noise_variance)
        print("----------------------------------------")


def BaseModule(cfg):
    model =  deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model
