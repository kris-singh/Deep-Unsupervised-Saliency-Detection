#!/usr/bin/env python
import logging

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101


class NoiseModule:
    def __init__(self, cfg):
        """
        Noise Module implements the noise mdoel from the paper.
        It maintains a variance for each pixel of each image.
        sd = sqrt(var), we store the variances for each of the prior distributions
        Sampling is done using var * N(0, 1)
        It is responsible for sampling from the noise distribution, loss calculation and updating the variance.
        Args
        ---
        cfg: (yacs.CfgNode) base configuration for the experiment.
        """
        super(NoiseModule, self).__init__()
        self.num_imgs = cfg.TRAIN.NUM_IMG
        self.num_maps = cfg.TRAIN.NUM_MAPS
        self.h, self.w = cfg.TRAIN.IMG_SIZE
        self.logger = logging.getLogger(str(cfg.SYSTEM.EXP_NAME) + ".noise_module")
        self.num_pixels = self.h * self.w
        self.alpha = cfg.NOISE.ALPHA
        self.device = cfg.SYSTEM.DEVICE
        # prior variance of the noise distribution, Initalized with zeros(!section 3.2 last para)
        self.noise_variance = torch.zeros(self.num_imgs * self.num_pixels)
        # emperical variance, observed variance between prediction and unsup labels
        self.emp_var = torch.zeros(self.num_imgs * self.num_pixels)

    def get_index(self, arr=None, img_idx=None):
        """
        Function for fetching indexes of pixels for img_idx
        Args
        ---
        img_idx: (int) index of image
        arr: (list) value array to fetch values from
        Returns
        ---
        values, idxs
        values: arr[idxs]
        idxs: starting and ending of pixels for the image
        """
        if arr is None:
            arr = self.noise_variance
        idx = img_idx * self.num_pixels
        return arr[idx:idx+self.num_pixels], np.arange(idx, idx+self.num_pixels)

    def get_index_multiple(self, arr=None, img_idxs=None):
        """
        Function for fetching indexes of pixels for img_idx
        Args
        ---
        img_idx: (list[int]) indexs of image
        arr: (list) value array to fetch values from
        Returns
        --- values: np.array, (len(img_idxs), num_pixels) idxs: starting and ending of pixels for the image
        """
        if arr is None:
            arr = self.noise_variance
        noise = np.zeros((len(img_idxs), self.num_pixels), dtype=np.float)
        noise_idx = np.zeros((len(img_idxs), self.num_pixels), dtype=np.float)
        for key, img_idx in enumerate(img_idxs):
            idx = img_idx * self.num_pixels
            noise[key] = arr[idx:idx+self.num_pixels]
            noise_idx[key] = np.arange(idx, idx+self.num_pixels)
        return noise, noise_idx

    def loss_fast(self, var1, var2):
        """
        Computes the loss using Eq 6
        Args
        ----
        var1: variance of q distribution, prior variance
        var2: variance of p distribution, predictive variance
        Returns
        ---
        noise loss:  float noise loss per image Note! different from paper which uses sum of noise
        """
        noise_loss = 0
        for idx in range(var1.shape[0]):
            covar1 = var1[idx].to(self.device) + 1e-6
            covar2 = var2[idx].to(self.device) + 1e-6
            ratio = 1. * (covar1 / covar2)
            loss = -0.5 * (torch.log(ratio) - ratio + 1).to(self.device)
            loss = abs(loss)
            # See Eq 6 from paper and https://github.com/pytorch/pytorch/blob/master/torch/distributions/kl.py#L394
            noise_loss += torch.sum(loss) / var1.shape[1]
        return noise_loss / var1.shape[0]

    def sample_noise(self, idxs):
        """
        Samples noise from Normal distribution with prior variance.
        Args
        ---
        idxs: List[int]
        sample from standard normal and transform using var for idxs
        Returns
        ---
        samples: np.array, noise samples of shape (len(idxs), NUM_MAPS, 128, 128)
        """
        samples = torch.zeros(len(idxs), self.num_maps, self.h, self.w)
        for idx, img_idx in enumerate(idxs):
            var, var_idx = self.get_index(self.noise_variance, img_idx)
            var = var.view(self.h, self.w).to(self.device)
            for map_idx in range(self.num_maps):
                # elementwise mutliplication with prior variance for each image of samples from N(0, 1)
                sample = var * torch.zeros(self.h, self.w).normal_().to(self.device)
                samples[idx][map_idx] = sample
        return samples

    def update(self):
        """
        Updates the prior variance for each pixel of each image by emp variance
        !Note: Emp variance needs to be updated before calling this.
        """
        print("Updating Noise")
        print("----------------------------------------")
        # import ipdb; ipdb.set_trace()
        # See equation 7, !Note: we have saved the variances and not sd, hence no need for squaring
        self.noise_variance = self.noise_variance + self.alpha * (self.emp_var - self.noise_variance)
        print(f'Max: {torch.max(self.noise_variance)}, Min :{torch.min(self.noise_variance)}')
        self.logger.info(f"Noise Variance: {self.noise_variance}")
        print("----------------------------------------")


def BaseModule(cfg):
    model = deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model
