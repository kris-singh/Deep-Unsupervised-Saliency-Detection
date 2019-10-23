#!/usr/bin/env python

import os
from PIL import Image
import torch
from torch.utils import data

from config import cfg
import numpy as np
import random


class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list, noise_root):
        self.sal_root = data_root
        self.sal_source = data_list
        self.noisy_path = noise_root
        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        im_name = self.sal_list[item % self.sal_num].split()[0]
        gt_name = self.sal_list[item % self.sal_num].split()[1]

        sal_image = load_image(os.path.join(self.sal_root, im_name))
        sal_label = load_sal_label(os.path.join(self.sal_root, gt_name))
        sal_noisy_label = []
        for noise_path in self.noisy_path:
            name = gt_name.split('/')[-1][:-4]
            sal_noisy_label.append(torch.Tensor(load_image(os.path.join(noise_path, name+'_ngt.png'), True)))

        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_noisy_label = torch.stack(sal_noisy_label)

        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'idx': item, 'sal_noisy_label': sal_noisy_label}
        return sample

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.img_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.img_list)

    def __getitem__(self, item):
        # import ipdb; ipdb.set_trace()
        im_name = self.img_list[item % self.image_num].split()[0]
        gt_name = self.img_list[item % self.image_num].split()[1]
        image = load_image(os.path.join(self.data_root, im_name))
        image = torch.Tensor(image)
        sal_label = load_sal_label(os.path.join(self.data_root, gt_name))
        sal_label = torch.Tensor(sal_label)

        return {'image': image, 'label': sal_label}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.TRAIN.ROOT, config.TRAIN.LIST, config.TRAIN.NOISE_ROOT)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.TRAIN.BATCH_SIZE,
                                      shuffle=shuffle, num_workers=config.SYSTEM.NUM_WORKERS,
                                      pin_memory=pin, drop_last=True)
    if mode == 'val':
        shuffle = True
        dataset = ImageDataTrain(config.VAL.ROOT, config.VAL.LIST, config.TRAIN.NOISE_ROOT)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.VAL.BATCH_SIZE,
                                      shuffle=shuffle, num_workers=config.SYSTEM.NUM_WORKERS,
                                      pin_memory=pin, drop_last=True)
    if mode == 'test':
        shuffle = False
        dataset = ImageDataTest(config.TEST.ROOT, config.TEST.LIST)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.TEST.BATCH_SIZE,
                                      shuffle=shuffle, num_workers=config.SYSTEM.NUM_WORKERS,
                                      pin_memory=pin, drop_last=True)
    return data_loader


def load_image(path, noise=False):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    im = im.resize(cfg.TRAIN.IMG_SIZE)
    in_ = np.array(im, dtype=np.float32)
    if not len(in_.shape)==3 and not noise:
        in_ = in_[np.newaxis, :, :]
        in_ = np.tile(in_, [3, 1, 1])
    if not noise:
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        in_ = in_.transpose((2,0,1))
    return in_


def load_sal_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    im = im.resize(cfg.TRAIN.IMG_SIZE)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label


def load_noisy_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    im = im.resize(cfg.TRAIN.IMG_SIZE)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label


def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
    return img, label
