#!/usr/bin/env python
import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from checkpoint import Checkpointer
from config import cfg
from dataloader import get_loader
from model import BaseModule, NoiseModule
from utils.early_stopping import check_cgt
from utils.math import max2d, min2d
from utils.save import save_config
from utils.setup_logger import setup_logger
from utils.visualise import visualize_results

EPS = 1e-3

h, w = cfg.TRAIN.IMG_SIZE[0], cfg.TRAIN.IMG_SIZE[1]
num_pixels = h*w
num_imgs = cfg.TRAIN.NUM_IMG
batch_size = cfg.TRAIN.BATCH_SIZE
num_batches = num_imgs // batch_size


def train(cfg, model, optimizer, loader, chkpt, writer, offset):
    train_loader, val_loader = loader
    device = cfg.SYSTEM.DEVICE
    pred_criterion = torch.nn.BCEWithLogitsLoss()
    train_noise = False
    wait = 0
    pred_model, noise_model = model
    for epoch in range(cfg.TRAIN.EPOCHS):
        pred_losses = []
        noise_losses = []
        losses = []
        val_losses = []
        for batch_idx, data in enumerate(train_loader):
            pred_model.train()
            writer_idx = batch_size * batch_idx + batch_size * epoch * num_batches
            item_idxs = data['idx']
            x = data['sal_image'].to(device)
            y = data['sal_label'].to(device)
            if cfg.SYSTEM.EXP_NAME in ['noise', 'real', 'overfit', 'overfit-noise']:
                y_noise = torch.unsqueeze(data['sal_noisy_label'][:, 0, :, :], 1).to(device)
            else:
                y_noise = data['sal_noisy_label'].to(device)
            y_n_min, y_n_max = min2d(y_noise), max2d(y_noise)
            y_noise = (y_noise - y_n_min) / (y_n_max - y_n_min)
            pred = pred_model(x)['out']
            y_pred = pred

            if cfg.SYSTEM.EXP_NAME == 'noise' or cfg.SYSTEM.EXP_NAME == 'real' or cfg.SYSTEM.EXP_NAME == 'full-update-pred-only' or cfg.SYSTEM.EXP_NAME == 'full-update':
                noise_prior = noise_model.sample_noise(item_idxs).to(device)
                pred = y_pred + noise_prior
                # truncate to lie in range[0, 1]
                pred_min, pred_max = min2d(pred), max2d(pred)
                pred_norm = (pred - pred_min) / (pred_max - pred_min)
            if cfg.SYSTEM.EXP_NAME == 'noise': # Works!
                pred_loss = pred_criterion(pred_norm, y_noise) / (cfg.TRAIN.BATCH_SIZE)
            if cfg.SYSTEM.EXP_NAME == 'real': # Works!
                pred_loss = pred_criterion(pred_norm, y) / (cfg.TRAIN.BATCH_SIZE)
            if cfg.SYSTEM.EXP_NAME == 'overfit': # Works!
                pred_loss = pred_criterion(pred, y) / (cfg.TRAIN.BATCH_SIZE)
            if cfg.SYSTEM.EXP_NAME == 'overfit-noise': # Works!
                pred_loss = pred_criterion(pred, y_noise) / (cfg.TRAIN.BATCH_SIZE)
            else:
                pred_loss = pred_criterion(pred_norm, y_noise) / (cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.NUM_MAPS)
            pred_losses.append(pred_loss.squeeze())

            noise_loss = 0.0
            if train_noise:
                pred_var = torch.var(y_noise - y_pred, 1).view(cfg.TRAIN.BATCH_SIZE, -1)
                if cfg.SYSTEM.EXP_NAME == 'full-update':
                    var, _ = noise_model.get_index_multiple(img_idxs=item_idxs)
                    var = torch.from_numpy(var).float()
                    noise_loss = noise_model.loss(pred_var, var)
                    noise_loss /= (var.shape[0] * num_pixels)
                    noise_losses.append(noise_loss.squeeze())
                if cfg.SYSTEM.EXP_NAME == 'full-update' or cfg.SYSTEM.EXP_NAME == 'full-update-pred-only':
                    logger.info('Training Noise Module')
                    noise_model.update(item_idxs, pred_var)


            optimizer.zero_grad()
            total_loss = pred_loss + (cfg.TRAIN.LAMBDA * noise_loss)
            losses.append(total_loss)
            total_loss.backward()
            optimizer.step()
            # writer.add_scalar('lr', optimizer.optimizer.param_groups[0]['lr'], writer_idx)

            if batch_idx % cfg.SYSTEM.LOG_FREQ == 0:
                logger.debug(f'epoch={epoch}, batch_id={batch_idx}, loss={total_loss}')
                writer.add_scalar('loss', pred_loss, writer_idx)
                visualize_results(pred_model, cfg, writer_idx, writer)

            cgt, wait = check_cgt(cfg, pred_model, val_loader, epoch, wait, val_losses)
            if cgt and train_noise == 0:
                train_noise = 1

            if cfg.SYSTEM.EXP_NAME in ['noise', 'real', 'overfit', 'overfit-noise']:
                train_noise = 0

        if epoch % cfg.SYSTEM.CHKPT_FREQ == 0:
            fn = f'checkpoint_epoch_{epoch+offset}'
            chkpt.save(fn)

        return np.mean(val_losses)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='path of config file', default=None, type=str)
    parser.add_argument('--clean_run', help='run from scratch', default=False, type=bool)
    parser.add_argument('opts', help='modify arguments', default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # config setup
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    if args.clean_run:
        if os.path.exists(f'../experiments/{cfg.SYSTEM.EXP_NAME}'):
            shutil.rmtree(f'../experiments/{cfg.SYSTEM.EXP_NAME}')
        if os.path.exists(f'../experiments/runs/{cfg.SYSTEM.EXP_NAME}'):
            shutil.rmtree(f'../experiments/runs/{cfg.SYSTEM.EXP_NAME}')
            time.sleep(30)
    device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(cfg.TRAIN.SEED)
    np.random.seed(cfg.TRAIN.SEED)
    logdir, chk_dir = save_config(cfg.TRAIN.SAVE_ROOT, cfg)
    writer = SummaryWriter(log_dir=logdir)

    # setup logger
    logger = setup_logger(cfg.SYSTEM.EXP_NAME)

    # load the data
    train_loader = get_loader(cfg, 'train')
    val_loader = get_loader(cfg, 'val')
    # Model
    prediction_model = BaseModule(cfg)
    noise_model = NoiseModule(cfg)
    prediction_model.to(device)
    # Optimizer
    optimizer = Adam(prediction_model.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS, eps=1e-08, weight_decay=0)
    # checkpointer
    chkpt = Checkpointer(prediction_model, optimizer, scheduler=None, save_dir=chk_dir, logger=logger, save_to_disk=True)
    model = [prediction_model, noise_model]
    fn = chkpt.get_checkpoint_file()
    offset = 0
    if not fn == '':
        start_idx = fn.rfind('_') + 1
        end_idx = fn.rfind('.')
        offset = int(fn[start_idx:end_idx])
    chkpt.load()
    loader = [train_loader, val_loader]
    train(cfg, model, optimizer, loader, chkpt, writer, offset)
