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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from checkpoint import Checkpointer
from config import cfg
from dataloader import get_loader
from model import BaseModule, NoiseModule
from utils.early_stopping import check_cgt
from utils.math import max2d, min2d
from utils.save import save_config
from utils.setup_logger import setup_logger
from utils.visualise import visualize_results
from utils.metrics import log_metrics

EPS = 1e-3


def train(cfg, model, optimizer, scheduler, loader, chkpt, writer, offset):
    h, w = cfg.TRAIN.IMG_SIZE[0], cfg.TRAIN.IMG_SIZE[1]
    num_pixels = h*w
    batch_size = cfg.TRAIN.BATCH_SIZE
    train_loader, val_loader = loader
    num_batches = len(train_loader)// batch_size
    device = cfg.SYSTEM.DEVICE
    pred_criterion = torch.nn.BCEWithLogitsLoss()
    train_noise = False
    pred_model, noise_model = model
    val_losses = []
    writer_idx = 0
    cgt, wait = 0, 0
    for epoch in range(cfg.TRAIN.EPOCHS):
        pred_losses = []
        noise_losses = []
        losses = []
        for batch_idx, data in enumerate(train_loader):
            pred_model.train()
            writer_idx = batch_idx * batch_size + (epoch * num_batches * batch_size)
            print(f"writer_idx: {writer_idx}, epoch:{epoch}, batch_idx:{batch_idx}")
            item_idxs = data['idx']
            x = data['sal_image'].to(device)
            y = data['sal_label'].to(device)
            if cfg.SYSTEM.EXP_NAME == 'noise':
                    y_noise = data['sal_noisy_label'].to(device)
                    x = torch.repeat_interleave(x, repeats=cfg.TRAIN.NUM_MAPS, dim=0)
                    y_noise = y_noise.view(-1).view(-1, 1, h, w)
            elif cfg.SYSTEM.EXP_NAME == 'avg':
                y_noise = data['sal_noisy_label'].to(device)
                y_noise = torch.mean(y_noise, dim=1, keepdim=True)
            elif cfg.SYSTEM.EXP_NAME == 'real':
                    y_noise = y
            else:
                y_noise = data['sal_noisy_label'].to(device)

            y_n_min, y_n_max = min2d(y_noise), max2d(y_noise)
            y_noise = (y_noise - y_n_min) / (y_n_max - y_n_min)

            pred = pred_model(x)['out']
            if not train_noise and cfg.SYSTEM.EXP_NAME == 'full':
                pred = torch.repeat_interleave(pred, repeats=cfg.TRAIN.NUM_MAPS, dim=1)

            if train_noise and cfg.SYSTEM.EXP_NAME == 'full':
                noise_prior = noise_model.sample_noise(item_idxs).to(device)
                noisy_pred = pred + noise_prior
                # truncate to lie in range[0, 1]
                noisy_min, noisy_max = min2d(noisy_pred), max2d(noisy_pred)
                noisy_pred = (pred - noisy_min) / (noisy_max - noisy_min)
                y_pred = noisy_pred
            else:
                y_pred = pred

            pred_loss = pred_criterion(y_pred, y_noise) / (cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.NUM_MAPS)
            pred_losses.append(pred_loss.squeeze())

            noise_loss = 0.0
            if train_noise:
                pred_var = torch.var(y_noise - pred, 1).view(cfg.TRAIN.BATCH_SIZE, -1)
                var, _ = noise_model.get_index_multiple(img_idxs=item_idxs)
                var = torch.from_numpy(var).float()
                noise_loss = noise_model.loss(pred_var, var)
                noise_loss /= (var.shape[0] * num_pixels)
                noise_losses.append(noise_loss.squeeze())

            optimizer.zero_grad()
            total_loss = pred_loss + (cfg.TRAIN.LAMBDA * noise_loss)
            losses.append(total_loss)
            total_loss.backward()
            optimizer.step()

            if batch_idx % cfg.SYSTEM.LOG_FREQ == 0:
                logger.debug(f'epoch={epoch}, batch_id={batch_idx}, loss={total_loss}')
                writer.add_scalar('loss', pred_loss, writer_idx)
                visualize_results(pred_model, cfg, writer_idx, writer)

            cgt, wait = check_cgt(cfg, pred_model, val_loader, epoch, wait, val_losses)
            if cfg.SYSTEM.EXP_NAME in ['real', 'avg', 'noise']:
                train_noise = 0
                if cgt:
                    return
            else:
                if cgt == 1 and train_noise == 0:
                    logger.info('Training Noise Module')
                    train_noise = 1
                    cgt, wait = 0, 0
                elif cgt==1 and train_noise == 1:
                    logger.info('Updating Noise Variance')
                    noise_model.update(item_idxs, pred_var)
                    cgt, wait = 0, 0
        scheduler.step(total_loss)
        writer.add_scalar('lr', scheduler.optimizer.param_groups[0]['lr'], writer_idx)
        metrics = log_metrics(pred, y)
        logger.info(str(metrics))
        writer.add_scalar(str(metrics), writer_idx)
        if epoch % cfg.SYSTEM.CHKPT_FREQ == 0:
            fn = f'checkpoint_epoch_{epoch+offset}'
            chkpt.save(fn)

def test(cfg, model, loader, writer):
    pred_model, _ = model
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_batches = len(loader) // batch_size
    batch_size = cfg.TRAIN.BATCH_SIZE
    for batch_idx, data in enumerate(loader):
            pred_model.eval()
            writer_idx = batch_idx * batch_size + num_batches * batch_size
            # import ipdb; ipdb.set_trace()
            x = data['image'].to(device)
            y = data['label'].to(device)
            pred = pred_model(x)['out']
            metrics = log_metrics(pred, y)
            logger.info(str(metrics))
            writer.add_scalar(str(metrics), writer_idx)


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
    test_loader = get_loader(cfg, 'test')

    # Model
    prediction_model = BaseModule(cfg)
    noise_model = NoiseModule(cfg)
    prediction_model.to(device)
    # Optimizer
    optimizer = Adam(prediction_model.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS, eps=1e-08, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-6, patience=50, factor=0.09)
    # checkpointer
    chkpt = Checkpointer(prediction_model, optimizer, scheduler=scheduler, save_dir=chk_dir, logger=logger, save_to_disk=True)
    model = [prediction_model, noise_model]
    fn = chkpt.get_checkpoint_file()
    offset = 0
    if not fn == '':
        start_idx = fn.rfind('_') + 1
        end_idx = fn.rfind('.')
        offset = int(fn[start_idx:end_idx])
    chkpt.load()
    loader = [train_loader, val_loader]
    train(cfg, model, optimizer, scheduler, loader, chkpt, writer, offset)
    test(cfg, model, test_loader, writer)
