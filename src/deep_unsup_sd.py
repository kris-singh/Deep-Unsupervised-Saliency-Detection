#!/usr/bin/env python

import argparse
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
from torch import optim as optim
from torch.utils.tensorboard import SummaryWriter
import optuna
from functools import partial

from checkpoint import Checkpointer
from config import cfg
from dataloader import get_loader
from model import BaseModule, NoiseModule
from utils.early_stopping import EarlyStopping
from utils.math_utils import max2d, min2d
from utils.metrics import log_metrics
from utils.save import save_config
from utils.setup_logger import setup_logger
from utils.visualise import visualize_results

EPS = 1e-3


def train(cfg, model, optimizer, scheduler, loader, chkpt, logger, writer, device, offset):
    h, w = cfg.SOLVER.IMG_SIZE[0], cfg.SOLVER.IMG_SIZE[1]
    num_pixels = h*w
    batch_size = cfg.SOLVER.BATCH_SIZE
    train_loader, val_loader = loader
    num_batches = len(train_loader) // batch_size
    pred_criterion = torch.nn.BCEWithLogitsLoss()
    train_noise = False
    pred_model, noise_model = model
    early_stopping = EarlyStopping(pred_model, val_loader, cfg)
    writer_idx = 0
    for epoch in range(cfg.SOLVER.EPOCHS):
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
                    x = torch.repeat_interleave(x, repeats=cfg.SOLVER.NUM_MAPS, dim=0)
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
                pred = torch.repeat_interleave(pred, repeats=cfg.SOLVER.NUM_MAPS, dim=1)

            if train_noise and cfg.SYSTEM.EXP_NAME == 'full':
                noise_prior = noise_model.sample_noise(item_idxs).to(device)
                noisy_pred = pred + noise_prior
                # truncate to lie in range[0, 1]
                noisy_min, noisy_max = min2d(noisy_pred), max2d(noisy_pred)
                noisy_pred = (pred - noisy_min) / (noisy_max - noisy_min)
                y_pred = noisy_pred
            else:
                y_pred = pred

            pred_loss = pred_criterion(y_pred, y_noise) / (cfg.SOLVER.BATCH_SIZE * cfg.SOLVER.NUM_MAPS)
            pred_losses.append(pred_loss.squeeze())

            noise_loss = 0.0
            if train_noise:
                pred_var = torch.var(y_noise - pred, 1).view(cfg.SOLVER.BATCH_SIZE, -1)
                var, _ = noise_model.get_index_multiple(img_idxs=item_idxs)
                var = torch.from_numpy(var).float()
                noise_loss = noise_model.loss(pred_var, var)
                noise_loss /= (var.shape[0] * num_pixels)
                noise_losses.append(noise_loss.squeeze())

            optimizer.zero_grad()
            total_loss = pred_loss + (cfg.SOLVER.LAMBDA * noise_loss)
            losses.append(total_loss)
            total_loss.backward()
            optimizer.step()

            if batch_idx % cfg.SYSTEM.LOG_FREQ == 0:
                logger.debug(f'epoch={epoch}, batch_id={batch_idx}, loss={total_loss}')
                writer.add_scalar('loss', pred_loss, writer_idx)
                visualize_results(pred_model, cfg, writer_idx, writer)
                if y_pred.shape == y.shape: # Hack since we can only compare precision this way
                    metrics = log_metrics(y_pred, y)
                    logger.info(str(metrics))
                    writer.add_scalar(str(metrics), writer_idx)

            early_stopping.validate(writer, writer_idx)

            if cfg.SYSTEM.EXP_NAME in ['real', 'avg', 'noise']:
                train_noise = 0
                if early_stopping.converged:
                    return
            else:
                if early_stopping.converged == 1 and train_noise == 0:
                    logger.info('Training Noise Module')
                    train_noise = 1
                    early_stopping.reset()
                elif early_stopping.converged == 1 and train_noise == 1:
                    logger.info('Updating Noise Variance')
                    noise_model.update(item_idxs, pred_var)
                    early_stopping.reset()
        scheduler.step(total_loss)
        writer.add_scalar('lr', scheduler.optimizer.param_groups[0]['lr'], writer_idx)
        if epoch % cfg.SYSTEM.CHKPT_FREQ == 0:
            fn = f'checkpoint_epoch_{cfg.SYSTEM.EXP_NAME}_{epoch+offset}'
            chkpt.save(fn, epoch=epoch)
    return early_stopping.val_loss.meters['val_loss'].avg


def test(cfg, model, loader, writer, logger):
    device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu'
    batch_size = cfg.SOLVER.BATCH_SIZE
    num_batches = len(loader) // batch_size
    for batch_idx, data in enumerate(loader):
            model.eval()
            writer_idx = batch_idx * batch_size + num_batches * batch_size
            # import ipdb; ipdb.set_trace()
            x = data['image'].to(device)
            y = data['label'].to(device)
            pred = model(x)['out']
            metrics = log_metrics(pred, y)
            logger.info(str(metrics))
            writer.add_scalar(str(metrics), writer_idx)


def objective(trial, cfg, model, search):
    device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu'
    # load the data
    train_loader = get_loader(cfg, 'train')
    val_loader = get_loader(cfg, 'val')
    prediction_model, noise_model = model
    prediction_model.to(device)
    lr = cfg.SOLVER.TRIAL if not search['lr'] else trial.suggest_loguniform('lr', low=1e-6, high=1e-1)
    momentum = cfg.SOLVER.MOMENTUM if not search['momentum'] else trial.suggest_uniform('momentum', low=0.5, high=0.9)
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    betas = cfg.SOLVER.BETAS
    step_size = cfg.SOLVER.STEP_SIZE if search['step_size'] else trial.suggest_discrete_uniform('step_size', low=10, high=100, q=10)
    factor = cfg.SOLVER.FACTOR if search['factor'] else trial.suggest_uniform('factor', low=0.1, high=0.4)

    # Optimizer
    if cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(prediction_model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(prediction_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    if cfg.SOLVER.SCHEDULER == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=factor)
    elif cfg.SOLVER.SCHEDULER == 'ReducePlateuLR':
        scheduler = optim.lr_scheduler.ReducePlateuLR(optimizer,
                                                      step_size=step_size,
                                                      min_lr=cfg.SOLVER.MIN_LR,
                                                      paitence=cfg.SOLVER.PAITENCE)
    # checkpointer
    chkpt = Checkpointer(prediction_model, optimizer, scheduler=scheduler, save_dir=chk_dir, logger=logger,
                         save_to_disk=True)
    offset = 0
    checkpointer = chkpt.load()
    if not checkpointer == {}:
        offset = checkpointer.pop('epoch')
    loader = [train_loader, val_loader]
    print('Same optimizer, {scheduler.optimizer == optimizer}')
    return train(cfg, model, optimizer, scheduler, loader, chkpt, logger, writer, device, offset)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='path of config file', default=None, type=str)
    parser.add_argument('--clean_run', help='run from scratch', default=False, type=bool)
    parser.add_argument('opts', help='modify arguments', default=None, nargs=argparse.REMAINDER)
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

    search = defaultdict()
    search['lr'], search['momentum'], search['factor'], search['step_size'] = [True]*4
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(cfg.SYSTEM.SEED)
    np.random.seed(cfg.SYSTEM.SEED)
    logdir, chk_dir = save_config(cfg.SAVE_ROOT, cfg)
    writer = SummaryWriter(log_dir=logdir)
    # setup logger
    logger = setup_logger(cfg.SYSTEM.EXP_NAME)
    # Model
    prediction_model = BaseModule(cfg)
    noise_model = NoiseModule(cfg)
    model = [prediction_model, noise_model]
    study = optuna.create_study(study_name=f"exp_name:{cfg.SYSTEM.EXP_NAME}", storage="sqlite:///example.db", load_if_exists=True)
    partial_objective = partial(objective, cfg=cfg, model=model, search=search)
    study.optimize(partial_objective, n_trials=100)
    test_loader = get_loader(cfg, 'test')
    test(cfg, prediction_model, test_loader, writer, logger)
