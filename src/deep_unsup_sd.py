#!/usr/bin/env python

import argparse
import logging
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import optim as optim
from torch.utils.tensorboard import SummaryWriter

from checkpoint import Checkpointer
from config import cfg
from dataloader import get_loader
from model import BaseModule, NoiseModule
from utils.basic import set_seeds, custom_lr
from utils.early_stopping import EarlyStopping
from utils.math_utils import max2d, min2d
from utils.metrics import log_metrics
from utils.save import save_config
from utils.setup_logger import setup_logger
from utils.visualise import visualize_results


def train(cfg, model, optimizer, scheduler, loader, chkpt, writer, offset):
    """
    Method for training deep unsupervised sailency detection model.
    Training module implements 4 training methodology
    1. Real: The backbone network is trainined with Ground Truth object, loss = mean((pred_y_i, y_gt_i))
    3. Noise: The backbone network is trained using losses on all the labels of unsupersived methods, loss = mean((pred_y_i, unup_y_i_m))
    3. Avg: The backbone network is trained  using avg of the all the unsupervised methods as ground truth, loss = mean((pred_y_i, mean(unup_y_i_m)))
    4. Full: The backbone network as well as the noise module is trained, the training proceeds as follows
    Training processeds in rounds,
       Round 1:
         Initalise the variane of the prior noise = 0.0
         Train the backbone network on all the unsuperversived methods loss = bcewithlogitloss(pred_y, unsup_y_i)
         Once the network is trained till converge
         Update the noise network using the update rule in Eq 7 in the paper
    Round i:
        Sample noise from the noise network
        Training the backbone network on all the unsupervised methods loss = bcewithlogitloss(pred_y + noise, unsup_y_i) + noise loss computed using Eq 6
        Train the backbone network till convergence
        Update the noise network using the update rule in Eq 7
    Args
    ---
    cfg: (CfgNode) Master configuration file
    model: (tuple) Consits of 2 model the backbone network and the noise module.
    optimizer: torch.optim Optimizer to train the network
    scheduler: torch.optim.lr_scheduler Scheduler for fixing learning rate
    loader: (tuple) traindataset loader and validation dataset loader
    chkpt: chekpointer
    writer: tensorboard writer
    offset: start point to save the model correctly
    """
    device = cfg.SYSTEM.DEVICE
    logger = logging.getLogger(cfg.SYSTEM.EXP_NAME)
    h, w = cfg.SOLVER.IMG_SIZE[0], cfg.SOLVER.IMG_SIZE[1]
    batch_size = cfg.SOLVER.BATCH_SIZE
    train_loader, val_loader = loader
    num_batches = len(train_loader.dataset) // batch_size
    pred_criterion = torch.nn.BCEWithLogitsLoss()
    train_noise = False
    pred_model, noise_model = model
    early_stopping = EarlyStopping(pred_model, val_loader, cfg)
    writer_idx = 0

    for epoch in range(cfg.SOLVER.EPOCHS):
        pred_model.train()
        for batch_idx, data in enumerate(train_loader):
            writer_idx = batch_idx * batch_size + (epoch * num_batches * batch_size)
            item_idxs = data['idx']
            # x: Input data, (None, 3, 128, 128)
            x = data['sal_image'].to(device)
            # y: GT label (None, 1, 128, 128), required for metrics
            y = data['sal_label'].to(device)
            orig_x = x  # used for computing metrics upon true data
            if cfg.SYSTEM.EXP_TYPE == 'noise':
                y_noise = data['sal_noisy_label'].to(device)
                # x: repeat input for each map, (None*NUM_MAPS, 3, 128, 128) !Note: Take care of batch size
                x = torch.repeat_interleave(x, repeats=cfg.SOLVER.NUM_MAPS, dim=0)
                # y_noise: Unsupervised labels, (None, 1, 128, 128)
                y_noise = y_noise.view(-1).view(-1, 1, h, w)
            elif cfg.SYSTEM.EXP_TYPE == 'avg':
                y_noise = data['sal_noisy_label'].to(device)
                # y_noise: Taking mean of all the noise labels, (None, 1, 128, 128)
                y_noise = torch.mean(y_noise, dim=1, keepdim=True)
            elif cfg.SYSTEM.EXP_TYPE == 'real':
                # y: GT label, (None, 1, 128, 128)
                y_noise = y
            else:
                y_noise = data['sal_noisy_label'].to(device)

            # Normalizing after addition of noise so that y_noise is between (0, +1)
            y_n_min, y_n_max = min2d(y_noise), max2d(y_noise)
            y_noise = (y_noise - y_n_min) / (y_n_max - y_n_min)
            # pred: (None, 1, 128, 128)
            pred = pred_model(x)['out']
            y_pred = pred

            if cfg.SYSTEM.EXP_TYPE == 'full':
                # Round 1
                if not train_noise:
                    # pred: In round 1 repeat along dim=1, (None, NUM_MAPS, 128, 128)
                    pred = torch.repeat_interleave(pred, repeats=cfg.SOLVER.NUM_MAPS, dim=1)
                    y_pred = pred
                # Round > 1
                else:
                    # noise_prior: Sampled Noise from Noise Module, (None, NUM_MAPS, 128, 128)
                    noise_prior = noise_model.sample_noise(item_idxs).to(device)
                    # noisy_pred: Noisy predictions after adding noise to predictions, (None, NUM_MAPS, 128, 128)
                    noisy_pred = pred + noise_prior
                    # truncate to lie in range[0, 1] see 3.2 after Eq 4
                    noisy_min, noisy_max = min2d(noisy_pred), max2d(noisy_pred)
                    noisy_pred = (noisy_pred - noisy_min) / (noisy_max - noisy_min)
                    y_pred = noisy_pred

            # Computing BCE Loss between pred and target, See Eq 4
            pred_loss = pred_criterion(y_pred, y_noise)

            noise_loss = 0.0
            if train_noise:
                # compute noise loss for batch using Eq 6
                emp_var = torch.var(y_noise - pred, 1).view(cfg.SOLVER.BATCH_SIZE, -1) + 1e-16
                prior_var, var_idx = noise_model.get_index_multiple(img_idxs=item_idxs)
                prior_var = torch.from_numpy(prior_var).float()
                # Important Order for loss var needs to get close to emp_var
                noise_loss = noise_model.loss_fast(prior_var, emp_var)

            optimizer.zero_grad()
            # total loss computed using Eq 2, this loss is used only for training the backbone network parameters
            total_loss = pred_loss + cfg.SOLVER.LAMBDA * noise_loss
            total_loss.backward()
            optimizer.step()

            if batch_idx % cfg.SYSTEM.LOG_FREQ == 0:
                logger.info(f'epoch:{epoch}, batch_idx:{batch_idx}, Noise Loss:{noise_loss}, Pred Loss: {pred_loss}')
                # logger.debug(f'epoch={epoch}, batch_id={batch_idx}, loss={total_loss}')
                writer.add_scalar('Loss', pred_loss, writer_idx)
                if cfg.SYSTEM.EXP_TYPE == 'full':
                    writer.add_scalars('Loss', {'pred_loss': pred_loss,
                                                'noise_loss': noise_loss,
                                                'total_loss': total_loss}, writer_idx)
                visualize_results(pred_model, cfg, writer_idx, writer)
                # compute metrics
                y_pred_t = pred_model(orig_x)['out']
                metrics = log_metrics(y_pred_t, y)
                writer.add_scalars("metrics", {"precision": metrics.meters['precision'].avg,
                                               "recall": metrics.meters['recall'].avg,
                                               "mae": metrics.meters['mae'].avg}, writer_idx)

        early_stopping.validate(writer, writer_idx)

        if cfg.SYSTEM.EXP_TYPE in ['real', 'avg', 'noise']:
            train_noise = 0
            if early_stopping.converged:
                logger.info("Converged")
                return
        else:
            if early_stopping.converged == 1 and train_noise == 0:
                # Update the noise variance using Eq 7. !Note: Importantly we do this for all images encountered, using the pred_variance
                update_full(cfg, model, train_loader)
                logger.info('Updating Noise Variance')
                train_noise = 1
                # Reset Early Stopping variables, training till next convergence
                early_stopping.reset()

        scheduler.step(early_stopping.val_loss.meters['val_loss'].avg)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], writer_idx)
        print(f"lr: {optimizer.param_groups[0]['lr']}")
        if epoch % cfg.SYSTEM.CHKPT_FREQ == 0:
            fn = f'checkpoint_epoch_{cfg.SYSTEM.EXP_NAME}_{epoch+offset}'
            chkpt.save(fn, epoch=epoch)


def update_full(cfg, model, train_loader):
    """
    This functions is used to model Eq 7 from the paper ie updating the noise.
    Args
    ---
    cfg: (CfgNode) Master configuration file
    model: (tuple) Consits of 2 model the backbone network and the noise module.
    train_loader: torch.utils.data.dataloader for training data
    """
    device = cfg.SYSTEM.DEVICE
    pred_model, noise_model = model
    for batch_idx, data in enumerate(train_loader):
        # x : input data, (None, 3, 128, 128)
        x = data['sal_image'].to(device)
        item_idxs = data['idx']
        # y_noise: Unsup labels, (None, NUM_MAPS, 128, 128)
        y_noise = data['sal_noisy_label'].to(device)
        # normalize noise value, !Note: If not done the convergence is poor in practice
        y_n_min, y_n_max = min2d(y_noise), max2d(y_noise)
        y_noise = (y_noise - y_n_min) / (y_n_max - y_n_min)
        # pred: (None, 1, 128, 128)
        pred = pred_model(x)['out']
        # emp_var: Emperical Variance for each pixel for each image, (None, 1, 128, 128)
        emp_var = torch.var(y_noise - pred, 1).view(cfg.SOLVER.BATCH_SIZE, -1).detach().cpu()
        _, var_idx = noise_model.get_index_multiple(img_idxs=item_idxs)
        noise_model.emp_var[var_idx.reshape(-1)] = emp_var.view(-1).to(noise_model.emp_var.device)
    # Note! We compute emperical variance for each image and update prior variance
    noise_model.update()


def test(cfg, model, loader, writer, logger):
    device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu'
    batch_size = cfg.SOLVER.BATCH_SIZE
    num_batches = len(loader) // batch_size
    batch_idx = 0
    writer_idx = batch_idx * batch_size + num_batches * batch_size
    for batch_idx, data in enumerate(loader):
            model.eval()
            writer_idx = batch_idx * batch_size + num_batches * batch_size
            # import ipdb; ipdb.set_trace()
            x = data['image'].to(device)
            y = data['label'].to(device)
            pred = model(x)['out']
            metrics = log_metrics(pred, y)
            logger.info(f"test_precision: {metrics.meters['precision'].avg},\
                        test_recall: {metrics.meters['recall'].avg},\
                        test_mae: {metrics.meters['mae'].avg}")

    writer.add_scalars("metrics", {"test_precision": metrics.meters['precision'].avg,
                                   "test_recall": metrics.meters['recall'].avg,
                                   "test_mae": metrics.meters['mae'].avg}, writer_idx)


def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='path of config file', default=None, type=str)
    parser.add_argument('--clean_run', help='run from scratch', default=False, type=bool)
    parser.add_argument('opts', help='modify arguments', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # config setup
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None: cfg.merge_from_list(args.opts)

    cfg.freeze()
    if args.clean_run:
        if os.path.exists(f'../experiments/{cfg.SYSTEM.EXP_NAME}'):
            shutil.rmtree(f'../experiments/{cfg.SYSTEM.EXP_NAME}')
        if os.path.exists(f'../experiments/runs/{cfg.SYSTEM.EXP_NAME}'):
            shutil.rmtree(f'../experiments/runs/{cfg.SYSTEM.EXP_NAME}')
            # Note!: Sleeping to make tensorboard delete it's cache.
            time.sleep(5)

    search = defaultdict()
    search['lr'], search['momentum'], search['factor'], search['step_size'] = [True]*4
    set_seeds(cfg)
    logdir, chk_dir = save_config(cfg.SAVE_ROOT, cfg)
    writer = SummaryWriter(log_dir=logdir)
    # setup logger
    logger_dir = Path(chk_dir).parent
    logger = setup_logger(cfg.SYSTEM.EXP_NAME, save_dir=logger_dir)
    # Model
    prediction_model = BaseModule(cfg)
    noise_model = NoiseModule(cfg)
    model = [prediction_model, noise_model]
    device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu'
    # load the data
    train_loader = get_loader(cfg, 'train')
    val_loader = get_loader(cfg, 'val')
    prediction_model, noise_model = model
    prediction_model.to(device)
    lr = cfg.SOLVER.LR
    momentum = cfg.SOLVER.MOMENTUM
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    betas = cfg.SOLVER.BETAS
    step_size = cfg.SOLVER.STEP_SIZE
    decay_factor = cfg.SOLVER.FACTOR

    # Optimizer
    if cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(prediction_model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(prediction_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    if cfg.SOLVER.SCHEDULER == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_factor)
    elif cfg.SOLVER.SCHEDULER == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=cfg.SOLVER.FACTOR,
                                                         min_lr=cfg.SOLVER.MIN_LR,
                                                         patience=cfg.SOLVER.PAITENCE,
                                                         cooldown = cfg.SOLVER.COOLDOWN,
                                                         threshold = cfg.SOLVER.THRESHOLD,
                                                         eps = 1e-24)
    # checkpointer
    chkpt = Checkpointer(prediction_model, optimizer, scheduler=scheduler, save_dir=chk_dir, logger=logger,
                         save_to_disk=True)
    offset = 0
    checkpointer = chkpt.load()
    if not checkpointer == {}:
        offset = checkpointer.pop('epoch')
    loader = [train_loader, val_loader]
    print(f'Same optimizer, {scheduler.optimizer == optimizer}')
    print(cfg)
    model = [prediction_model, noise_model]
    train(cfg, model, optimizer, scheduler, loader, chkpt, writer, offset)
    test_loader = get_loader(cfg, 'test')
    test(cfg, prediction_model, test_loader, writer, logger)

if __name__ == "__main__":
    main()
