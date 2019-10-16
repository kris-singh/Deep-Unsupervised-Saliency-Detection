#!/usr/bin/env python
import torch
import logging


def check_cgt(cfg, model, val_loader, epoch_idx, wait, val_losses):
    logger = logging.getLogger(cfg.SYSTEM.EXP_NAME)
    wait_period = cfg.SYSTEM.WAIT_TIME
    cgt = 0
    loss = validate(cfg, model, val_loader, epoch_idx, logger)
    val_losses.append(loss.detach().cpu())
    if len(val_losses) > 2 and val_losses[-1] >= val_losses[-2]:
        if wait > wait_period:
            cgt, wait = 1, 0
        else:
            wait = wait+1
            cgt, wait = cgt, wait
    else:
        cgt, wait = 0, wait
    return cgt, wait


def validate(cfg, model, val_loader, epoch, logger):
    model.eval()
    pred_loss = torch.nn.BCEWithLogitsLoss()
    loss = 0.0
    for idx, data in enumerate(val_loader):
        x = data['sal_image'].to(cfg.SYSTEM.DEVICE)
        y = data['sal_label'].to(cfg.SYSTEM.DEVICE)
        y_pred = model(x)['out']
        loss += pred_loss(y_pred, y)
        if idx % cfg.SYSTEM.LOG_FREQ == 0:
            logger.debug(f'Validation Loss: epoch={epoch}, batch_id={idx}, loss={loss}')
    return loss / cfg.VAL.BATCH_SIZE
