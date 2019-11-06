#!/usr/bin/env python
import logging

import torch

from utils.meter import MetricLogger
from utils.metrics import log_metrics


class EarlyStopping:
    def __init__(self, model, val_loader, cfg):
        self.model = model
        self.val_loader = val_loader
        self.best_f1 = None
        self.cfg = cfg
        self.logger = logging.getLogger(str(cfg.SYSTEM.EXP_NAME) + '.utils.early_stopping')
        self.paitence = cfg.SYSTEM.WAIT_TIME
        self.wait = 0
        self.f1 = None
        self.converged = False
        self.val_loss = MetricLogger()

    def reset(self):
        self.converged = False
        self.wait = 0

    def check_cgt(self):
        if self.best_f1 is None:
            self.best_f1 = self.f1
        elif self.best_f1 < self.f1:
            if self.wait > self.paitence:
                self.converged = True
                self.wait = 0
            else:
                self.wait += 1
        else:
            self.wait = 0
            self.best_f1 = self.f1

    def validate(self, writer, epoch_idx):
        self.model.eval()
        pred_loss = torch.nn.BCEWithLogitsLoss()
        metrics = None
        for idx, data in enumerate(self.val_loader):
            x = data['sal_image'].to(self.cfg.SYSTEM.DEVICE)
            y = data['sal_label'].to(self.cfg.SYSTEM.DEVICE)
            y_pred = self.model(x)['out']
            loss = pred_loss(y_pred, y) / self.cfg.SOLVER.BATCH_SIZE
            self.val_loss.update(**{'val_loss': loss})
            if idx % self.cfg.SYSTEM.LOG_FREQ:
                self.logger.debug(f'Validation Loss: {loss}, Avg: {self.val_loss.meters["val_loss"].avg}')

            metrics = log_metrics(y_pred, y)
        self.f1 = 1. / (metrics.meters['precision'].avg+1e-16) + 1. / (metrics.meters['recall'].avg+1e-16)
        writer.add_scalars('val_metrics', {'val_precision': metrics.meters['precision'].avg,
                                            'val_recall': metrics.meters['recall'].avg,
                                            'val_f1': self.f1,
                                            'val_mae': metrics.meters['mae'].avg}, epoch_idx)
        writer.add_scalar('val_loss', loss, epoch_idx)
        self.check_cgt()
