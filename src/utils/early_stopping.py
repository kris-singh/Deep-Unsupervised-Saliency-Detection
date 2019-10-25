#!/usr/bin/env python
import logging

import torch

from utils.meter import MetricLogger
from utils.metrics import log_metrics


class EarlyStopping:
    def __init__(self, model, val_loader, cfg):
        self.model = model
        self.val_loader = val_loader
        self.best_loss = None
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.SYSTEM.EXP_NAME)
        self.paitence = cfg.SYSTEM.WAIT_TIME
        self.wait = 0
        self.converged = False
        self.val_loss = MetricLogger()

    def reset(self):
        self.converged = False
        self.wait = 0

    def check_cgt(self):
        if self.best_loss is None:
            self.best_loss = self.val_loss.meters["val_loss"].avg
        elif self.best_loss <= self.val_loss.meters["val_loss"].avg:
            if self.wait > self.paitence:
                self.converged = True
                self.wait = 0
            else:
                self.wait += 1

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
            # import ipdb; ipdb.set_trace()
            if idx % self.cfg.SYSTEM.LOG_FREQ:
                # self.logger.debug(f'Validation Loss: {loss}, Avg: {self.val_loss.meters["val_loss"].avg}')
                if metrics is not None:
                    pass
                    # self.logger.info(str(metrics))

            metrics = log_metrics(y_pred, y)
            writer.add_scalar(str(metrics), epoch_idx + idx)
            writer.add_scalar('val_loss', loss, epoch_idx + idx)
        self.check_cgt()
