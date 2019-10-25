#!/usr/bin/env python

import numpy as np
from utils.meter import MetricLogger
from utils.basic import to_np, count_nonzero

meter = MetricLogger()


def mae(pred, gt):
    """
    pred: np.array (B, 1, H, W)
    gt: np.array(B, 1, H, W)
    """
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    mae = np.mean(np.abs(pred-gt))
    return float(mae)


def precision(pred, gt):
    """
    Defined as portion of correctly classified fg pixels.
    tp / tp + fp
    pred: np.array (B, 1, H, W)
    gt: np.array(B, 1, H, W)
    """
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    pred[pred<=0.5] = 0.0
    pred[pred>0.5] = 1.0
    diff = pred * gt
    prec = 0.0
    for im in range(pred.shape[0]):
        tp = count_nonzero(gt[im])
        prec += count_nonzero(diff[im]) / (count_nonzero(pred[im]) + 1e-6)

    return prec/pred.shape[0]


def recall(pred, gt):
    """
    Defined as the portion of
    tp / tp + fn
    pred: np.array size (B, 1, H, W)
    gt: np.array size (B, 1, H, W)
    """
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    pred[pred<=0.5] = 0.0
    pred[pred>0.5] = 1.0
    diff = pred * gt
    recall = 0.0
    for im in range(pred.shape[0]):
        recall += count_nonzero(diff[im]) / (count_nonzero(gt[im]) + 1e-6)

    return recall / pred.shape[0]


def f1(precision, recall, beta):
    """
    precision: float
    recall: float
    beta: float
    """
    return (1+beta**2) * (precision*recall) / ((beta**2 * precision) + recall)


def log_metrics(pred, gt):
    avg_meter = MetricLogger()
    pred = to_np(pred)
    gt = to_np(gt)
    prec = precision(pred, gt)
    rec = recall(pred, gt)
    avg_meter.update(precision=prec)
    avg_meter.update(recall=rec)
    avg_meter.update(mae=mae(pred, gt))
    return avg_meter


def main():
    pred = np.ones((5, 1, 28, 28))
    gt = np.ones((5, 1, 28, 28))
    print(precision(pred, gt))
    print(recall(pred, gt))
    print(mae(pred, gt))
    pred = np.zeros((5, 1, 28, 28))
    print(precision(pred, gt))
    print(recall(pred, gt))
    print(mae(pred, gt))
    pred = np.zeros((5, 1, 28, 28))
    pred[:,:, ::2, ::2] = 1
    print(precision(pred, gt))
    print(recall(pred, gt))
    print(mae(pred, gt))


if __name__ == "__main__":
        main()
