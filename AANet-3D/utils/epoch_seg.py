import sys
import torch
import torch.nn as nn

from tqdm import tqdm as tqdm
from utils.meter import AverageValueMeter
from utils.optim import *
import model_utils
import numpy as np
from scipy.spatial.distance import cdist
import time


class TrainEpoch:

    def __init__(self, model, loss, seg_metrics, optimizer, stage_name='train', device='cpu',
                 noise=None, grad_fn=None, verbose=True):
        self.model = model
        self.loss = loss

        self.seg_metrics = seg_metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.optimizer = optimizer
        self.noise = noise
        self.grad_fn = grad_fn
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        if self.noise is not None:
            self.noise.to(self.device)
        for seg_metrics in self.seg_metrics:
            seg_metrics.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, image, label):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        logs = {}
        loss_meters = {'seg_loss': AverageValueMeter()}

        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.seg_metrics}

        self.model.train()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose), ncols=200) as iterator:
            for sample in iterator:
                for key in sample:
                    sample[key] = sample[key].to(self.device, non_blocking=True).float()

                self.optimizer.zero_grad()

                result_dict = self.model.forward(sample['image'])

                loss = self.loss(result_dict, sample)
                loss.backward()
                if self.grad_fn is not None:
                    self.grad_fn(self.model.parameters())
                self.optimizer.step()

                loss_meters['seg_loss'].add(loss.cpu().detach().numpy())
                loss_logs = {k: v.mean for k, v in loss_meters.items()}
                logs.update(loss_logs)

                for metric_fn in self.seg_metrics:
                    metric_value = metric_fn(result_dict['segs'], sample['label']).item()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class ValidEpoch:

    def __init__(self, model, loss, seg_metrics, stage_name='valid', device='cpu', verbose=True):
        self.model = model
        self.loss = loss

        self.seg_metrics = seg_metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for seg_metrics in self.seg_metrics:
            seg_metrics.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, image, label):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        logs = {}
        loss_meters = {'seg_loss': AverageValueMeter()}

        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.seg_metrics}

        self.model.eval()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose), ncols=200) as iterator:
            for sample in iterator:
                for key in sample:
                    sample[key] = sample[key].float().to(self.device, non_blocking=True)

                with torch.no_grad():
                    result_dict = self.model.forward(sample['image'])

                loss = self.loss(result_dict, sample)

                loss_meters['seg_loss'].add(loss.cpu().detach().numpy())
                loss_logs = {k: v.mean for k, v in loss_meters.items()}
                logs.update(loss_logs)

                for metric_fn in self.seg_metrics:
                    metric_value = metric_fn(result_dict['segs'], sample['label']).item()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs
