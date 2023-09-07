import sys
import torch
import torch.nn as nn

from tqdm import tqdm as tqdm
from utils.meter import AverageValueMeter
from utils.optim import *
import model_utils
import numpy as np
from scipy.spatial.distance import cdist
import torch.distributed as dist


class TrainEpoch:

    def __init__(self, model, loss, seg_metrics, optimizer, stage_name='train', device='cpu',
                 grad_fn=None, verbose=True, master=True):
        self.model = model
        self.loss = loss

        self.seg_metrics = seg_metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.master = master
        self.optimizer = optimizer
        self.grad_fn = grad_fn
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
        loss_meters = {'cls_loss': AverageValueMeter(), 'reg_loss': AverageValueMeter(),
                       'seg_loss': AverageValueMeter()}

        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.seg_metrics}

        self.model.train()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose), ncols=200) as iterator:
            for sample in iterator:
                image = sample['image']
                label = sample['label'].float()
                annot = sample['annot']

                image = image.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                annot = annot.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                result_dict = self.model.forward(image)
                cls_loss, reg_loss, seg_loss = self.loss(result_dict, annot, label)
                loss = cls_loss + reg_loss + seg_loss

                loss.backward()
                if self.grad_fn is not None:
                    self.grad_fn(self.model.parameters())
                self.optimizer.step()

                cls_loss = cls_loss.detach()
                reg_loss = reg_loss.detach()
                seg_loss = seg_loss.detach()

                segs = result_dict['segs'].detach()
                world_size = dist.get_world_size()

                label_gather = [torch.zeros_like(label) for _ in range(world_size)]
                dist.all_gather(label_gather, label)
                label_gather = torch.cat(label_gather, dim=0)

                segs_gather = [torch.zeros_like(segs) for _ in range(world_size)]
                dist.all_gather(segs_gather, segs)
                segs_gather = torch.cat(segs_gather, dim=0)

                cls_loss_gather = [torch.zeros_like(cls_loss) for _ in range(world_size)]
                dist.all_gather(cls_loss_gather, cls_loss)
                cls_loss_gather = torch.stack(cls_loss_gather).mean()

                reg_loss_gather = [torch.zeros_like(reg_loss) for _ in range(world_size)]
                dist.all_gather(reg_loss_gather, reg_loss)
                reg_loss_gather = torch.stack(reg_loss_gather).mean()

                seg_loss_gather = [torch.zeros_like(seg_loss) for _ in range(world_size)]
                dist.all_gather(seg_loss_gather, seg_loss)
                seg_loss_gather = torch.stack(seg_loss_gather).mean()
                if self.master:
                    loss_meters['cls_loss'].add(cls_loss_gather.cpu())
                    loss_meters['reg_loss'].add(reg_loss_gather.cpu())
                    loss_meters['seg_loss'].add(seg_loss_gather.cpu())
                    loss_logs = {k: v.mean for k, v in loss_meters.items()}
                    logs.update(loss_logs)

                    for metric_fn in self.seg_metrics:
                        metric_value = metric_fn(segs_gather, label_gather).item()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class ValidEpoch:

    def __init__(self, model, loss, seg_metrics, stage_name='valid', device='cpu',
                 verbose=True, master=True):
        self.model = model
        self.loss = loss
        self.seg_metrics = seg_metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.master = master
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
        loss_meters = {'cls_loss': AverageValueMeter(), 'reg_loss': AverageValueMeter(),
                       'seg_loss': AverageValueMeter()}

        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.seg_metrics}

        self.model.eval()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose), ncols=200) as iterator:
            for sample in iterator:
                image = sample['image']
                label = sample['label'].float()
                annot = sample['annot']

                image = image.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                annot = annot.to(self.device, non_blocking=True)

                with torch.no_grad():
                    result_dict = self.model.forward(image)

                cls_loss, reg_loss, seg_loss = self.loss(result_dict, annot, label)

                cls_loss = cls_loss.detach()
                reg_loss = reg_loss.detach()
                seg_loss = seg_loss.detach()

                segs = result_dict['segs'].detach()
                world_size = dist.get_world_size()

                label_gather = [torch.zeros_like(label) for _ in range(world_size)]
                dist.all_gather(label_gather, label)
                label_gather = torch.cat(label_gather, dim=0)

                segs_gather = [torch.zeros_like(segs) for _ in range(world_size)]
                dist.all_gather(segs_gather, segs)
                segs_gather = torch.cat(segs_gather, dim=0)

                cls_loss_gather = [torch.zeros_like(cls_loss) for _ in range(world_size)]
                dist.all_gather(cls_loss_gather, cls_loss)
                cls_loss_gather = torch.stack(cls_loss_gather).mean()

                reg_loss_gather = [torch.zeros_like(reg_loss) for _ in range(world_size)]
                dist.all_gather(reg_loss_gather, reg_loss)
                reg_loss_gather = torch.stack(reg_loss_gather).mean()

                seg_loss_gather = [torch.zeros_like(seg_loss) for _ in range(world_size)]
                dist.all_gather(seg_loss_gather, seg_loss)
                seg_loss_gather = torch.stack(seg_loss_gather).mean()
                if self.master:
                    loss_meters['cls_loss'].add(cls_loss_gather.cpu())
                    loss_meters['reg_loss'].add(reg_loss_gather.cpu())
                    loss_meters['seg_loss'].add(seg_loss_gather.cpu())
                    loss_logs = {k: v.mean for k, v in loss_meters.items()}
                    logs.update(loss_logs)

                    for metric_fn in self.seg_metrics:
                        metric_value = metric_fn(segs_gather, label_gather).item()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

                del result_dict

        return logs
