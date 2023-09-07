import pandas as pd

import dataIO
import transform
from utils import epoch_seg, scheduler, optim, metrics_inference, model_io, metrics, losses
import model_utils
from models import AANet

from sync_batchnorm import convert_model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed
import numpy as np
import torchvision.models
from tensorboardX import SummaryWriter
from optparse import OptionParser
import os
import functools
import random
import warnings
from inference import test_model

torch.backends.cudnn.benchmark = True

GLOBAL_SEED = 0
warnings.filterwarnings('ignore')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    set_seed(seed + worker_id)


if 'lustre' in os.getcwd():
    print('On Lustre')
    crop_size = [96, 96, 96]
else:
    print('On Local')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    crop_size = [96, 96, 96]

DEVICE = 'cuda'

image_root = './PEData/processed_itk/image'
label_root = './PEData/processed_itk/label'
vessel_root = './PEData/processed_itk/vessel'

save_model_dir = './save_models'

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--epochs',
                      dest='epochs',
                      default=500,
                      type='int',
                      help='number of epochs')
    parser.add_option('-b',
                      '--batch_size',
                      dest='batch_size',
                      default=8,
                      type='int',
                      help='batch size, CT in one batch')
    parser.add_option('-s',
                      '--sample_num',
                      dest='sample_num',
                      default=2,
                      type='int',
                      help='sampled patch per CT. The actual image batch size is batch_size*sample_num')
    parser.add_option('-w',
                      '--num_workers',
                      dest='num_workers',
                      default=4,
                      type='int',
                      help='num_workers')
    parser.add_option('-l',
                      '--learn_rate',
                      dest='learn_rate',
                      default=1e-4,
                      type='float',
                      help='learn rate before warmup, will be *10 after warmup')
    parser.add_option('--hu_low',
                      dest='hu_low',
                      default=-100,
                      type='float',
                      help='hu_low')
    parser.add_option('--hu_high',
                      dest='hu_high',
                      default=500,
                      type='float',
                      help='hu_high')
    parser.add_option('-o',
                      '--log-path',
                      type='str',
                      dest='log_path',
                      default='./log/',
                      help='log path')
    parser.add_option('-u',
                      '--unique_name',
                      type='str',
                      dest='unique_name',
                      default='det2',
                      help='name of this experiment')
    parser.add_option('--seed',
                      type='int',
                      dest='seed',
                      default='1',
                      help='global random seed')
    parser.add_option('--valid',
                      type='int',
                      dest='valid',
                      default='1',
                      help='if valid')

    (options, args) = parser.parse_args()
    set_seed(options.seed)
    unique_name = options.unique_name
    model = AANet.Net(n_filters=[64, 96, 128, 160], en_blocks=[2, 3, 3, 3], de_blocks=[1, 2, 2],
                      stem_filters=16, aspp_filters=32)

    pretrain = True
    if pretrain == True:
        state_dict = torch.load('./save_models/luna16_pretrain.pth')
        model.load_state_dict(state_dict, strict=False)
        model.init_head()

    model = nn.parallel.DataParallel(model)
    model.to(DEVICE)

    "Post Transform"
    transform_list_train = [
        transform.Pad(output_size=crop_size),
        transform.RandomCrop(output_size=crop_size)]
    transform_list_valid = []

    train_transform = torchvision.transforms.Compose(transform_list_train)
    valid_transform = torchvision.transforms.Compose(transform_list_valid)

    "Rib Crop Tool"
    crop_fn_train = dataIO.crop_itk.SegCrop2(crop_size=crop_size, tp_ratio=0.75, spacing=[1., 1., 1.],
                                             rand_translation=[20, 20, 20], rand_rotation=[30, 30, 30],
                                             rand_space=[0.8, 1.2], rand_rotation90=[True, True, True],
                                             rand_flip=[True, True, True], rand_transpose=[False, False, False],
                                             hu_low=options.hu_low, hu_high=options.hu_high,
                                             sample_num=options.sample_num, obj_crop=True, blank_side=0)
    crop_fn_valid = dataIO.crop_itk.SegCrop2(crop_size=crop_size, tp_ratio=0.75, spacing=[1., 1., 1.],
                                             hu_low=options.hu_low, hu_high=options.hu_high,
                                             sample_num=options.sample_num, obj_crop=False, blank_side=0)

    print('Dataset Prepare!')
    train_split = np.array(pd.read_csv('./PEData/train_split_cad.csv'))[:, 0].tolist()
    val_split = np.array(pd.read_csv('./PEData/test_split_cad.csv'))[:, 0].tolist()

    train_dataset = dataIO.dataset.DatasetITKV(image_root, label_root, vessel_root, train_split,
                                               crop_fn=crop_fn_train, transform_post=train_transform)
    valid_dataset = dataIO.dataset.DatasetITKV(image_root, label_root, vessel_root, val_split,
                                               crop_fn=crop_fn_valid, transform_post=valid_transform)

    print('Data Loader Prepare!')
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True,
                              collate_fn=dataIO.dataset.collate_fn_dicts, worker_init_fn=worker_init_fn,
                              num_workers=options.num_workers, pin_memory=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=options.batch_size, shuffle=False,
                              collate_fn=dataIO.dataset.collate_fn_dicts, worker_init_fn=worker_init_fn,
                              num_workers=options.num_workers, pin_memory=False)

    print('Loss, Metrics and Optimizer Prepare!')
    seg_metrics = [metrics.Fscore(threshold=0.4)]

    loss = 0.5 * losses.TverskyLoss(square=True, batch_dice=True, alpha=0.5, beta=0.5) + \
           0.5 * losses.TverskyLoss(square=True, batch_dice=False, alpha=0.5, beta=0.5, drop_bg=True)

    seg_loss = model_utils.losses.SegLoss(['segs', 'segvs'], ['label', 'mask'], [1, 1], [[0], [0]],
                                          [loss, loss], [0.75, 0.25])

    optimizer = optim.AdamW(params=model.parameters(), lr=options.learn_rate, weight_decay=1e-4)
    scheduler_reduce = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options.epochs, eta_min=1e-5)
    scheduler_warm = scheduler.GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=4,
                                                      after_scheduler=scheduler_reduce)
    grad_fn = functools.partial(optim.clip_grad_norm_, max_norm=100)

    print('Epoch Prepare!')

    train_epoch = epoch_seg.TrainEpoch(
        model=model,
        loss=seg_loss,
        seg_metrics=seg_metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
        grad_fn=grad_fn
    )

    valid_epoch = epoch_seg.ValidEpoch(
        model,
        loss=seg_loss,
        seg_metrics=seg_metrics,
        device=DEVICE,
        verbose=True,
    )

    print('Start Train!')
    set_seed(options.seed)
    writer = SummaryWriter(options.log_path + unique_name)
    for i in range(0, options.epochs):
        print('\nEpoch: {}'.format(i))
        print('current lr:', optimizer.param_groups[0]['lr'])
        train_logs = train_epoch.run(train_loader)

        model_io.Save(model.state_dict(), os.path.join(save_model_dir, unique_name + '_ms.pth'))

        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], i)
        for item in train_logs:
            writer.add_scalar('Train/' + item, train_logs[item], i)

        if options.valid:
            valid_logs = valid_epoch.run(valid_loader)
            for item in valid_logs:
                writer.add_scalar('Valid/' + item, valid_logs[item], i)

        scheduler_warm.step()

    "Test!"
    model.eval()
    test_model(model,
               '../PEData/processed_itk/test_split_cad.csv',
               unique_name + '_sth50',
               seg_thresh=0.5,
               hu_low=options.hu_low, hu_high=options.hu_high)
