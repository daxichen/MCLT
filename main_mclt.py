import argparse
import builtins
import math
import os
import sys
import random
import shutil
import time
import warnings
import numpy as np
import scipy.io as sio
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
import time

import moco.builder
import moco.loader
import moco.optimizer

from vit_pytorch import VisionTransformer as vits
from utils import loadData, save_model
from PIL import Image
import torch.optim as optim
from thop import profile


def parse_option():
    parser = argparse.ArgumentParser(description='MCLT')
    parser.add_argument('--dataset', type=str, default='Sandiego2',
                        choices=['Sandiego2', 'Sandiego100', 'Beach4','MUUFL'], help='dataset')

    # epoch Sandiego2=Sandiego100=Beach4=200, MUUFL=100
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run ')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=480, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when Sandiego2=480; Sandiego100=400; panelHIMp1=512, Urban1=400, Airport=400, Beach4=600, MUUFL=1300'
                         'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')   # Sandiego2=Sandiego100=Beach4=0.05, MUUFL=0.05
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
    parser.add_argument('--moco-mlp-dim', default=256, type=int,
                    help='hidden dimension in MLPs (default: 256)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
    parser.add_argument('--moco-k', default=14400, type=int,
                    help='queue size; number of negative keys (default: Sandiego2=14400; Sandiego100=10000, panelHIMp1=4096, Urban1=10000, Airport=10000, Beach4=12000, MUUFL=71500/13000)')
    parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
    parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 1.0)')

    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs用于记录多次运行的 id')

    args = parser.parse_args()

    args.model_path = './save/MCLT/{}_models'.format(args.dataset)
    args.model_name = '{}_lr_{}_bsz_{}_temp_{}_trial_{}'.\
           format(args.dataset, args.learning_rate,
               args.batch_size, args.moco_t, args.trial)
    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    return args

def set_model(args):
    # Sandiego2
    v = vits(
        img_size=189,
        patch_size=9,
        embed_dim=128,
        depth=2,
        num_heads=8,
        representation_size=None,
        num_classes=128
    )

    # Sandiego100
    # v = vits(
    #     img_size=189,
    #     patch_size=9,
    #     embed_dim=128,
    #     depth=2,
    #     num_heads=8,
    #     representation_size=None,
    #     num_classes=128
    # )


    # MUUFL
    # v = vits(
    #     img_size=64,
    #     patch_size=4,
    #     embed_dim=128,
    #     depth=2,
    #     num_heads=8,
    #     representation_size=None,
    #     num_classes=128
    # )

    # Beach4
    # v = vits(
    #     img_size=102,
    #     patch_size=6,
    #     embed_dim=128,
    #     depth=2,
    #     num_heads=8,
    #     representation_size=None,
    #     num_classes=128
    # )

    model = moco.builder.MoCo_ViT(
            v,
            dim=args.moco_dim, mlp_dim=args.moco_mlp_dim, T=args.moco_t, K=args.moco_k, m=args.moco_m)

    
    criterion = nn.CrossEntropyLoss()


    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, args):


    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, learning_rates, losses],
    #     prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    losses = AverageMeter()
    bsz = args.batch_size


    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (images1,images2) in enumerate(train_loader):
        images1 = images1.to('cuda')
        images2 = images2.to('cuda')
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        output, target = model(im_q=images1, im_k=images2)

        loss = criterion(output, target)

        losses.update(loss.item(), bsz)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if (i + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), loss=losses))
            sys.stdout.flush()

    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m

def set_optimizer(args, model):
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=args.learning_rate,
    #                       momentum=args.momentum,
    #                       weight_decay=args.weight_decay)

    # optimizer = optim.Adam(model.parameters(),
    #                       lr=args.learning_rate,
    #                       weight_decay=args.weight_decay)


    optimizer = moco.optimizer.LARS(model.parameters(), args.learning_rate,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    return optimizer

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, Datapath1, Datapath2):
        self.DataList1 = Datapath1/1.0
        self.DataList2 = Datapath2/1.0
        
    def __getitem__(self, index):
        index = index
        Data = (self.DataList1[index])
        Data2 = (self.DataList2[index])
       
        return torch.FloatTensor(Data), torch.FloatTensor(Data2)
    def __len__(self):
        return len(self.DataList1)

def set_loader(args):
    # construct data loader 构造数据加载器

    data, label = loadData(args.dataset)
    data = data
    data = np.transpose(data,(2,0,1))
    cdata = data.shape

    augmentation1 = [
        transforms.RandomApply([moco.loader.GaussianBlur(0.005*cdata[1])], p=1.0)
    ]

    augmentation2 = [
        transforms.RandomApply([moco.loader.GaussianBlur(0.005*cdata[1])], p=0.0)
    ]

    data_transforme1 = transforms.Compose(augmentation1)
    data_transforme2 = transforms.Compose(augmentation2)

    number = cdata[0]
    aug_data1 = []
    aug_data2 = []
    for i in range(number):
        image_new = Image.fromarray(data[i,:,:]/1.0)
        aug1 = data_transforme1(image_new)
        aug1 = np.asarray(aug1)
        aug_data1.append(aug1[None,:])

        aug2 = data_transforme2(image_new)
        aug2 = np.asarray(aug2)
        aug_data2.append(aug2[None,:])

    aug_x1 = np.vstack(aug_data1)
    aug_x2 = np.vstack(aug_data2)


    aug_x1 = np.transpose(aug_x1,(1,2,0))
    aug_x2 = np.transpose(aug_x2,(1,2,0))

    aug_x1 = np.reshape(aug_x1,(cdata[1]*cdata[2],cdata[0]))
    aug_x2 = np.reshape(aug_x2,(cdata[1]*cdata[2],cdata[0]))

    train_dataset = moco.loader.TwoCropsTransform(aug_x1, aug_x2)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, sampler=train_sampler)

    return train_loader


def main():
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_option()
    # build data loader
    train_loader = set_loader(args)
    
    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(args, model)

    # training routine
    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer, epoch, args)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)
    
    end = time.time()
    print (end-start)

    # save the last model
    save_file = os.path.join(
        args.save_folder, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)

if __name__ == '__main__':
    main()
