# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Updated for running Gaussian Mixture Model
# Adapted from https://github.com/amdegroot/ssd.pytorch


from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss_GMM
from ssd_gmm import build_ssd_gmm
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC300', choices=['VOC300', 'VOC512'],
                    type=str, help='VOC300 or VOC512')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--id', default=1, type=int,
                    help='the id of the experiment')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'VOC300':
        cfg = voc300
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))
    else:
        cfg = voc512
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    net = build_ssd_gmm('train', cfg['min_dim'], cfg['num_classes'])

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(args.resume))
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        net.vgg.load_state_dict(vgg_weights)
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        net.extras.apply(weights_init)
        net.loc_mu_1.apply(weights_init)
        net.conf_mu_1.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    if args.cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
        cudnn.benchmark = True

    criterion = MultiBoxLoss_GMM(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()

    loc_loss = 0
    conf_loss = 0
    epoch = 0

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # warm-up
        if iteration < 1000:
            lr = args.lr * ((iteration+1)/1000.0)**4
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            targets = [ann for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data
        conf_loss += loss_c.data

        if (float(loss) > 100) or torch.isinf(loss) or torch.isnan(loss):
            # if the net diverges, go back to point 0 and train from scratch
            break

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')
            print('loss: %.4f , loss_c: %.4f , loss_l: %.4f , lr : %.4f\n' % (
                  loss.data, loss_c.data, loss_l.data, float(optimizer.param_groups[0]['lr'])))

        if iteration != 0 and (iteration+1) % 10000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), 'weights/ssd300' + '_id_' + str(args.id) + '_' + repr(iteration + 1) + '.pth')


def adjust_learning_rate(
    optimizer,
    gamma,
    step,
):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(
    _xlabel,
    _ylabel,
    _title,
    _legend,
):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(
    iteration,
    loc,
    conf,
    window1,
    window2,
    update_type,
    epoch_size=1
):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
