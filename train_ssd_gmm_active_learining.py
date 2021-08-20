# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss_GMM
from ssd_gmm import build_ssd_gmm
from utils.test_voc import *
from active_learning_loop import *
import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import math
import random
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from subset_sequential_sampler import SubsetSequentialSampler


random.seed(314) # need to change manually
torch.manual_seed(314) # need to change manually


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='VOC dataset root directory path')
parser.add_argument('--coco_root', default=COCO_ROOT,
                    help='COCO dataset root directory path')
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
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--eval_save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--use_cuda', default=True,
                    help='if True use GPU, otherwise use CPU')
parser.add_argument('--id', default=1, type=int,
                    help='the id of the experiment')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.dataset == 'VOC':
    cfg = voc300_active
else:
    cfg = coco300_active

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
if not os.path.exists(args.eval_save_folder):
    os.mkdir(args.eval_save_folder)


def create_loaders():
    num_train_images = cfg['num_total_images']
    indices = list(range(num_train_images))
    random.shuffle(indices)
    labeled_set = indices[:cfg['num_initial_labeled_set']]
    unlabeled_set = indices[cfg['num_initial_labeled_set']:]

    if cfg['name'] == 'VOC':
        supervised_dataset = VOCDetection(root=args.voc_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        unsupervised_dataset = VOCDetection(args.voc_root, [('2007', 'trainval')],
                                            BaseTransform(300, MEANS),
                                            VOCAnnotationTransform())
    else:
        supervised_dataset = COCODetection(root=args.coco_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        unsupervised_dataset = COCODetection(args.coco_root,
                                             transform=BaseTransform(300, MEANS))

    supervised_data_loader = data.DataLoader(supervised_dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             sampler=SubsetRandomSampler(labeled_set),
                                             collate_fn=detection_collate,
                                             pin_memory=True)

    unsupervised_data_loader = data.DataLoader(unsupervised_dataset, batch_size=1,
                                               num_workers=args.num_workers,
                                               sampler=SubsetSequentialSampler(unlabeled_set),
                                               collate_fn=detection_collate,
                                               pin_memory=True)
    return supervised_dataset, supervised_data_loader, unsupervised_dataset, unsupervised_data_loader, indices, labeled_set, unlabeled_set


def change_loaders(
    supervised_dataset,
    unsupervised_dataset,
    labeled_set,
    unlabeled_set,
):
    supervised_data_loader = data.DataLoader(supervised_dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             sampler=SubsetRandomSampler(labeled_set),
                                             collate_fn=detection_collate,
                                             pin_memory=True)

    unsupervised_data_loader = data.DataLoader(unsupervised_dataset, batch_size=1,
                                               num_workers=args.num_workers,
                                               sampler=SubsetSequentialSampler(unlabeled_set),
                                               collate_fn=detection_collate,
                                               pin_memory=True)
    return supervised_data_loader, unsupervised_data_loader


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


def load_net_optimizer_multi(cfg):
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
        net = nn.DataParallel(net)
        net = net.cuda()
    return net, optimizer


def train(
    labeled_set,
    supervised_data_loader,
    indices,
    cfg,
    criterion,
):
    finish_flag = True
    while finish_flag:
        net, optimizer = load_net_optimizer_multi(cfg)
        net = net.train()
        loc_loss = 0
        conf_loss = 0
        supervised_flag = 1
        step_index = 0

        batch_iterator = iter(supervised_data_loader)
        for iteration in range(args.start_iter, cfg['max_iter']):
            # warm-up
            if iteration < 1000:
                lr = args.lr * ((iteration+1)/1000.0)**4
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(supervised_data_loader)
                images, targets = next(batch_iterator)

            images = images.to(device)
            targets = [ann.to(device) for ann in targets]

            # forward
            t0 = time.time()
            out = net(images)

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

            if iteration != 0 and (iteration + 1) % 10000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(), 'weights/ssd300_AL_' + cfg['name'] + '_id_' + str(args.id) +
                           '_num_labels_' + str(len(labeled_set)) + '_' + repr(iteration + 1) + '.pth')

            if ((iteration + 1) == cfg['max_iter']):
                finish_flag = False
        else:
            finish_flag = False
    return net

def main():
    if args.cuda:
        cudnn.benchmark = True
    print(args)
    supervised_dataset, supervised_data_loader, unsupervised_dataset, unsupervised_data_loader, indices, labeled_set, unlabeled_set = create_loaders()
    criterion = MultiBoxLoss_GMM(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    print(len(labeled_set), len(unlabeled_set))
    net = train(labeled_set, supervised_data_loader, indices, cfg, criterion)

    # # active learning loop
    for i in range(cfg['num_cycles']):
        if cfg['name'] == 'VOC':
            # select the best weight
            list_iter = ['90000', '100000', '110000', '120000']
            list_weights = []
            for loop in list_iter:
                name = 'weights/ssd300_AL_' + cfg['name'] + '_id_' + str(args.id) + '_num_labels_' + str(len(labeled_set)) + '_' + loop + '.pth'
                list_weights.append(str(name))

            list_mean = []
            for loop in list_weights:
                net = build_ssd_gmm('test', cfg['min_dim'], cfg['num_classes'])
                net = nn.DataParallel(net)
                print('loading trained weight {}...'.format(loop))
                net.load_state_dict(torch.load(loop))
                net.eval()
                test_dataset = VOCDetection(args.voc_root, [('2007', 'test')], BaseTransform(300, MEANS), VOCAnnotationTransform())
                mean_ap = test_net(args.eval_save_folder, net, args.cuda,
                                   test_dataset, BaseTransform(300, MEANS),
                                   args.top_k, 300, thresh=args.confidence_threshold)
                list_mean.append(float(mean_ap))
            best_weight = list_weights[list_mean.index(max(list_mean))]

            # active learning
            net = build_ssd_gmm('train', cfg['min_dim'], cfg['num_classes'])
            net = nn.DataParallel(net)
            print('loading best weight {}...'.format(best_weight))
            net.load_state_dict(torch.load(best_weight))

        net.eval()
        batch_iterator = iter(unsupervised_data_loader)
        labeled_set, unlabeled_set = active_learning_cycle(
            batch_iterator,
            labeled_set,
            unlabeled_set,
            net,
            cfg["num_classes"],
            acquisition_budget=cfg['acquisition_budget'],
            num_total_images=cfg['num_total_images'],
        )

        # save the labeled training set list
        f = open("labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt", 'w')
        for i in range(len(labeled_set)):
            f.write(str(labeled_set[i]))
            f.write("\n")
        f.close()

        # change the loaders
        supervised_data_loader, unsupervised_data_loader = change_loaders(supervised_dataset, unsupervised_dataset, labeled_set, unlabeled_set)
        print(len(labeled_set), len(unlabeled_set))

        args.resume = None
        args.start_iter = 0
        net = train(labeled_set, supervised_data_loader, indices, cfg, criterion)

if __name__ == '__main__':
    main()
