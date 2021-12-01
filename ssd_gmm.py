# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc300, voc512, coco
import os


class SSD_GMM(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD_GMM, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        if(size==300):
            self.cfg = (coco, voc300)[num_classes == 21]
        else:
            self.cfg = (coco, voc512)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        # localization GMM parameters
        self.loc_mu_1 = nn.ModuleList(head[0])
        self.loc_var_1 = nn.ModuleList(head[1])
        self.loc_pi_1 = nn.ModuleList(head[2])
        self.loc_mu_2 = nn.ModuleList(head[3])
        self.loc_var_2 = nn.ModuleList(head[4])
        self.loc_pi_2 = nn.ModuleList(head[5])
        self.loc_mu_3 = nn.ModuleList(head[6])
        self.loc_var_3 = nn.ModuleList(head[7])
        self.loc_pi_3 = nn.ModuleList(head[8])
        self.loc_mu_4 = nn.ModuleList(head[9])
        self.loc_var_4 = nn.ModuleList(head[10])
        self.loc_pi_4 = nn.ModuleList(head[11])

        # Classification GMM parameters
        self.conf_mu_1 = nn.ModuleList(head[12])
        self.conf_var_1 = nn.ModuleList(head[13])
        self.conf_pi_1 = nn.ModuleList(head[14])
        self.conf_mu_2 = nn.ModuleList(head[15])
        self.conf_var_2 = nn.ModuleList(head[16])
        self.conf_pi_2 = nn.ModuleList(head[17])
        self.conf_mu_3 = nn.ModuleList(head[18])
        self.conf_var_3 = nn.ModuleList(head[19])
        self.conf_pi_3 = nn.ModuleList(head[20])
        self.conf_mu_4 = nn.ModuleList(head[21])
        self.conf_var_4 = nn.ModuleList(head[22])
        self.conf_pi_4 = nn.ModuleList(head[23])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_GMM(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected.

            train:
                list of concat outputs from:
                    1: confidence layers
                    2: localization layer
                    3: priorbox layers
        """
        sources = list()
        loc_mu_1 = list()
        loc_var_1 = list()
        loc_pi_1 = list()
        loc_mu_2 = list()
        loc_var_2 = list()
        loc_pi_2 = list()
        loc_mu_3 = list()
        loc_var_3 = list()
        loc_pi_3 = list()
        loc_mu_4 = list()
        loc_var_4 = list()
        loc_pi_4 = list()
        conf_mu_1 = list()
        conf_var_1 = list()
        conf_pi_1 = list()
        conf_mu_2 = list()
        conf_var_2 = list()
        conf_pi_2 = list()
        conf_mu_3 = list()
        conf_var_3 = list()
        conf_pi_3 = list()
        conf_mu_4 = list()
        conf_var_4 = list()
        conf_pi_4 = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l_mu_1, l_var_1, l_pi_1, l_mu_2, l_var_2, l_pi_2, l_mu_3, l_var_3, l_pi_3, l_mu_4, l_var_4, l_pi_4, \
        c_mu_1, c_var_1, c_pi_1, c_mu_2, c_var_2, c_pi_2, c_mu_3, c_var_3, c_pi_3, c_mu_4, c_var_4, c_pi_4) in zip(sources, \
        self.loc_mu_1, self.loc_var_1, self.loc_pi_1, self.loc_mu_2, self.loc_var_2, self.loc_pi_2, \
        self.loc_mu_3, self.loc_var_3, self.loc_pi_3, self.loc_mu_4, self.loc_var_4, self.loc_pi_4, \
        self.conf_mu_1, self.conf_var_1, self.conf_pi_1, self.conf_mu_2, self.conf_var_2, self.conf_pi_2, \
        self.conf_mu_3, self.conf_var_3, self.conf_pi_3, self.conf_mu_4, self.conf_var_4, self.conf_pi_4):
            loc_mu_1.append(l_mu_1(x).permute(0, 2, 3, 1).contiguous())
            loc_var_1.append(l_var_1(x).permute(0, 2, 3, 1).contiguous())
            loc_pi_1.append(l_pi_1(x).permute(0, 2, 3, 1).contiguous())
            loc_mu_2.append(l_mu_2(x).permute(0, 2, 3, 1).contiguous())
            loc_var_2.append(l_var_2(x).permute(0, 2, 3, 1).contiguous())
            loc_pi_2.append(l_pi_2(x).permute(0, 2, 3, 1).contiguous())
            loc_mu_3.append(l_mu_3(x).permute(0, 2, 3, 1).contiguous())
            loc_var_3.append(l_var_3(x).permute(0, 2, 3, 1).contiguous())
            loc_pi_3.append(l_pi_3(x).permute(0, 2, 3, 1).contiguous())
            loc_mu_4.append(l_mu_4(x).permute(0, 2, 3, 1).contiguous())
            loc_var_4.append(l_var_4(x).permute(0, 2, 3, 1).contiguous())
            loc_pi_4.append(l_pi_4(x).permute(0, 2, 3, 1).contiguous())
            conf_mu_1.append(c_mu_1(x).permute(0, 2, 3, 1).contiguous())
            conf_var_1.append(c_var_1(x).permute(0, 2, 3, 1).contiguous())
            conf_pi_1.append(c_pi_1(x).permute(0, 2, 3, 1).contiguous())
            conf_mu_2.append(c_mu_2(x).permute(0, 2, 3, 1).contiguous())
            conf_var_2.append(c_var_2(x).permute(0, 2, 3, 1).contiguous())
            conf_pi_2.append(c_pi_2(x).permute(0, 2, 3, 1).contiguous())
            conf_mu_3.append(c_mu_3(x).permute(0, 2, 3, 1).contiguous())
            conf_var_3.append(c_var_3(x).permute(0, 2, 3, 1).contiguous())
            conf_pi_3.append(c_pi_3(x).permute(0, 2, 3, 1).contiguous())
            conf_mu_4.append(c_mu_4(x).permute(0, 2, 3, 1).contiguous())
            conf_var_4.append(c_var_4(x).permute(0, 2, 3, 1).contiguous())
            conf_pi_4.append(c_pi_4(x).permute(0, 2, 3, 1).contiguous())

        loc_mu_1 = torch.cat([o.view(o.size(0), -1) for o in loc_mu_1], 1)
        loc_var_1 = torch.cat([o.view(o.size(0), -1) for o in loc_var_1], 1)
        loc_pi_1 = torch.cat([o.view(o.size(0), -1) for o in loc_pi_1], 1)
        loc_mu_2 = torch.cat([o.view(o.size(0), -1) for o in loc_mu_2], 1)
        loc_var_2 = torch.cat([o.view(o.size(0), -1) for o in loc_var_2], 1)
        loc_pi_2 = torch.cat([o.view(o.size(0), -1) for o in loc_pi_2], 1)
        loc_mu_3 = torch.cat([o.view(o.size(0), -1) for o in loc_mu_3], 1)
        loc_var_3 = torch.cat([o.view(o.size(0), -1) for o in loc_var_3], 1)
        loc_pi_3 = torch.cat([o.view(o.size(0), -1) for o in loc_pi_3], 1)
        loc_mu_4 = torch.cat([o.view(o.size(0), -1) for o in loc_mu_4], 1)
        loc_var_4 = torch.cat([o.view(o.size(0), -1) for o in loc_var_4], 1)
        loc_pi_4 = torch.cat([o.view(o.size(0), -1) for o in loc_pi_4], 1)
        conf_mu_1 = torch.cat([o.view(o.size(0), -1) for o in conf_mu_1], 1)
        conf_var_1 = torch.cat([o.view(o.size(0), -1) for o in conf_var_1], 1)
        conf_pi_1 = torch.cat([o.view(o.size(0), -1) for o in conf_pi_1], 1)
        conf_mu_2 = torch.cat([o.view(o.size(0), -1) for o in conf_mu_2], 1)
        conf_var_2 = torch.cat([o.view(o.size(0), -1) for o in conf_var_2], 1)
        conf_pi_2 = torch.cat([o.view(o.size(0), -1) for o in conf_pi_2], 1)
        conf_mu_3 = torch.cat([o.view(o.size(0), -1) for o in conf_mu_3], 1)
        conf_var_3 = torch.cat([o.view(o.size(0), -1) for o in conf_var_3], 1)
        conf_pi_3 = torch.cat([o.view(o.size(0), -1) for o in conf_pi_3], 1)
        conf_mu_4 = torch.cat([o.view(o.size(0), -1) for o in conf_mu_4], 1)
        conf_var_4 = torch.cat([o.view(o.size(0), -1) for o in conf_var_4], 1)
        conf_pi_4 = torch.cat([o.view(o.size(0), -1) for o in conf_pi_4], 1)

        if self.phase == "test":
            loc_var_1 = torch.sigmoid(loc_var_1)
            loc_var_2 = torch.sigmoid(loc_var_2)
            loc_var_3 = torch.sigmoid(loc_var_3)
            loc_var_4 = torch.sigmoid(loc_var_4)

            loc_pi_1 = loc_pi_1.view(-1, 4)
            loc_pi_2 = loc_pi_2.view(-1, 4)
            loc_pi_3 = loc_pi_3.view(-1, 4)
            loc_pi_4 = loc_pi_4.view(-1, 4)

            pi_all = torch.stack(
                [
                    loc_pi_1.reshape(-1),
                    loc_pi_2.reshape(-1),
                    loc_pi_3.reshape(-1),
                    loc_pi_4.reshape(-1)
                ]
            )
            pi_all = pi_all.transpose(0,1)
            pi_all = (torch.softmax(pi_all, dim=1)).transpose(0,1).reshape(-1)
            (
                loc_pi_1,
                loc_pi_2,
                loc_pi_3,
                loc_pi_4
            ) = torch.split(pi_all, loc_pi_1.reshape(-1).size(0), dim=0)
            loc_pi_1 = loc_pi_1.view(-1, 4)
            loc_pi_2 = loc_pi_2.view(-1, 4)
            loc_pi_3 = loc_pi_3.view(-1, 4)
            loc_pi_4 = loc_pi_4.view(-1, 4)

            conf_var_1 = torch.sigmoid(conf_var_1)
            conf_var_2 = torch.sigmoid(conf_var_2)
            conf_var_3 = torch.sigmoid(conf_var_3)
            conf_var_4 = torch.sigmoid(conf_var_4)

            conf_pi_1 = conf_pi_1.view(-1, 1)
            conf_pi_2 = conf_pi_2.view(-1, 1)
            conf_pi_3 = conf_pi_3.view(-1, 1)
            conf_pi_4 = conf_pi_4.view(-1, 1)

            conf_pi_all = torch.stack(
                [
                    conf_pi_1.reshape(-1),
                    conf_pi_2.reshape(-1),
                    conf_pi_3.reshape(-1),
                    conf_pi_4.reshape(-1)
                ]
            )
            conf_pi_all = conf_pi_all.transpose(0,1)
            conf_pi_all = (torch.softmax(conf_pi_all, dim=1)).transpose(0,1).reshape(-1)
            (
                conf_pi_1,
                conf_pi_2,
                conf_pi_3,
                conf_pi_4
            ) = torch.split(conf_pi_all, conf_pi_1.reshape(-1).size(0), dim=0)
            conf_pi_1 = conf_pi_1.view(-1, 1)
            conf_pi_2 = conf_pi_2.view(-1, 1)
            conf_pi_3 = conf_pi_3.view(-1, 1)
            conf_pi_4 = conf_pi_4.view(-1, 1)

            output = self.detect.apply(
                self.priors.type(type(x.data)),
                loc_mu_1.view(loc_mu_1.size(0), -1, 4),
                loc_var_1.view(loc_var_1.size(0), -1, 4),
                loc_pi_1.view(loc_var_1.size(0), -1, 4),
                loc_mu_2.view(loc_mu_2.size(0), -1, 4),
                loc_var_2.view(loc_var_2.size(0), -1, 4),
                loc_pi_2.view(loc_var_2.size(0), -1, 4),
                loc_mu_3.view(loc_mu_3.size(0), -1, 4),
                loc_var_3.view(loc_var_3.size(0), -1, 4),
                loc_pi_3.view(loc_var_3.size(0), -1, 4),
                loc_mu_4.view(loc_mu_4.size(0), -1, 4),
                loc_var_4.view(loc_var_4.size(0), -1, 4),
                loc_pi_4.view(loc_var_4.size(0), -1, 4),
                self.softmax(conf_mu_1.view(conf_mu_1.size(0), -1, self.num_classes)),
                conf_var_1.view(conf_var_1.size(0), -1, self.num_classes),
                conf_pi_1.view(conf_var_1.size(0), -1, 1),
                self.softmax(conf_mu_2.view(conf_mu_2.size(0), -1, self.num_classes)),
                conf_var_2.view(conf_var_2.size(0), -1, self.num_classes),
                conf_pi_2.view(conf_var_2.size(0), -1, 1),
                self.softmax(conf_mu_3.view(conf_mu_3.size(0), -1, self.num_classes)),
                conf_var_3.view(conf_var_3.size(0), -1, self.num_classes),
                conf_pi_3.view(conf_var_3.size(0), -1, 1),
                self.softmax(conf_mu_4.view(conf_mu_4.size(0), -1, self.num_classes)),
                conf_var_4.view(conf_var_4.size(0), -1, self.num_classes),
                conf_pi_4.view(conf_var_4.size(0), -1, 1)
            )

        else:
            output = (
                self.priors,
                loc_mu_1.view(loc_mu_1.size(0), -1, 4),
                loc_var_1.view(loc_var_1.size(0), -1, 4),
                loc_pi_1.view(loc_pi_1.size(0), -1, 4),
                loc_mu_2.view(loc_mu_2.size(0), -1, 4),
                loc_var_2.view(loc_var_2.size(0), -1, 4),
                loc_pi_2.view(loc_pi_2.size(0), -1, 4),
                loc_mu_3.view(loc_mu_3.size(0), -1, 4),
                loc_var_3.view(loc_var_3.size(0), -1, 4),
                loc_pi_3.view(loc_pi_3.size(0), -1, 4),
                loc_mu_4.view(loc_mu_4.size(0), -1, 4),
                loc_var_4.view(loc_var_4.size(0), -1, 4),
                loc_pi_4.view(loc_pi_4.size(0), -1, 4),
                conf_mu_1.view(conf_mu_1.size(0), -1, self.num_classes),
                conf_var_1.view(conf_var_1.size(0), -1, self.num_classes),
                conf_pi_1.view(conf_pi_1.size(0), -1, 1),
                conf_mu_2.view(conf_mu_2.size(0), -1, self.num_classes),
                conf_var_2.view(conf_var_2.size(0), -1, self.num_classes),
                conf_pi_2.view(conf_pi_2.size(0), -1, 1),
                conf_mu_3.view(conf_mu_3.size(0), -1, self.num_classes),
                conf_var_3.view(conf_var_3.size(0), -1, self.num_classes),
                conf_pi_3.view(conf_pi_3.size(0), -1, 1),
                conf_mu_4.view(conf_mu_4.size(0), -1, self.num_classes),
                conf_var_4.view(conf_var_4.size(0), -1, self.num_classes),
                conf_pi_4.view(conf_pi_4.size(0), -1, 1)
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            elif v=='K':
                layers += [nn.Conv2d(in_channels, 256,
                           kernel_size=4, stride=1, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    vgg_source = [21, -2]
    loc_mu_1_layers = []
    loc_var_1_layers = []
    loc_pi_1_layers = []
    loc_mu_2_layers = []
    loc_var_2_layers = []
    loc_pi_2_layers = []
    loc_mu_3_layers = []
    loc_var_3_layers = []
    loc_pi_3_layers = []
    loc_mu_4_layers = []
    loc_var_4_layers = []
    loc_pi_4_layers = []
    conf_mu_1_layers = []
    conf_var_1_layers = []
    conf_pi_1_layers = []
    conf_mu_2_layers = []
    conf_var_2_layers = []
    conf_pi_2_layers = []
    conf_mu_3_layers = []
    conf_var_3_layers = []
    conf_pi_3_layers = []
    conf_mu_4_layers = []
    conf_var_4_layers = []
    conf_pi_4_layers = []

    for k, v in enumerate(vgg_source):
        loc_mu_1_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_var_1_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_pi_1_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_mu_2_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_var_2_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_pi_2_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_mu_3_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_var_3_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_pi_3_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_mu_4_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_var_4_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_pi_4_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_mu_1_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_var_1_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_pi_1_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 1, kernel_size=3, padding=1)]
        conf_mu_2_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_var_2_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_pi_2_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 1, kernel_size=3, padding=1)]
        conf_mu_3_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_var_3_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_pi_3_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 1, kernel_size=3, padding=1)]
        conf_mu_4_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_var_4_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_pi_4_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 1, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_mu_1_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        loc_var_1_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        loc_pi_1_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        loc_mu_2_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        loc_var_2_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        loc_pi_2_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        loc_mu_3_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        loc_var_3_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        loc_pi_3_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        loc_mu_4_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        loc_var_4_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        loc_pi_4_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
        conf_mu_1_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_var_1_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_pi_1_layers += [nn.Conv2d(v.out_channels, cfg[k] * 1, kernel_size=3, padding=1)]
        conf_mu_2_layers += [nn.Conv2d(v.out_channels,  cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_var_2_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_pi_2_layers += [nn.Conv2d(v.out_channels, cfg[k] * 1, kernel_size=3, padding=1)]
        conf_mu_3_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_var_3_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_pi_3_layers += [nn.Conv2d(v.out_channels, cfg[k] * 1, kernel_size=3, padding=1)]
        conf_mu_4_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_var_4_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        conf_pi_4_layers += [nn.Conv2d(v.out_channels, cfg[k] * 1, kernel_size=3, padding=1)]

    return vgg, extra_layers, (
        loc_mu_1_layers, loc_var_1_layers, loc_pi_1_layers, loc_mu_2_layers, loc_var_2_layers, loc_pi_2_layers, \
        loc_mu_3_layers, loc_var_3_layers, loc_pi_3_layers, loc_mu_4_layers, loc_var_4_layers, loc_pi_4_layers, \
        conf_mu_1_layers, conf_var_1_layers, conf_pi_1_layers, conf_mu_2_layers, conf_var_2_layers, conf_pi_2_layers, \
        conf_mu_3_layers, conf_var_3_layers, conf_pi_3_layers, conf_mu_4_layers, conf_var_4_layers, conf_pi_4_layers
    )


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'K'],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],
}


def build_ssd_gmm(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)

    return SSD_GMM(phase, size, base_, extras_, head_, num_classes)
