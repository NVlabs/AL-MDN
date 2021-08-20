# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp
import math

def Gaussian(y, mu, var):
    eps = 0.3
    result = (y-mu)/var
    result = (result**2)/2*(-1)
    exp = torch.exp(result)
    result = exp/(math.sqrt(2*math.pi))/(var + eps)

    return result

def NLL_loss(bbox_gt, bbox_pred, bbox_var):
        bbox_var = torch.sigmoid(bbox_var)
        prob = Gaussian(bbox_gt, bbox_pred, bbox_var)

        return prob

class MultiBoxLoss_GMM(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, cls_type='Type-1'):
        super(MultiBoxLoss_GMM, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.cls_type = cls_type

    def forward(self, predictions, targets):
        priors, loc_mu_1, loc_var_1, loc_pi_1, loc_mu_2, loc_var_2, loc_pi_2, \
        loc_mu_3, loc_var_3, loc_pi_3, loc_mu_4, loc_var_4, loc_pi_4, \
        conf_mu_1, conf_var_1, conf_pi_1, conf_mu_2, conf_var_2, conf_pi_2, \
        conf_mu_3, conf_var_3, conf_pi_3, conf_mu_4, conf_var_4, conf_pi_4 = predictions

        num = loc_mu_1.size(0)
        priors = priors[:loc_mu_1.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold,
                  truths,
                  defaults,
                  self.variance,
                  labels,
                  loc_t,
                  conf_t,
                  idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_mu_1)
        loc_mu_1_ = loc_mu_1[pos_idx].view(-1, 4)
        loc_mu_2_ = loc_mu_2[pos_idx].view(-1, 4)
        loc_mu_3_ = loc_mu_3[pos_idx].view(-1, 4)
        loc_mu_4_ = loc_mu_4[pos_idx].view(-1, 4)

        loc_t = loc_t[pos_idx].view(-1, 4)

        # localization loss
        loss_l_1 = NLL_loss(loc_t, loc_mu_1_, loc_var_1[pos_idx].view(-1, 4))
        loss_l_2 = NLL_loss(loc_t, loc_mu_2_, loc_var_2[pos_idx].view(-1, 4))
        loss_l_3 = NLL_loss(loc_t, loc_mu_3_, loc_var_3[pos_idx].view(-1, 4))
        loss_l_4 = NLL_loss(loc_t, loc_mu_4_, loc_var_4[pos_idx].view(-1, 4))

        loc_pi_1_ = loc_pi_1[pos_idx].view(-1, 4)
        loc_pi_2_ = loc_pi_2[pos_idx].view(-1, 4)
        loc_pi_3_ = loc_pi_3[pos_idx].view(-1, 4)
        loc_pi_4_ = loc_pi_4[pos_idx].view(-1, 4)

        pi_all = torch.stack([
                    loc_pi_1_.reshape(-1),
                    loc_pi_2_.reshape(-1),
                    loc_pi_3_.reshape(-1),
                    loc_pi_4_.reshape(-1)
                    ])
        pi_all = pi_all.transpose(0,1)
        pi_all = (torch.softmax(pi_all, dim=1)).transpose(0,1).reshape(-1)
        (
            loc_pi_1_,
            loc_pi_2_,
            loc_pi_3_,
            loc_pi_4_
        ) = torch.split(pi_all, loc_pi_1_.reshape(-1).size(0), dim=0)
        loc_pi_1_ = loc_pi_1_.view(-1, 4)
        loc_pi_2_ = loc_pi_2_.view(-1, 4)
        loc_pi_3_ = loc_pi_3_.view(-1, 4)
        loc_pi_4_ = loc_pi_4_.view(-1, 4)

        _loss_l = (
            loc_pi_1_*loss_l_1 +
            loc_pi_2_*loss_l_2 +
            loc_pi_3_*loss_l_3 +
            loc_pi_4_*loss_l_4
        )

        epsi = 10**-9
        # balance parameter
        balance = 2.0
        loss_l = -torch.log(_loss_l + epsi)/balance
        loss_l = loss_l.sum()

        if self.cls_type == 'Type-1':
            # Classification loss (Type-1)
            conf_pi_1_ = conf_pi_1.view(-1, 1)
            conf_pi_2_ = conf_pi_2.view(-1, 1)
            conf_pi_3_ = conf_pi_3.view(-1, 1)
            conf_pi_4_ = conf_pi_4.view(-1, 1)

            conf_pi_all = torch.stack([
                            conf_pi_1_.reshape(-1),
                            conf_pi_2_.reshape(-1),
                            conf_pi_3_.reshape(-1),
                            conf_pi_4_.reshape(-1)
                            ])
            conf_pi_all = conf_pi_all.transpose(0,1)
            conf_pi_all = (torch.softmax(conf_pi_all, dim=1)).transpose(0,1).reshape(-1)
            (
                conf_pi_1_,
                conf_pi_2_,
                conf_pi_3_,
                conf_pi_4_
            ) = torch.split(conf_pi_all, conf_pi_1_.reshape(-1).size(0), dim=0)
            conf_pi_1_ = conf_pi_1_.view(conf_pi_1.size(0), -1)
            conf_pi_2_ = conf_pi_2_.view(conf_pi_2.size(0), -1)
            conf_pi_3_ = conf_pi_3_.view(conf_pi_3.size(0), -1)
            conf_pi_4_ = conf_pi_4_.view(conf_pi_4.size(0), -1)

            conf_var_1 = torch.sigmoid(conf_var_1)
            conf_var_2 = torch.sigmoid(conf_var_2)
            conf_var_3 = torch.sigmoid(conf_var_3)
            conf_var_4 = torch.sigmoid(conf_var_4)

            rand_val_1 = torch.randn(conf_var_1.size(0), conf_var_1.size(1), conf_var_1.size(2))
            rand_val_2 = torch.randn(conf_var_2.size(0), conf_var_2.size(1), conf_var_2.size(2))
            rand_val_3 = torch.randn(conf_var_3.size(0), conf_var_3.size(1), conf_var_3.size(2))
            rand_val_4 = torch.randn(conf_var_4.size(0), conf_var_4.size(1), conf_var_4.size(2))

            batch_conf_1 = (conf_mu_1+torch.sqrt(conf_var_1)*rand_val_1).view(-1, self.num_classes)
            batch_conf_2 = (conf_mu_2+torch.sqrt(conf_var_2)*rand_val_2).view(-1, self.num_classes)
            batch_conf_3 = (conf_mu_3+torch.sqrt(conf_var_3)*rand_val_3).view(-1, self.num_classes)
            batch_conf_4 = (conf_mu_4+torch.sqrt(conf_var_4)*rand_val_4).view(-1, self.num_classes)

            loss_c_1 = log_sum_exp(batch_conf_1) - batch_conf_1.gather(1, conf_t.view(-1, 1))
            loss_c_2 = log_sum_exp(batch_conf_2) - batch_conf_2.gather(1, conf_t.view(-1, 1))
            loss_c_3 = log_sum_exp(batch_conf_3) - batch_conf_3.gather(1, conf_t.view(-1, 1))
            loss_c_4 = log_sum_exp(batch_conf_4) - batch_conf_4.gather(1, conf_t.view(-1, 1))

            loss_c = (
                loss_c_1 * conf_pi_1_.view(-1, 1) +
                loss_c_2 * conf_pi_2_.view(-1, 1) +
                loss_c_3 * conf_pi_3_.view(-1, 1) +
                loss_c_4 * conf_pi_4_.view(-1, 1)
            )
            loss_c = loss_c.view(pos.size()[0], pos.size()[1])
            loss_c[pos] = 0  # filter out pos boxes for now  : true -> zero
            loss_c = loss_c.view(num, -1)

            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_mu_1)
            neg_idx = neg.unsqueeze(2).expand_as(conf_mu_1)

            batch_conf_1_ = conf_mu_1+torch.sqrt(conf_var_1)*rand_val_1
            batch_conf_2_ = conf_mu_2+torch.sqrt(conf_var_2)*rand_val_2
            batch_conf_3_ = conf_mu_3+torch.sqrt(conf_var_3)*rand_val_3
            batch_conf_4_ = conf_mu_4+torch.sqrt(conf_var_4)*rand_val_4

            conf_pred_1 = batch_conf_1_[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_2 = batch_conf_2_[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_3 = batch_conf_3_[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_4 = batch_conf_4_[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)

            targets_weighted = conf_t[(pos+neg).gt(0)]

            loss_c_1 = log_sum_exp(conf_pred_1) - conf_pred_1.gather(1, targets_weighted.view(-1, 1))
            loss_c_2 = log_sum_exp(conf_pred_2) - conf_pred_2.gather(1, targets_weighted.view(-1, 1))
            loss_c_3 = log_sum_exp(conf_pred_3) - conf_pred_3.gather(1, targets_weighted.view(-1, 1))
            loss_c_4 = log_sum_exp(conf_pred_4) - conf_pred_4.gather(1, targets_weighted.view(-1, 1))

            _conf_pi_1 = conf_pi_1_[(pos+neg).gt(0)]
            _conf_pi_2 = conf_pi_2_[(pos+neg).gt(0)]
            _conf_pi_3 = conf_pi_3_[(pos+neg).gt(0)]
            _conf_pi_4 = conf_pi_4_[(pos+neg).gt(0)]

            loss_c = (
                loss_c_1 * _conf_pi_1.view(-1, 1) +
                loss_c_2 * _conf_pi_2.view(-1, 1) +
                loss_c_3 * _conf_pi_3.view(-1, 1) +
                loss_c_4 * _conf_pi_4.view(-1, 1)
            )
            loss_c = loss_c.sum()

        else:
            # Classification loss (Type-2)
            # more details are in our supplementary material
            conf_pi_1_ = conf_pi_1.view(-1, 1)
            conf_pi_2_ = conf_pi_2.view(-1, 1)
            conf_pi_3_ = conf_pi_3.view(-1, 1)
            conf_pi_4_ = conf_pi_4.view(-1, 1)

            conf_pi_all = torch.stack([
                            conf_pi_1_.reshape(-1),
                            conf_pi_2_.reshape(-1),
                            conf_pi_3_.reshape(-1),
                            conf_pi_4_.reshape(-1)
                            ])
            conf_pi_all = conf_pi_all.transpose(0,1)
            conf_pi_all = (torch.softmax(conf_pi_all, dim=1)).transpose(0,1).reshape(-1)
            (
                conf_pi_1_,
                conf_pi_2_,
                conf_pi_3_,
                conf_pi_4_
            ) = torch.split(conf_pi_all, conf_pi_1_.reshape(-1).size(0), dim=0)
            conf_pi_1_ = conf_pi_1_.view(conf_pi_1.size(0), -1)
            conf_pi_2_ = conf_pi_2_.view(conf_pi_2.size(0), -1)
            conf_pi_3_ = conf_pi_3_.view(conf_pi_3.size(0), -1)
            conf_pi_4_ = conf_pi_4_.view(conf_pi_4.size(0), -1)

            conf_var_1 = torch.sigmoid(conf_var_1)
            conf_var_2 = torch.sigmoid(conf_var_2)
            conf_var_3 = torch.sigmoid(conf_var_3)
            conf_var_4 = torch.sigmoid(conf_var_4)

            rand_val_1 = torch.randn(conf_var_1.size(0), conf_var_1.size(1), conf_var_1.size(2))
            rand_val_2 = torch.randn(conf_var_2.size(0), conf_var_2.size(1), conf_var_2.size(2))
            rand_val_3 = torch.randn(conf_var_3.size(0), conf_var_3.size(1), conf_var_3.size(2))
            rand_val_4 = torch.randn(conf_var_4.size(0), conf_var_4.size(1), conf_var_4.size(2))

            batch_conf_1 = (conf_mu_1+torch.sqrt(conf_var_1)*rand_val_1).view(-1, self.num_classes)
            batch_conf_2 = (conf_mu_2+torch.sqrt(conf_var_2)*rand_val_2).view(-1, self.num_classes)
            batch_conf_3 = (conf_mu_3+torch.sqrt(conf_var_3)*rand_val_3).view(-1, self.num_classes)
            batch_conf_4 = (conf_mu_4+torch.sqrt(conf_var_4)*rand_val_4).view(-1, self.num_classes)

            soft_max = nn.Softmax(dim=1)

            epsi = 10**-9
            weighted_softmax_out = (
                        soft_max(batch_conf_1)*conf_pi_1_.view(-1, 1) +
                        soft_max(batch_conf_2)*conf_pi_2_.view(-1, 1) +
                        soft_max(batch_conf_3)*conf_pi_3_.view(-1, 1) +
                        soft_max(batch_conf_4)*conf_pi_4_.view(-1, 1)
            )
            softmax_out_log = -torch.log(weighted_softmax_out+epsi)
            loss_c = softmax_out_log.gather(1, conf_t.view(-1,1))

            loss_c = loss_c.view(pos.size()[0], pos.size()[1])
            loss_c[pos] = 0  # filter out pos boxes for now  : true -> zero
            loss_c = loss_c.view(num, -1)

            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_mu_1)
            neg_idx = neg.unsqueeze(2).expand_as(conf_mu_1)

            batch_conf_1_ = conf_mu_1+torch.sqrt(conf_var_1)*rand_val_1
            batch_conf_2_ = conf_mu_2+torch.sqrt(conf_var_2)*rand_val_2
            batch_conf_3_ = conf_mu_3+torch.sqrt(conf_var_3)*rand_val_3
            batch_conf_4_ = conf_mu_4+torch.sqrt(conf_var_4)*rand_val_4

            conf_pred_1 = batch_conf_1_[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_2 = batch_conf_2_[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_3 = batch_conf_3_[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_4 = batch_conf_4_[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)

            targets_weighted = conf_t[(pos+neg).gt(0)]

            _conf_pi_1 = conf_pi_1_[(pos+neg).gt(0)]
            _conf_pi_2 = conf_pi_2_[(pos+neg).gt(0)]
            _conf_pi_3 = conf_pi_3_[(pos+neg).gt(0)]
            _conf_pi_4 = conf_pi_4_[(pos+neg).gt(0)]

            weighted_softmax_out = (
                        soft_max(conf_pred_1)*_conf_pi_1.view(-1, 1) +
                        soft_max(conf_pred_2)*_conf_pi_2.view(-1, 1) +
                        soft_max(conf_pred_3)*_conf_pi_3.view(-1, 1) +
                        soft_max(conf_pred_4)*_conf_pi_4.view(-1, 1)
            )
            softmax_out_log = -torch.log(weighted_softmax_out+epsi)
            loss_c = softmax_out_log.gather(1, targets_weighted.view(-1,1))
            loss_c = loss_c.sum()

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
