# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from data import *
from layers.box_utils import decode, nms
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
import math
from statistics import mean, median, variance, pstdev


def active_learning_cycle(
    batch_iterator,
    labeled_set,
    unlabeled_set,
    net,
    num_classes,
    acquisition_budget,
    num_total_images,
):
    """Active learning cycle for Mixture Density Networks.

    Collect aleatoric and epistemic uncertainties of both tasks (localization and classification)
    and normalize each uncertainty values using z-score for having similar scale. Afte that,
    the maximum value among the four uncertaintiesc will be the final score for the current image.
    """
    # lists of aleatoric and epistemic uncertainties of localization
    list_loc_al = []
    list_loc_ep = []
    # lists of aleatoric and epistemic uncertainties of classification
    list_conf_al = []
    list_conf_ep = []

    # filtering threshold of confidence score
    thresh = 0.5
    checker = 0
    for j in range(len(batch_iterator)):
        print(j)
        images, _ = next(batch_iterator)
        images = images.cuda()

        out = net(images)
        priors, loc, loc_var, loc_pi, loc_2, loc_var_2, loc_pi_2, \
        loc_3, loc_var_3, loc_pi_3, loc_4, loc_var_4, loc_pi_4, \
        conf, conf_var, conf_pi, conf_2, conf_var_2, conf_pi_2, \
        conf_3, conf_var_3, conf_pi_3, conf_4, conf_var_4, conf_pi_4 = out

        # confidence score of classification
        # use a softmax function to make valus in probability space
        conf = torch.softmax(conf, dim=2)
        conf_2 = torch.softmax(conf_2, dim=2)
        conf_3 = torch.softmax(conf_3, dim=2)
        conf_4 = torch.softmax(conf_4, dim=2)

        # mixture weight of classification
        conf_p_pi = conf_pi.view(-1, 1)
        conf_p_2_pi = conf_pi_2.view(-1, 1)
        conf_p_3_pi = conf_pi_3.view(-1, 1)
        conf_p_4_pi = conf_pi_4.view(-1, 1)

        conf_var = torch.sigmoid(conf_var)
        conf_var_2 = torch.sigmoid(conf_var_2)
        conf_var_3 = torch.sigmoid(conf_var_3)
        conf_var_4 = torch.sigmoid(conf_var_4)

        # use a softmax function to keep pi in probability space and split mixture weights
        (
            conf_pi,
            conf_pi_2,
            conf_pi_3,
            conf_pi_4
        ) = stack_softamx_unbind(
            pi=conf_p_pi,
            pi_2=conf_p_2_pi,
            pi_3=conf_p_3_pi,
            pi_4=conf_p_4_pi,
        )
        conf_pi = conf_pi.view(conf.size(0), -1, 1)
        conf_pi_2 = conf_pi_2.view(conf.size(0), -1, 1)
        conf_pi_3 = conf_pi_3.view(conf.size(0), -1, 1)
        conf_pi_4 = conf_pi_4.view(conf.size(0), -1, 1)

        # classification score
        new_conf = conf_pi*conf + conf_pi_2*conf_2 + conf_pi_3*conf_3 + conf_pi_4*conf_4

        # aleatoric uncertainty of classification
        cls_al_uc = conf_pi*conf_var + conf_pi_2*conf_var_2 + conf_pi_3*conf_var_3 + conf_pi_4*conf_var_4

        # epistemic uncertainty of classification
        cls_ep_uc = (
            conf_pi*(conf-new_conf)**2 +
            conf_pi_2*(conf_2-new_conf)**2 +
            conf_pi_3*(conf_3-new_conf)**2 +
            conf_pi_4*(conf_4-new_conf)**2
        )
        new_conf = new_conf.view(loc.size(0), priors.size(0), num_classes).transpose(2, 1)
        cls_al_uc = cls_al_uc.view(loc.size(0), priors.size(0), num_classes).transpose(2, 1)
        cls_ep_uc = cls_ep_uc.view(loc.size(0), priors.size(0), num_classes).transpose(2, 1)

        # aleatoric uncertainty of localizaiton
        # use a sigmoid function to satisfy the positiveness constraint
        loc_var = torch.sigmoid(loc_var)
        loc_var_2 = torch.sigmoid(loc_var_2)
        loc_var_3 = torch.sigmoid(loc_var_3)
        loc_var_4 = torch.sigmoid(loc_var_4)

        # mixture weight of localizaiton
        loc_p_pi = loc_pi.view(-1, 4)
        loc_p_2_pi = loc_pi_2.view(-1, 4)
        loc_p_3_pi = loc_pi_3.view(-1, 4)
        loc_p_4_pi = loc_pi_4.view(-1, 4)

        # use a softmax function to keep pi in probability space and split mixture weights
        (
            pi_1_after,
            pi_2_after,
            pi_3_after,
            pi_4_after
        ) = stack_softamx_unbind(
            pi=loc_p_pi,
            pi_2=loc_p_2_pi,
            pi_3=loc_p_3_pi,
            pi_4=loc_p_4_pi,
        )

        pi_1_after = pi_1_after.view(loc.size(0), -1, 4)
        pi_2_after = pi_2_after.view(loc.size(0), -1, 4)
        pi_3_after = pi_3_after.view(loc.size(0), -1, 4)
        pi_4_after = pi_4_after.view(loc.size(0), -1, 4)

        # localization coordinates
        new_loc = pi_1_after*loc + pi_2_after*loc_2 + pi_3_after*loc_3 + pi_4_after*loc_4

        # aleatoric uncertainty of localization
        al_uc = (
            pi_1_after*loc_var +
            pi_2_after*loc_var_2 +
            pi_3_after*loc_var_3 +
            pi_4_after*loc_var_4
        )

        # epistemic uncertainty of localization
        ep_uc = (
            pi_1_after*(loc-new_loc)**2 +
            pi_2_after*(loc_2-new_loc)**2 +
            pi_3_after*(loc_3-new_loc)**2 +
            pi_4_after*(loc_4-new_loc)**2
        )

        num = loc.size(0)
        output = torch.zeros(num, num_classes, 200, 15)
        variance = [0.1, 0.2]
        for i in range(num):
            decoded_boxes = decode(new_loc[i], priors, variance)
            conf_scores = new_conf[i]
            loc_al_uc_clone = al_uc[i]
            loc_ep_uc_clone = ep_uc[i]
            conf_al_clone = cls_al_uc[i]
            conf_ep_clone = cls_ep_uc[i]

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(0.01)
                # confidence score
                scores = conf_scores[cl][c_mask]
                # aleatoric and epistemic uncertainties of classification
                conf_al = conf_al_clone[cl][c_mask]
                conf_ep = conf_ep_clone[cl][c_mask]
                if scores.size(0) == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # aleatoric and epistemic uncertainties of localization
                loc_al_uc = loc_al_uc_clone[l_mask].view(-1, 4)
                loc_ep_uc = loc_ep_uc_clone[l_mask].view(-1, 4)

                ids, count = nms(boxes.detach(), scores.detach(), 0.45, 200)
                output[i, cl, :count] = torch.cat(
                    (
                        scores[ids[:count]].unsqueeze(1),
                        boxes[ids[:count]],
                        loc_al_uc[ids[:count]],
                        loc_ep_uc[ids[:count]],
                        conf_al[ids[:count]].unsqueeze(1),
                        conf_ep[ids[:count]].unsqueeze(1)
                    ),
                    1
                )

        # store the maximum value of each uncertainty in each jagged list
        for p in range(output.size(1)):
            q = 0
            if j == checker:
                list_loc_al.append([])
                list_loc_ep.append([])
                list_conf_al.append([])
                list_conf_ep.append([])
                checker = j + 1
            while output[0, p, q, 0] >= thresh:
                UC_max_al_temp = torch.max(output[0, p, q, 5:9]).item()
                UC_max_ep_temp = torch.max(output[0, p, q, 9:13]).item()
                UC_max_conf_al_temp = torch.max(output[0, p, q, 13:14]).item()
                UC_max_conf_ep_temp = torch.max(output[0, p, q, 14:15]).item()
                list_loc_al[j].append(UC_max_al_temp)
                list_loc_ep[j].append(UC_max_ep_temp)
                list_conf_al[j].append(UC_max_conf_al_temp)
                list_conf_ep[j].append(UC_max_conf_ep_temp)
                q += 1

    # z-score normalization and the deciding labeled and unlabeled dataset
    labeled_set, unlabeled_set = normalization_and_select_dataset(
        labeled_set=labeled_set,
        unlabeled_set=unlabeled_set,
        list_loc_al=list_loc_al,
        list_loc_ep=list_loc_ep,
        list_conf_al=list_conf_al,
        list_conf_ep=list_conf_ep,
        acquisition_budget=acquisition_budget,
        num_total_images=num_total_images,
    )

    return labeled_set, unlabeled_set


def stack_softamx_unbind(
    pi,
    pi_2,
    pi_3,
    pi_4,
):
    """Softmax and split mixture weights (pi)."""
    pi_all = torch.stack([pi, pi_2, pi_3, pi_4])
    pi_all = torch.softmax(pi_all, dim=0)
    (
        pi,
        pi_2,
        pi_3,
        pi_4
    ) = torch.unbind(pi_all, dim=0)

    return pi, pi_2, pi_3, pi_4


def normalization_and_select_dataset(
    labeled_set,
    unlabeled_set,
    list_loc_al,
    list_loc_ep,
    list_conf_al,
    list_conf_ep,
    acquisition_budget,
    num_total_images,
):
    """Z-score normalization and selecting labeled and unlabeled dataset.

    Args:
        labeled_set: current labeled list
        unlabeled_set: current unlabeled list
        list_loc_al: aleatoric uncertainty of localization (jagged list)
        list_loc_ep: epistemic uncertainty of localization (jagged list)
        list_conf_al: aleatoric uncertainty of classification (jagged list)
        list_conf_ep: epistemic uncertainty of classification (jagged list)
        acquisition_budget: selection budget for unlabeled dataset
        num_total_images: number of total dataset
    """
    # calculate the mean and variance of each uncertainty list for z-score normalization
    mean_loc_al = mean([val for sub in list_loc_al for val in sub])
    stdev_loc_al = pstdev([val for sub in list_loc_al for val in sub])
    mean_loc_ep = mean([val for sub in list_loc_ep for val in sub])
    stdev_loc_ep = pstdev([val for sub in list_loc_ep for val in sub])
    mean_conf_al = mean([val for sub in list_conf_al for val in sub])
    stdev_conf_al = pstdev([val for sub in list_conf_al for val in sub])
    mean_conf_ep = mean([val for sub in list_conf_ep for val in sub])
    stdev_conf_ep = pstdev([val for sub in list_conf_ep for val in sub])

    # minimum value of z-score (manually selected value)
    uc_min = -99999.0
    # insert minimum value into empty list in jagged list
    # find max value of each index in jagged list
    uncertainties = [list_loc_al, list_loc_ep, list_conf_al, list_conf_ep]
    for i in range(len(uncertainties)):
        uncertainty = uncertainties[i]
        for _ in range(uncertainty.count([])):
            uncertainty[uncertainty.index([])] = [uc_min]
        uncertainties[i] = [max(val) for val in uncertainty]

    # z-score normalization
    uncertainties[0] = [(val-mean_loc_al)/stdev_loc_al for val in uncertainties[0]]
    uncertainties[1] = [(val-mean_loc_ep)/stdev_loc_ep for val in uncertainties[1]]
    uncertainties[2] = [(val-mean_conf_al)/stdev_conf_al for val in uncertainties[2]]
    uncertainties[3] = [(val-mean_conf_ep)/stdev_conf_ep for val in uncertainties[3]]

    # make the minimum value converted by z-score normalization to the original minimum value
    # need this part because we need to calculate the maximum of the total 4 uncertainties
    for _ in range(uncertainties[0].count((uc_min-mean_loc_al)/stdev_loc_al)):
        uncertainties[0][uncertainties[0].index((uc_min-mean_loc_al)/stdev_loc_al)] = uc_min
    for _ in range(uncertainties[1].count((uc_min-mean_loc_ep)/stdev_loc_ep)):
        uncertainties[1][uncertainties[1].index((uc_min-mean_loc_ep)/stdev_loc_ep)] = uc_min
    for _ in range(uncertainties[2].count((uc_min-mean_conf_al)/stdev_conf_al)):
        uncertainties[2][uncertainties[2].index((uc_min-mean_conf_al)/stdev_conf_al)] = uc_min
    for _ in range(uncertainties[3].count((uc_min-mean_conf_ep)/stdev_conf_ep)):
        uncertainties[3][uncertainties[3].index((uc_min-mean_conf_ep)/stdev_conf_ep)] = uc_min

    uncertainties = torch.FloatTensor(uncertainties)
    uc_list = torch.stack([uncertainties[0], uncertainties[1], uncertainties[2], uncertainties[3]], dim=1)
    uc_list = np.array(uc_list)
    criterion_UC = np.max(uc_list, axis=1)
    sorted_indices = np.argsort(criterion_UC)[::-1]

    labeled_set += list(np.array(unlabeled_set)[sorted_indices[:acquisition_budget]])
    unlabeled_set = list(np.array(unlabeled_set)[sorted_indices[acquisition_budget:]])

    # assert that sizes of lists are correct and that there are no elements that are in both lists
    assert len(list(set(labeled_set) | set(unlabeled_set))) == num_total_images
    assert len(list(set(labeled_set) & set(unlabeled_set))) == 0

    return labeled_set, unlabeled_set
