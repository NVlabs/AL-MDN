# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
from torch.autograd import Function
from ..box_utils import decode, nms


class Detect_GMM(Function):
    """At test time, Detect is the final layer of SSD. Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = [0.1, 0.2]


    def forward(self, prior_data, \
    loc_mu_1=None, loc_var_1=None, loc_pi_1=None, loc_mu_2=None, loc_var_2=None, loc_pi_2=None, \
    loc_mu_3=None, loc_var_3=None, loc_pi_3=None, loc_mu_4=None, loc_var_4=None, loc_pi_4=None, \
    conf_mu_1=None, conf_var_1=None, conf_pi_1=None, conf_mu_2=None, conf_var_2=None, conf_pi_2=None, \
    conf_mu_3=None, conf_var_3=None, conf_pi_3=None, conf_mu_4=None, conf_var_4=None, conf_pi_4=None
    ):
        num = loc_mu_1.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 15)

        new_loc = (
            loc_pi_1*loc_mu_1 +
            loc_pi_2*loc_mu_2 +
            loc_pi_3*loc_mu_3 +
            loc_pi_4*loc_mu_4
        )
        al_uc = (
            loc_pi_1*loc_var_1 +
            loc_pi_2*loc_var_2 +
            loc_pi_3*loc_var_3 +
            loc_pi_4*loc_var_4
        )
        ep_uc = (
            loc_pi_1*(loc_mu_1-new_loc)**2 +
            loc_pi_2*(loc_mu_2-new_loc)**2 +
            loc_pi_3*(loc_mu_3-new_loc)**2 +
            loc_pi_4*(loc_mu_4-new_loc)**2
        )

        new_conf = (
            conf_pi_1*conf_mu_1 +
            conf_pi_2*conf_mu_2 +
            conf_pi_3*conf_mu_3 +
            conf_pi_4*conf_mu_4
        )
        cls_al_uc = (
            conf_pi_1*conf_var_1 +
            conf_pi_2*conf_var_2 +
            conf_pi_3*conf_var_3 +
            conf_pi_4*conf_var_4
        )
        cls_ep_uc = (
            conf_pi_1*(conf_mu_1-new_conf)**2 +
            conf_pi_2*(conf_mu_2-new_conf)**2 +
            conf_pi_3*(conf_mu_3-new_conf)**2 +
            conf_pi_4*(conf_mu_4-new_conf)**2
        )

        new_conf = new_conf.view(num, num_priors, self.num_classes).transpose(2, 1)
        cls_al_uc = cls_al_uc.view(num, num_priors, self.num_classes).transpose(2, 1)
        cls_ep_uc = cls_ep_uc.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(new_loc[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = new_conf[i].clone()
            conf_al_clone = cls_al_uc[i].clone()
            conf_ep_clone = cls_ep_uc[i].clone()
            loc_al_uc_clone = al_uc[i].clone()
            loc_ep_uc_clone = ep_uc[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                conf_al = conf_al_clone[cl][c_mask]
                conf_ep = conf_ep_clone[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                loc_al_uc = loc_al_uc_clone[l_mask].view(-1, 4)
                loc_ep_uc = loc_ep_uc_clone[l_mask].view(-1, 4)

                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
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

        flt = output.contiguous().view(num, -1, 15)
        _, idx = flt[:, :, 0].sort(1, descending = True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        return output
