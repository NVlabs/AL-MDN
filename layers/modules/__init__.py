# Updated multibox_loss_gmm for training
# Adapted from https://github.com/amdegroot/ssd.pytorch

from .l2norm import L2Norm
from .multibox_loss_gmm import MultiBoxLoss_GMM

__all__ = ['L2Norm', 'MultiBoxLoss_GMM']
