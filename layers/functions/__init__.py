# Updated detection_gmm for detection
# Adapted from https://github.com/amdegroot/ssd.pytorch

from .detection_gmm import Detect_GMM
from .prior_box import PriorBox


__all__ = ['Detect_GMM', 'PriorBox']
