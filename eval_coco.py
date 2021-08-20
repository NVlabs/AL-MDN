# Originated from https://github.com/amdegroot/ssd.pytorch/issues/422
"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License
"""

import torch
from torch.autograd import Variable
from data import COCODetection_eval, COCO_ROOT_EVAL
from data import COCO_CLASSES as labelmap
import torch.nn as nn
from ssd_gmm import build_ssd_gmm
import cv2

import os
import time
import pickle
import argparse
import numpy as np

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--dataset_root',
                    default=COCO_ROOT_EVAL, type=str,
                    help='path to your coco2017 data')
parser.add_argument('--model_type',
                    default='SSD', type=str,
                    help='tested model')
parser.add_argument('--trained_model',
                    default='weights/trained_COCO_name.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--retest', default=False, type=str2bool,
                    help='test the result on result file')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        rgb_std: std of the dataset
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, resize, rgb_means, rgb_std=(1, 1, 1), swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.std = rgb_std
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):
        img = cv2.resize(np.array(img), (self.resize, self.resize)).astype(np.float32)
        img -= self.means
        img /= self.std
        img = img.transpose(self.swap)
        return torch.from_numpy(img)


def test_net(save_folder, net, cuda, testset, transform):
    with torch.no_grad():
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # dump predictions and assoc. ground truth to text file for now
        num_images = len(testset)
        num_classes = 81
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(num_classes)]

        _t = {'im_detect': Timer(), 'misc': Timer()}
        det_file = os.path.join(save_folder, 'detections.pkl')

        if args.retest:
            f = open(det_file, 'rb')
            all_boxes = pickle.load(f)
            print('Evaluating detections')
            testset.evaluate_detections(all_boxes, save_folder)
            return

        for i in range(num_images):
            img, h, w = testset.pull_image(i)
            x = Variable(transform(img).unsqueeze(0))
            if cuda:
                x = x.cuda()

            _t['im_detect'].tic()
            detections = net(x).data  # [1, class, top_k, 5]
            detect_time = _t['im_detect'].toc(average=False)

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :, :]  # [top_k, 5]
                mask = dets[:, 0].gt(0.).expand(15, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 15)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:5]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32, copy=False)
                all_boxes[j][i] = cls_dets

            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time), end='\r')

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1  # +1 for background
    print(num_classes)
    net = build_ssd_gmm('test', size=300, num_classes=num_classes)
    net = nn.DataParallel(net)
    print(torch.cuda.is_available())

    if args.trained_model:
        print("Loading weight:", args.trained_model)
        ckp = torch.load(args.trained_model)
        net.load_state_dict(ckp['weight'] if 'weight' in ckp.keys() else ckp)
    net.eval()
    print('Finished loading model!')

    # test on coco2017 VAL set (5000 images)
    testset = COCODetection_eval(args.dataset_root, [('2017', 'val')], None)

    # save the test result here (those detected bounding box, etc.)
    save_folder = os.path.join(args.save_folder, 'coco')
    test_save_dir = os.path.join(save_folder, args.model_type)

    test_net(test_save_dir, net, args.cuda, testset,
             BaseTransform(300, rgb_means=(123, 117, 104), rgb_std=(1, 1, 1), swap=(2, 0, 1)))
