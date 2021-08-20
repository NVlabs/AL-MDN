# Updated for running Gaussian Mixture Model
# Adapted from https://github.com/amdegroot/ssd.pytorch


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
from ssd_gmm import build_ssd_gmm
from matplotlib import pyplot as plt
from data import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT, COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector demo')
parser.add_argument('--trained_model', default='weights/trained_VOC_name.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='VOC dataset root directory path')
parser.add_argument('--coco_root', default=COCO_ROOT,
                    help='COCO dataset root directory path')
args = parser.parse_args()


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# initialize SSD
if args.dataset == 'VOC':
    net = build_ssd_gmm('test', 300, 21)
else:
    net = build_ssd_gmm('test', 300, 81)
net = nn.DataParallel(net)
net.load_state_dict(torch.load(args.trained_model))

# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
if args.dataset == 'VOC':
    testset = VOCDetection(args.voc_root, [('2007', 'trainval')], None, VOCAnnotationTransform())
else:
    testset = COCODetection(args.coco_root, 'train2014', None, VOCAnnotationTransform())

img_id = 7
image = testset.pull_image(img_id)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# View the sampled input image before transform
plt.figure(figsize=(10,10))
plt.imshow(rgb_image)

x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
plt.imshow(x)
x = torch.from_numpy(x).permute(2, 0, 1)

# wrap tensor in Variable
xx = Variable(x.unsqueeze(0))
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)

top_k=10
plt.figure(figsize=(10,10))
if args.dataset == 'VOC':
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
else:
    colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.5:
        score = detections[0,i,j,0]
        if args.dataset == 'VOC':
            label_name = VOC_CLASSES[i-1]
        else:
            label_name = COCO_CLASSES[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:5]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        print(score)
        print(pt[0], pt[1], pt[2], pt[3])
        print('----')
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1
plt.savefig('out_img.png')
