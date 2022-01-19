# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 21:27:18 2022

@author: Yuhen
"""

from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
from torchsummary import summary
from ResNetFPN import *
from dataloader import *
from GIoU import *

transform = transforms.Compose([transforms.ToTensor()]) # image in 0-1 range

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def dimension_convertor(images):
    x = []
    for i in range(len(images)):
        img = np.array([images[i,:,:,j] for j in range(3)])
        img = img / 255.
        x.append(torch.from_numpy(img).double())
    return x

#backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#backbone.out_channels = 1280 # FasterRCNN needs to know the number of output channels in a backbone. For mobilenet_v2, it's 1280

backbone = resnet_fpn_backbone('resnet18', pretrained=True)
backbone.out_channels = 512 # https://stackoverflow.com/questions/58362892/resnet-18-as-backbone-in-faster-r-cnn

anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
anchor_generator = AnchorGenerator(sizes=anchor_sizes, # Tuple[Tuple[int]] # https://github.com/pytorch/vision/issues/3246
                                   aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)) # RPN generate 5 x 3 anchors per spatial location, with 5 different sizes (32,64,128,256,512) and 3 different aspect ratios.

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], # backbone return: OrderedDict[Tensor]
                                                output_size=7, 
                                                sampling_ratio=2)

model = FasterRCNN(backbone,
                   num_classes=2, 
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

# For training
"""
images: List[Tensor], shape=(num_samples, num_channels, H, W)
bboxes: FloatTensor, shape=(num_samples, num_boxes, 4)
labels: Int64Tensor, low=0, high=0, shape=(num_samples, num_boxes)
"""

label_type = ['binary', 'bboxes', 'masks']
total_samples = 2210
train_set = MyDataset(root_path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\train', label_type=label_type, transform=transform)
trainloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)

raw_images = train_set.load_data_from_h5(path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\train\images.h5', index=np.arange(total_samples))
raw_bboxes = train_set.load_data_from_h5(path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\train\bboxes.h5', index=np.arange(total_samples))
# bbox shape: [x1, y1, x2, y2], top-left and bottom-right corners.
# constrains: 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
raw_labels = train_set.load_data_from_h5(path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\train\binary.h5', index=np.arange(total_samples))
binary_label = ["dog", "cat"]

images_basket = dimension_convertor(raw_images)
bboxes_basket = raw_bboxes
labels_basket = raw_labels

def img_convertor(batch):    
    images = torch.rand(10, 3, 256, 256) # batch_size = 10
    for i in range(10):
        images[i] = images_basket[batch*10+i]
        
    images = list(image for image in images)
    return images

def bbox_convertor(batch):
    bboxes = torch.rand(10, 1, 4)
    for i in range(10):
        bboxes[i][0] = torch.from_numpy(bboxes_basket[batch*10+i]).double()
    return bboxes
    
def label_convertor(batch):
    labels = torch.randint(0, 2, (10, 1))
    for i in range(10):
        labels[i] = int(labels_basket[batch*10+i])
    return labels

def target_convertor():
    targets = [] # List[Dict]
    for i in range(len(bboxes)):
        d = {}
        d['boxes'] = bboxes[i]
        d['labels'] = labels[i]
        targets.append(d)
    return targets

# loss and optimiser
criterion = torch.nn.CrossEntropyLoss()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 10
   
# output = model(images, targets)
for epoch in range(10):
    batch = 0
    while batch < total_samples / batch_size:
        images = img_convertor(batch)
        bboxes = bbox_convertor(batch)
        labels = label_convertor(batch)
        targets = target_convertor()
        optimizer.zero_grad()
        output = model(images, targets) # Dict[Tensor]: dict_keys(['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'])
        losses = sum(loss for loss in output.values()) # https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train#Train
        loss_value = losses.item()
        losses.backward()
        optimizer.step()
        print('Epoch {}, batch {}, loss: {}'.format(epoch+1, batch+1, loss_value))
        batch += 1
print('Training done.')

# For inference
model = model.double() # need to be double, not float.
model.eval()

test_set = MyDataset(root_path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\test', label_type=label_type, transform=transform)
testloader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=0)
num_examples = 2
img_example = test_set.load_data_from_h5(path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\test\images.h5', index=np.arange(num_examples))
labels = torch.Tensor([0, 1])
# reshape input: expected to be a list of tensors, each of shape [C, H, W].

x = dimension_convertor(img_example)

#x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

predictions = model(x) # List[Dict["boxes":FloatTensor[N, 4] of [x1, y1, x2, y2], "labels":Int64Tensor[N], "scores":Tensor[N]]]

# draw prediction
pred_bbox = []
pred_scores = []
pred_labels = []
# binary_label = ["dog", "cat"]
for i in range(len(x)):
    bbox = predictions[i]["boxes"][0]
    coord = [(float(bbox[0]),float(bbox[1])),(float(bbox[2]), float(bbox[3]))]
    pred_bbox.append(coord)
    pred_labels.append(binary_label[int(predictions[i]["labels"][0])])
    pred_scores.append(float(predictions[i]["scores"][0]))

#img_example = test_set.load_data_from_h5(path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\test\images.h5', index=np.arange(2))
#img = Image.fromarray(np.concatenate([img_example[i,...] for i in range(2)],axis=1).astype(np.uint8))
j = 0
while j < 2:
    img = Image.fromarray(img_example[j,...].astype(np.uint8))
    #offset = j * 256
    shape = pred_bbox[j]
    # bbox_example[i][0]+, bbox_example[i][1]+
    bbox = ImageDraw.Draw(img)
    bbox.rectangle(shape, fill=None, outline='red')
    ImageDraw.Draw(img).text((256*i, 0), "label:{}, scores:{}".format(pred_labels[j], pred_scores[j]), (0, 0, 0))
    img.show()
    j += 1

#img.show()



     