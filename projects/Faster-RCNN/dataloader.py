# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:39:30 2022

@author: Yuheng
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

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
     ])

class MyDataset(Dataset):
    def __init__(self, root_path, label_type, transform):
        self.root_path = root_path
        self.label_type = label_type
        self.transform = transform
        
        self.img_path = self.root_path + '/images.h5'
        self.label_path = [self.root_path + '/' + self.label_type[i] + '.h5' for i in range(len(self.label_type))]
        
    def load_data_from_h5(self, path, index):
        with h5py.File(path, 'r') as file:
            key = list(file.keys())[0] # ['images'][0]
            elems = file[key][index]
        return elems
    
    def __getitem__(self, index):
        # The __getitem__ function loads and returns a sample from the dataset at the given index.
        images = self.load_data_from_h5(self.img_path, index)
        #images = self.transform(np.array(images, dtype=np.uint8))
        labels = [self.load_data_from_h5(self.label_path, index) for i in range(len(self.label_path))]
        if self.transform:
            images = self.transform(images)
        
        return images, labels
    
    def __len__(self):
        # The __len__ function returns the number of samples in our dataset.
        with h5py.File(self.img_path, 'r') as file:
            key = list(file.keys())[0]
            lens = file[key].len()
        return int(lens)



"""   
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1)/2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None
        
    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, bn=False):
        super(VGG16, self).__init__()
"""
    
if __name__ == '__main__':
    
    label_type = ['binary', 'bboxes', 'masks']
    train_set = MyDataset(root_path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\train', label_type=label_type, transform=transform)
    test_set = MyDataset(root_path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\test', label_type=label_type, transform=transform)
    val_set = MyDataset(root_path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\val', label_type=label_type, transform=transform)

    trainloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    testloader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=2)
    valloader = DataLoader(val_set, batch_size=16, shuffle=True, num_workers=2)

    num_examples = 4
    img_example = train_set.load_data_from_h5(path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\train\images.h5', index=np.arange(num_examples))
    bbox_example = train_set.load_data_from_h5(path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\train\bboxes.h5', index=np.arange(num_examples))
    # bbox shape: [x1, y1, x2, y2], top-left and bottom-right corners.
    # constrains: 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
    binary_example = train_set.load_data_from_h5(path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\train\binary.h5', index=np.arange(num_examples))
    mask_example = train_set.load_data_from_h5(path=r'E:\UCL Postgraduate\0090 Intro to DL\CW2\train\masks.h5', index=np.arange(num_examples))
    binary_label = ["dog", "cat"]
    img = Image.fromarray(np.concatenate([img_example[i,...] for i in range(num_examples)],axis=1).astype(np.uint8))
    i = 0
    while i < num_examples:
        offset = i * 256
        # shape = [(bbox_example[i][:2][0]+offset,bbox_example[i][:2][1]), (bbox_example[i][2]+offset, bbox_example[i][3])]
        # bbox_example[i][0]+, bbox_example[i][1]+
        shape = [(bbox_example[i][0]+offset,bbox_example[i][1]), (bbox_example[i][2]+offset, bbox_example[i][3])]
        bbox = ImageDraw.Draw(img)
        bbox.rectangle(shape, fill=None, outline='red')
        label_index = int(binary_example[i][0])
        ImageDraw.Draw(img).text((256*i, 0), "label:{}".format(binary_label[label_index]), (0, 0, 0))
        i += 1
    img.show()
    