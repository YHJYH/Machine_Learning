# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:09:55 2022

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

__all__ = ["ResNet",
           "resnet18",
           "resnet101"]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)    
        
        return out
        
class ResNet(nn.Module):
        
    def __init__(self, block, layers, num_classes=4): # change num_classes from 2 to 4 for four coords in object detection bbox case
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # initialize weights (mean, std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) # initialize weight to one
                m.bias.data.zero_() # initialize bias to zero
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        out = self.avgpool(x4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out

def _resnet(block, layers, num_classes, pretrained, PATH='saved_model_binary.pt', device=torch.device('cpu')):
    """
    block: BasicBlock or Bottleneck,
    layers: List[int],
    pretrained: Bool,
    PATH: Str, in torch.save(model.state_dict(), PATH)
    device: torch.device('cpu') or torch.device('cuda')
    """
    model = ResNet(block=block, layers=layers, num_classes=num_classes)
    if pretrained is True:
        if device == torch.device('cpu'):
            model.load_state_dict(torch.load(PATH, map_location=device)) # save on CPU and load on CPU
        elif device == torch.device('cuda'):
            model.load_state_dict(torch.load(PATH)) # save on GPU and load on GPU
            model.to(device)
        # other case: save on CPU load on GPU
    return model
    

def resnet18(pretrained=False):
    return _resnet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1, pretrained=pretrained)

def resnet101(pretrained=False):
    return _resnet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=1, pretrained=pretrained)

def resnet34(pretrained=False):
    return _resnet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=4, pretrained=pretrained) # change corresponding num_classes

def resnet50(pretrained=False):
    return _resnet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1, pretrained=pretrained)

def resnet152(pretrained=False):
    return _resnet(block=Bottleneck, layers=[3, 8, 36, 3], num_classes=1, pretrained=pretrained)

        
if __name__ == "__main__":
    
    #resnet18_model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
    #summary(resnet18_model, (3, 256, 256))
    #resnet101_model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])
    # print(resnet101_model)
    #summary(resnet101_model, (3, 256, 256))
    resnet34_model = resnet34()
    summary(resnet34_model, (3, 256, 256))
