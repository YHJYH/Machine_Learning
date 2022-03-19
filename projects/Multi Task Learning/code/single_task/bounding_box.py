import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import box_iou

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from data_loader import *


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
        
    def __init__(self, block, layers, num_classes=4):
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
        

if __name__ == '__main__':
    # choose device
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load data
    train_set, trainloader = load_data('../datasets-oxpet/data_new/train', batch_size=16, num_workers=2)
    val_set, valloader = load_data('../datasets-oxpet/data_new/val', batch_size=16, num_workers=2)
    test_set, testloader = load_data('../datasets-oxpet/data_new/test', batch_size=16, num_workers=2)
    
    # Run the model
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3]).to(device) # resnet34
    optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)
    criterion = nn.SmoothL1Loss()
    
    epochs = 100
    train_loss_list = []
    train_iou_list = []
    test_loss_list = []
    test_iou_list = []
    
    for epoch in range(epochs):
        loss_avg = 0.
        iou_avg = 0.
        progress_bar = tqdm(trainloader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch+1))
            images = images.to(device)
            true_box = labels[1].to(device)
            
            output_box = model(images)
            loss = criterion(output_box, true_box)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()
            iou_metric = box_iou(output_box, true_box)
            iou = torch.mean(torch.diagonal(iou_metric)).item()
            iou_avg += iou

            progress_bar.set_postfix(
                    loss='%.3f' % (loss_avg / (i + 1)),
                    iou='%.3f' % (iou_avg / (i + 1)))
            
        train_loss_list.append(loss_avg/(i+1))
        train_iou_list.append(iou_avg/(i+1))

        # we monitor test performance per epoch
        with torch.no_grad():
            iou_avg = 0.
            loss_avg = 0.
            progress_bar_test = tqdm(testloader)
            for i, (images, labels) in enumerate(progress_bar_test):
                progress_bar_test.set_description('Test Epoch ' + str(epoch+1))
                images = images.to(device)
                true_box = labels[1].to(device)
                
                output_box = model(images)
                loss = criterion(output_box, true_box)
                loss_avg += loss.item()
                iou_metric = box_iou(output_box, true_box)
                iou = torch.mean(torch.diagonal(iou_metric)).item()
                iou_avg += iou
                progress_bar_test.set_postfix(
                        test_loss='%.3f' % (loss_avg / (i + 1)),
                        test_iou='%.3f' % (iou_avg / (i + 1)))
                
            test_loss_list.append(loss_avg/(i+1))
            test_iou_list.append(iou_avg/(i+1))

    print('Training done.')
    torch.save(model.state_dict(), 'saved_model_bbox.pt')
    print("Model Saved.")
    
    # Plot iou figure
    plt.plot(np.arange(len(train_iou_list)), train_iou_list, color='orange', label='training IoU of bounding box', linestyle='-')
    plt.plot(np.arange(len(test_iou_list)), test_iou_list, color='orange', label='testing IoU of bounding box', linestyle='--')
    plt.legend()
    plt.ylabel('Metric')
    plt.xlabel('Epochs')
    plt.grid()
    plt.title("Metric vs Epochs(100) of bounding-box task")
    plt.savefig('IoU_single_bbox', dpi=500)
    plt.show()