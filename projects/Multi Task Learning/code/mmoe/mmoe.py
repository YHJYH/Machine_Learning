import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import box_iou

import os
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import *

class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
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

class ContextGating(nn.Module):
    def __init__(self, size):
        super(ContextGating, self).__init__()
        
        self.cg_layer = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.sigmoid(self.cg_layer(x))
        return weight * x
    
class ExpertCla(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ExpertCla, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

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
        
        return x3
    
    


class TowerCla(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(TowerCla, self).__init__()
        self.inplanes = 256 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.context_gating1 = ContextGating(512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.context_gating2 = ContextGating(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # initialize weights (mean, std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) # initialize weight to one
                m.bias.data.zero_() # initialize bias to zero

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
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



    def forward(self, x3):
        x4 = self.layer4(x3)
        
        out = self.avgpool(x4)
        out = out.view(out.size(0), -1)
        out = self.context_gating1(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.context_gating2(out)
        out = self.fc3(out)
        
        return out
    

class ExpertBox(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ExpertBox, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

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
        
        return x3
    
    


class TowerBox(nn.Module):
    def __init__(self, block, layers, num_classes=4):
        super(TowerBox, self).__init__()
        self.inplanes = 256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.context_gating1 = ContextGating(512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.context_gating2 = ContextGating(64)
        self.fc3 = nn.Linear(64, num_classes)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # initialize weights (mean, std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) # initialize weight to one
                m.bias.data.zero_() # initialize bias to zero

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
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

    def forward(self, x3):
        x4 = self.layer4(x3)
        out = self.avgpool(x4)
        out = out.view(out.size(0), -1)
        out = self.context_gating1(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.context_gating2(out)
        out = self.fc3(out)
        
        return out

class ExpertUnet(nn.Module):
    def __init__(self, n_channels):
        super(ExpertUnet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]



class TowerUnet(nn.Module):
    def __init__(self, n_classes):
        super(TowerUnet, self).__init__()
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
    
    def forward(self, x):
        x_ = self.up1(x[4], x[3])
        x_ = self.up2(x_, x[2])
        x_ = self.up3(x_, x[1])
        x_ = self.up4(x_, x[0])
        x_ = self.outc(x_)
        x_ = nn.Softmax(dim=1)(x_)
        return x_


class Gate(nn.Module):
    def __init__(self, in_channel, out_channel, num_task=3):
        super(Gate, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel*4, kernel_size=8)
        self.bn2 = nn.BatchNorm2d(out_channel*4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cg1 = ContextGating(out_channel*4)
        self.fc = nn.Linear(out_channel*4, num_task)
        self.cg2 = ContextGating(num_task)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x_fc = self.avgpool(x2)
        x_fc = torch.squeeze(x_fc)
        x = self.cg1(x_fc)
        x = self.fc(x)
        x = self.cg2(x)
        x_out = nn.Softmax(dim=1)(x)

        return x_out



class MMoE(nn.Module):
    def __init__(self, num_expert=3, num_gate=3):
        super(MMoE, self).__init__()
        self.num_expert = num_expert
        self.num_gate = num_gate

        self.expert_cla = ExpertCla(BasicBlock, layers=[2, 2, 2, 2])
        self.tower_cla = TowerCla(BasicBlock, layers=[2, 2, 2, 2])

        self.expert_Unet = ExpertUnet(n_channels=3)
        self.tower_Unet = TowerUnet(n_classes=2)

        self.expert_box = ExpertBox(BasicBlock, layers=[3, 4, 6, 3])
        self.tower_box = TowerBox(BasicBlock, layers=[3, 4, 6, 3])

        self.gate_cla = Gate(3, 64)
        self.gate_Unet = Gate(3, 64)
        self.gate_box = Gate(3, 64)

        self.tranconv_cla = nn.Conv2d(256, 1024, kernel_size=1)
        self.bn_cla = nn.BatchNorm2d(1024)

        self.tranconv_U_cla = nn.Conv2d(1024, 256, kernel_size=1)
        self.bn_U_cla = nn.BatchNorm2d(256)

        self.tranconv_U_box = nn.Conv2d(1024, 256, kernel_size=1)
        self.bn_U_box = nn.BatchNorm2d(256)

        self.tranconv_box = nn.Conv2d(256, 1024, kernel_size=1)
        self.bn_box = nn.BatchNorm2d(1024)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        gate_cla = self.gate_cla(input)
        gate_Unet = self.gate_Unet(input)
        gate_box = self.gate_box(input)

        expert_cla = self.expert_cla(input)
        expert_Unet = self.expert_Unet(input)
        expert_box = self.expert_box(input)
        
        #transition of output of classification to Unet
        expert_cla_tran = self.relu(self.bn_cla(self.tranconv_cla(expert_cla)))
        #transition of output of Unet to classification
        expert_Unet_tran_cla = self.relu(self.bn_U_cla(self.tranconv_U_cla(expert_Unet[-1])))
        #transition of output of Unet to box
        expert_Unet_tran_box = self.relu(self.bn_U_box(self.tranconv_U_box(expert_Unet[-1])))
        #transition of output of box to Unet
        expert_box_tran = self.relu(self.bn_box(self.tranconv_box(expert_box)))

        #open up transformation of output of tasks
        expert_cla_tran = torch.unsqueeze(expert_cla_tran, 1)
        expert_Unet_tran_cla = torch.unsqueeze(expert_Unet_tran_cla, 1)
        expert_Unet_tran_box = torch.unsqueeze(expert_Unet_tran_box, 1)
        expert_box_tran = torch.unsqueeze(expert_box_tran, 1)

        #open up outputs of tasks
        expert_cla = torch.unsqueeze(expert_cla, 1)
        expert_Unet[-1] = torch.unsqueeze(expert_Unet[-1], 1)
        expert_box = torch.unsqueeze(expert_box, 1)

        size_cla = torch.cat((expert_cla, expert_Unet_tran_cla, expert_box), 1).size()
        size_U = torch.cat((expert_Unet[-1], expert_cla_tran, expert_box_tran), 1).size()
        size_box = torch.cat((expert_box, expert_Unet_tran_box, expert_box), 1).size()

        gate_cla = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(gate_cla, 2), 3), 4)
        gate_Unet = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(gate_Unet, 2), 3), 4)
        gate_box = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(gate_box, 2), 3), 4)

        gate_cla = gate_cla.repeat(1, 1, size_cla[2], size_cla[3], size_cla[4])
        gate_Unet = gate_Unet.repeat(1, 1, size_U[2], size_U[3], size_U[4])
        gate_box = gate_box.repeat(1, 1, size_box[2], size_box[3], size_box[4])

        expert_out_cla = torch.sum(gate_cla * torch.cat((expert_cla, expert_Unet_tran_cla, expert_box), 1), 1)
        expert_out_Unet = torch.sum(gate_Unet * torch.cat((expert_Unet[-1], expert_cla_tran, expert_box_tran), 1), 1)
        expert_out_box = torch.sum(gate_box * torch.cat((expert_box, expert_Unet_tran_box, expert_cla), 1), 1)

        expert_Unet[-1] = expert_out_Unet

        tower_Unet = self.tower_Unet(expert_Unet)
        tower_cla = self.tower_cla(expert_out_cla)
        tower_box = self.tower_box(expert_out_box)


        return tower_Unet, tower_cla, tower_box

    
def get_iou(predict, label):
    predict_f = torch.flatten(predict)
    label_f = torch.flatten(label)
    intersection = torch.sum(predict_f*label_f)
    iou = intersection/(torch.sum(predict_f) + torch.sum(label_f) - intersection)
    return iou

if __name__ == '__main__':
    # choose device
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load data
    train_set, trainloader = load_data('../datasets-oxpet/data_new/train', batch_size=8, num_workers=2)
    val_set, valloader = load_data('../datasets-oxpet/data_new/val', batch_size=8, num_workers=2)
    test_set, testloader = load_data('../datasets-oxpet/data_new/test', batch_size=8, num_workers=2)
    
    # Run the model
    mmoe = MMoE().to(device)
    
    # loss and optimiser
    gate_cla_params = list(map(id, mmoe.gate_cla.parameters()))
    gate_seg_params = list(map(id, mmoe.gate_Unet.parameters()))
    gate_box_params = list(map(id, mmoe.gate_box.parameters()))
    expert_cla_params = list(map(id, mmoe.expert_cla.parameters()))
    expert_Unet_params = list(map(id, mmoe.expert_Unet.parameters()))
    expert_box_params = list(map(id, mmoe.expert_box.parameters()))
    tower_cla_params = list(map(id, mmoe.tower_cla.parameters()))
    tower_Unet_params = list(map(id, mmoe.tower_Unet.parameters()))
    tower_box_params = list(map(id, mmoe.tower_box.parameters()))
    base_params = filter(lambda p: id(p) not in gate_cla_params + gate_seg_params + gate_box_params + 
                        expert_cla_params + expert_Unet_params + expert_box_params + 
                        tower_cla_params + tower_Unet_params + tower_box_params, 
                        mmoe.parameters())
    criterion_Unet = torch.nn.CrossEntropyLoss()
    criterion_cla = nn.BCEWithLogitsLoss()
    criterion_box = nn.SmoothL1Loss()
    # In training, it turns out that manually modification of lr will let the result performs better
    # After 60 epochs, set all learning rate to its 1/10 (5e-6)
    lr = 5e-5
    optimizer = optim.Adam([
                {'params': base_params},
                {'params': mmoe.gate_cla.parameters(), 'lr': lr},
                {'params': mmoe.gate_Unet.parameters(), 'lr': lr},
                {'params': mmoe.gate_box.parameters(), 'lr': lr},
                {'params': mmoe.expert_cla.parameters(), 'lr': lr},
                {'params': mmoe.tower_cla.parameters(), 'lr': lr},
                {'params': mmoe.expert_Unet.parameters(), 'lr': lr},
                {'params': mmoe.tower_Unet.parameters(), 'lr': lr}, 
                {'params': mmoe.expert_box.parameters(), 'lr': 1e-4},
                {'params': mmoe.tower_box.parameters(), 'lr': 1e-4}],              
                lr=lr, eps=1e-5)

    epochs = 100
    train_acc_list = []
    train_loss_seg_list = []
    train_loss_cla_list = []
    train_loss_box_list = []
    train_iou = []
    train_bbox_iou_list = []
    test_acc_list = []
    test_loss_seg_list = []
    test_loss_cla_list = []
    test_loss_box_list = []
    test_iou = []
    test_bbox_iou_list = []

    # train
    for epoch in range(epochs):
        mmoe.train()
        loss_avg_seg = 0
        loss_avg_cla = 0
        loss_avg_box = 0
        correct = 0.
        total = 0.
        iou_avg = 0.
        bbox_iou_avg = 0.
        progress_bar = tqdm(trainloader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch+1))
            images = images.to(device)
            masks = torch.squeeze(labels[0]).to(device)
            binarys = labels[2].to(device)
            boxes = labels[1].to(device)
            optimizer.zero_grad()
            output_seg, output_cla, output_box = mmoe(images)

            _, predicted_masks = torch.max(output_seg.data, 1)
            
            loss_seg = criterion_Unet(output_seg, masks.long())
            loss_cla = criterion_cla(output_cla, binarys.type(torch.cuda.FloatTensor))
            loss_box = criterion_box(output_box, boxes.long())

            # loss = loss_seg/(loss_seg.detach() + eps)  + loss_cla/(loss_cla.detach() + eps)
            loss = (1/3) * loss_seg + (1/3) * loss_cla + (1/3) * loss_box
            loss.backward()
            optimizer.step()

            loss_avg_seg += loss_seg.item()
            loss_avg_cla += loss_cla.item()
            loss_avg_box += loss_box.item()

            
            seg_iou = get_iou(predicted_masks, masks)
            iou_avg += seg_iou.item()
            bbox_iou_metric = box_iou(output_box, boxes)
            bbox_iou = torch.mean(torch.diagonal(bbox_iou_metric)).item()
            bbox_iou_avg += bbox_iou

            pred = (output_cla > 0.5) + 0
            total += binarys.size(0)
            correct += (pred == binarys.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(
                    loss_seg='%.3f' % (loss_avg_seg / (i + 1)),
                    loss_cla='%.3f' % (loss_avg_cla / (i + 1)),
                    loss_box='%.3f' % (loss_avg_box / (i + 1)),
                    acc='%.3f' % accuracy,
                    seg_iou='%.3f' % (iou_avg / (i + 1)),
                    bbox_iou='%.3f' % (bbox_iou_avg / (i + 1)))
            
        train_acc_list.append(accuracy)
        train_loss_seg_list.append(loss_avg_seg / (i + 1))
        train_loss_cla_list.append(loss_avg_cla / (i + 1))
        train_loss_box_list.append(loss_avg_box / (i + 1))
        train_iou.append(iou_avg / (i + 1))
        train_bbox_iou_list.append(bbox_iou_avg.item() / (i + 1))


        with torch.no_grad():
            mmoe.eval()
            loss_avg_seg = 0
            loss_avg_cla = 0
            loss_avg_box = 0
            correct = 0.
            total = 0.
            iou_avg = 0.
            bbox_iou_avg = 0.
            progress_bar_test = tqdm(testloader)
            for i, (images, labels) in enumerate(progress_bar_test):
                progress_bar_test.set_description('Test Epoch ' + str(epoch+1))
                images = images.to(device)
                masks = torch.squeeze(labels[0]).to(device)
                binarys = labels[2].to(device)
                boxes = labels[1].to(device)
                output_seg, output_cla, output_box = mmoe(images)
                _, predicted_masks = torch.max(output_seg.data, 1)
                iou = get_iou(predicted_masks, masks)
                iou_avg += iou.item()
                bbox_iou_metric = box_iou(output_box, boxes)
                bbox_iou = torch.mean(torch.diagonal(bbox_iou_metric)).item()
                bbox_iou_avg += bbox_iou

                pred = (output_cla > 0.5) + 0
                total += binarys.size(0)
                correct += (pred == binarys.data).sum().item()
                accuracy = correct / total

                loss_seg = criterion_Unet(output_seg, masks.long())
                loss_cla = criterion_cla(output_cla, binarys.type(torch.cuda.FloatTensor))
                loss_box = criterion_box(output_box, boxes.long())

                loss_avg_seg += loss_seg.item()
                loss_avg_cla += loss_cla.item()
                loss_avg_box += loss_box.item()


                progress_bar_test.set_postfix(
                        test_loss_seg='%.3f' % (loss_avg_seg / (i + 1)),
                        test_loss_cla='%.3f' % (loss_avg_cla / (i + 1)),
                        test_loss_box='%.3f' % (loss_avg_box / (i + 1)),
                        test_acc='%.3f' % accuracy,
                        test_iou='%.3f' % (iou_avg / (i + 1)),
                        test_bbox_iou='%.3f' % (bbox_iou_avg / (i + 1)))
                
            test_acc_list.append(accuracy)
            test_loss_seg_list.append(loss_avg_seg / (i + 1))
            test_loss_cla_list.append(loss_avg_cla/ (i + 1))
            test_loss_box_list.append(loss_avg_box / (i + 1))
            test_iou.append(iou_avg / (i + 1))
            test_bbox_iou_list.append(bbox_iou_avg.item() / (i + 1))

    print('Training done.')
    torch.save(mmoe.state_dict(), 'saved_mmoe.pt')
    
    plt.plot(np.arange(100), train_acc_list, color='b', label='training accuracy of classification', linestyle="-")
    plt.plot(np.arange(100), test_acc_list, color='b', label='testing accuracy of classification', linestyle="--")
    plt.plot(np.arange(100), train_iou, color='orange', label='training IoU of segmentation', linestyle='-')
    plt.plot(np.arange(100), test_iou, color='orange', label='testing IoU of segmentation', linestyle='--')
    plt.plot(np.arange(100), train_bbox_iou_list, color='green', label='training IoU of bounding box', linestyle='-')
    plt.plot(np.arange(100), test_bbox_iou_list, color='green', label='testing IoU of bounding box', linestyle='--')
    plt.legend()
    plt.grid()
    plt.ylabel('Metric')
    plt.xlabel('Epochs')
    plt.title("Metric vs Epochs(100) of MMoE")
    plt.savefig('mmoe.png', dpi=500)
    plt.show()