import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import *

def get_iou(predict, label):
    predict_f = torch.flatten(predict)
    label_f = torch.flatten(label)
    intersection = torch.sum(predict_f*label_f)
    iou = intersection/(torch.sum(predict_f) + torch.sum(label_f) - intersection)
    return iou

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

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = F.log_softmax(x, 1)
        return x



if __name__ == '__main__':
    # choose the device
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load data
    train_set, trainloader = load_data('../datasets-oxpet/data_new/train', batch_size=16, num_workers=2)
    val_set, valloader = load_data('../datasets-oxpet/data_new/val', batch_size=16, num_workers=2)
    test_set, testloader = load_data('../datasets-oxpet/data_new/test', batch_size=16, num_workers=2)
    
    # Run the model
    model = UNet(n_channels=3, n_classes=2).to(device)

    # loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, eps=1e-5)

    train_loss_list = []
    train_iou_list = []
    test_loss_list = []
    test_iou_list = []
    epochs = 100

    # train
    for epoch in range(epochs):
        iou_avg = 0.
        loss_avg = 0
        progress_bar = tqdm(trainloader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch+1))
            images = images.to(device)
            masks = torch.squeeze(labels[0]).to(device)
            optimizer.zero_grad()
            output =  model(images)
            _, predicted_masks = torch.max(output.data, 1)
            loss = criterion(output, masks.long())
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()

            iou = get_iou(predicted_masks, masks)
            iou_avg += iou.item()
            progress_bar.set_postfix(
                    loss='%.3f' % (loss_avg / (i + 1)),
                    iou='%.3f' % (iou_avg / (i + 1)))
            
        train_loss_list.append(loss_avg/(i+1))
        train_iou_list.append(iou_avg/(i+1))

        # monitor test performance per epoch
        with torch.no_grad():
            correct = 0
            total = 0
            iou_avg = 0.
            loss_avg = 0.
            progress_bar_test = tqdm(testloader)
            for i, (images, labels) in enumerate(progress_bar_test):
                progress_bar_test.set_description('Test Epoch ' + str(epoch+1))
                images = images.to(device)
                masks = torch.squeeze(labels[0]).to(device)
                output = model(images)
                _, predicted_masks = torch.max(output.data, 1)
                loss = criterion(output, masks.long())
                loss_avg += loss.item()
                iou = get_iou(predicted_masks, masks)
                iou_avg += iou.item()

                progress_bar_test.set_postfix(
                        test_loss='%.3f' % (loss_avg / (i + 1)),
                        test_iou='%.3f' % (iou_avg / (i + 1)))
                
            test_loss_list.append(loss_avg/(i+1))
            test_iou_list.append(iou_avg/(i+1))
            
    print('Training done.')        
    torch.save(model.state.dict(), 'saved_model_seg.pt')
    print("Model Saved.")
    
    plt.plot(np.arange(len(train_iou_list)), train_iou_list, color='b', label='training iou of segmentation', linestyle="-")
    plt.plot(np.arange(len(test_iou_list)), test_iou_list, color='b', label='testing iou of segmentation', linestyle="--")
    plt.legend()
    plt.grid()
    plt.ylabel('IoU')
    plt.xlabel('Epochs')
    plt.title("IoU vs Epochs(100) of segmentation")
    plt.savefig('iou_seg.png', dpi=500)
    plt.show()