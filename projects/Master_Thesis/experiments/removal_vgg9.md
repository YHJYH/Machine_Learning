### remove ~18% params or 7 layers (VGG9)
5 exps: [0.7561, 0.7125, 0.7392, 0.7323, 0.7262] mean±std = (0.7333, 0.0144)
```
Total params: 27,502,272
Trainable params: 27,502,272
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.09
Params size (MB): 104.91
Estimated Total Size (MB): 106.02
----------------------------------------------------------------
```
%reduced params = (33638218-27502272)/33638218\*100 = 18.2410% <br>
num. of reduced params = 33638218-27502272 = 6135946
```
def forward(self, x):
        
        x1 = self.conv1(x)
        
        #x2 = F.relu(x1)
        #x3 = self.conv2(x2)
        
        x4 = F.relu(x1)
        x5 = F.max_pool2d(x4, kernel_size=2, stride=2)
        x6 = self.conv3(x5)
        
        #x7 = F.relu(x6)
        #x8 = self.conv4(x7)
        
        x9 = F.relu(x6)
        x10 = F.max_pool2d(x9, kernel_size=2, stride=2)
        x11 = self.conv5(x10)
        
        #x12 = F.relu(x11)
        #x13 = self.conv6(x12)
        
        #x14 = F.relu(x13)
        #x15 = self.conv7(x14)
        
        x16 = F.relu(x11)
        x17 = F.max_pool2d(x16, kernel_size=2, stride=2)
        x18 = self.conv8(x17)
        
        #x19 = F.relu(x18)
        #x20 = self.conv9(x19)
        
        x21 = F.relu(x18)
        x22 = self.conv10(x21)
        
        x23 = F.relu(x22)
        x24 = F.max_pool2d(x23, kernel_size=2, stride=2)
        x25 = self.conv11(x24)
        
        x26 = F.relu(x25)
        x27 = self.conv12(x26)
        
        #x28 = F.relu(x27)
        #x29 = self.conv13(x28)
        
        x30 = F.relu(x27)
        x31 = F.max_pool2d(x30, kernel_size=2, stride=2)
        x32 = torch.reshape(torch.flatten(x31), (-1, 512))
        x33 = self.fc1(x32)
        
        x34 = F.relu(x33)
        x35 = self.fc2(x34)
        
        #x36 = F.relu(x35)
        #x37 = self.fc3(x36)
        
        x38 = F.log_softmax(x35, dim=1)
        
        feature_map = [x1, x6, x11, x18, x22, x25, x27, x33, x35]
        
        return (feature_map, x38)
    # total params: 27,502,272
```
train_model5_x
```
5_1
Train loss: 0.584666, Valid loss: 0.830359
Updating model file...
Early stopping at: 16

5_2
Train loss: 0.529756, Valid loss: 0.844136
Updating model file...
Early stopping at: 16
----------------------------------------------
5_3
Train loss: 0.689438, Valid loss: 0.869324
Updating model file...
Early stopping at: 14
----------------------------------------------
5_4
Train loss: 0.566496, Valid loss: 0.820620
Updating model file...
Early stopping at: 16
----------------------------------------------
5_5
Train loss: 0.640914, Valid loss: 0.832715
Updating model file...
Early stopping at: 14
----------------------------------------------
```
features5_x.pt
```
5_1
Test average loss: 2.0481, acc: 0.7561

5_2
Test average loss: 1.2710, acc: 0.7125
----------
5_3
Test average loss: 0.9462, acc: 0.7392
----------
5_4
Test average loss: 0.9763, acc: 0.7323
----------
5_5
Test average loss: 0.9981, acc: 0.7262
----------
```
CKA (Linear) plot:<br>
![5_1linear](5_1linear.png) ![5_2linear](5_2linear.png) ![5_3linear](5_3linear.png) ![5_4linear](5_4linear.png) ![5_5linear](5_5linear.png)

CKA (RBF) plot:<br>
![5_1rbf](5_1rbf.png) ![5_2rbf](5_2rbf.png) ![5_3rbf](5_3rbf.png) ![5_4rbf](5_4rbf.png) ![5_5rbf](5_5rbf.png)
