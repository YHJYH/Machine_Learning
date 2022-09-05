### remove ~10% params or 4 conv layers
5 exps: [0.7314, 0.7350, 0.7105, 0.7353, 0.7189] mean±std = (0.7262,0.0099)

time: (36.602819999999994, 0.6637402787235366)

```
Total params: 29,939,392
Trainable params: 29,939,392
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.61
Params size (MB): 114.21
Estimated Total Size (MB): 115.83
----------------------------------------------------------------
-data_type=cifar10 -model_type=vgg16_N -learning_rate=0.001 -momentum=0.9 -num_epoch=50 -patience=50
```
VGG16_N model (VGG12)<br>
%reduced params = (33638218-29939392)/33638218\*100 = 10.9959%<br>
num. of reduced params = 33638218-29939392 = 3698826
```
def forward(self, x):
        
        x1 = self.conv1(x)
        
        x2 = F.relu(x1)
        x3 = self.conv2(x2)
        
        x4 = F.relu(x3)
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
        
        x28 = F.relu(x27)
        x29 = self.conv13(x28)
        
        x30 = F.relu(x29)
        x31 = F.max_pool2d(x30, kernel_size=2, stride=2)
        x32 = torch.reshape(torch.flatten(x31), (-1, 512))
        x33 = self.fc1(x32)
        
        x34 = F.relu(x33)
        x35 = self.fc2(x34)
        
        x36 = F.relu(x35)
        x37 = self.fc3(x36)
        
        x38 = F.log_softmax(x37, dim=1)
        
        feature_map = [x1, x3, x6, x11, x18, x22, x25, x27, x29, x33, x35, x37]
        
        return (feature_map, x38)
    # total params: 29,939,392
```
train_model2_x
```
2_1
Train loss: 0.840259, Valid loss: 0.978522
Updating model file...
Early stopping at: Epoch 13

2_2
Train loss: 0.601131, Valid loss: 0.943132
Updating model file...
Early stopping at: 16

2_3
Train loss: 0.615519, Valid loss: 0.964807
Updating model file...
Early stopping at: 16
----------------------------------------------
2_4
Train loss: 0.593968, Valid loss: 0.939733
Updating model file...
Early stopping at: 16
----------------------------------------------
2_5
Train loss: 0.494378, Valid loss: 0.967905
Updating model file...
Early stopping at: 17
```
features2_x.pt
```
2_1
Test average loss: 2.4430, acc: 0.7314
Test time: 37.4056 s
----------
2_2
Test average loss: 2.3425, acc: 0.7350
Test time: 35.9279 s
----------
2_3
Test average loss: 2.0945, acc: 0.7105
Test time: 35.7321 s
----------
2_4
Test average loss: 2.3861, acc: 0.7353
Test time: 37.1596 s
----------
2_5
Test average loss: 2.4244, acc: 0.7189
Test time: 36.7889 s
----------
```
CKA Linear plot (averaged over 5 exps): <br>
![vgg12_linear](vgg12_linear.png)

CKA RBF plot (averaged over 5 exps): <br>
![vgg12_rbf](vgg12_rbf.png)

CKA (Linear) plot:<br>
![2_1linear](2_1linear.png) ![2_2linear](2_2linear.png) ![2_3linear](2_3linear.png) ![2_4linear](2_4linear.png) ![2_5linear](2_5linear.png)

CKA (RBF) plot:<br>
![2_1rbf](2_1rbf.png) ![2_2rbf](2_2rbf.png) ![2_3rbf](2_3rbf.png) ![2_4rbf](2_4rbf.png) ![2_5rbf](2_5rbf.png)
