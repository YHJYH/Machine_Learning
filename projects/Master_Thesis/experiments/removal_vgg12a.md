# vgg12a remove 1fc and 3 convs
acc. = [0.7173, 0.7243, 0.7038, 0.7219, 0.7024] mean:0.7139 std: 0.0091

remained: 13752000/33638218\*100 = 40.8820705% <br>
reduced: 59.1179295%
```
Total params: 13,752,000
Trainable params: 13,752,000
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.75
Params size (MB): 52.46
Estimated Total Size (MB): 54.22
----------------------------------------------------------------
```

train_model14_x
```
14_1
Train loss: 0.651281, Valid loss: 0.892203
Updating model file...
Early stopping at: 16
----------------------------------------------
14_2
Train loss: 0.639813, Valid loss: 0.859704
Updating model file...
Early stopping at: 16
----------------------------------------------
14_3
Train loss: 0.725766, Valid loss: 0.895957
Updating model file...
Early stopping at: 15
----------------------------------------------
14_4
Train loss: 0.637085, Valid loss: 0.911723
Updating model file...
Early stopping at: 16
----------------------------------------------
14_5
Train loss: 0.695956, Valid loss: 0.878469
Updating model file...
Early stopping at: 15
----------------------------------------------
```

features14_x
```
14_1
Test average loss: 1.0495, acc: 0.7173
----------
14_2
Test average loss: 1.0757, acc: 0.7243
----------
14_3
Test average loss: 1.0432, acc: 0.7038
----------
14_4
Test average loss: 1.0400, acc: 0.7219
----------
14_5
Test average loss: 1.0353, acc: 0.7024
----------
```

CKA linear avg:<br>
![vgg12a_linear](vgg12a_linear.png)

CKA RBF avg: <br>
![vgg12a_rbf](vgg12a_rbf.png)

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
        
        x14 = F.relu(x11)
        x15 = self.conv7(x14)
        
        x16 = F.relu(x15)
        x17 = F.max_pool2d(x16, kernel_size=2, stride=2)
        x18 = self.conv8(x17)
        
        x19 = F.relu(x18)
        x20 = self.conv9(x19)
        
        x21 = F.relu(x20)
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
        
        #x34 = F.relu(x33)
        #x35 = self.fc2(x34)
        
        x36 = F.relu(x33)
        x37 = self.fc3(x36)
        
        x38 = F.log_softmax(x37, dim=1)
        
        feature_map = [x1, x3, x6, x11, x15, x18, x20, x22, x25, x27, x33, x37]
        
        return (feature_map, x38)
```
