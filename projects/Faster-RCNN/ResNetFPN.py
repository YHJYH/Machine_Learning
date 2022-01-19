# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:16:36 2022

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
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock
from _utils import IntermediateLayerGetter
from torchvision.ops import misc
import resnet

class BackboneWithFPN(nn.Module):
    # Feature Pyramid Network
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None):
        """
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int]
        out_channels: int,
        extra_blocks: ExtraFPNBlock = None
        """
        super(BackboneWithFPN, self).__init__()
        
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
            
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, 
                                         out_channels=out_channels, 
                                         extra_blocks=extra_blocks
                                         )
        self.out_channels = out_channels  
        
    def forward(self, x):
        x = self.body(x)
        
        x = self.fpn(x)
        return x

def resnet_fpn_backbone(backbone_name, pretrained, trainable_layers=3,  # norm_layer=misc.FrozenBatchNorm2d, 
                        returned_layers=None, extra_blocks=None):
    """
    backbone_name: str,
    pretrained: bool,
    """
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained) #  norm_layer=norm_layer otherwise: TypeError: resnet18() got an unexpected keyword argument 'norm_layer'
    return _resnet_fpn_extractor(backbone, trainable_layers, returned_layers, extra_blocks)
        
def _resnet_fpn_extractor(backbone, trainable_layers, returned_layers, extra_blocks):
    assert 0 <= trainable_layers <= 5
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters(): # iterator over module parameters
        if all([not name.startswith(layer) for layer in layers_to_train]): # Check if all items in a list are True.
            parameter.requires_grad_(False) # If autograd should record operations on this tensor.
            
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool() # Applies a max_pool2d on top of the last feature map.
        
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)} # f-string
    
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i-1) for i in returned_layers]
    out_channels = 512 # 256 change to 512
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

if __name__ == "__main__":
    """
    resnet18_model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
    summary(resnet18_model, (3, 256, 256))
    resnet101_model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])
    # print(resnet101_model)
    summary(resnet101_model, (3, 256, 256))
    """
    # test 
    backbone = resnet_fpn_backbone('resnet18', pretrained=False)
    x = torch.rand(1,3,64,64)
    output = backbone(x)
    print([(k, v.shape) for k, v in output.items()])
    
    