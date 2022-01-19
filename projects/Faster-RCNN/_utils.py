# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 18:55:27 2022

@author: Yuhen
"""

from collections import OrderedDict
from torch import nn

class IntermediateLayerGetter(nn.ModuleDict):
    
    def __init__(self, model, return_layers):
        """
        model: nn.Module,
        return_layers: Dict[str, str]
        """
        if not set(return_layers).issubset([name for name, _ in model.named_children()]): # set(Dict): {Dict.keys()}; set(List): {List.elements}
            raise ValueError("return_layers are not present in model")
            
        original_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()} # .items(): [(name k: module v)]
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
            
        super().__init__(layers)
        self.return_layers = original_return_layers
        
    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
    
# def _make_divisible():