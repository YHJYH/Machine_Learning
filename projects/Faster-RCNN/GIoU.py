# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 23:27:23 2022

@author: Yuheng

GIoU: indicated if new, better prediction was closer to the ground truth than the previous prediction, 
even in cases of no intersection.
"""
import torch
from torchvision.ops import generalized_box_iou, box_iou
import numpy as np


def GIOU(pred_box, true_box):
    """
    Parameters
    ----------
    pred_box : Tensor of shape (N, 4)
        N is the batch_size. [x1, y1, x2, y2]
    true_box : Tensor of shape (N, 4).

    Returns
    -------
    giou: IoU - (Ac - U)/Ac where Ac is the area of smallest enclosing box, U is the union area.
    giou_loss: 1 - GIoU.
    
    Note
    -------
    giou is used as a metric, while giou_loss is used as a loss. Ref: https://giou.stanford.edu/
    
    area_pred = box_area(pred_box) # Tensor[N]
    area_true = box_area(true_box) # Tensor[N]
    
    N = len(pred_box)
    inter_box = torch.randn(N, 4)
    
    for i in range(N):
        inter_box[i][0] = torch.max(pred_box[i][0], true_box[i][0])
        inter_box[i][1] = torch.max(pred_box[i][1], true_box[i][1])
        inter_box[i][2] = torch.min(pred_box[i][2], true_box[i][2])
        inter_box[i][3] = torch.min(pred_box[i][3], true_box[i][3])
    
    if (inter_box[:,2:] >= inter_box[:,:2]).all():
        area_inter = box_area(inter_box)
    else:
        area_inter = 0
    
    enclose_box = torch.randn(N, 4)
    for j in range(N):
        enclose_box[j][0] = torch.min(pred_box[i][0], true_box[i][0])
        enclose_box[j][1] = torch.min(pred_box[i][1], true_box[i][1])
        enclose_box[j][2] = torch.max(pred_box[i][2], true_box[i][2])
        enclose_box[j][3] = torch.max(pred_box[i][3], true_box[i][3])
    area_enclose = box_area(enclose_box)
    
    area_union = area_pred + area_true - area_inter
    iou = area_inter - area_union
    giou = iou - (area_enclose - area_union) / area_enclose
    
    giou_loss = 1 - giou
    """
    
    giou_metric = generalized_box_iou(pred_box, true_box)
    giou = torch.mean(torch.diagonal(giou_metric))
    giou_loss = 1 - giou
    iou_metric = box_iou(pred_box, true_box)
    iou = torch.mean(torch.diagonal(iou_metric))
    iou_loss = 1 - iou
    
    
    return giou, giou_loss, iou, iou_loss

