#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：CiT-Net 
@File    ：loss.py
@Author  ：songliqiang
@Date    ：2024/3/15 14:44 
'''
import torch
import torch.nn as nn

# 均方误差损失（MSE）
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target):
        return torch.mean((input - target) ** 2)

# Dice损失
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # 计算输入和目标的张量形状的交集
        intersection = (input * target).sum(dim=1).view(-1)
        # 计算输入和目标张量的元素数量
        cardinality = input.view(-1) + target.view(-1)
        # 避免除以0的情况
        cardinality[cardinality == 0] = 1
        # 计算Dice损失
        loss = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - torch.mean(loss)
