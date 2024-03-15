#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：CiT-Net 
@File    ：train.py
@Author  ：songliqiang
@Date    ：2024/3/15 14:24 
'''
import torch
import torch.nn as nn
from torch.optim import Adam
import loss  as loss
import CiT_Net_T as md
import utils.util as util
import math

def calculate_lambda(delta, epoch):
    """
    计算权重系数λ。
    :param delta: δ值，默认为1。
    :param epoch: 当前epoch数。
    :return: 权重系数λ。
    """
    lambda_coefficient = delta * math.exp(-5 * (1 - epoch) ** 2)
    return lambda_coefficient

delta = 1
epoch = 10  # 当前epoch数

lambda_coefficient = calculate_lambda(delta, epoch)
print("权重系数λ为:", lambda_coefficient)

def train():
    path = 'F:/datasets01/2d'
    num_epochs=10;
    delta = 1
    batchSize = 32
    model =md.CIT().to('cuda')
    optimizer = Adam(model.parameters(), lr=0.001)
    train_image,train_label = util.load_list_data(path,batchSize)
    dataloader = zip(train_image, train_label)
    # MSE损失
    criterion_mse = loss.MSELoss()

    # Dice损失
    criterion_dice = loss.DiceLoss()

    num_BatchSize = train_image.size(0)/batchSize
    for epoch in range(num_epochs):
        for i in range(num_BatchSize):
            for inputs, targets in dataloader:
                inputs = inputs.to('cuda')
                targets = targets.to('cuda')
                start_index = i * batchSize
                end_index = (i + 1) * batchSize
                batch_data = inputs[start_index:end_index]
                targets =targets[start_index:end_index]
                # 前向传播
                Cit_outputs,CNN_out,Trans_out = model(batch_data)

                # # MSE损失
                # loss_mse = criterion_mse(outputs, targets)
                # #
                # # # Dice损失
                # loss_dice = criterion_dice(outputs, targets)

                loss_Tec = criterion_mse(Cit_outputs, targets)+criterion_dice(Cit_outputs, targets)
                loss_cnn =criterion_mse(CNN_out, targets)+criterion_dice(CNN_out, targets)
                loss_tran =criterion_mse(Trans_out, targets)+criterion_dice(Trans_out, targets)
                lambda_coefficient =calculate_lambda(delta,epoch)
                loss_Total =lambda_coefficient*loss_Tec+(((1-lambda_coefficient))/2)*loss_cnn+(((1-lambda_coefficient))/2)*loss_tran
                # 反向传播和优化
                optimizer.zero_grad()
                loss_Total.backward()
                optimizer.step()
                print(f"epoch:{epoch}+loss_sum:{loss_Total.item():.4f} loss_mse:{loss_cnn.item():.4f} loss_dice:{loss_tran.item():.4f}")
        torch.save(model,'model_save/model_epoch{}.pth'.format(epoch))

if __name__ == "__main__":
    train()