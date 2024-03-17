#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：CiT-Net 
@File    ：test.py
@Author  ：songliqiang
@Date    ：2024/3/17 11:12 
'''
import os.path

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
    save_path = 'F:/data/'
    list_path =['cit_out','tran_out','cnn_ut']
    # for lpath in list_path:
    #     if os.path.exists(os.path.join(save_path,lpath)):
    #         os.makedirs(os.path.join(save_path,lpath))
    num_epochs=100;
    delta = 1
    batchSize = 32
    assert torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =md.CIT().to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    # train_image = torch.rand(2 , 1, 224, 224).to("cuda")
    # targets = torch.rand(2, 1, 224, 224).to("cuda")
    train_image,targets = util.load_list_data(path,batchSize,1)
    train_image=train_image.to(device)
    targets =targets.to(device)
    # dataloader = zip(train_image, train_label)
    # MSE损失

    # model =CIT().to("cuda")

    # out_result, _, _ = model(input)
    # print(out_result.shape)

    # flops, params = profile(model, (input,))
    criterion_mse = loss.MSELoss().to(device)

    # Dice损失
    criterion_dice = loss.DiceLoss().to(device)

    num_BatchSize = train_image.size(0)/batchSize
    for epoch in range(num_epochs):

        # for inputs, targets in dataloader:
        # inputs = inputs.to('cuda')
        # targets = targets.to('cuda')
        # start_index = i * batchSize
        # end_index = (i + 1) * batchSize
        # batch_data = inputs[start_index:end_index]
        # targets =targets[start_index:end_index]
        # # 前向传播
        Cit_outputs,CNN_out,Trans_out = model(train_image)

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
        if os.path.exists(os.path.join(save_path,f'cit_out/{epoch}')):
            os.makedirs(os.path.join(save_path,f'cit_out/{epoch}'))
            util.TensorToImage(Cit_outputs, os.path.join(save_path,f'cit_out/{epoch}'))
        else:
            util.TensorToImage(Cit_outputs, os.path.join(save_path, f'cit_out/{epoch}'))

        print(f"epoch:{epoch}+loss_sum:{loss_Total.item():.4f} loss_mse:{loss_cnn.item():.4f} loss_dice:{loss_tran.item():.4f}")
    if os.path.exists(os.path.join(save_path,f'cnn_out/{epoch}')):
        os.makedirs(os.path.join(save_path,f'cnn_out/{epoch}'))
        util.TensorToImage(CNN_out, os.path.join(save_path,f'cnn_out/{epoch}'))
    else:
        util.TensorToImage(CNN_out, os.path.join(save_path, f'cnn_out/{epoch}'))
    if os.path.exists(os.path.join(save_path,f'tran_out/{epoch}')):
        os.makedirs(os.path.join(save_path,f'tran_out/{epoch}'))
        util.TensorToImage(Trans_out, os.path.join(save_path,f'tran_out/{epoch}'))
    else:
        util.TensorToImage(Trans_out, os.path.join(save_path, f'tran_out/{epoch}'))
    if os.path.exists(os.path.join(save_path, f'train_image/{epoch}')):
        os.makedirs(os.path.join(save_path, f'train_image/{epoch}'))
        util.TensorToImage(train_image, os.path.join(save_path, f'train_image/{epoch}'))
    else:
        util.TensorToImage(train_image, os.path.join(save_path, f'train_image/{epoch}'))
    if os.path.exists(os.path.join(save_path, f'targets/{epoch}')):
        os.makedirs(os.path.join(save_path, f'targets/{epoch}'))
        util.TensorToImage(targets, os.path.join(save_path, f'targets/{epoch}'))
    else:
        util.TensorToImage(targets, os.path.join(save_path, f'targets/{epoch}'))

    torch.save(model,'model_epoch{}.pth'.format(epoch))

if __name__ == "__main__":
    train()