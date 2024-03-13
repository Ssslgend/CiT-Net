#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Transformers 
@File    ：util.py
@Author  ：songliqiang
@Date    ：2024/3/13 15:06 
'''
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

IMAGE_EXTENSION = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                   '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
path ='/datasets01/archive/train.X1/n01440764'

def isImageFile(fname):
   return any(fname.endswith(extension) for extension in IMAGE_EXTENSION)

def isMatFile(fname):
   return fname.endswith('.mat')

def isImageFolder(path):
   assert os.path.isdir(path),"{:s} path is invalid".format(path)
   for fname in os.listdir(path):
      if isImageFile(fname):
         return True
   return False
def getImageFromPath(path):
   #     assert os.path.isdir(path),"{:s} path is invalid".format(path)
   assert os.path.isdir(path), f"{path} path is invalid"
   images = []
   #    os,walk 会返回一个三元组所以得用三元组接受
   for root, dirPath, fnames in sorted(os.walk(path)):
      for fname in sorted(fnames):
         if isImageFile(fname):
            img_path = os.path.join(root, fname)
            images.append(img_path)
   assert images, '{:s} has no valid image file'.format(path)

   return sorted(images)
def getMartFromPath(path):
   #     assert os.path.isdir(path),"{:s} path is invalid".format(path)
   assert os.path.isdir(path), f"{path} path is invalid"
   images = []
   #    os,walk 会返回一个三元组所以得用三元组接受
   for root, dirPath, fnames in sorted(os.walk(path)):
      for fname in sorted(fnames):
         if isMatFile(fname):
            img_path = os.path.join(root, fname)
            images.append(img_path)
   # assert images, '{:s} has no valid image file'.format(img_path)

   return sorted(images)

def transformImage(images, size=256):
   # 假设 images 是一个包含多个图像路径的列表

   preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    )
   # 预处理所有图像并转换为 (batch, channel, width, height) 格式
   preprocess_tensor =[]
   for image_path in images:
      try:
         tensor = preprocess(Image.open(image_path))
         if tensor.shape == (3, 224, 224):
            preprocess_tensor.append(tensor)
      except Exception as e:
         print(f"{image_path} is not a valid image file {e}")
   # batch_images = torch.stack([preprocess(Image.open(image_path)) for image_path in images])
   batch_images = torch.stack(preprocess_tensor)
   # 查看批次图像张量形状
   print(batch_images.shape)  # 输出: torch.Size([batch_size, 3, 224, 224])


if __name__ == '__main__':
   images = getImageFromPath(path)
   transformImage(images)