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

IMAGE_EXTENSION = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                   '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


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
   assert images, '{:s} has no valid image file'.format(path)

   return sorted(images)