from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import torch
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(".."))
from utils.utils import *

def default_loader(image, resize_shape, change_scale = False):
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])
    # image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (h_original, w_original) = image.shape
    image = cv2.resize(image, dsize=(resize_shape,resize_shape), interpolation=cv2.INTER_CUBIC)


    np_image_data = np.asarray(image)
    image_tensor = trans(np_image_data)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.permute(1,0,2,3)
    
    return image_tensor

def get_gt_tensor(this_gt, size):
    this_gt = this_gt +180
    gt_tensor_self = torch.zeros(size,size)
    angle_convert = this_gt*size/360
    angle_index = angle_convert//1 + (angle_convert%1+0.5)//1
    if angle_index.long() == size:
        angle_index = size-1
        gt_tensor_self[angle_index,0] = 1
    else:
        gt_tensor_self[angle_index.long(),0] = 1
    # print("angle_index", angle_index)

    return gt_tensor_self
def cv_bridge( img_msg):
        """ cv_bridge does not support python3 and this is extracted from the
            cv_bridge file to convert the msg::Img to np.ndarray
        """
        color_msg = img_msg
        print("encoding=====", img_msg.encoding)
        #set different dtype based on different encoding type
        if 'C' in img_msg.encoding:
            map_dtype = {'U': 'uint', 'S': 'int', 'F': 'float'}
            dtype_str, n_channels_str = img_msg.encoding.split('C')
            n_channels = int(n_channels_str)
            dtype = np.dtype(map_dtype[dtype_str[-1]] + dtype_str[:-1])
        elif img_msg.encoding == 'bgr8' or img_msg.encoding == 'rgb8':
            n_channels = 3
            dtype = np.dtype('uint8')
        elif img_msg.encoding == 'mono8':
            n_channels = 1
            dtype = np.dtype('uint8')

            
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        img1 = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                        dtype=dtype)#, buffer=img_msg.data)
        img1 = np.squeeze(img1)
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            img1 = img1.byteswap().newbyteorder()

        #convert RGB to BGR
        if img_msg.encoding == 'rgb8':
            img0 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding == 'mono8':
            img0 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img0 = img1
        return img0