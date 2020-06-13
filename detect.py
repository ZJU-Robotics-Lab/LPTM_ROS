#!/home/jessy104/miniconda3/envs/yolo3pytorch/bin/python3
from collections import defaultdict
import torch.nn.functional as F
import rospy
import cv_bridge
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import time
import copy
from unet.pytorch_lptm import FFT2, UNet, LogPolar, PhaseCorr, Corr2Softmax
from data.dataset_LPTM import *
import numpy as np
import shutil
from utils.utils import *
import kornia
from data.dataset import *
from utils.detect_utils import *
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image


def callback1(data):
        global done1, template_msg
        template_msg = cv_bridge(data)
        done1 = 1
        # print("in callback1", done1)
def callback2(data):
        global done2, source_msg
        source_msg = cv_bridge(data)
        done2 = 1
        # print("in callback2", done2)
def detect_model(template_path, source_path, model_template, model_source, model_corr2softmax,\
             model_trans_template, model_trans_source, model_trans_corr2softmax):
    global done1, done2, template_msg, source_msg
    batch_size_inner = 1

    since = time.time()

    # Each epoch has a training and validation phase
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    phase = "val"

    model_template.eval()   # Set model to evaluate mode
    model_source.eval()
    model_corr2softmax.eval()
    model_trans_template.eval()
    model_trans_source.eval()
    model_trans_corr2softmax.eval()
    rospy.Subscriber(template_path, Image, callback1)
    rospy.Subscriber(source_path, Image, callback2)
    while not rospy.is_shutdown():
        with torch.no_grad():
            if done1 and done2:
                # print("in detect")
                template= default_loader(template_msg, 256)
                source= default_loader(source_msg, 256)
                template = template.to(device)
                source = source.to(device)
                since = time.time()
                rotation_cal, scale_cal = detect_rot_scale(template, source,\
                                             model_template, model_source, model_corr2softmax, device )
                tranformation_y, tranformation_x = detect_translation(template, source, rotation_cal, scale_cal, \
                                                    model_trans_template, model_trans_source, model_trans_corr2softmax, device)
                time_elapsed = time.time() - since
                done1, done2 = 0, 0
                # print('in detection time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print("in detection time", time_elapsed)





if __name__ == '__main__':
    checkpoint_path = "./checkpoints/laser_sat_qsdjt_9epoch.pt"
    template_path = "/stereo_grey/left/image_raw"
    source_path = "/stereo_grey/right/image_raw"

    load_pretrained =True
    rospy.init_node('Detecter', anonymous=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The devices that the code is running on:", device)
    # device = torch.device("cpu")
    done1, done2 = 0, 0
    batch_size = 1
    num_class = 1
    start_epoch = 0
    model_template = UNet(num_class).to(device)
    model_source = UNet(num_class).to(device)
    model_corr2softmax = Corr2Softmax(200., 0.).to(device)
    model_trans_template = UNet(num_class).to(device)
    model_trans_source = UNet(num_class).to(device)
    model_trans_corr2softmax = Corr2Softmax(200., 0.).to(device)

    optimizer_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=2e-4)
    optimizer_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=2e-4)
    optimizer_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)
    optimizer_trans_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=2e-4)
    optimizer_trans_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=2e-4)
    optimizer_trans_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)

    if load_pretrained:
        model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
        _, _, _, _, _, _,\
            start_epoch = load_checkpoint(\
                                        checkpoint_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                        optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s, device)


    detect_model(template_path, source_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax)