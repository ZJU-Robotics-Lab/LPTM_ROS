#!/home/mav-lab/miniconda3/envs/rostorch/bin/python3
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
from image_mcl.msg import image_coords, coords_weights


# def callback1(data):
#         global done1, template_msg
#         template_msg = cv_bridge(data)
#         done1 = 1
#         # print("in callback1", done1)
# def callback2(data):
#         global done2, source_msg
#         source_msg = cv_bridge(data)
#         done2 = 1
        # print("in callback2", done2)
def callback_all(data):
        global done_all, template_msg, source_msg, x_coords, y_coords, particle_number, header
        template_msg = data.TemplateImage
        source_msg = data.SourceImage
        header = data.header
        x_coords = data.x_position_of_particle
        y_coords = data.y_position_of_particle
        particle_number = data.particle_number

        source_msg = cv_bridge(source_msg)
        template_msg = cv_bridge(template_msg)
        done_all = 1
        print("in callback")
def detect_model(template_path, source_path, model_template, model_source, model_corr2softmax,\
             model_trans_template, model_trans_source, model_trans_corr2softmax):
    global done1, done2, done_all, mcl_topic, template_msg, source_msg, x_coords, y_coords, particle_number, header, coords_weights_pub, weights_pub
    batch_size_inner = 1

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
    # rospy.Subscriber(template_path, Image, callback1)
    # rospy.Subscriber(source_path, Image, callback2)
    rospy.Subscriber(mcl_topic, image_coords, callback_all)
    # first_flag = 1
    while not rospy.is_shutdown():
        with torch.no_grad():
            if done_all:
                # print("in detect")
                since = time.time()
                template= default_loader(template_msg, 256)
                source= default_loader(source_msg, 256)
                template = template.to(device)
                source = source.to(device)
                rotation_cal, scale_cal, corr_result_rot = detect_rot_scale(template, source,\
                                             model_template, model_source, model_corr2softmax, device )
                tranformation_y, tranformation_x, corr_result_trans = detect_translation(template, source, rotation_cal, scale_cal, \
                                                    model_trans_template, model_trans_source, model_trans_corr2softmax, device)
                print("particle number", particle_number)
                for i in range(particle_number):
                    weights = corr_result_trans[0, y_coords[i], x_coords[i]]
                    weights_for_particle.append(weights.cpu().numpy())
                    print("weight for", i, "is", weights)

                coords_weights_pub.header = header
                coords_weights_pub.particle_number = particle_number
                coords_weights_pub.weights_for_particle = weights_for_particle
                print("published", coords_weights_pub)
                weights_pub.publish(coords_weights_pub)
                # if first_flag:
                #     weights_pub.publish(coords_weights_pub)
                #     first_flag = 0
                time_elapsed = time.time() - since
                done1, done2, done_all = 0, 0, 0
                # print('in detection time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print("in detection time", time_elapsed, header)



if __name__ == '__main__':
    checkpoint_path = "./checkpoints/laser_sat_qsdjt_9epoch.pt"
    template_path = "/stereo_grey/left/image_raw"
    source_path = "/stereo_grey/right/image_raw"
    mcl_topic = "/particle_pose"
    weights_topic = "/LPTM/weights_for_particles"

    load_pretrained =True
    rospy.init_node('Detecter', anonymous=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The devices that the code is running on:", device)
    # device = torch.device("cpu")
    done1, done2, done_all = 0, 0,0
    coords_weights_pub = coords_weights()
    weights_for_particle = []
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

    weights_pub = rospy.Publisher(weights_topic, coords_weights)

    if load_pretrained:
        model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
        _, _, _, _, _, _,\
            start_epoch = load_checkpoint(\
                                        checkpoint_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                        optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s, device)


    detect_model(template_path, source_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax)