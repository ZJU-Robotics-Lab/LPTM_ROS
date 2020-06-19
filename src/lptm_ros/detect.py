#!/home/mav-lab/miniconda3/envs/rostorch/bin/python3
from collections import defaultdict
import torch.nn.functional as F
import rospy
# import cv_bridge
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
from lptm_ros.srv import ComputePtWeights, ComputePtWeightsResponse
import cv2


def handle_compute_weight(req):
    global done_all, template_msg, source_msg, x_coords, y_coords, particle_number, header
    print("handle_compute_weight", req.particle_number)
    
    template_msg = req.TemplateImage
    source_msg = req.SourceImage
    x_coords = req.x_position_of_particle
    y_coords = req.y_position_of_particle
    particle_number = req.particle_number
    source_msg = cv_bridge(source_msg)
    template_msg = cv_bridge(template_msg)
    # header = req.header
    done_all = 1

    coords_weights_pub = detect_model(template_path, source_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax)
    return ComputePtWeightsResponse(coords_weights_pub)

def add_server():
    s = rospy.Service('compute_weight_server', ComputePtWeights, handle_compute_weight)
    print("Ready to provide service")
    rospy.spin()

# def callback_all(data):
#         global done_all, template_msg, source_msg, x_coords, y_coords, particle_number, header
#         template_msg = data.TemplateImage
#         source_msg = data.SourceImage
#         header = data.header
#         x_coords = data.x_position_of_particle
#         y_coords = data.y_position_of_particle
#         particle_number = data.particle_number

#         source_msg = cv_bridge(source_msg)
#         template_msg = cv_bridge(template_msg)
#         done_all = 1
#         print("in callback")

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
    # rospy.Subscriber(mcl_topic, image_coords, callback_all)

    while not rospy.is_shutdown():
        with torch.no_grad():
            if done_all:
                weights_for_particle = []
                since = time.time()
                # cv2.imshow("source", source_msg)
                # cv2.waitKey(1000)
                # print("source", source_msg)
                # cv2.imshow("template", template_msg)
                # cv2.waitKey(1000)
                template= default_loader(template_msg, 256)
                source= default_loader(source_msg, 256)
                # imshow(template)
                # plt.show()
                # imshow(source)
                # plt.show()                
                template = template.to(device)
                source = source.to(device)
                # rotation_cal, scale_cal, corr_result_rot = detect_rot_scale(template, source,\
                #                              model_template, model_source, model_corr2softmax, device )
                # print("rotation_cal", rotation_cal)
                rotation_cal, scale_cal = torch.Tensor([168]), torch.Tensor([1.02])
                print("rotation_cal", rotation_cal)
                tranformation_y, tranformation_x, corr_result_trans = detect_translation(template, source, rotation_cal, scale_cal, \
                                                    model_trans_template, model_trans_source, model_trans_corr2softmax, device)
                soft_corr_trans = model_trans_corr2softmax(corr_result_trans)
                soft_corr_trans.view(-1)
                m = nn.Softmax(dim=1)
                soft_corr_trans = m(soft_corr_trans)
                soft_corr_trans.reshape([1,256,256])
                # soft_corr_trans = softmax2d(soft_corr_trans, device)

                # print("soft_corr_trans", soft_corr_trans)
                print("particle number", particle_number)
                # imshow(corr_result_trans[0,:,:])
                # plt.show()
                # plt.close()
                for i in range(particle_number):
                    # print("x", x_coords[i], "y", y_coords[i])
                    if y_coords[i]>=180 or y_coords[i] <= 0 or x_coords[i] >= 180 or x_coords[i] <= 0:
                        weights = torch.Tensor([0]).to(device)
                    else:
                        weights = soft_corr_trans[0, 255-int(y_coords[i]*256/180), 255-int(x_coords[i]*256/180)]
                    # print("Weights", weights)
                    weights_for_particle.append(weights.cpu().numpy())
                    # print("weight for", i, "is", weights)
                    # print("coords",  y_coords[i], x_coords[i])

                # coords_weights_pub.header = header
                # coords_weights_pub.particle_number = particle_number
                coords_weights_pub.weights_for_particle = weights_for_particle
                # print("published", coords_weights_pub)
                # weights_pub.publish(coords_weights_pub)

                time_elapsed = time.time() - since
                done1, done2, done_all = 0, 0, 0
                # print('in detection time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print("in detection time", time_elapsed)
                return weights_for_particle



if __name__ == '__main__':
    checkpoint_path = "./checkpoints/dsnt_x_y_overfit.pt"
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
    coords_weights_pub = ComputePtWeightsResponse()
    
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

    # weights_pub = rospy.Publisher(weights_topic, coords_weights)
    

    if load_pretrained:
        model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
        _, _, _, _, _, _,\
            start_epoch = load_checkpoint(\
                                        checkpoint_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                        optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s, device)

    add_server()
    # detect_model(template_path, source_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax)