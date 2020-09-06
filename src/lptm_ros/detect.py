#!/home/mav-lab/miniconda3/envs/rostorch/bin/python3
from collections import defaultdict
import torch.nn.functional as F
import rospy
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
import time
import threading
import multiprocessing
from kornia.filters.kernels import get_gaussian_kernel2d
from matplotlib import cm
 
def get_jet():
 
    colormap_int = np.zeros((256, 3), np.uint8)
    colormap_float = np.zeros((256, 3), np.float)
 
    for i in range(0, 256, 1):
       colormap_float[i, 2] = cm.jet(i)[0]
       colormap_float[i, 1] = cm.jet(i)[1]
       colormap_float[i, 0] = cm.jet(i)[2]
 
       colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[0] * 255.0))
       colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
       colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[2] * 255.0))
    return colormap_int

def handle_compute_weight(req):
    global done_all, template_msg, source_msg, x_coords, y_coords, particle_number, header
    # print("handle_compute_weight", req.particle_number)
    
    template_msg = req.TemplateImage
    source_msg = req.SourceImage
    x_coords = req.x_position_of_particle
    y_coords = req.y_position_of_particle
    particle_number = req.particle_number
    source_msg = cv_bridge(source_msg)
    template_msg = cv_bridge(template_msg)
    # header = req.header
    done_all = 1

    coords_weights_pub, imgmsg = detect_model(template_path, source_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax)
    corrmap_pub.publish(imgmsg)
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
                rotation_cal, scale_cal = torch.Tensor([0.0]), torch.Tensor([300./300.]) # qsdjt rot:-108.6  scale:384./300.  gym rot:165  scale:220./200.

                # print("rotation_cal", rotation_cal)

                tranformation_y, tranformation_x, corr_result_trans = detect_translation(template, source, rotation_cal, scale_cal, \
                                                    model_trans_template, model_trans_source, model_trans_corr2softmax, device)
                soft_corr_trans = model_trans_corr2softmax(corr_result_trans)
                soft_corr_trans = soft_corr_trans.unsqueeze(0)
                gauss = kornia.filters.GaussianBlur2d((49, 49), (10, 10))
                soft_corr_trans = gauss(soft_corr_trans)
                soft_corr_trans = soft_corr_trans.squeeze(0)
                
                # def update():
                #     # clear
                #     imshow(soft_corr_trans[0,:,:])
                #     plt.show()
                #     plt.close()

                # # use thread
                # t = threading.Thread(target=update)
                # t.start()

                # imshow(soft_corr_trans[0,:,:])
                # plt.show()
                # plt.pause(0.1)
                # plt.close()
                for i in range(particle_number):
                    
                    if y_coords[i]>=template_msg.shape[0] or y_coords[i] <= 0 or x_coords[i] >= template_msg.shape[0] or x_coords[i] <= 0:
                        weights = torch.Tensor([1e-10]).to(device)
                    else:
                        # print("x", x_coords[i], "y", y_coords[i])
                        weights = soft_corr_trans[0, int(float(y_coords[i])*256.0/float(template_msg.shape[0])), int(float(x_coords[i])*256.0/float(template_msg.shape[0]))]
                    weights_for_particle.append(weights.cpu().numpy())
                    # print("coords",  255-int(float(y_coords[i])*256.0/float(template_msg.shape[0])), 255-int(float(x_coords[i])*256.0/float(template_msg.shape[0])))
                grey_map = soft_corr_trans[0,...].cpu().numpy()
                grey_map = 255 * (grey_map - np.min(grey_map))/(np.max(grey_map)-np.min(grey_map))

                # print("grey_map",np.max(grey_map))

                corr_map = np.zeros((grey_map.shape[0], grey_map.shape[1],3),np.uint8)
                color_map = get_jet()
                for i in range(0, grey_map.shape[0]):
                    for j in range(0, grey_map.shape[1]):
                        corr_map[i, j] = color_map[int(grey_map[i, j])]

                imgmsg = cv2_to_imgmsg(corr_map)
                # coords_weights_pub.header = header
                # coords_weights_pub.particle_number = particle_number
                # print("max", weights_for_particle.index(max(weights_for_particle)), np.max(weights_for_particle))
                # print("coords", int(float(y_coords[weights_for_particle.index(max(weights_for_particle))])*256.0/float(template_msg.shape[0])), int(float(x_coords[weights_for_particle.index(max(weights_for_particle))])*256.0/float(template_msg.shape[0])))

                coords_weights_pub.weights_for_particle = weights_for_particle

                time_elapsed = time.time() - since
                done1, done2, done_all = 0, 0, 0

                # print("in detection time", time_elapsed)

                return weights_for_particle, imgmsg



if __name__ == '__main__':
    checkpoint_path = "./checkpoints/qsdjt_mse_16epoch_1w_3k.pt"
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

    corrmap_pub = rospy.Publisher("corr_map", Image, queue_size=1)
    

    if load_pretrained:
        model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
        _, _, _, _, _, _,\
            start_epoch = load_checkpoint(\
                                        checkpoint_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                        optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s, device)

    add_server()
    # detect_model(template_path, source_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax)
