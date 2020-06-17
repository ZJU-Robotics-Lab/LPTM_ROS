from collections import defaultdict
import torch.nn.functional as F
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
from utils.validate_utils import *


def val_model(model_template, model_source, model_corr2softmax,\
             model_trans_template, model_trans_source, model_trans_corr2softmax, writer_val, iters, epoch):
    batch_size_inner = 4
    iters -= 7000


    # Each epoch has a training and validation phase
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    phase = "val"
    loss_list = []
    rot_list = []
    model_template.eval()   # Set model to evaluate mode
    model_source.eval()
    model_corr2softmax.eval()
    model_trans_template.eval()
    model_trans_source.eval()
    model_trans_corr2softmax.eval()
    acc_x = np.zeros(20)
    acc_y = np.zeros(20)
    acc = 0.

    with torch.no_grad():

        for template, source, groundTruth_number, gt_scale,  gt_trans in LPTMdataloader(batch_size_inner)[phase]:
            template = template.to(device)
            source = source.to(device)
            iters += 1    
            # imshow(template[0,:,:])
            # plt.show()
            # imshow(source[0,:,:])
            # plt.show()
            # print("gtSCALE~~~~",gt_scale)
            loss_rot, loss_scale, scale_cal, loss_l1_rot, loss_mse_rot, loss_l1_scale, loss_mse_scale \
                    = validate_rot_scale(template.clone(), source.clone(), groundTruth_number.clone(), gt_scale.clone(),\
                                         model_template, model_source, model_corr2softmax, device )
            loss_y, loss_x, total_loss, loss_l1_x,loss_l1_y,loss_mse_x, loss_mse_y \
                    = validate_translation(template.clone(), source.clone(), groundTruth_number.clone(), gt_scale.clone(), gt_trans.clone(), \
                                            model_trans_template, model_trans_source, model_trans_corr2softmax,acc_x, acc_y, device)


            # loss = compute_loss(corr_final, gt_angle)
            total_rs_loss = loss_rot + loss_scale
            loss_list.append(total_rs_loss.tolist())
            writer_val.add_scalar('LOSS ROTATION', loss_rot.detach().cpu().numpy(), iters)
            writer_val.add_scalar('LOSS SCALE', loss_scale.detach().cpu().numpy(), iters)
            writer_val.add_scalar('LOSS X', loss_x.detach().cpu().numpy(), iters)
            writer_val.add_scalar('LOSS Y', loss_y.detach().cpu().numpy(), iters)

            writer_val.add_scalar('LOSS ROTATION L1', loss_l1_rot.item(), iters)
            writer_val.add_scalar('LOSS ROTATION MSE', loss_mse_rot.item(), iters)
            writer_val.add_scalar('LOSS SCALE L1', loss_l1_scale.item(), iters)
            writer_val.add_scalar('LOSS SCALE MSE', loss_mse_scale.item(), iters)

            writer_val.add_scalar('LOSS X L1', loss_l1_x.item(), iters)
            writer_val.add_scalar('LOSS X MSE', loss_mse_x.item(), iters)
            writer_val.add_scalar('LOSS Y L1', loss_l1_y.item(), iters)
            writer_val.add_scalar('LOSS Y MSE', loss_mse_y.item(), iters)

           
            # writer_val.add_scalar('ACC X', ACC_x.item(), iters)
            # writer_val.add_scalar('ACC Y', ACC_y.item(), iters)
            # writer_val.add_scalar('ACC ROTATION', ACC_rot.item(), iters)
            # writer_val.add_scalar('ACC SCALE', ACC_scale.item(), iters)
            
            # writer_val.add_scalar('angle_loss', loss_a.detach().cpu().numpy(), iters)
            # writer_val.add_scalar('x_loss', loss_x.detach().cpu().numpy(), iters)
            # writer_val.add_scalar('y_loss', loss_y.detach().cpu().numpy(), iters)
           
            # writer_val.add_image("temp_input", template[0,:,:].cpu(), iters)
            # writer_val.add_image("src_input", source[0,:,:].cpu(), iters)
            # writer_val.add_image("unet_temp_rot", template_visual_rot[0,:,:].cpu(), iters)
            # writer_val.add_image("unet_src_rot", source_visual_rot[0,:,:].cpu(), iters)
            # writer_val.add_image("unet_temp_trans", template_visual_trans[0,:,:].cpu(), iters)
            # writer_val.add_image("unet_src_trans", source_visual_trans[0,:,:].cpu(), iters)
            # rot_list.append(rotation_cal.tolist())

            # groundTruth_list.append(groundTruth_number.tolist())
    # X = np.linspace(0, 19, 20)
    # fig = plt.figure()
    # plt.bar(X,acc_x/1000)
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")    
    
    # plt.savefig("./checkpoints/barChart/celoss+l1/x/"+ str(epoch) + "_yyl_barChartX_top2.jpg")

    # Y = np.linspace(0, 19, 20)
    # fig = plt.figure()
    # plt.bar(Y,acc_y/1000)
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")    
    
    # plt.savefig("./checkpoints/barChart/celoss+l1/y/"+ str(epoch) + "_yyl_barChartY_top2.jpg")
    return loss_list


single_val = False
if single_val:
    epoch = 1
    checkpoint_path = "./checkpoints/laser_sat_qsdjt_9epoch.pt"
    load_pretrained =True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("The devices that the code is running on:", device)
    writer_val = SummaryWriter(log_dir="./checkpoints/log/val/")
    batch_size = 1
    num_class = 1
    start_epoch = 0
    iters = 0
    model_template = UNet(num_class).to(device)
    model_source = UNet(num_class).to(device)
    model_corr2softmax = Corr2Softmax(200., 0.).to(device)
    model_trans_template = UNet(num_class).to(device)
    model_trans_source = UNet(num_class).to(device)
    model_trans_corr2softmax = Corr2Softmax(11.72, 0.).to(device)

    optimizer_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=2e-4)
    optimizer_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=2e-4)
    optimizer_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)
    optimizer_trans_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=2e-4)
    optimizer_trans_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=2e-4)
    optimizer_trans_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)

    if load_pretrained:
        model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
        optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s,\
            start_epoch = load_checkpoint(\
                                        checkpoint_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                        optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s, device)

    loss_list = val_model(model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax, writer_val, iters, epoch)
            

                                     





