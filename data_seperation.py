#-*- coding: utf-8 -*-

#!/usr/bin/env python

# By Hyunki Seong.
# Email : hynkis@kaist.ac.kr

import argparse
import datetime
import time
from matplotlib.pyplot import contour

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tools.custom_dataset_eval import customDataset

from bil import BIL
from utils import *

"""
Data

Straight + Obstacle: 755, 860



"""
parser = argparse.ArgumentParser(description="Neural Decision Network")

parser.add_argument('--cpu',           action="store_true", help='run on CPU (default: False)')
parser.add_argument('--gpu_num',          type=int, default=0,     metavar='N', help='GPU number for training')
parser.add_argument('--seed',           type=int, default=1234567, metavar='N',  help='random seed (default: 123456)')
parser.add_argument('--load_past_data', action="store_true",                    help='Load Past Training data (default: False)')
parser.add_argument('--save_interval',  type=int,  default=100,   metavar='N', help='Saving all data interval (default: 10)')
parser.add_argument('--num_workers',  type=int,  default=8,   metavar='N', help='Number of workers for dataloader (default: 4)')
parser.add_argument('--shuffle_batch',  type=bool,  default=False,   metavar='N', help='Whether shuffle batch of not (default: True)')

parser.add_argument('--batch_size',    type=int, default=1,     metavar='N', help='batch size (default: 128)')
parser.add_argument('--max_epoch',     type=int, default=1,     metavar='N', help='max_epoch (default: 1000)')
parser.add_argument('--lr',             type=float, default=3e-4, metavar='G', help='learning rate. (default:1e-4 / 0.0003)')
parser.add_argument('--lr_lambda',     type=float, default=1.0, metavar='G', help='lambda for lr scheduler. (default: 0.99)')
parser.add_argument('--lr_interval',   type=int, default=2400,     metavar='N', help='max_epoch (default: 1000)')
parser.add_argument('--max_grad_norm',     type=float, default=5.0, metavar='G', help='maximum grad norm. (default: 5)')
parser.add_argument('--boltz_alpha',   type=float, default=4.0,     metavar='N', help='temperature factor for Boltzmann Distribution (smooth softmax) (default: 5.0)')

parser.add_argument('--save_prefix',     default="8head_noTanh", help='prefix at save path (default: )')

args = parser.parse_args()

# Pytorch config
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# Path
DATASET_PATH        = "./dataset"

# DATASET_PATH        = "./dataset/train/img_data"
# IMG_NAME_PATH       = "./dataset/train/img_data.txt"
# LABEL_PATH          = "./dataset/train/img_data.txt"

# LOAD_MODEL_PATH          = './model/bpn_model_good_cnnDim1128Only_ControlTanh_2021-05-31_03-30-41_epoch_1903_iteration_200000.pth'
# LOAD_MODEL_PATH          = './model/bpn_model_good_cnnDim1128Only_DecisionTanh_ControlTanh_2021-06-01_22-25-52_epoch_1903_iteration_200000.pth'
# LOAD_MODEL_PATH          = './model/bpn_model_good_cnnDim1128Only_DecisionTanh_ControlTanh_RouteHeadPi2Pi_2021-06-02_17-04-00_epoch_523_iteration_55000.pth'
# LOAD_MODEL_PATH          = './model/bpn_model_good_AugmentBlur_cnnDim1128Only_DecisionTanh_ControlTanh_RouteHeadPi2Pi_2021-06-02_22-32-56_epoch_524_iteration_55000.pth'
# LOAD_MODEL_PATH          = './model/bpn_model_good_largeData_cnnDim1128Only_DecisionTanh_ControlTanh_RouteHeadPi2Pi_2021-06-03_16-31-06_epoch_387_iteration_51000.pth'
# LOAD_MODEL_PATH          = './model/bpn_model_good_largeData_cnnDim1128Only_ControlTanh_RouteHeadPi2Pi_2021-06-05_00-56-04_epoch_584_iteration_77000.pth'
# LOAD_MODEL_PATH          = './model/bpn_model_good_AugmentBlur_largeData_good_config_2021-06-06_22-34-50_epoch_341_iteration_45000.pth'

# LOAD_MODEL_PATH          = './model/bpn_model_attOnlyDecision_AugmentNoiseShiftBlur_2021-06-11_03-00-41_epoch_4957_iteration_653000.pth'
# LOAD_MODEL_PATH          = './model/bpn_model_attOnlyDecision_scaleDotMean_cnn64_AugmentNoiseShiftBlur_2021-06-22_23-48-02_epoch_2125_iteration_280000.pth'
# LOAD_MODEL_PATH          = './model/bpn_model_attOnlyDecision_scaleDot_cnn64_AugmentNoiseShiftBlur_2021-06-18_10-57-33_epoch_4866_iteration_641000.pth'

# - 1D decision
# LOAD_MODEL_PATH          = './model/2022-07-12_18-29-02_singleGPU_BRM_SpatoalContext_v4_2/2022-07-12_18-29-02_singleGPU_BRM_SpatoalContext_v4_2_epoch_10075_iteration_800000.pth'
# LOAD_MODEL_PATH          = './model/2022-07-12_18-29-02_singleGPU_BRM_SpatoalContext_v4_2/2022-07-12_18-29-02_singleGPU_BRM_SpatoalContext_v4_2_epoch_20151_iteration_1600000.pth'
LOAD_MODEL_PATH          = './model/2022-08-14_22-39-30_singleGPU_BRM_SpatoalContext_v5/2022-08-14_22-39-30_singleGPU_BRM_SpatoalContext_v5_epoch_4969_iteration_269000.pth'


START_INDEX = 0
END_INDEX = 5000

print("Load path")
print("DATASET_PATH         :", DATASET_PATH)
print("LOAD_MODEL_PATH        :", LOAD_MODEL_PATH)

# Parameters
image_shape = (224,224)
# device = torch.device("cuda:" + str(args.gpu_num))
device = torch.device("cuda")
if args.cpu:
    device = torch.device("cpu")

transforms = None
torch.autograd.set_detect_anomaly(True)

def main():
    # Define model
    model = BIL(img_shape=image_shape, args=args)

    # Load Dataset
    dataset = customDataset(dataset_dir=DATASET_PATH,
                            transforms=transforms,
                            )
    # Define Dataloader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle_batch,
                            num_workers=8,
                            drop_last=True
                            )

    # Load model and training info

    # - load model weight
    model.load_model(LOAD_MODEL_PATH)
    model.bpn.eval()
    print("Load model done")

    # - Index for data seperation
    WF_list = np.array([])
    OL_list = np.array([])
    OR_list = np.array([])
    ST_list = np.array([])
    LT_list = np.array([])
    RT_list = np.array([])
    
    # -- WF
    WF_list = np.hstack([WF_list, np.arange(20, 60, 1)])
    WF_list = np.hstack([WF_list, np.arange(320, 375, 1)])
    WF_list = np.hstack([WF_list, np.arange(1870, 2000, 1)])
    WF_list = np.hstack([WF_list, np.arange(2340, 2370, 1)])
    WF_list = np.hstack([WF_list, np.arange(3170, 3340, 1)])
    WF_list = np.hstack([WF_list, np.arange(3515, 3580, 1)])
    WF_list = np.hstack([WF_list, np.arange(4000, 4030, 1)])
    WF_list = np.hstack([WF_list, np.arange(4420, 4550, 1)])
    WF_list = np.hstack([WF_list, np.arange(4900, 4960, 1)])

    # -- OL
    OL_list = np.hstack([OL_list, np.arange(2730, 2785, 1)])

    # -- OR
    OR_list = np.hstack([OR_list, np.arange(999999, 999999, 1)])

    # -- ST
    ST_list = np.hstack([ST_list, np.arange(880, 960, 1)])
    ST_list = np.hstack([ST_list, np.arange(3055, 3105, 1)])

    # -- LT
    LT_list = np.hstack([LT_list, np.arange(260, 295, 1)])
    LT_list = np.hstack([LT_list, np.arange(3905, 3940, 1)])
    LT_list = np.hstack([LT_list, np.arange(4255, 4280, 1)])
    LT_list = np.hstack([LT_list, np.arange(4820, 4870, 1)])

    # -- RT
    RT_list = np.hstack([RT_list, np.arange(600, 640, 1)])
    RT_list = np.hstack([RT_list, np.arange(1200, 1260, 1)])
    RT_list = np.hstack([RT_list, np.arange(2215, 2285, 1)])
    RT_list = np.hstack([RT_list, np.arange(2580, 2630, 1)])

    # Data loading
    with torch.no_grad():
        # Training iteration (Update iteration)
        for i, (data, _) in enumerate(dataloader):
            print("===============================")
            print("===== %d th data =====" %(i))
            print("===============================")
            if i < START_INDEX:
                print("Pass this data")
                continue
            if i >= END_INDEX:
                print("Done!")
                break

            print(data["image"].size())  # input image  (batch, 224, 224)
            print(data["state"].size())  # input state  (batch, 7) 
            print(data["label"].size())  # output label (batch, 2)

            if i in WF_list:
                save_dir = "/home/seong/seong_workspace/bil/dataset/data_split/WF"
            elif i in OL_list:
                save_dir = "/home/seong/seong_workspace/bil/dataset/data_split/OL"
            elif i in OR_list:
                save_dir = "/home/seong/seong_workspace/bil/dataset/data_split/OR"
            elif i in ST_list:
                save_dir = "/home/seong/seong_workspace/bil/dataset/data_split/ST"
            elif i in LT_list:
                save_dir = "/home/seong/seong_workspace/bil/dataset/data_split/LT"
            elif i in RT_list:
                save_dir = "/home/seong/seong_workspace/bil/dataset/data_split/RT"
            else:
                save_dir = "/home/seong/seong_workspace/bil/dataset/data_split/else"
            save_seperate_data(front_img=data["image"],
                                bev_img=data["bev"],
                                veh_state=data["state"][:, [1, 2]],
                                label=data["label"],
                                index=i,
                                save_dir=save_dir)
            data_name = str(i).zfill(8) + ".npz"
            data_path = os.path.join(save_dir, data_name)
            print("Saved file :", data_path)

            if i >= 4889:
                print("Break point")
                break


if __name__ == '__main__':
    main()
