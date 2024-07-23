# -*- coding: utf-8 -*-

# Reference: https://stackoverflow.com/questions/56774582/adding-custom-labels-to-pytorch-dataloader-dataset-does-not-work-for-custom-data
# Auther : Hyunki Seong, hynkis@kaist.ac.kr

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import copy
import random

import cv2

from utils_postprocess import *


SEQUENCE_LENGTH = 4
SEQUENCE_RESOLUSION = 5

# - Params
RESIZE_SHAPE = (224,224)
RESIZE_SHAPE_BEV = (64,64) # dataset is (60,60)
CROP_POINT   = (100,120)
CROP_SHAPE   = (440,240) # (640,480) to (440,240) # (640,480) to (440,440)
SHIFT_PIXEL  = 50 # shifting pixel size to left/right for image shifting augmentation. (total 2 times)

STEER_NEUTRAL    = 1560 # (1960~1560~1160)
THROTTLE_NEUTRAL = 1460 # (1860~1460~1060)
MAX_ROUTE_DIST = 10
MIN_ROUTE_DIST = 2

# - Load img and label data with multiple trajectory data
dataset_csv = None
image_list = None  # input - image
state_array = None # input - state, route_info
label_array = None # output - steer, throttle

# Drop data size
DROP_INITIAL_DATA_SIZE = 50
DROP_TERMINAL_DATA_SIZE = 50

dataset_dir = "../../dataset"

# - Path of root dataset dir
dataset_dir_csv = os.path.join(dataset_dir, 'csv')
dataset_dir_img = os.path.join(dataset_dir, 'img')
dataset_dir_bev = os.path.join(dataset_dir, 'bev')
dataset_dir_npz = os.path.join(dataset_dir, 'npz')

# ================================================ #
# ===== 4. Post-processing (Sequential data) ===== #
# ================================================ #

# Load csv data
file_list = os.listdir(dataset_dir_csv)
print("file_list :")
print(file_list)
file_list_csv = [file for file in file_list if file.endswith(".csv")]

dataset_csv = None

for file_csv in file_list_csv:
    # Load and Concat each rollout
    file_path_csv = os.path.join(dataset_dir_csv, file_csv)
    tmp_dataset_csv = pd.read_csv(file_path_csv)
    dataset_length = tmp_dataset_csv.shape[0]

    # Remove initial several data
    tmp_dataset_csv = tmp_dataset_csv.drop(labels=range(0, DROP_INITIAL_DATA_SIZE), axis=0)
    tmp_dataset_csv = tmp_dataset_csv.drop(labels=range(dataset_length-DROP_TERMINAL_DATA_SIZE, dataset_length), axis=0)
    print("tmp_dataset_csv shape :", tmp_dataset_csv.shape)

    if dataset_csv is None:
        dataset_csv = tmp_dataset_csv
    else:
        dataset_csv = pd.concat([dataset_csv, tmp_dataset_csv], ignore_index=True)

    # Set 'done' True at terminal state (index for 'done' is 13)
    dataset_csv['done'].iloc[-1] = True
    print("set done := True at the terminal state of each rollout")

    print("dataset_csv shape :", dataset_csv.shape)

# Reset index
dataset_csv = dataset_csv.reset_index(drop=True)
print(dataset_csv)


# Check valid index (if SEQUENCE_LENGTH > 1)
# - reference: https://discuss.pytorch.org/t/how-to-skip-wrong-document-in-getitem-in-dataset-class/110659 
valid_idx = list(range(0, dataset_csv.shape[0]))
if SEQUENCE_LENGTH > 1:
    # - check invaild indices (initial data)
    invalid_idx = list(range(0, SEQUENCE_LENGTH*SEQUENCE_RESOLUSION))
    # - check invalid indices (done data)
    done_indices = dataset_csv[(dataset_csv['done']==1.0)].index
    invalid_start_list = np.array(done_indices) + 1
    invalid_end_list = np.array(done_indices) + SEQUENCE_LENGTH*SEQUENCE_RESOLUSION
    for ii in range(len(invalid_start_list)):
        partial_invalid_idx = list(range(invalid_start_list[ii], invalid_end_list[ii]))
        invalid_idx = invalid_idx + partial_invalid_idx
    # - substract invalid from valid
    valid_idx = [x for x in valid_idx if x not in invalid_idx]            

# - List of images to load in csv files
image_list = dataset_csv['img_path']

# - Array of states in csv files
state_array_raw = dataset_csv[['pose_x',
                               'pose_y',
                               'pose_yaw',
                               'target_wpt_x_t',
                               'target_wpt_y_t',
                               'target_wpt_x_t1',
                               'target_wpt_y_t1',
                               ]]

pose_x_list = state_array_raw['pose_x'].to_numpy()
pose_y_list = state_array_raw['pose_y'].to_numpy()
pose_yaw_list = state_array_raw['pose_yaw'].to_numpy()
target_wpt_x_t_list = state_array_raw['target_wpt_x_t'].to_numpy()
target_wpt_y_t_list = state_array_raw['target_wpt_y_t'].to_numpy()
target_wpt_x_t1_list = state_array_raw['target_wpt_x_t1'].to_numpy()
target_wpt_y_t1_list = state_array_raw['target_wpt_y_t1'].to_numpy()
route_dist_t, route_heading_t = \
        calc_route_info(pose_x_list, pose_y_list, pose_yaw_list, target_wpt_x_t_list, target_wpt_y_t_list)
route_dist_t1, route_heading_t1 = \
        calc_route_info(pose_x_list, pose_y_list, pose_yaw_list, target_wpt_x_t1_list, target_wpt_y_t1_list)

state_array = dataset_csv[['imu_accel_x', 'imu_accel_y', 'imu_yaw_rate']]
# state_array['route_dist_t'] = route_dist_t
# state_array['route_heading_t'] = route_heading_t
# state_array['route_dist_t1'] = route_dist_t1
# state_array['route_heading_t1'] = route_heading_t1

route_progress_t  = map_route_progress(route_dist_t, max_route_dist=MAX_ROUTE_DIST, min_route_dist=MIN_ROUTE_DIST)
route_progress_t1 = map_route_progress(route_dist_t1, max_route_dist=MAX_ROUTE_DIST, min_route_dist=MIN_ROUTE_DIST)
state_array['route_progress_t']  = route_progress_t
state_array['route_heading_t']   = route_heading_t
state_array['route_progress_t1'] = route_progress_t1
state_array['route_heading_t1']  = route_heading_t1

state_array['done'] = dataset_csv[['done']].astype('float').to_numpy()

# - Array of label in csv files
label_array = dataset_csv[['steer_t', 'throttle_t']]
# normalize steer and throttle
label_array.loc[:,'steer_t'] -= STEER_NEUTRAL
label_array.loc[:,'steer_t'] /= 400
label_array.loc[:,'throttle_t'] -= THROTTLE_NEUTRAL
label_array.loc[:,'throttle_t'] /= 400


def get_data(index):
    # - Loop during sequence length
    tmp_image_list = []
    tmp_bev_list = []

    valid_index = valid_idx[index]

    # - Load state data at every iteration (from i-seq-1 to i)
    state_list_np = state_array.to_numpy()
    state_list = state_list_np[valid_index-SEQUENCE_LENGTH*SEQUENCE_RESOLUSION+1:valid_index+1:SEQUENCE_RESOLUSION, :]

    # - Load label data at every iteration (from i-seq-1 to i)
    label_list_np = label_array.to_numpy()
    label_list = label_list_np[valid_index-SEQUENCE_LENGTH*SEQUENCE_RESOLUSION+1:valid_index+1:SEQUENCE_RESOLUSION, :]
    
    # - Load image data at every iteration with random shifting augmentation
    img_name = image_list[valid_index-SEQUENCE_LENGTH*SEQUENCE_RESOLUSION+1:valid_index+1:SEQUENCE_RESOLUSION]

    image_path = [os.path.join(dataset_dir_img, img_name_path) for img_name_path in img_name]

    # - Load Bird-Eye's View topological map image
    bev_path = [os.path.join(dataset_dir_bev, img_name_path) for img_name_path in img_name] # bev file name is same with camera image.

    for i in range(len(image_path)):
        image, _, _ = load_image_gray_w_crop(path=image_path[i], state=state_list[i], label=label_list[i],
                                             normalize=True, crop_points=CROP_POINT, crop_shape=CROP_SHAPE, resize_size=RESIZE_SHAPE)
        bev = load_image_rgb(path=bev_path[i], normalize=True, resize=True, resize_size=RESIZE_SHAPE_BEV)

        # - stack data
        tmp_image_list.append(image)
        tmp_bev_list.append(bev)

    data = {
            'image': np.array(tmp_image_list), # (T,1,224,224) [-1,+1]
            'state': np.array(state_list),
            'bev': np.array(tmp_bev_list), # (T,3,64,64) [-1,+1]
            'label': np.array(label_list),
            }

    return data


def main():
    valid_data_length = len(valid_idx)
    for ii in range(valid_data_length):
        data = get_data(ii)
        print("data shape; image: ", data['image'].shape, "state :", data['state'].shape, "bev :", data['bev'].shape, "label :", data['label'].shape)
        file_name = str(ii).zfill(8) + '.npz'
        save_path = os.path.join(dataset_dir_npz, file_name)

        np.savez_compressed(save_path,
                            image=data['image'],
                            state=data['state'],
                            bev=data['bev'],
                            label=data['label'],
                            )
        print("save data :", save_path, ii, valid_data_length)

if __name__ == "__main__":
    main()