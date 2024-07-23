# -*- coding: utf-8 -*-

# Reference: https://stackoverflow.com/questions/56774582/adding-custom-labels-to-pytorch-dataloader-dataset-does-not-work-for-custom-data
# Auther : Hyunki Seong, hynkis@kaist.ac.kr

"""
Dataset
    - Raw data:
        -- trajectories of vehicle with (image, IMU, trajectory, steering_angle, throttle)
        -- route waypoints (intersection points)
    - Preprocess:
        -- set target route points(current, next)
        -- calculate route information
        0 : 'img_stamp'
        1 : 'img_path'
        2 : 'steer_t'
        3 : 'throttle_t'
        4 : 'imu_accel_x'
        5 : 'imu_accel_y'
        6 : 'imu_yaw_rate'
        7 : 'pose_x'
        8 : 'pose_y'
        9 : 'pose_yaw'
        10 : 'closest_wpt_x'
        11 : 'closest_wpt_y'
        12 : 'is_arrive'
        13 : 'target_wpt_x_t'
        14 : 'target_wpt_y_t'
        15 : 'target_wpt_x_t1'
        16 : 'target_wpt_y_t1'
        
Dataloader 
    - Inputs:
        -- Front camera image (cv_img I_front)
        -- IMU                (float a_x, float a_y, float yaw_rate)
        -- Route Information  (float Rp_t0, float Rh_t0, float Rp_t1, float Rh_t1)
        *Rp_t: Progress to current target route waypoint
        *Rh_t: Heading to current target route waypoint
    
    - Outputs:
        -- Steering angle
        -- Throttle
    TODO:
    - Data augmentation
        -- brightness
Dataset Analyzer
    - Visualize distribution of Steer/Thorttle
    - 
"""

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

class customDataset(Dataset):
    """
    Load Image and label dataset
    """
    # def __init__(self, dataset_dir, img_name_txt, label_txt, loss_function="ELBO", transforms=None):
    def __init__(self, dataset_dir, transforms=None, window=0):
        # - Path of root dataset dir
        self.dataset_dir_npz = os.path.join(dataset_dir, 'npz')
        # self.dataset_dir_csv = os.path.join(dataset_dir, 'csv')
        # self.dataset_dir_img = os.path.join(dataset_dir, 'img')
        # self.dataset_dir_bev = os.path.join(dataset_dir, 'bev')

        self.file_list = os.listdir(self.dataset_dir_npz)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # - load npz data
        file_path = self.file_list[index]
        sample_data = np.load(os.path.join(self.dataset_dir_npz, file_path), mmap_mode='r')
        data = {
                'image': torch.FloatTensor(sample_data['image']), # (T,1,224,224) [-1,+1]
                'state': torch.FloatTensor(sample_data['state']),
                'bev': torch.FloatTensor(sample_data['bev']), # (T,3,64,64) [-1,+1]
                'label': torch.FloatTensor(sample_data['label']),
                }

        return data, index

    def custom_pil_imshow(self, pil_img):
        pil_img_cp = copy.deepcopy(pil_img)
        cv_img = np.array(pil_img_cp)
        # Convert RGB to BGR
        cv_img = cv_img[:,:,::-1].copy()
        cv2.imshow("img", cv_img)
        cv2.waitKey()

    def custom_torch_imshow_rgb(self, torch_img, denormalize=False):
        torch_img_cp = copy.deepcopy(torch_img)
        np_img = torch_img_cp.numpy() # ndarray is 0~1 for cv imshow, not 0~255
        if denormalize:
            np_img += 1 # [-1,+1] to [0,1]
            np_img *= 0.5
            
        cv_img = np_img.copy().transpose(1, 2, 0) # (C,H,W) to (H,W,C) Convert RGB to BGR
        cv2.imshow("img", cv_img)
        cv2.waitKey()

    def custom_torch_imshow_gray(self, torch_img, denormalize=False):
        torch_img_cp = copy.deepcopy(torch_img)
        np_img = torch_img_cp.numpy().transpose(1, 2, 0) # (C,H,W) to (H,w,C) ndarray is 0~1 for cv imshow, not 0~255
        if denormalize:
            np_img += 1 # [-1,+1] to [0,1]
            np_img *= 0.5
            
        cv_img = np_img.copy() # Convert RGB to BGR
        cv2.imshow("img", cv_img)
        cv2.waitKey()

    def torch_to_cv_img(self, torch_img, denormalize=False):
        torch_img_cp = copy.deepcopy(torch_img) # (C,H,W)
        np_img = torch_img_cp.numpy().transpose(1, 2, 0) # (C,H,W) to (H,W,C) ndarray is 0~1 for cv imshow, not 0~255
        if denormalize:
            np_img += 1 # [-1,+1] to [0,1]
            np_img *= 0.5

        cv_img = np_img[:,:,::-1].copy() # Convert RGB to BGR

        return cv_img

    def cv_to_torch_img(self, cv_img):
        cv_img_cp = copy.deepcopy(cv_img)
        cv_img_cp = cv2.cvtColor(cv_img_cp, cv2.COLOR_BGR2RGB).astype(np.float32)
        cv_img_cp /= 255
        cv_img_cp *= 2
        cv_img_cp -= 1
        cv_img_cp = cv_img_cp.transpose(2,0,1) # (H, W, C) to (C H W)
        torch_img = torch.FloatTensor(cv_img_cp)
        return torch_img

    
    def add_noise(self, image, noise_type='gauss', dropout=0.5):
        """
        - noise_type: 
        - image: 
        """
        # Random dropout w.r.t. adding noise
        random_toss = random.random()
        if random_toss < dropout:
            # No adding noise
            return image

        else:
            if noise_type == "gauss":
                row,col,ch= image.shape
                mean = 0
                var = random.choice([0.01, 0.02])
                sigma = var**0.5
                gauss = np.random.normal(mean,sigma,(row,col,ch))
                gauss = gauss.reshape(row,col,ch)
                noisy = image + gauss
                # No overflow
                noisy = np.minimum(1., np.maximum(-1., noisy))
                return noisy

            elif noise_type == "motion_blur":
                # generating the kernel
                size = 8
                kernel_motion_blur = np.zeros((size, size))
                kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
                kernel_motion_blur = kernel_motion_blur / size

                # applying the kernel to the input image
                output = cv2.filter2D(image, -1, kernel_motion_blur)
                return output

            elif noise_type == "s&p":
                row,col,ch = image.shape
                s_vs_p = 0.5
                amount = 0.004
                out = np.copy(image)
                # Salt mode
                num_salt = np.ceil(amount * image.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt))
                        for i in image.shape]
                out[coords] = 1

                # Pepper mode
                num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper))
                        for i in image.shape]
                out[coords] = 0
                return out
            
            else:
                raise NotImplementedError

class DatasetVisualizer():
    def __init__(self, dataset_csv):
        self.dataset_csv = dataset_csv
        self.label_array = self.dataset_csv[['steer_t',
                                             'throttle_t',
                                             ]]
        self.STEER_NEUTRAL    = 1560 # (1960~1560~1160)
        self.THROTTLE_NEUTRAL = 1460 # (1860~1460~1060)
    
    def draw_label_distribution(self):
        self.label_array.loc[:,'steer_t'] -= self.STEER_NEUTRAL
        self.label_array.loc[:,'steer_t'] /= 400
        self.label_array.loc[:,'throttle_t'] -= self.THROTTLE_NEUTRAL
        self.label_array.loc[:,'throttle_t'] /= 400

        steer    = self.label_array["steer_t"]
        throttle = self.label_array["throttle_t"]

        plt.hist(steer, bins=100, density=False, alpha=0.7, histtype='stepfilled')
        plt.hist(throttle, bins=100, density=False, alpha=0.7, histtype='stepfilled')
        plt.show()

def main():
    dataset_dir  = "../dataset" # "../dataset"

    # Brightness / contrast / saturation claim nonnegative, and the hue in claim 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5 $.
    # new_transforms = torch.nn.Sequential(
        # transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
        # transforms.GaussianBlur(99),
        # transforms.RandomErasing(p=0.9, scale=(0.02, 0.03), ratio=(1.0, 1.0), value=0, inplace=False),
    # )
    # new_transforms = torch.jit.script(new_transforms)

    new_transforms = True

    BATCH_SIZE = 1000
    SEQUENCE_LENGTH = 4
    SHUFFLE_BATCH = True

    dataset = customDataset(dataset_dir=dataset_dir,
                            transforms=new_transforms,
                            window=3,
                            )


    # Sampler
    print("Total size of the dataset :", len(dataset))
    num_samples = len(dataset)

    # # - test sampler
    # sampled_indices = list(prioritized_sampler)
    # print("Count index '1000' :", sampled_indices.count(1000))
    # raise NotImplementedError 

    dataloader = DataLoader(dataset=dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=2,
                            drop_last=True,
                            # shuffle=SHUFFLE_BATCH,
                            )

    # # for getting one batch from dataloader without iteration
    # images, labels = next(iter(dataloader))

    # # Visualize dataset distribution
    # dataset_visualizer = DatasetVisualizer(dataset.dataset_csv)
    # dataset_visualizer.draw_label_distribution()

    for epoch in range(4):
        print("Epoch :", epoch)

        for i, (data, idx) in enumerate(dataloader):
            print("===== %d th batch in dataloader =====" %(i))
            print("Sampled indices :", idx)
            print(data["image"].size())  # input image
            print(data["bev"].size())    # input bev
            print(data["state"].size())  # input state
            print(data["label"].size())  # output label
            print("state dyn")
            print(data["state"][:,0:3])
            print("state route")
            print(data["state"][:,3:])
            for j in range(BATCH_SIZE):
                print("---------- %d th data in a batch" %(j))
                for k in range(SEQUENCE_LENGTH):
                    print("-------------- imu_accel_x : %f, imu_accel_y : %f, imu_yaw_rate : %f" \
                            %(data["state"][j,k,0], data["state"][j,k,1], data["state"][j,k,2]))
                    print("-------------- route_progress_t : %f, route_head_t : %f, route_progress_t1 : %f, route_head_t1 : %f" \
                            %(data["state"][j,k,3], data["state"][j,k,4], data["state"][j,k,5], data["state"][j,k,6]))
                    print("-------------- steer : %f, throttle : %f" %(data["label"][j,k,0], data["label"][j,k,1]))
                    dataset.custom_torch_imshow_rgb(data["image"][j,k], denormalize=True)
                    # dataset.custom_torch_imshow_rgb(data["bev"][j,k], denormalize=True)

if __name__ == '__main__':
    main()