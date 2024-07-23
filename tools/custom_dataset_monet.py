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

class PriotizedDataSampler(object):
    def __init__(self, num_total_data, alpha=0.6, beta=0.4):
        # Parameters
        self.num_total_data = num_total_data
        self.probs   = np.ones(self.num_total_data) / self.num_total_data # ndarray (N,)
        self.prios   = np.ones(self.num_total_data)*1e5        # ndarray (N,). all the inital samples have large priority
        self.alpha = alpha
        self.beta  = beta
        
        # Sampler
        self.sampler = WeightedRandomSampler(weights=self.probs,
                                            num_samples=self.num_total_data,
                                            replacement=True,
                                            )
    
    def compute_weights(self, batch_indices):
        """
        Compute importance sampling weights after each epoch
        (Before every update iteration)
            - w_j = (N * P(j))^-beta / max_i(w_i)
        """
        weights = (self.num_total_data * self.probs[batch_indices]) ** (-self.beta)
        weights /= weights.max()

        return weights

    def update_prios(self, losses, batch_indices):
        """
        Update sample priorities after each update iteration
        (After every update iteration)
        """
        self.prios[batch_indices] = np.abs(losses) + 1e-5 # p = abs(loss) + epsilon

    def update_probs(self):
        """
        Update Sampling probability after each epoch
        (After finishing an epoch)
        """
        self.probs = self.prios ** self.alpha
        self.probs /= self.probs.sum()

    def update_sampler(self):
        """
        Update Sampler
        (After finishing an epoch)
        """
        self.sampler = WeightedRandomSampler(weights=self.probs,
                                             num_samples=self.num_total_data,
                                             replacement=True,
                                             )

class customDataset(Dataset):
    """
    Load Image and label dataset
    """
    # def __init__(self, dataset_dir, img_name_txt, label_txt, loss_function="ELBO", transforms=None):
    def __init__(self, dataset_dir, transforms=None, window=0, roi_aug_prob=0.5):
        # - Params
        self.RESIZE_SHAPE = (224,224)
        self.RESIZE_SHAPE_BEV = (64,64) # dataset is (60,60)
        self.CROP_POINT   = (100,70) # (100,120)
        self.CROP_SHAPE   = (440,240) # (640,480) to (440,240) # (640,480) to (440,440)
        self.SHIFT_PIXEL  = 50 # shifting pixel size to left/right for image shifting augmentation. (total 2 times)
        
        self.STEER_NEUTRAL    = 1415 # (left 1100~1415~1900 right)
        self.THROTTLE_NEUTRAL = 1500 # (slow 1900~1500~1100 fast) 1455
        self.DEL_STEER_MAX = 400
        self.DEL_THROTTLE_MAX = 400
        self.ROI_AUG_PROB = roi_aug_prob # prob of roi augmentation
        self.ROI_AUG_STEER_THERS = 0.02 # based on steer label histogram result.

        # - Load img and label data with multiple trajectory data
        self.dataset_csv = None
        self.image_list = None  # input - image
        self.state_array = None # input - state, route_info
        self.label_array = None # output - steer, throttle

        self.window = window # 10

        # - Path of root dataset dir
        self.dataset_dir_csv = os.path.join(dataset_dir, 'csv')
        self.dataset_dir_img = os.path.join(dataset_dir, 'img')
        self.dataset_dir_bev = os.path.join(dataset_dir, 'bev')

        file_list = os.listdir(self.dataset_dir_csv)
        print("file_list :")
        print(file_list)
        file_list_csv = [file for file in file_list if file.endswith(".csv")]

        for file_csv in file_list_csv:
            file_path_csv = os.path.join(self.dataset_dir_csv, file_csv)
            dataset_csv = pd.read_csv(file_path_csv)
            if self.dataset_csv is None:
                self.dataset_csv = dataset_csv
            else:
                self.dataset_csv = pd.concat([self.dataset_csv, dataset_csv], ignore_index=True)
            print("dataset shape :", self.dataset_csv.shape)
        
        # - List of images to load in csv files
        self.image_list = self.dataset_csv['img_path']

        # - Array of states in csv files
        self.state_array_raw = self.dataset_csv[['pose_x',
                                                 'pose_y',
                                                 'pose_yaw',
                                                 ]]

        pose_x_list = self.state_array_raw['pose_x'].to_numpy()
        pose_y_list = self.state_array_raw['pose_y'].to_numpy()
        pose_yaw_list = self.state_array_raw['pose_yaw'].to_numpy()

        self.state_array = self.dataset_csv[['imu_accel_x', 'imu_accel_y', 'imu_yaw_rate']]

        # - Array of label in csv files
        self.label_array = self.dataset_csv[['steer_t',
                                             'throttle_t',
                                             ]]
        # normalize steer and throttle
        # - steer
        norm_steer = self.label_array.loc[:,'steer_t'] - self.STEER_NEUTRAL
        norm_steer = norm_steer / self.DEL_STEER_MAX
        norm_steer = np.clip(norm_steer, -1, 1)
        self.label_array.loc[:,'steer_t'] = norm_steer
        # - throttle
        norm_throttle = self.label_array.loc[:,'throttle_t'] - self.THROTTLE_NEUTRAL
        norm_throttle = norm_throttle / self.DEL_THROTTLE_MAX
        norm_throttle = np.clip(norm_throttle, -1, 1)
        self.label_array.loc[:,'throttle_t'] = norm_throttle

        # Apply transforms if you need
        self.transforms = transforms

        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # - Load state data at every iteration
        state = self.state_array.loc[index,:].to_numpy()

        # - Load label data at every iteration
        label = self.label_array.loc[index,:].to_numpy()
        
        # # - Load image data at every iteration
        # img_name = self.image_list[index]
        # image_path = os.path.join(self.dataset_dir_img, img_name)
        # image = self.load_image_gray(path=image_path, normalize=True)

        # - Load image data at every iteration with random shifting augmentation
        img_name = self.image_list[index]
        image_path = os.path.join(self.dataset_dir_img, img_name)
        # image = self.load_image_gray(path=image_path, normalize=True)
        image, state, label = self.load_image_gray_w_randAugment(path=image_path, state=state, label=label, normalize=True)

        if self.transforms:
            rand_noise_type = random.choice(['gauss', 'motion_blur'])  
            image = self.add_noise(image, noise_type=rand_noise_type, dropout=0.5)

        # - Load Bird-Eye's View topological map image
        bev_path = os.path.join(self.dataset_dir_bev, img_name) # bev file name is same with camera image.
        bev = self.load_image_rgb(path=bev_path, normalize=True, resize=True, resize_size=self.RESIZE_SHAPE_BEV)

        data = {
                'image': torch.FloatTensor(image), # (1,224,224) [-1,+1]
                'state': torch.FloatTensor(state),
                'bev': torch.FloatTensor(bev), # (3,64,64) [-1,+1]
                'label': torch.FloatTensor(label),
                }

        return data, index

    def load_image_rgb(self, path=None, normalize=False, resize=True, resize_size=None):
        # can use any other library too like OpenCV as long as you are consistent with it
        raw_image = Image.open(path) # RGB image
        # raw_image = np.transpose(raw_image.resize(self.RESIZE_SHAPE), (2,1,0)) # (H,W,C) to (C,W,H)
        if resize:
            raw_image = np.transpose(raw_image.resize(resize_size), (2,0,1)) # (H,W,C) to (C,H,W)
        else:
            raw_image = np.transpose(raw_image, (2,0,1)) # (H,W,C) to (C,H,W)

        imx_t = np.array(raw_image, dtype=np.float32) / 255.
        if normalize:
            imx_t -= 0.5
            imx_t *= 2.

        return imx_t

    def load_image_gray(self, path=None, normalize=False):
        # can use any other library too like OpenCV as long as you are consistent with it
        raw_image = Image.open(path) # grayscale image
        # raw_image = np.transpose(raw_image.resize(self.RESIZE_SHAPE), (1,0)) # (H,W,C) to (C,W,H)
        raw_image = np.array(raw_image.resize(self.RESIZE_SHAPE)) # to (H,W)
        raw_image = np.expand_dims(raw_image, axis=0) # (224,224) to (1,224,224): (C,H,W)
        imx_t = np.array(raw_image, dtype=np.float32) / 255.
        if normalize:
            imx_t -= 0.5
            imx_t *= 2.

        return imx_t

    def load_image_gray_w_randAugment(self, path, state, label, normalize=False):
        raw_image = Image.open(path) # grayscale image
        # print("raw_image :", raw_image.size)
        # - Random image shifting (only if steer label is small. The value is based on the steer label histogram result.)
        if np.random.rand() < self.ROI_AUG_PROB and abs(label[0]) < self.ROI_AUG_STEER_THERS:
            # ROI shifting
            rand_shift = random.choice([-2, -1.5, -1, 1, 1.5, 2])
            shifting_size = int(self.SHIFT_PIXEL * rand_shift) # 50 * [-2,-1,0,+1,+2]
            label[0] -= shifting_size * 0.5 * 1/self.DEL_STEER_MAX # 50/400 * 0.5 * [-2,-1,0,+1,+2] --> [-0.125, -0.062, 0, +0.062, +0.125]
        else:
            shifting_size = 0

        # # - Random image shifting (only straight driving. not in intersection)
        # if state[3] < 0.5:
        #     # ROI shifting when route progress < 0.5
        #     rand_shift = random.choice([-2, -1.5, -1, 0, 1, 1.5, 2])
        #     shifting_size = int(self.SHIFT_PIXEL * rand_shift) # 50 * [-2,-1,0,+1,+2]
        #     label[0] += shifting_size * 1/self.DEL_STEER_MAX. # 50/400 * [-2,-1,0,+1,+2]
        # else:
        #     shifting_size = 0
            
        # PIL crop: (leftup_x, leftup_y, rightdown_x, rightdown_y)
        img_crop_area = (self.CROP_POINT[0]+shifting_size, self.CROP_POINT[1], self.CROP_POINT[0]+shifting_size + self.CROP_SHAPE[0], self.CROP_POINT[1]+self.CROP_SHAPE[1])
        raw_image = raw_image.crop(img_crop_area)

        # raw_image = np.transpose(raw_image.resize(self.RESIZE_SHAPE), (1,0)) # (H,W,C) to (C,W,H)
        raw_image = np.array(raw_image.resize(self.RESIZE_SHAPE)) # to (H,W)
        raw_image = np.expand_dims(raw_image, axis=0) # (224,224) to (1,224,224): (C,H,W)
        imx_t = np.array(raw_image, dtype=np.float32) / 255.
        if normalize:
            imx_t -= 0.5
            imx_t *= 2.

        return imx_t, state, label

    def custom_pil_imshow(self, pil_img, window_name="img"):
        pil_img_cp = copy.deepcopy(pil_img)
        cv_img = np.array(pil_img_cp)
        # Convert RGB to BGR
        cv_img = cv_img[:,:,::-1].copy()
        cv2.imshow(window_name, cv_img)
        cv2.waitKey(0)

    def custom_torch_imshow_rgb(self, torch_img, denormalize=False, window_name="img"):
        torch_img_cp = copy.deepcopy(torch_img)
        np_img = torch_img_cp.numpy() # ndarray is 0~1 for cv imshow, not 0~255
        if denormalize:
            np_img += 1 # [-1,+1] to [0,1]
            np_img *= 0.5
            
        cv_img = np_img.copy().transpose(1, 2, 0) # (C,H,W) to (H,W,C) Convert RGB to BGR
        cv2.imshow(window_name, cv_img)
        cv2.waitKey(0)

    def custom_torch_imshow_gray(self, torch_img, denormalize=False, window_name="img"):
        torch_img_cp = copy.deepcopy(torch_img)
        np_img = torch_img_cp.numpy().transpose(1, 2, 0) # (C,H,W) to (H,w,C) ndarray is 0~1 for cv imshow, not 0~255
        if denormalize:
            np_img += 1 # [-1,+1] to [0,1]
            np_img *= 0.5
            
        cv_img = np_img.copy() # Convert RGB to BGR
        cv2.imshow(window_name, cv_img)
        cv2.waitKey(0)

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

    def calc_route_info(self, x_list, y_list, yaw_list, target_x_list, target_y_list):
        route_dist = np.sqrt(np.power(x_list - target_x_list, 2) + np.power(y_list - target_y_list, 2))
        route_heading = np.arctan2(target_y_list - y_list, target_x_list - x_list) - yaw_list
        # Pi to Pi
        route_heading = self.pi_to_pi_array(route_heading)

        return route_dist, route_heading
    
    def pi_to_pi_array(self, np_array):
        np_array[np.where(np_array>np.pi)] -= 2*np.pi
        np_array[np.where(np_array<-np.pi)] += 2*np.pi

        return np_array

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
        self.STEER_NEUTRAL    = 1415 # (1900~1415~1100)
        self.THROTTLE_NEUTRAL = 1500 # (1900~1500~1100) 1455
        self.DEL_STEER_MAX = 400
        self.DEL_THROTTLE_MAX = 400
    
    def draw_label_distribution(self):
        # normalize steer and throttle
        # - steer
        norm_steer = self.label_array.loc[:,'steer_t'] - self.STEER_NEUTRAL
        norm_steer = norm_steer / self.DEL_STEER_MAX
        norm_steer = np.clip(norm_steer, -1, 1)
        self.label_array.loc[:,'steer_t'] = norm_steer
        # - throttle
        norm_throttle = self.label_array.loc[:,'throttle_t'] - self.THROTTLE_NEUTRAL
        norm_throttle = norm_throttle / self.DEL_THROTTLE_MAX
        norm_throttle = np.clip(norm_throttle, -1, 1)
        self.label_array.loc[:,'throttle_t'] = norm_throttle

        # self.label_array.loc[:,'steer_t'] -= self.STEER_NEUTRAL
        # self.label_array.loc[:,'steer_t'] /= self.DEL_STEER_MAX
        # self.label_array.loc[:,'throttle_t'] -= self.THROTTLE_NEUTRAL
        # self.label_array.loc[:,'throttle_t'] /= self.DEL_THROTTLE_MAX

        steer    = self.label_array["steer_t"]
        throttle = self.label_array["throttle_t"]

        plt.hist(steer, bins=500, density=False, alpha=0.7, histtype='stepfilled')
        # plt.hist(throttle, bins=100, density=False, alpha=0.7, histtype='stepfilled')
        plt.show()

def main():
    # dataset_dir  = "../dataset"
    dataset_dir  = "../dataset/dataset_train"

    # Brightness / contrast / saturation claim nonnegative, and the hue in claim 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5 $.
    # new_transforms = torch.nn.Sequential(
        # transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
        # transforms.GaussianBlur(99),
        # transforms.RandomErasing(p=0.9, scale=(0.02, 0.03), ratio=(1.0, 1.0), value=0, inplace=False),
    # )
    # new_transforms = torch.jit.script(new_transforms)

    transforms = False # gaussian noise augmentation (ROI shifting is default)

    BATCH_SIZE = 1
    SHUFFLE_BATCH = False

    dataset = customDataset(dataset_dir=dataset_dir,
                            transforms=transforms,
                            window=3,
                            )


    # Sampler
    print("Total size of the dataset :", len(dataset))
    num_samples = len(dataset)
    # num_samples = 10
    prio_sampler = PriotizedDataSampler(num_total_data=num_samples)

    # # - test sampler
    # sampled_indices = list(prioritized_sampler)
    # print("Count index '1000' :", sampled_indices.count(1000))
    # raise NotImplementedError 

    dataloader = DataLoader(dataset=dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=2,
                            drop_last=True,
                            # sampler=prio_sampler.sampler,
                            shuffle=SHUFFLE_BATCH,
                            )

    # # for getting one batch from dataloader without iteration
    # images, labels = next(iter(dataloader))

    # Visualize dataset distribution
    dataset_visualizer = DatasetVisualizer(dataset.dataset_csv)
    dataset_visualizer.draw_label_distribution()

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
            print(data["state"][:,:3])
            for j in range(BATCH_SIZE):
                print("---------- %d th image in a batch" %(j))
                print("-------------- imu_accel_x : %f, imu_accel_y : %f, imu_yaw_rate : %f" \
                                             %(data["state"][j,0], data["state"][j,1], data["state"][j,2]))
                print("-------------- steer : %f, throttle : %f" %(data["label"][j,0], data["label"][j,1]))
                dataset.custom_torch_imshow_rgb(data["image"][j], denormalize=True, window_name="img")
                dataset.custom_torch_imshow_rgb(data["bev"][j], denormalize=True, window_name="bev")
            
            # Update phase
            weights = prio_sampler.compute_weights(batch_indices=idx)
            # do network update...
            losses = np.random.rand(len(idx))
            prio_sampler.update_prios(losses=losses, batch_indices=idx)
            print("weights :", weights)
            print("prios   :", prio_sampler.prios)

        # Update priority
        prio_sampler.update_probs()
        prio_sampler.update_sampler()
        print("probs :", prio_sampler.probs)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=1,
                                drop_last=True,
                                sampler=prio_sampler.sampler,
                                # shuffle=SHUFFLE_BATCH,
                                )
        print("============= dataloader is updated. =============")

if __name__ == '__main__':
    main()