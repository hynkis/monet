# imports
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils import *


from torch.utils.data.sampler import SubsetRandomSampler
from RAdam import radam

import cv2
import matplotlib.image as mpimg
import numpy as np
import csv
import requests
import zipfile
import time
import pandas as pd

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
        2 : 'imu_accel_x'
        3 : 'imu_accel_y'
        4 : 'imu_yaw_rate'
        5 : 'pose_x'
        6 : 'pose_y'
        7 : 'pose_yaw'
        8 : 'closest_wpt_x'
        9 : 'closest_wpt_y'
        10 : 'is_arrive'
        11 : 'target_wpt_x_t'
        12 : 'target_wpt_y_t'
        13 : 'target_wpt_x_t1'
        14 : 'target_wpt_y_t1'
        
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
"""

class DataDownloader:
    def __init__(self):
        pass


class CustomDataset(data.Dataset):
    def __init__(self, csv_file_path, image_dir, transform = None):
        self.csv_file_path = csv_file_path
        self.image_dir = image_dir
        self.transform = transform

        self.examples = []

        with open(self.csv_file_path) as csvfile:
          reader = csv.reader(csvfile)
          next(reader, None)
          for line in reader:
              self.examples.append(line)

    def __getitem__(self, index):
        example = self.examples[index]
        center, left, right = example[0], example[1], example[2]
        steering_angle = float(example[3])

        if np.random.rand() < 0.6:
            image, steering_angle = augument(self.image_dir, center, left, right, steering_angle)
        else:
            image = load_image(self.image_dir, center) 
    
        image = preprocess(image)
        
        if self.transform is not None:
            image = self.transform(image)
           
        return image, steering_angle

    def __len__(self):
        return len(self.examples)


