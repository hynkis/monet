# -*- coding: utf-8 -*-

# Reference: https://stackoverflow.com/questions/56774582/adding-custom-labels-to-pytorch-dataloader-dataset-does-not-work-for-custom-data
# Auther : Hyunki Seong, hynkis@kaist.ac.kr

import numpy as np
import torch
from PIL import Image
import copy
import random

import cv2

def load_image_rgb(path=None, normalize=False, resize=True, resize_size=None):
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

def load_image_gray(path=None, normalize=False, resize_size=None):
    # can use any other library too like OpenCV as long as you are consistent with it
    raw_image = Image.open(path) # grayscale image
    # raw_image = np.transpose(raw_image.resize(self.RESIZE_SHAPE), (1,0)) # (H,W,C) to (C,W,H)
    raw_image = np.array(raw_image.resize(resize_size)) # to (H,W)
    raw_image = np.expand_dims(raw_image, axis=0) # (224,224) to (1,224,224): (C,H,W)
    imx_t = np.array(raw_image, dtype=np.float32) / 255.
    if normalize:
        imx_t -= 0.5
        imx_t *= 2.

    return imx_t

def load_image_gray_w_crop(path, state, label, normalize=False, crop_points=None, crop_shape=None, resize_size=None):
    raw_image = Image.open(path) # grayscale image
    # print("raw_image :", raw_image.size)
    
    # crop image without shifting
    shifting_size = 0
        
    # PIL crop: (leftup_x, leftup_y, rightdown_x, rightdown_y)
    img_crop_area = (crop_points[0]+shifting_size, crop_points[1], crop_points[0]+shifting_size + crop_shape[0], crop_points[1]+crop_shape[1])
    raw_image = raw_image.crop(img_crop_area)

    # raw_image = np.transpose(raw_image.resize(self.RESIZE_SHAPE), (1,0)) # (H,W,C) to (C,W,H)
    raw_image = np.array(raw_image.resize(resize_size)) # to (H,W)
    raw_image = np.expand_dims(raw_image, axis=0) # (224,224) to (1,224,224): (C,H,W)
    imx_t = np.array(raw_image, dtype=np.float32) / 255.
    if normalize:
        imx_t -= 0.5
        imx_t *= 2.

    return imx_t, state, label


def custom_pil_imshow(pil_img):
    pil_img_cp = copy.deepcopy(pil_img)
    cv_img = np.array(pil_img_cp)
    # Convert RGB to BGR
    cv_img = cv_img[:,:,::-1].copy()
    cv2.imshow("img", cv_img)
    cv2.waitKey()

def custom_torch_imshow_rgb(torch_img, denormalize=False):
    torch_img_cp = copy.deepcopy(torch_img)
    np_img = torch_img_cp.numpy() # ndarray is 0~1 for cv imshow, not 0~255
    if denormalize:
        np_img += 1 # [-1,+1] to [0,1]
        np_img *= 0.5
        
    cv_img = np_img.copy().transpose(1, 2, 0) # (C,H,W) to (H,W,C) Convert RGB to BGR
    cv2.imshow("img", cv_img)
    cv2.waitKey()

def custom_torch_imshow_gray(torch_img, denormalize=False):
    torch_img_cp = copy.deepcopy(torch_img)
    np_img = torch_img_cp.numpy().transpose(1, 2, 0) # (C,H,W) to (H,w,C) ndarray is 0~1 for cv imshow, not 0~255
    if denormalize:
        np_img += 1 # [-1,+1] to [0,1]
        np_img *= 0.5
        
    cv_img = np_img.copy() # Convert RGB to BGR
    cv2.imshow("img", cv_img)
    cv2.waitKey()

def torch_to_cv_img(torch_img, denormalize=False):
    torch_img_cp = copy.deepcopy(torch_img) # (C,H,W)
    np_img = torch_img_cp.numpy().transpose(1, 2, 0) # (C,H,W) to (H,W,C) ndarray is 0~1 for cv imshow, not 0~255
    if denormalize:
        np_img += 1 # [-1,+1] to [0,1]
        np_img *= 0.5

    cv_img = np_img[:,:,::-1].copy() # Convert RGB to BGR

    return cv_img

def cv_to_torch_img(cv_img):
    cv_img_cp = copy.deepcopy(cv_img)
    cv_img_cp = cv2.cvtColor(cv_img_cp, cv2.COLOR_BGR2RGB).astype(np.float32)
    cv_img_cp /= 255
    cv_img_cp *= 2
    cv_img_cp -= 1
    cv_img_cp = cv_img_cp.transpose(2,0,1) # (H, W, C) to (C H W)
    torch_img = torch.FloatTensor(cv_img_cp)
    return torch_img


def pi_to_pi_array(np_array):
    np_array[np.where(np_array>np.pi)] -= 2*np.pi
    np_array[np.where(np_array<-np.pi)] += 2*np.pi

    return np_array

def calc_route_info(x_list, y_list, yaw_list, target_x_list, target_y_list):
    route_dist = np.sqrt(np.power(x_list - target_x_list, 2) + np.power(y_list - target_y_list, 2))
    route_heading = np.arctan2(target_y_list - y_list, target_x_list - x_list) - yaw_list
    # Pi to Pi
    route_heading = pi_to_pi_array(route_heading)

    return route_dist, route_heading

def map_route_progress(route_dist, max_route_dist, min_route_dist):
    # route_dist: ndarray
    route_dist = np.minimum(max_route_dist, np.maximum(min_route_dist, route_dist))
    # route_dist -> route_progress = 
    route_progress = -1./(max_route_dist - min_route_dist) * (route_dist - max_route_dist)
    route_progress = 2*route_progress - 1 # 0~1 to -1~1

    return route_progress

def add_noise(image, noise_type='gauss', dropout=0.5):
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

