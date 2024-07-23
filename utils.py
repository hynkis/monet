
# imports
import os
import sys
import math

import torch
import torch.linalg as linalg

import cv2
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import seaborn as sns

import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
import ot # for computing wasserstein distance
from scipy.stats import wasserstein_distance as wasser # for computing wasserstein distance

# Algorithms
def calc_2_WD(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''

    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvalsh(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()

def compute_WD(data1, data2):
    """
    data1: A = np.array([0.3, 0.2, 0.1, 0.4])
    data2: B = np.array([0.3, 0.2, 0.2, 0.3])

    # for moving cost from ith index in A to jth index in B
    ia = np.arange(0, A.shape[0], 1)
    ib = np.arange(0, B.shape[0], 1)
    n = A.shape[0]
    M = ot.dist(ia.reshape((n, 1)), ib.reshape((n, 1)), 'euclidean')
    
    # compute wasserstein distance
    W = ot.emd2(A, B, M)
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    # for moving cost from ith index in A to jth index in B
    ia = np.arange(0, data1.shape[0], 1)
    ib = np.arange(0, data2.shape[0], 1)
    n = data1.shape[0]
    M = ot.dist(ia.reshape((n, 1)), ib.reshape((n, 1)), 'euclidean')

    # compute wasserstein distance
    W = ot.emd2(data1, data2, M)
    # W = ot.sinkhorn2(data1, data2, M, 1)
    # W = ot.sinkhorn2(data1, data2, M)
    return W

    # Wsci = wasser(data1, data2)
    # return Wsci

# reorder tensor elements
def reorder_tensor(tensor):
    # if input tensor  : [[0,0,0], [1,1,1], [2,2,2]]
    #    output tensor : [[2,2,2], [0,0,0], [1,1,1]]

    batch = tensor.shape[0]
    reorder_ind = [batch-1] + list(range(batch-1)) # if batch=5, reorder_ind = [4, 0, 1, 2, 3]
    reorder_ind = torch.tensor(reorder_ind)
    _, inds = torch.sort(reorder_ind)

    return tensor[inds]

# calculate entropy
def calc_entropy(dist, ets=1e-15):
	return -torch.sum(dist * torch.log(dist + ets), dim=-1, keepdim=True)

# Helper defs
class PlotterClass:
    def gen_plot_img(self, data, viz_type="decision", plot_fig=False, title="", to_cv_image=False, resize=True):
        """Create a pyplot plot and save to buffer."""
        plt.figure(figsize=(5, 5))
        plt.plot(data)
        if viz_type == "att_max_pool":
            plt.ylim(0.0, 0.1)

        elif viz_type == "att_mean_pool":
            plt.ylim(0.0, 0.04)

        elif viz_type == "decision":
            # plt.ylim(-0.05, 0.35) # for softmax decision
            plt.ylim(-1.5, 1.5) # for noPlan latent decision
            # plt.ylim(-1.0, 1.0) # for additive latent decision

            # plt.ylim(-0.1, 0.3) # for softmax decision
            # plt.ylim(-1.1, 1.1) # for tanh latent decision
        else:
            raise NotImplementedError

        plt.title(title)
        plt.grid()
        plt.xlim(-1, 16)

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        pil_image = PIL.Image.open(buf)
        if resize:
            pil_image = pil_image.resize((224, 224))    
        # image = ToTensor()(image).unsqueeze(0)
        image = ToTensor()(pil_image) # (C H W), not (1 C H W)

        if plot_fig is True:
            plt.show()

        plt.cla()
        plt.clf()
        plt.close('all')

        if to_cv_image == True:
            # use numpy to convert the pil_image into a numpy array
            numpy_image=np.array(pil_image)  

            # convert to a openCV2 image and convert from RGB to BGR format
            opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            return opencv_image

        return image

    def gen_heatmap_img(self, data, is_att=False, plot_fig=False, title=""):
        """Create a pyplot heatmap and save to buffer."""
        plt.figure()
        ax = sns.heatmap(data)
        ax.invert_yaxis()
        plt.title(title)

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = image.resize((224, 224))
        # image = ToTensor()(image).unsqueeze(0)
        image = ToTensor()(image) # (C H W), not (1 C H W)

        if plot_fig is True:
            plt.show()

        plt.cla()
        plt.clf()
        plt.close('all')

        return image

    def gen_bar_img(self, data, title='Classification Probabilites', to_cv_image=False, resize=True):
        """Create a pyplot bar plot."""
        plt.figure(figsize=(5, 5))
        # Class info
        class_label = ['ST', 'LT', 'RT', 'CA']

        sns.barplot(x=class_label, y=data, width=0.5)

        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.ylim([0, 1.0])

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        pil_image = PIL.Image.open(buf)
        if resize:
            pil_image = pil_image.resize((224, 224))    
        # image = ToTensor()(image).unsqueeze(0)
        image = ToTensor()(pil_image) # (C H W), not (1 C H W)

        plt.cla()
        plt.clf()
        plt.close('all')

        if to_cv_image == True:
            # use numpy to convert the pil_image into a numpy array
            numpy_image=np.array(pil_image)  

            # convert to a openCV2 image and convert from RGB to BGR format
            opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            return opencv_image

        return image


# --------------------------------------------------------------

def save_seperate_data(front_img, bev_img, veh_state, label, index, save_dir):
    data_name = str(index).zfill(8) + ".npz"
    data_path = os.path.join(save_dir, data_name)
    np.savez_compressed(data_path,
                        front_img = front_img,
                        bev_img = bev_img,
                        veh_state = veh_state,
                        label = label,
                        index = index,
                        )

# ---------------------------------------------------------------

def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    name = image_file.split('/')[-1]
    return mpimg.imread(os.path.join(data_dir, name))

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]  # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    print(image.shape)
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    # image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle
