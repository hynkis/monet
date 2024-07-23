#-*- coding: utf-8 -*-

#!/usr/bin/env python

# By Hyunki Seong.
# Email : hynkis@kaist.ac.kr

"""
- Collect deployment dataset
- Parse deployment dataset (using get_sync_data_from_inference_bag_monet.py)
- run eval.py

"""

import argparse
import datetime
import time
from matplotlib.pyplot import contour

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tools.custom_dataset_monet_eval import customDataset

# from baseline_model.MoNet_v01.codes.learner import IL
# from baseline_model.MoNet_v01_add.codes.learner import IL
from baseline_model.MoNet_v01_add_LGC.codes.learner_additive import IL
# from baseline_model.MoNet_v01_noLGC.codes.learner import IL

from utils import *

# for task classifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description="Neural Decision Network")

parser.add_argument('--cpu',           action="store_true", help='run on CPU (default: False)')
parser.add_argument('--gpu_num',          type=int, default=0,     metavar='N', help='GPU number for training')
parser.add_argument('--seed',           type=int, default=1234567, metavar='N',  help='random seed (default: 123456)')
parser.add_argument('--load_past_data', action="store_true",                    help='Load Past Training data (default: False)')
parser.add_argument('--save_interval',  type=int,  default=100,   metavar='N', help='Saving all data interval (default: 10)')
parser.add_argument('--num_workers',  type=int,  default=1,   metavar='N', help='Number of workers for dataloader (default: 4)')
parser.add_argument('--shuffle_batch',  type=bool,  default=False,   metavar='N', help='Whether shuffle batch of not (default: True)')

parser.add_argument('--batch_size',    type=int, default=1,     metavar='N', help='batch size (default: 128)')
parser.add_argument('--max_epoch',     type=int, default=100000,     metavar='N', help='max_epoch (default: 1000)')
parser.add_argument('--lr',             type=float, default=3e-4, metavar='G', help='learning rate. (default:1e-4 / 0.0003)')
parser.add_argument('--lr_lambda',     type=float, default=1.0, metavar='G', help='lambda for lr scheduler. (default: 0.99)')
parser.add_argument('--lr_interval',   type=int, default=2400,     metavar='N', help='max_epoch (default: 1000)')
parser.add_argument('--max_grad_norm',     type=float, default=50.0, metavar='G', help='maximum grad norm. (default: 5)')

parser.add_argument('--decision_std', type=float, default=0.1, metavar='N', help='standard deviation for Gaussian decision. heuristic: 1 / size_decision (default: 0.5)')
parser.add_argument('--ent_beta',     type=float, default=1e-4, metavar='N', help='weight for decision entropy loss (default: 1e-2)')
parser.add_argument('--boltz_alpha',  type=float, default=2.0, metavar='N', help='temperature factor for Boltzmann Distribution (smooth softmax) (default: 5.0)')

parser.add_argument('--roi_aug_prob',    type=float, default=0.5,     metavar='N', help='ROI augmentation prob (default: 0.5)')
parser.add_argument('--w_dec_loss',  type=float, default=5e-4, metavar='N', help='weight for contrastive decision loss (default: 1e-3)')
parser.add_argument('--w_ent_loss',  type=float, default=1e-4, metavar='N', help='weight for decision entropy loss (default: 1e-4; 1e-2)')
parser.add_argument('--use_dec_loss',  type=bool, default=True, metavar='N', help='Whether use decision loss or not (default: False)')
parser.add_argument('--use_ent_loss',  type=bool, default=True, metavar='N', help='Whether use entropy loss or not (default: False)')

parser.add_argument('--max_update_iter',   type=int, default=500000,     metavar='N', help='maximum update iter for annealing (default: 5.0)')
parser.add_argument('--save_prefix',     default="MoNet_v01", help='prefix at save path (default: )')

args = parser.parse_args()

# Pytorch config
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# Path (dataset)
# # - for task-wise sample data
# BASE_DATASET_PATH = "./eval_data/indoor_rccar/test_data_v1"
# DATASET_TOTAL_PATH = BASE_DATASET_PATH + "/total"
# - for rollout data
BASE_DATASET_PATH  = "./eval_data/indoor_rccar/result_rollout"
DATASET_TOTAL_PATH = BASE_DATASET_PATH + "/filtered"

# Path (model)
BASE_MODEL_PATH = "./baseline_model/MoNet_v01_add_LGC"
# BASE_MODEL_PATH = "./baseline_model/MoNet_v01_add"
# BASE_MODEL_PATH = "./baseline_model/MoNet_v01_noLGC"
# BASE_MODEL_PATH = "./baseline_model/MoNet_v01"

# Path (samples)
BASE_PATH = BASE_MODEL_PATH + '/results/final_model_result'
# DATASET_SAMPLE_PATH = BASE_PATH + '/MoNet_v01_data.csv'
DATASET_SAMPLE_PATH = BASE_PATH + '/MoNet_v01_add_LGC_data.csv'

index_CA_left_end  = 31
index_CA_right_end = 31+36
index_LT_end       = 31+36+59
index_RT_end       = 31+36+59+59
index_ST_end       = 31+36+59+59+64
index_ST_wall_end  = 31+36+59+59+64+318

data_idx_CA = list(range(0, index_CA_right_end))
data_idx_LT = list(range(index_CA_right_end, index_LT_end))
data_idx_RT = list(range(index_LT_end, index_RT_end))
data_idx_ST = list(range(index_RT_end, index_ST_end))
data_idx_ST_wall = list(range(index_ST_end, index_ST_wall_end))
data_idx_total = [data_idx_ST_wall + data_idx_ST, data_idx_LT, data_idx_RT, data_idx_CA]
# data_idx_total = [data_idx_ST_wall, data_idx_ST, data_idx_LT, data_idx_RT, data_idx_CA]

# - load decision data
data = pd.read_csv(DATASET_SAMPLE_PATH, header=None).to_numpy()
print("data :", data.shape)
# - parse latent decision data
data_decision = data[:,:16]

""" Train decision decoder """
DATASET_SAMPLE_PATH =  BASE_PATH + '/MoNet_v01_add_LGC_data_label.csv'
# DATASET_SAMPLE_PATH =  BASE_PATH + '/MoNet_v01_data_label.csv'
data_sample_decision = pd.read_csv(DATASET_SAMPLE_PATH, header=None)

x_decision = data_sample_decision.to_numpy()[:,:16]
y_decision = data_sample_decision.to_numpy()[:,-1:]

xtrain, xtest, ytrain, ytest = train_test_split(x_decision, y_decision, test_size=0.15)

svc = LinearSVC(dual=False, random_state=123)
params_grid = {"C": [10**k for k in range(-3, 4)]}

clf = GridSearchCV(svc, params_grid)
cclf = CalibratedClassifierCV(base_estimator=clf , cv=5)
cclf.fit(xtrain, ytrain.ravel())
print("Train decision decoder done!")

# Path (model)
LOAD_MODEL_DIR = BASE_MODEL_PATH + '/models' # models are not yet

SAVE_DATA_PATH = BASE_DATASET_PATH + '/MoNet_v01_add_LGC_rollout.csv'
# SAVE_DATA_PATH = BASE_DATASET_PATH + '/MoNet_v01_rollout.csv'
# SAVE_DATA_PATH = BASE_MODEL_PATH + '/results/MoNet_v01_data.csv'

LOAD_MODEL_PATH = LOAD_MODEL_DIR + '/2023-08-07_20-59-02_MoNet_v01_add_LGC_N1_total_230807_epoch_4754_iteration_650000.pth'
# LOAD_MODEL_PATH = LOAD_MODEL_DIR + '/2023-05-26_19-32-35_MoNet_v01_N1_total_230526_epoch_4754_iteration_650000.pth'

SAVE_DATA       = False
SHOW_IMG        = True
SHOW_DEC_TRAJ   = True
SHOW_DECISION   = True
WAIT_EACH_IMAGE = False

print("Load path")
print("DATASET_TOTAL_PATH :", DATASET_TOTAL_PATH)
print("LOAD_MODEL_PATH    :", LOAD_MODEL_PATH)

# Parameters
image_shape = (224,224)
bev_shape   = (64,64)
device = torch.device("cuda:" + str(args.gpu_num))
# device = torch.device("cuda")
if args.cpu:
    device = torch.device("cpu")

transforms = None
torch.autograd.set_detect_anomaly(True)

# Parameters
# - already filtered
MAX_EVAL_DATA_SIZE = 1500 # 1500
EVAL_DATA_START_INDEX = 0
EVAL_DATA_END_INDEX = EVAL_DATA_START_INDEX + MAX_EVAL_DATA_SIZE - 1

def to_np_img(img):
    # (re-normalize [-1,+1] to [0,1])
    img += 1 # (C,H,W)
    img *= 0.5
    return img

def to_att_img(att_map, shape=(224,224)):
    att_map = np.maximum(att_map, 0, dtype='float')
    att_map /= np.max(att_map)
    att_map = cv2.resize(att_map, shape) # (8,8) to (224,224)
    att_map = cv2.applyColorMap(np.uint8(255*att_map), cv2.COLORMAP_JET)
    return att_map

def predict_decision(curr_decision):
    input_decision = curr_decision.reshape(1,-1)
    test_prob = cclf.predict_proba(input_decision)
    ypred     = cclf.predict(input_decision)

    return test_prob, ypred


def main():
    # Define model
    model = IL(img_shape=image_shape, args=args)

    # Load Dataset
    dataset = customDataset(dataset_dir=DATASET_TOTAL_PATH,
                            transforms=transforms,
                            )
    # Define Dataloader
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle_batch,
                            num_workers=8,
                            drop_last=True
                            )

    # Eval data
    latent_decision_list = []
    decision_prob_list = []
    decision_ent_list = []
    action_output_list = []

    # Load model and training info
    # - load model weight
    model.load_model(LOAD_MODEL_PATH)
    model.monet.eval()
    print("Load model done")

    # Evaluation iteration
    with torch.no_grad():
        # Training iteration (Update iteration)
        for i, (data, _) in enumerate(dataloader):
            print("======================")
            print("----- %d th data -----" %(i))
            print("======================")
            if i < EVAL_DATA_START_INDEX:
                print("Pass this data")
                continue
            print(data["image"].size())  # input image  (batch, 1, 224, 224)
            print(data["bev"].size())    # input bev    (batch, 3, 60, 60)
            print(data["label"].size())  # output label (batch, 2)

            # to Device
            # - input
            X_img   = data["image"].to(device)
            X_bev   = data["bev"].to(device)
            X_state = data["state"][:, [1, 2]].to(device) # {a_y, yaw_rate}
            # - ground truth
            y       = data["label"]

            # - save attention map as distribution figure
            for j in range(args.batch_size):
                test_X_img       = X_img[0].unsqueeze(dim=0) # (1,224,224) -> (1,1,224,224)
                test_X_bev       = X_bev[0].unsqueeze(dim=0) # (3,60,60) -> (1,3,60,60)
                test_X_state     = X_state[0].unsqueeze(dim=0)
                a, _, _ = model.monet(test_X_img, test_X_bev)
                a_np = a.detach().cpu().numpy() # (steer, throttle), (1,2)
                
                print("%d th image in a batch" %(j))
                print("imu_accel_x : %f, imu_accel_y : %f, imu_yaw_rate : %f" \
                        %(data["state"][j,0], data["state"][j,1], data["state"][j,2]))
                print("steer_infer : %d, throttle_infer : %d" %(a_np[j,0]*400, a_np[j,1]*400))
                print("steer_label : %d, throttle_label : %d" %(y[j,0]*400, y[j,1]*400))

                # - save sample image (re-normalize [-1,+1] to [0,1])
                raw_img = to_np_img(test_X_img[0].cpu()) # (C,H,W)
                raw_bev = to_np_img(test_X_bev[0].cpu()) # (C,H,W)

                # - save sample image (numpy image. 0~1)
                raw_img_cv = to_np_img(test_X_img[0].cpu()) # (C,H,W)
                raw_img_cv = raw_img_cv.repeat(3, 1, 1) # (1,224,224) -> (3,224,224)
                raw_img_cv = raw_img_cv.permute(1, 2, 0).numpy() # (C,H,W) to (H,W,C)
                # raw_img_cv = cv2.applyColorMap(np.uint8(raw_img_cv*255.), cv2.COLORMAP_JET)
                raw_img_cv = np.uint8(raw_img_cv*255.)

                raw_bev_cv = to_np_img(test_X_bev[0].cpu()) # (C,H,W) -> # (3,224,224)
                raw_bev_cv = raw_bev_cv.permute(1, 2, 0).numpy() # (C,H,W) to (H,W,C)
                raw_bev_cv = np.uint8(raw_bev_cv*255.)

                # RGB to BGR (originally, route is red, but blue in the saved image by PIL)
                raw_img_cv_BGR = raw_img_cv[:,:,[2,1,0]]
                raw_bev_cv_BGR = raw_bev_cv[:,:,[2,1,0]]
                if SHOW_IMG:
                    cv2.imshow("raw_img_cv", raw_img_cv_BGR)
                    cv2.imshow("raw_bev_cv", raw_bev_cv_BGR)

                # - save attention map as distribution figure
                # if model.bpn.att_map is not None and model.bpn.att_map_d is not None:
                img_shape = (raw_img_cv.shape[1], raw_img_cv.shape[0])
                if model.monet.z_decision is not None \
                    and model.monet.att_map_sp is not None \
                    and model.monet.att_map_ct is not None:
                    # Define plotter class (for reducing memory leak)
                    plotter = PlotterClass()

                    # Visualization of the Spatial attention
                    #   - (batch, head, N, N(soft)) -(index:0)-> (head, N, N(soft)) -(head:i)-> (N, N(soft)) -(index:0)-> (N(soft),)

                    # - attention (mean pool)
                    spatial_att_head0_mean = model.monet.att_map_sp[0][0].mean(dim=0).view(6,6).cpu().numpy() # (64+5,)
                    spatial_att_head0_mean = np.maximum(spatial_att_head0_mean, 0, dtype='float')
                    spatial_att_head0_mean /= np.max(spatial_att_head0_mean)
                    spatial_att_head0_mean = cv2.resize(spatial_att_head0_mean, (raw_img_cv_BGR.shape[1], raw_img_cv_BGR.shape[0]))
                    spatial_att_head0_mean = cv2.applyColorMap(np.uint8(255*spatial_att_head0_mean), cv2.COLORMAP_JET)
                    spatial_att_head0_mean_BGR = spatial_att_head0_mean[:,:,[2,1,0]] # RGB to BGR
                    spatial_att_head0_mean_img_cv = cv2.addWeighted(spatial_att_head0_mean_BGR, 0.5, raw_img_cv_BGR, 0.5, 0)
                    spatial_att_head0_mean_img_cv = spatial_att_head0_mean_img_cv[:,:,::-1].copy() / 255. # numpy image
                    if SHOW_IMG:
                        cv2.imshow("att_head0", spatial_att_head0_mean_img_cv)

                    # Visualization of the Decision
                    # - (batch, N(soft)) -> (N(soft),)
                    z_decision_head0 = model.monet.z_decision[0].cpu().numpy() # (batch,dim) -> (dim,)
                    z_decision_head0_cv = plotter.gen_plot_img(z_decision_head0, viz_type="decision", title="", plot_fig=False, to_cv_image=True, resize=False)

                    # Real-time decision decoding
                    decision_prob, pred_decision = predict_decision(z_decision_head0)
                    print("decision_prob :", decision_prob)
                    print("pred_decision :", pred_decision)
                    z_decision_head0_bar_cv = plotter.gen_bar_img(decision_prob[0], title="", to_cv_image=True, resize=False)
                    latent_decision_list.append(z_decision_head0)
                    decision_prob_list.append(decision_prob[0])

                    # Calculate entropy of current decision
                    decision_ent = calc_entropy(torch.tensor(decision_prob[0]))
                    print("decision_ent :", decision_ent)
                    decision_ent_list.append(decision_ent.numpy()) # for (N, 1)

                    if SHOW_IMG:
                        cv2.imshow("z_decision_head0", z_decision_head0_cv)
                        cv2.imshow("z_decision_head0_bar", z_decision_head0_bar_cv)

                    # here, label is already inferred actions by learned model during real-world deployment.
                    action_output_list.append(y[0].numpy()) # for (1,2) tensor -> (2,) numpy; (N,1)
                    # action_output_list.append(a_np[0]) # for (1,2) -> (2,); (N,1)

                if WAIT_EACH_IMAGE:
                    cv2.waitKey(0)
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # Break
            if len(latent_decision_list) >= MAX_EVAL_DATA_SIZE:
                break
    
    latent_decision_list = np.array(latent_decision_list) # (N, dim)
    decision_prob_list   = np.array(decision_prob_list) # (N, dim)
    decision_ent_list = np.array(decision_ent_list) # (N, 1)
    action_output_list = np.array(action_output_list) # (N, 2)
    
    print("latent_decision_list :", latent_decision_list.shape)
    print("decision_prob_list   :", decision_prob_list.shape)
    print("decision_ent_list    :", decision_ent_list.shape)
    print("action_output_list   :", action_output_list.shape)
    # dataset_np:
    # - latent_decision_list[:,:16] # (N, 16)
    # - decision_prob_list[:,16:20] # (N, 4)
    # - decision_ent_list[:,20:21] # (N, 1)
    # - action_output_list[:,21:23] # (N, 2)
    dataset_np = np.hstack([latent_decision_list, decision_prob_list, decision_ent_list, action_output_list])
    dataset_df = pd.DataFrame(dataset_np) # (N, dim+2)
    
    # Plot decision trajectory
    if SHOW_DEC_TRAJ:
        # raw latent decision
        plt.figure()
        latent_decision_df_viz = pd.DataFrame(latent_decision_list.T) # (dim, N)
        sns.heatmap(latent_decision_df_viz, robust=True, cmap='binary', xticklabels=10,  yticklabels=False, cbar=True)
        # sns.heatmap(latent_decision_df, robust=True, cmap='YlGnBu', xticklabels=100,  yticklabels=False, cbar=True)

        # probability of decoded decision
        plt.figure()
        prob_decision_df_viz = pd.DataFrame(decision_prob_list.T) # (dim, N)
        sns.heatmap(prob_decision_df_viz, robust=True, cmap='binary', xticklabels=10,  yticklabels=False, cbar=True)

        # entropy of decoded decision
        plt.figure()
        plt.plot(decision_ent_list[:, 0], '-')

        # action output
        plt.figure()
        plt.plot(action_output_list[:, 0], '-')

        plt.show()

    # Save data
    if SAVE_DATA:
        dataset_df.to_csv(SAVE_DATA_PATH, index=False, header=False)
        print("Saving data is done")


if __name__ == '__main__':
    main()
