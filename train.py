#-*- coding: utf-8 -*-

#!/usr/bin/env python

# By Hyunki Seong.
# Email : hynkis@kaist.ac.kr

# for training, run (-np means # of gpus):
# CUDA_VISIBLE_DEVICES=1 python3 train.py

import argparse
import datetime
import time
import os
import pathlib
import shutil
import tracemalloc
from flask import Flask
app = Flask(__name__) # https://stackoverflow.com/questions/55113041/attributeerror-module-flask-app-has-no-attribute-route
import gc

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trans
from tensorboardX import SummaryWriter
# from tools.custom_dataset_processed import customDataset
# from tools.custom_dataset import customDataset
from tools.custom_dataset_monet import customDataset

from learner import IL
from utils import *

"""
Training log

2023.05.09
- train N1 total data
- no entropy loss 

2023.05.13
- add unsupervised decision loss (cosine similarity)

2023.05.14
- add decision loss + entropy loss (regularization)

2023.05.14
- w_dec_loss = 5e-3
- w_ent_loss = 5e-4
- target_label = torch.sign(cos_sim(z_i_flat, z_j_flat))
- target_weight = torch.abs(cos_sim(z_i_flat, z_j_flat))
- L_cos = torch.sum(target_weight * criterion(h_i, h_j, target_label)) / batch

2023.05.15
- w_dec_loss = 1e-2
- w_ent_loss = None

2023.05.15
- w_dec_loss = 5e-3

2023.05.15
- rollback (L_cos = torch.sum(sign * similarity) / batch)
- rollback (similarity := 2*cos_sim - 1)
- use dec ent losses

2023.05.17
- add another last layer (modulate the middle of layers)
: fc (relu) -> fc (relu) -> fc (modulation) -> fc (relu) -> fc (tanh)

2023.05.25
- perception similarity (similarity := 2*cos_sim - 1) to (similarity := cos_sim)
==> bad

2023.05.26
- rollback perception similarity (similarity := 2*cos_sim - 1)
- w_dec_loss: 1e-3 --> 5e-4

2023.06.03
- no entropy loss

TODO

"""

tracemalloc.start()
my_snapshot = None

@app.route("/karyogram", methods=["POST"])
def karyogram():
    # keep using it so use global variable
    global my_snapshot
    if not my_snapshot:
        # save init memory usage
        my_snapshot = tracemalloc.take_snapshot()
    else:
        lines = []
        # statistics of the memory usage change comparing to the init
        top_stats = tracemalloc.take_snapshot().compare_to(my_snapshot, 'lineno')
        # top 10 memory usages
        for stat in top_stats[:10]:
            lines.append(str(stat))
        print('\n'.join(lines), flush=True)

@app.route("/infer", methods=["POST"])
def infer():
    # create statistic of current memory usage; print top 5 usages
    snapshot = tracemalloc.take_snapshot()
    for idx, stat in enumerate(snapshot.statistics('lineno')[:5], 1):
        print(str(stat), flush=True)
    # details of the top memory usages
    traces = tracemalloc.take_snapshot().statistics('traceback')
    for stat in traces[:1]:
        print("memory_blocks=", stat.count, "size_kB=", stat.size / 1024, flush=True)
        for line in stat.traceback.format():
            print(line, flush=True)

parser = argparse.ArgumentParser(description="Neural Decision Network")

parser.add_argument('--cpu',           action="store_true", help='run on CPU (default: False)')
parser.add_argument('--gpu_num',          type=int, default=0,     metavar='N', help='GPU number for training')
parser.add_argument('--seed',           type=int, default=123456, metavar='N',  help='random seed (default: 123456)')
parser.add_argument('--load_past_data', action="store_true",                    help='Load Past Training data (default: False)')
parser.add_argument('--save_interval',  type=int,  default=50,   metavar='N', help='Saving all data interval (default: 100)')
parser.add_argument('--num_workers',  type=int,  default=2,   metavar='N', help='Number of workers for dataloader (default: 4)')
parser.add_argument('--shuffle_batch',  type=bool,  default=True,   metavar='N', help='Whether shuffle batch of not (default: True)')
parser.add_argument('--roi_aug_prob',    type=float, default=0.5,     metavar='N', help='ROI augmentation prob (default: 0.5)')

parser.add_argument('--batch_size',    type=int, default=512,     metavar='N', help='batch size (default: 1024)')
parser.add_argument('--max_epoch',     type=int, default=100000,     metavar='N', help='max_epoch (default: 1000)')
parser.add_argument('--lr',             type=float, default=3e-4, metavar='G', help='learning rate. (default:1e-4 / 0.0003)')
parser.add_argument('--lr_lambda',     type=float, default=1.0, metavar='G', help='lambda for lr scheduler. (default: 0.99)')
parser.add_argument('--lr_interval',   type=int, default=2400,     metavar='N', help='max_epoch (default: 1000)')

parser.add_argument('--w_dec_loss',  type=float, default=5e-4, metavar='N', help='weight for contrastive decision loss (default: 1e-3)')
parser.add_argument('--w_ent_loss',  type=float, default=1e-4, metavar='N', help='weight for decision entropy loss (default: 1e-4; 1e-2)')
parser.add_argument('--use_dec_loss',  type=bool, default=True, metavar='N', help='Whether use decision loss or not (default: False)')
parser.add_argument('--use_ent_loss',  type=bool, default=False, metavar='N', help='Whether use entropy loss or not (default: False)')

parser.add_argument('--boltz_alpha', type=float, default=2.0, metavar='N', help='temperature factor for Boltzmann Distribution (smooth softmax) (default: 5.0)')

parser.add_argument('--max_update_iter',   type=int, default=500000,     metavar='N', help='maximum update iter for annealing (default: 5.0)')

parser.add_argument('--max_grad_norm',     type=float, default=50.0, metavar='G', help='maximum grad norm. (default: 5)')
parser.add_argument('--decision_std', type=float, default=0.1, metavar='N', help='standard deviation for Gaussian decision. heuristic: 1 / size_decision (default: 0.5)')

# parser.add_argument('--save_prefix',     default="singleGPU_MoNet_v1_gumbel", help='(22.11.02) v1.0')
# parser.add_argument('--save_prefix',     default="singleGPU_MoNet_v00", help='(22.11.03) v1.0')
# parser.add_argument('--save_prefix',     default="singleGPU_MoNet_v00", help='(22.11.05) v1.0 (rollback "BRM_SpatialTempContext_v2")')
# parser.add_argument('--save_prefix',     default="singleGPU_MoNet_v00_SpCt_Gaussian", help='(22.11.09) v1.0 (no LSTM, gaussian decision, entropy regularization)')
# parser.add_argument('--save_prefix',     default="singleGPU_MoNet_v00_SpCt_Gaussian", help='(23.04.03) v1.0 (reproduce prev best config)')
# parser.add_argument('--save_prefix',     default="singleGPU_MoNet_v00_stoch", help='(23.04.07) v1.0 (stochastic decision during training)')
# parser.add_argument('--save_prefix',     default="singleGPU_MoNet_v00_noEntropy", help='(23.0412) v1.0 (deterministic decision during training)')
# parser.add_argument('--save_prefix',     default="singleGPU_MoNet_v00_noEntropyFix", help='(23.04.15) v1.0 (deterministic decision during training)')

# parser.add_argument('--save_prefix',     default="MoNet_v00_N1_total_230509", help='(deterministic decision during training)')
# parser.add_argument('--save_prefix',     default="MoNet_v01_N1_total_230514", help='(add unsupervised decision loss)')
# parser.add_argument('--save_prefix',     default="MoNet_v01_N1_total_230515", help='(add unsupervised decision loss)')
# parser.add_argument('--save_prefix',     default="MoNet_v01_N1_total_230525", help='(add unsupervised decision loss)')
# parser.add_argument('--save_prefix',     default="MoNet_v01_N1_total_230526", help='(add unsupervised decision loss)')
parser.add_argument('--save_prefix',     default="MoNet_v01_NoEnt_N1_total_230603", help='(add unsupervised decision loss)')

args = parser.parse_args()

# Pytorch config
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.empty_cache()
    # print("Empty cuda cache!")

# Path
TRAIN_DATASET_PATH = "./dataset/dataset_train"
VALID_DATASET_PATH = "./dataset/dataset_valid"

# Codes to backup
code_file_list = ['eval_WD.py',
                  'eval.py',
                  'modules.py',
                  'tail.py',
                  'train.py',
                  'transformer.py',
                  'tsne.py',
                  'utils.py',
                  'visualization.py',
                  ]

BASE_DIR_FOR_LOAD = '2023-06-03_00-16-07_MoNet_v01_NoEnt_N1_total_230603' 
FILE_NAME = '_epoch_1122_iteration_153500'
LOAD_LOG_PATH            = './runs/' + BASE_DIR_FOR_LOAD + '_log_'
LOAD_MODEL_PATH          = './model/' + BASE_DIR_FOR_LOAD + '/' + BASE_DIR_FOR_LOAD + FILE_NAME + '.pth'
LOAD_TRAINING_INFO_PATH  = './training_info/' + BASE_DIR_FOR_LOAD + '/' + BASE_DIR_FOR_LOAD + '_train_info' + FILE_NAME + '.npz' 
LOAD_BASE_MODEL_DIR = './model/' + BASE_DIR_FOR_LOAD + '/' + BASE_DIR_FOR_LOAD
LOAD_BASE_TRAINING_INFO_DIR = './training_info/' + BASE_DIR_FOR_LOAD + '/' + BASE_DIR_FOR_LOAD + '_train_info'
LOAD_BASE_CODE_DIR = './code/' + BASE_DIR_FOR_LOAD

# set data directory
if args.load_past_data:
    SAVE_LOG_PATH = LOAD_LOG_PATH 
    SAVE_MODEL_PATH = LOAD_BASE_MODEL_DIR 
    SAVE_TRAINING_INFO_PATH = LOAD_BASE_TRAINING_INFO_DIR
    SAVE_CODE_DIR_PATH = LOAD_BASE_CODE_DIR
else:
    # Make directory (if exist, pass without any error)
    BASE_MODEL_DIR = './model/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + args.save_prefix
    BASE_TRAINING_INFO_DIR = './training_info/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + args.save_prefix
    BASE_CODE_DIR = './code/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + args.save_prefix
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    os.makedirs(BASE_TRAINING_INFO_DIR, exist_ok=True)
    os.makedirs(BASE_CODE_DIR, exist_ok=True)

    SAVE_LOG_PATH    = './runs/{}_{}_log_'.format(
                                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                args.save_prefix,
                                                )
    SAVE_MODEL_PATH  = BASE_MODEL_DIR + '/{}_{}'.format(
                                                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                        args.save_prefix,
                                                        )
    SAVE_TRAINING_INFO_PATH = BASE_TRAINING_INFO_DIR + '/{}_{}_train_info'.format(
                                                                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                                args.save_prefix,
                                                                                )
    SAVE_CODE_DIR_PATH = BASE_CODE_DIR

# backup codes
for code_file in code_file_list:
    shutil.copy(pathlib.Path('.', code_file), pathlib.Path(SAVE_CODE_DIR_PATH, code_file))
print("Backup code files!")


print("Save/Load path")
print("TRAIN_DATASET_PATH :", TRAIN_DATASET_PATH)
print("VALID_DATASET_PATH :", VALID_DATASET_PATH)

print("LOAD_MODEL_PATH         :", LOAD_MODEL_PATH)
print("LOAD_LOG_PATH           :", LOAD_LOG_PATH)
print("LOAD_TRAINING_INFO_PATH :", LOAD_TRAINING_INFO_PATH)

print("SAVE_MODEL_PATH         :", SAVE_MODEL_PATH)
print("SAVE_LOG_PATH           :", SAVE_LOG_PATH)
print("SAVE_TRAINING_INFO_PATH :", SAVE_TRAINING_INFO_PATH)
print("SAVE_CODE_DIR_PATH      :", SAVE_CODE_DIR_PATH)

# Parameters
image_shape = (224,224)
device = torch.device("cuda:" + str(args.gpu_num))
if args.cpu:
    device = torch.device("cpu")

transforms = False # gaussian noise augmentation (ROI shifting is default)

def main():
    # Define model
    model = IL(img_shape=image_shape, args=args)

    # Load Dataset
    train_dataset = customDataset(
        dataset_dir=TRAIN_DATASET_PATH,
        transforms=transforms,
        roi_aug_prob=args.roi_aug_prob,
    )
    valid_dataset = customDataset(
        dataset_dir=VALID_DATASET_PATH,
        transforms=transforms,
        roi_aug_prob=0.0, # no augmentation
    )

    # Define Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_batch,
        num_workers=args.num_workers,
        drop_last=True
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_batch,
        num_workers=1,
        drop_last=True
    )

    # Training info
    epoch = 0
    update_iteration = 0
    train_dataset_size = len(train_dataset)

    # Load model and training info
    if args.load_past_data:
        # - load training info
        load_training_info = np.load(LOAD_TRAINING_INFO_PATH)
        epoch            = load_training_info['epoch'] + 1
        update_iteration = load_training_info['update_iteration'] + 1
        load_training_info.close()
        print("Load Training info done")
        # - load model weight
        model.load_model(LOAD_MODEL_PATH, is_eval=False)
        print("Load model done")


    # Tensorboard log data and Writer
    if args.load_past_data:
        writer = SummaryWriter(log_dir=LOAD_LOG_PATH)
        print("Load SummaryWriter done")
    else:
        writer = SummaryWriter(log_dir=SAVE_LOG_PATH)

    # Training iteration (Epoch)
    while epoch < args.max_epoch:
        # Training iteration (Update iteration)
        for i, (data, idx) in enumerate(train_dataloader):
            print("=====================================")
            print("----- %d th data in %d th epoch -----" %(i, epoch))
            print("=====================================")
            print(data["image"].size())  # input image  (batch, 1, 224, 224)
            print(data["bev"].size())    # input bev    (batch, 3, 60, 60)
            print(data["label"].size())  # output label (batch, 2)
            # print(data["state"].size())  # input state  (batch, 7)

            # to Device
            X_img   = data["image"].to(device)
            X_bev   = data["bev"].to(device)
            # y       = data["label"][:, -1, :].to(device)
            y       = data["label"].to(device)
            # X_state = data["state"][:, :, [1, 2]].to(device) # {a_y, yaw_rate}

            # Update model
            print("Compute loss & update model")
            loss, loss_dec, loss_ent = model.update([X_img, X_bev], y, is_decision_loss=args.use_dec_loss, is_entropy_loss=args.use_ent_loss, is_stoch_decision=False)

            print("Updated (Train). epoch: {}, update_iteration: {}, loss: {}, loss_dec: {}, loss_ent: {}".format(
                epoch, update_iteration, round(loss, 6), round(loss_dec, 6), round(loss_ent, 6)
                ))

            # Save interval (each iteration interval)
            # - add rank condition (only on worker 0 to prevent other workers from corrupting them.)
            #   : https://github.com/horovod/horovod/issues/73
            if update_iteration % args.save_interval == 0 \
                and update_iteration != 0:
                # Save model and training info less times than logs
                if update_iteration % (args.save_interval*10) == 0 and update_iteration != 0:
                    # Save model
                    model.save_model(SAVE_MODEL_PATH + "_epoch_" + str(epoch) + "_iteration_" + str(update_iteration) + ".pth")

                    # Save training info
                    np.savez_compressed(SAVE_TRAINING_INFO_PATH + "_epoch_" + str(epoch) + "_iteration_" + str(update_iteration),
                                        epoch            = epoch,
                                        update_iteration = update_iteration,
                                        )

                # Save logs                
                timestamp = time.time()

                writer.add_scalar('loss', loss**2, global_step=update_iteration, walltime=timestamp)
                writer.add_scalar('loss_L1', loss, global_step=update_iteration, walltime=timestamp)
                writer.add_scalar('loss_dec', loss_dec, global_step=update_iteration, walltime=timestamp)
                writer.add_scalar('loss_ent', loss_ent, global_step=update_iteration, walltime=timestamp)
                
                #   - test inference
                with torch.no_grad():
                    # Test mode
                    model.monet.eval()
                    print("Test mode. trainable? :", model.monet.perception_s_img.training, "Validation epoch start!")
                    # Eval one epoch
                    loss_valid_L1_list = []
                    loss_valid_dec_list = []
                    loss_valid_ent_list = []
                    for valid_iter, (data, idx) in enumerate(valid_dataloader):
                        # to Device
                        valid_X_img   = data["image"].to(device)
                        valid_X_bev   = data["bev"].to(device)
                        valid_y       = data["label"].to(device)
                        # Compute loss only
                        loss_valid_L1, loss_valid_dec, loss_valid_ent = model.calc_loss([valid_X_img, valid_X_bev], valid_y, is_stoch_decision=False)
                        loss_valid_L1_list.append(loss_valid_L1)
                        loss_valid_dec_list.append(loss_valid_dec)
                        loss_valid_ent_list.append(loss_valid_ent)

                        print("Validation. epoch: {}, valid_iter: {}, loss L1: {}, loss dec: {}, loss ent: {}".format(
                            epoch, valid_iter, round(loss_valid_L1, 6), round(loss_valid_dec, 6), round(loss_valid_ent, 6)))

                    # Compute average valid loss
                    ave_loss_L1_valid = np.mean(loss_valid_L1_list)
                    ave_loss_dec_valid = np.mean(loss_valid_dec_list)
                    ave_loss_ent_valid = np.mean(loss_valid_ent_list)

                    writer.add_scalar('loss_L1/valid', ave_loss_L1_valid, global_step=update_iteration, walltime=timestamp)
                    writer.add_scalar('loss_dec/valid', ave_loss_dec_valid, global_step=update_iteration, walltime=timestamp)
                    writer.add_scalar('loss_ent/valid', ave_loss_ent_valid, global_step=update_iteration, walltime=timestamp)
                    print("Validation done. epoch: {}, update_iteration: {}, loss L1: {}, loss dec: {}, loss ent: {}".format(
                        epoch, update_iteration, round(ave_loss_L1_valid, 6), round(ave_loss_dec_valid, 6), round(ave_loss_ent_valid, 6)))
                        
                    # Visualize info (attention, latent decision)
                    model.monet.z_decision = None
                    test_idx = np.random.choice(range(args.batch_size))
                    test_X_img = valid_X_img[test_idx].unsqueeze(dim=0) # (T,1,224,224) -> (1,T,1,224,224)
                    test_X_bev = valid_X_bev[test_idx].unsqueeze(dim=0) # (T,3,60,60) -> (1,T,3,60,60)
                    # - eval with deterministic decision
                    a, _, _ = model.monet(test_X_img, test_X_bev, is_stoch_decision=False)

                    # - save sample image (re-normalize [-1,+1] to [0,1])
                    raw_img = test_X_img[0].cpu() # (C,H,W)
                    # raw_img = test_X_img[0][-1].cpu() # (C,H,W)
                    raw_img += 1
                    raw_img *= 0.5

                    raw_bev = test_X_bev[0].cpu() # (C,H,W)
                    # raw_bev = test_X_bev[0][-1].cpu() # (C,H,W)
                    raw_bev += 1
                    raw_bev *= 0.5

                    # - save sample image (numpy image. 0~1)
                    raw_img_cv = test_X_img[0].cpu() # (C,H,W)
                    # raw_img_cv = test_X_img[0][-1].cpu() # (C,H,W)
                    raw_img_cv += 1.
                    raw_img_cv *= 0.5
                    raw_img_cv = raw_img_cv.repeat(3, 1, 1) # (1,224,224) -> (3,224,224)
                    raw_img_cv = raw_img_cv.permute(1, 2, 0).numpy() # (C,H,W) to (H,W,C)

                    # raw_img_cv = cv2.applyColorMap(np.uint8(raw_img_cv*255.), cv2.COLORMAP_JET)
                    raw_img_cv = np.uint8(raw_img_cv*255.)

                    # - save attention map as distribution figure
                    if model.monet.z_decision is not None \
                        and model.monet.att_map_sp is not None \
                        and model.monet.att_map_ct is not None:
                        # Define plotter class (for reducing memory leak)
                        plotter = PlotterClass()

                        # Visualization of the Spatial attention
                        #   - (batch, head, N, N(soft)) -(index:0)-> (head, N, N(soft)) -(head:i)-> (N, N(soft)) -(index:0)-> (N(soft),)
                        spatial_att_head0_0 = model.monet.att_map_sp[0][0][0].view(6,6).cpu().numpy() # (64+5,)
                        # #   - (batch, T, head, N, N(soft)) -(index:0)-> (T, head, N, N(soft)) -(index:-1)-> (head, N, N(soft))  -(head:i)-> (N, N(soft)) -(index:0)-> (N(soft),)
                        spatial_att_head0_0 = np.maximum(spatial_att_head0_0, 0, dtype='float')
                        spatial_att_head0_0 /= np.max(spatial_att_head0_0)
                        spatial_att_head0_0 = cv2.resize(spatial_att_head0_0, (raw_img_cv.shape[1], raw_img_cv.shape[0]))
                        spatial_att_head0_0 = cv2.applyColorMap(np.uint8(255*spatial_att_head0_0), cv2.COLORMAP_JET)
                        spatial_att_head0_0_img_cv = cv2.addWeighted(spatial_att_head0_0, 0.5, raw_img_cv, 0.5, 0)
                        spatial_att_head0_0_img_cv = spatial_att_head0_0_img_cv[:,:,::-1].copy() / 255. # numpy image

                        #   - (batch, head, N, N(soft)) -(index:0)-> (head, N, N(soft)) -(head:i)-> (N, N(soft)) -(max(dim=0))-> (N(soft),)
                        spatial_att_head0 = model.monet.att_map_sp[0][0].max(dim=0)[0].view(6,6).cpu().numpy() # (64+5,)
                        # #   - (batch, T, head, N, N(soft)) -(index:0)-> (T, head, N, N(soft)) -(index:-1)-> (head, N, N(soft))  -(head:i)-> (N, N(soft)) -(max(dim=0))-> (N(soft),)
                        spatial_att_head0 = np.maximum(spatial_att_head0, 0, dtype='float')
                        spatial_att_head0 /= np.max(spatial_att_head0)
                        spatial_att_head0 = cv2.resize(spatial_att_head0, (raw_img_cv.shape[1], raw_img_cv.shape[0]))
                        spatial_att_head0 = cv2.applyColorMap(np.uint8(255*spatial_att_head0), cv2.COLORMAP_JET)
                        spatial_att_head0_img_cv = cv2.addWeighted(spatial_att_head0, 0.5, raw_img_cv, 0.5, 0)
                        spatial_att_head0_img_cv = spatial_att_head0_img_cv[:,:,::-1].copy() / 255. # numpy image

                        spatial_att_head0_mean = model.monet.att_map_sp[0][0].mean(dim=0).view(6,6).cpu().numpy() # (64+5,)
                        spatial_att_head0_mean = np.maximum(spatial_att_head0_mean, 0, dtype='float')
                        spatial_att_head0_mean /= np.max(spatial_att_head0_mean)
                        spatial_att_head0_mean = cv2.resize(spatial_att_head0_mean, (raw_img_cv.shape[1], raw_img_cv.shape[0]))
                        spatial_att_head0_mean = cv2.applyColorMap(np.uint8(255*spatial_att_head0_mean), cv2.COLORMAP_JET)
                        spatial_att_head0_mean_img_cv = cv2.addWeighted(spatial_att_head0_mean, 0.5, raw_img_cv, 0.5, 0)
                        spatial_att_head0_mean_img_cv = spatial_att_head0_mean_img_cv[:,:,::-1].copy() / 255. # numpy image

                        spatial_att_head0_min = model.monet.att_map_sp[0][0].min(dim=0)[0].view(6,6).cpu().numpy() # (64+5,)
                        spatial_att_head0_min = np.maximum(spatial_att_head0_min, 0, dtype='float')
                        spatial_att_head0_min /= np.max(spatial_att_head0_min)
                        spatial_att_head0_min = cv2.resize(spatial_att_head0_min, (raw_img_cv.shape[1], raw_img_cv.shape[0]))
                        spatial_att_head0_min = cv2.applyColorMap(np.uint8(255*spatial_att_head0_min), cv2.COLORMAP_JET)
                        spatial_att_head0_min_img_cv = cv2.addWeighted(spatial_att_head0_min, 0.5, raw_img_cv, 0.5, 0)
                        spatial_att_head0_min_img_cv = spatial_att_head0_min_img_cv[:,:,::-1].copy() / 255. # numpy image

                        # - concat images (att_map_ct head0,1,2,3)
                        head0123_up = torch.cat([
                                                 trans.ToTensor()(np.float32(spatial_att_head0_img_cv)),
                                                 trans.ToTensor()(np.float32(spatial_att_head0_mean_img_cv)),
                                                 ], dim=2) # (C,H,W)

                        head0123_down = torch.cat([
                                                   trans.ToTensor()(np.float32(spatial_att_head0_min_img_cv)),
                                                   trans.ToTensor()(np.float32(spatial_att_head0_0_img_cv)),
                                                   ], dim=2) # (C,H,W)

                        head0123 = torch.cat([head0123_up, head0123_down], dim=1)# (C,H,W)

                        # Visualization of the Contextual attention (batch, head, N_c+1, N_c+1(soft))
                        # - dict[depth]: (batch, head, N_c+1, N_c+1(soft)) (batch is 1)
                        #   - (batch, head, N_c+1, N_c+1(soft)) (batch is 1)
                        #   -(index:0)-> (head, N_c+1, N_c+1(soft))
                        #     -(head:i)-> (N_c+1, N_c+1(soft))
                        #       -(index [0])-> (N_c+1(soft),) ==> Look at the total attention between the class token, and the context patches
                        #         -(index [1:])-> (N_c(soft),) ==> attention w.r.t cls token which is at index:0

                        att_context_map_head0 = model.monet.att_map_ct[0][0].max(dim=0)[0].cpu().numpy() # (64+5,)
                        contextual_att_img_max = plotter.gen_plot_img(att_context_map_head0, viz_type="att_max_pool", title="Attention Map (Max)")

                        att_context_map_head0 = model.monet.att_map_ct[0][0].mean(dim=0).cpu().numpy() # (64+5,)
                        contextual_att_img_mean = plotter.gen_plot_img(att_context_map_head0, viz_type="att_mean_pool", title="Attention Map (Mean)")

                        # att_context_map_head0 = model.monet.att_map_ct[0][0].min(dim=0)[0].cpu().numpy() # (64+5,)
                        # contextual_att_img_min = plotter.gen_plot_img(att_context_map_head0, viz_type="att_max_pool", title="Attention Map (Min)")

                        # att_context_map_head0 = model.monet.att_map_ct[0][0][0].cpu().numpy() # (64+5,)
                        # contextual_att_img_0 = plotter.gen_plot_img(att_context_map_head0, viz_type="att_mean_pool", title="Attention Map (0 index)")

                        # Visualization of the Decision
                        # - (batch, N(soft)) -> (N(soft),)
                        z_decision_head0 = model.monet.z_decision[0].cpu().numpy() # (batch,dim) -> (dim,)
                        z_decision_head0 = plotter.gen_plot_img(z_decision_head0, viz_type="decision", title="Decision Feature", plot_fig=False)

                        # concat images (raw, att, decision output)
                        raw_img = raw_img.repeat(3, 1, 1) # (1,224,224) -> (3,224,224)
                        raw_bev_resize = torch.zeros_like(raw_img) # (3,60,60) --> (3,224,224) with zero padding
                        raw_bev_resize[:,:raw_bev.shape[1],:raw_bev.shape[2]] = raw_bev
                        spatial_att_map = trans.ToTensor()(np.float32(spatial_att_head0_mean_img_cv))
                        
                        raw_att_dec = torch.cat([raw_img, raw_bev_resize, spatial_att_map, contextual_att_img_mean, z_decision_head0], dim=2) # (C H W)
                        # raw_att_dec = torch.cat([raw_img, raw_bev_resize, contextual_att_img_max, contextual_att_img_mean, z_decision_head0], dim=2) # (C H W)
                        # raw_att_dec = torch.cat([raw_img, contextual_att_img_max, contextual_att_img_mean, contextual_att_img_min, contextual_att_img_0, z_decision_head0], dim=2) # (C H W)

                        # writes images on tensorboard
                        writer.add_image('raw_att_dec', raw_att_dec, global_step=update_iteration, walltime=timestamp) # tensor image
                        writer.add_image('head0123', head0123, global_step=update_iteration, walltime=timestamp) # tensor image
                        # writer.add_image('bev', raw_bev, global_step=update_iteration, walltime=timestamp)
                       
                print("----------------------")
                print("Check Memory Increment")
                print("----------------------")
                karyogram()

                print("--------------------------")
                print("Check Current Memory Usage")
                print("--------------------------")
                infer()
                
                print("--------------")
                print("Save Data Done")
                print("--------------")

                # for reduce memory leak
                del plotter, att_context_map_head0, head0123
                del raw_img, raw_att_dec, raw_bev, raw_bev_resize
                del spatial_att_head0, spatial_att_head0_mean, spatial_att_head0_min, spatial_att_head0_0
                del contextual_att_img_max, contextual_att_img_mean
                # del contextual_att_img_max, contextual_att_img_mean, contextual_att_img_min, contextual_att_img_0
                # dummy string
                plottor = ""
                spatial_att_head0 = ""
                spatial_att_head0_mean = ""
                spatial_att_head0_min = ""
                spatial_att_head0_0 = ""
                contextual_att_img_max = ""
                contextual_att_img_mean = ""
                # contextual_att_img_min = ""
                # contextual_att_img_0 = ""
                del plottor
                del spatial_att_head0, spatial_att_head0_mean, spatial_att_head0_min, spatial_att_head0_0
                del contextual_att_img_max, contextual_att_img_mean
                # del contextual_att_img_max, contextual_att_img_mean, contextual_att_img_min, contextual_att_img_0
                gc.collect
                
                # empty cache
                torch.cuda.empty_cache()
                print("torch.cuda.empty_cache() Done")
                
                # Train mode
                model.monet.train()
                print("Train mode. trainable? :", model.monet.perception_s_img.training)

            # Update training info
            update_iteration += 1
            epoch = (update_iteration * args.batch_size) // train_dataset_size

        if epoch % args.lr_interval == 0 and epoch != 0:
            # Update learning rate schedule
            print("Before updating lr scheduler :", model.monet_optim.param_groups[0]["lr"])
            model.monet_lr_scheduler.step()
            print("After updating lr scheduler :", model.monet_optim.param_groups[0]["lr"])

    # Done training
    print("Training is done! (Max epoch)")

if __name__ == '__main__':
    main()
