#-*- coding: utf-8 -*-

#!/usr/bin/env python

# Task Similarity Score

# By Hyunki Seong.
# Email : hynkis@kaist.ac.kr

import numpy as np
import torch
from utils import compute_WD

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import time

"""
Load Data

"""

# MoNet data
# BASE_PATH = "./dataset/rosbag_n1_3rd_5th/samples_bag/total_filtered_bag/result"
# DATA_PATH = BASE_PATH + "/tsne_MoNet_v01_data.csv"

# index_CA_left_end  = 73
# index_CA_right_end = 73+82
# index_LT_end       = 73+82+99
# index_RT_end       = 73+82+99+97
# index_ST_end       = 73+82+99+97+24
# index_ST_near_end  = 73+82+99+97+24+40
# index_ST_wall_end  = 73+82+99+97+24+40+247

SAVE_RESULT = True
USE_SOFTMAX = True

# Path (result data)
MODEL_NAME = 'MoNet_v01_add_LGC'
# MODEL_NAME = 'MoNet_v01'
BASE_PATH = './baseline_model/' + MODEL_NAME + '/results/final_model_result'
DATA_PATH = BASE_PATH + '/' + MODEL_NAME + '_data.csv'

index_CA_left_end  = 31
index_CA_right_end = 31+36
index_LT_end       = 31+36+59
index_RT_end       = 31+36+59+59
index_ST_end       = 31+36+59+59+64
index_ST_wall_end  = 31+36+59+59+64+318

# - load decision data
data = pd.read_csv(DATA_PATH, header=None).to_numpy()
print("data :", data.shape)
# - parse latent decision data
data_decision = data[:,:16]

"""
Compute Representational Similarity Matrix (cosine similarity)
score_sum = 0
for ind_i in data_idx_TASK1:
    for ind_j in data_idx_TASK2:
        score = compute_score(data_idx_TASK1[ind_i], data_idx_TASK2[ind_j])
        score_sum += score
score_avg = score_sum / (len(data_idx_TASK1) * len(data_idx_TASK2))

"""

# Data
SAVE_PATH = BASE_PATH + "/rsm_result.csv" # "./results/rsm/rsm.csv"
data_idx_CA = list(range(0, index_CA_right_end))
data_idx_LT = list(range(index_CA_right_end, index_LT_end))
data_idx_RT = list(range(index_LT_end, index_RT_end))
data_idx_ST = list(range(index_RT_end, index_ST_end))
data_idx_ST_wall = list(range(index_ST_end, index_ST_wall_end))

data_idx_total = [data_idx_ST_wall, data_idx_ST, data_idx_LT, data_idx_RT, data_idx_CA]
print("data_idx_total length :", len(data_idx_total))

"""
Compute Representational Similarity Matrix (RSM)

"""
tic = time.time()
# Compute cosine similarity
cos_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

# Compute Representational Similarity Matrix
RSM = np.ones((5, 5)) * -1 # initialize as -1
# - cluster A
for i, data_idx_A in enumerate(data_idx_total):
    print("data_idx_A")
    print(data_idx_A)
    # - cluster A with cluster B
    for j, data_idx_B in enumerate(data_idx_total):
        print("Data cluster A: {} (len:{}) <--> B: {} (len:{})".format(i, len(data_idx_A), j, len(data_idx_B)))
        # - check redundant loop
        if RSM[i,j] != -1 or RSM[j,i] != -1:
            print("pass this loop! RSM[i,j]: {}, RSM[j,i]: {}".format(RSM[i,j], RSM[j,i]))
            if RSM[i,j] == -1:
                RSM[i,j] = RSM[j,i]
            else:
                RSM[j,i] = RSM[i,j]
            print("update done using pre-computed value")
            continue
        sim_score_sum = 0
        # - compute sum of sim_score
        # -- data a in cluster A
        for data_idx_a in data_idx_A:
            # -- data b in cluster B
            for data_idx_b in data_idx_B:
                # # Wasser Dist
                # data_a = data_decision[data_idx_a]
                # data_b = data_decision[data_idx_b]
                # # !make it sure the sum is 1
                # diff_data_a = np.sum(data_a) - 1
                # diff_data_b = np.sum(data_b) - 1
                # data_a[0] = data_a[0] - diff_data_a 
                # data_b[0] = data_b[0] - diff_data_b
                # sim_score = compute_WD(data_a, data_b)
                # cosine Dist
                data_a = torch.tensor(data_decision[data_idx_a])
                data_b = torch.tensor(data_decision[data_idx_b])
                sim_score = cos_similarity(data_a, data_b) # data_idx_a,b are the indices of the total data(data_decision)

                sim_score_sum += sim_score
    
        # Update RSM data
        RSM[i,j] = sim_score_sum / (len(data_idx_A) * len(data_idx_B))

toc = time.time()
print("Computing RSM is done! Duration :", toc - tic)

"""
Softmax for normalization
"""
def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

if USE_SOFTMAX:
    for i in range(RSM.shape[0]):
        RSM[i,:] = softmax(RSM[i,:])

"""
Save Representational Similarity Matrix (RSM)

"""
if SAVE_RESULT:
    result_rsm = RSM

    df = pd.DataFrame(result_rsm)
    df.to_csv(SAVE_PATH, index=False, header=False)
    print("RSM result is saved!")

"""
Visualize Representational Similarity Matrix (RSM)

"""
label_X = ["ST", "SI", "LT", "RT", "CA"]
label_Y = ["ST", "SI", "LT", "RT", "CA"]

fig, ax = plt.subplots()
ax = sns.heatmap(RSM,
                 robust=True,
                 cmap="Blues",
                 xticklabels=label_X,
                 yticklabels=label_Y,
                 cbar=True,
                 annot=True,
                 fmt='.2f',
                 cbar_kws={'label': 'Normalized Similarity'},
                #  cbar_kws={"orientation": "horizontal"},
                 )
ax.xaxis.tick_top() # x axis on top
ax.set_xlabel('Compared Class')
ax.set_ylabel('Reference Class')
ax.xaxis.set_label_position('top')

plt.yticks(rotation=0)

# ax.set_title("Representational Similarity Map")
# fig.tight_layout()
plt.show()