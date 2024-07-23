#-*- coding: utf-8 -*-

#!/usr/bin/env python

# Search a representative latent decision (decision prior)
# for Representational Similarity Analysis (RSA)

# By Hyunki Seong.
# Email : hynkis@kaist.ac.kr

import numpy as np
import torch
from utils import compute_WD

import matplotlib.pyplot as plt
import pandas as pd
from matplot_selector import SelectFromCollection
import time

# MoNet data
BASE_PATH = "./dataset/rosbag_n1_3rd_5th/samples_bag/230607_n1_5th_MoNet/filtered/result"
DATA_PATH = BASE_PATH + "/tsne_MoNet_v01_data.csv"

index_CA_left_end  = 54
index_CA_right_end = 54+48
index_LT_end       = 54+48+59
index_RT_end       = 54+48+59+59
index_ST_end       = 54+48+59+59+24
index_ST_near_end  = 54+48+59+59+24+40
index_ST_wall_end  = 54+48+59+59+24+40+361

# - load decision data
data = pd.read_csv(DATA_PATH, header=None).to_numpy()

data_idx_CA = list(range(0, index_CA_right_end))
data_idx_LT = list(range(index_CA_right_end, index_LT_end))
data_idx_RT = list(range(index_LT_end, index_RT_end))
data_idx_ST = list(range(index_RT_end, index_ST_end))
data_idx_ST_wall = list(range(index_ST_end, index_ST_wall_end))
# data_idx_CA_left = list(range(0, index_CA_left_end))
# data_idx_CA_right = list(range(index_CA_left_end, index_CA_right_end))
# data_idx_ST_near = list(range(index_ST_end, index_ST_near_end))
# data_idx_ST_wall = list(range(index_ST_near_end, index_ST_wall_end))

# - parse latent decision data
print("data :", data.shape)
data_decision = data[:,:16]


"""
Compute Decision Prior (cosine similarity)

for ind_i in data_idx_TASK:
    score_list = []
    score_median = []
    for ind_j in data_idx_TASK:
        similarity_score = cos(data_decision[ind_i], data_decision[ind_j])
        score_list.append(similarity_score)
    score_median = np.median(score_list)
decision_TASK = data_decision[argmin(score_median)]

"""

# Data
SAVE_RESULT = True
SAVE_PATH = BASE_PATH + "/CA" + "_decision_prior.csv"
data_idx = data_idx_CA

tic = time.time()
# Compute cosine similarity
cos_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

score_median_list = []
score_mean_list = []
score_max_list = []
for i, ind_i in enumerate(data_idx):
    score_list = []
    print("data index i {}, {} th in total {}".format(ind_i, i, len(data_idx)))
    for ind_j in data_idx:
        if ind_i == ind_j:
            continue
        latent_plan_i = torch.tensor(data_decision[ind_i])
        latent_plan_j = torch.tensor(data_decision[ind_j])
        cos_sim_torch = cos_similarity(latent_plan_i, latent_plan_j)
        score_list.append(cos_sim_torch.numpy())
    score_median_list.append(np.median(score_list))
    score_mean_list.append(np.mean(score_list))
    score_max_list.append(np.max(score_list))
    
ind_score_median_min = np.argmin(score_median_list)
ind_score_median_median = np.argwhere(score_median_list == np.percentile(score_median_list, 50, interpolation='nearest'))[0,0] # [[index]] --> index
ind_score_median_max = np.argmax(score_median_list)

ind_score_mean_min = np.argmin(score_mean_list)
ind_score_mean_median = np.argwhere(score_mean_list == np.percentile(score_mean_list, 50, interpolation='nearest'))[0,0] # [[index]] --> index
ind_score_mean_max = np.argmax(score_mean_list)

ind_score_max_min = np.argmin(score_max_list)
ind_score_max_median = np.argwhere(score_max_list == np.percentile(score_max_list, 50, interpolation='nearest'))[0,0] # [[index]] --> index
ind_score_max_max = np.argmax(score_max_list)

toc = time.time()

if SAVE_RESULT:
    result_dec_prior = np.array([data_idx[ind_score_median_min],
                                data_idx[ind_score_median_median],
                                data_idx[ind_score_median_max],
                                data_idx[ind_score_mean_min],
                                data_idx[ind_score_mean_median],
                                data_idx[ind_score_mean_max],
                                data_idx[ind_score_max_min],
                                data_idx[ind_score_max_median],
                                data_idx[ind_score_max_max]])
    df = pd.DataFrame(result_dec_prior)
    df.to_csv(SAVE_PATH, index=False, header=False)
    print("decision similarity result is saved!")

print("Target data :", str(data_idx), "process time :", toc - tic)
print("score min,       min median_score, data index :", ind_score_median_min, score_median_list[ind_score_median_min], data_idx[ind_score_median_min])
print("score median, median median_score, data index :", ind_score_median_median, score_median_list[ind_score_median_median], data_idx[ind_score_median_median])
print("score max index, max median_score, data index :", ind_score_median_max, score_median_list[ind_score_median_max], data_idx[ind_score_median_max])

print("score min,       min mean_score, data index :", ind_score_mean_min, score_mean_list[ind_score_mean_min], data_idx[ind_score_mean_min])
print("score median, median mean_score, data index :", ind_score_mean_median, score_mean_list[ind_score_mean_median], data_idx[ind_score_mean_median])
print("score max index, max mean_score, data index :", ind_score_mean_max, score_mean_list[ind_score_mean_max], data_idx[ind_score_mean_max])

print("score min,       min max_score, data index :", ind_score_max_min, score_max_list[ind_score_max_min], data_idx[ind_score_max_min])
print("score median, median max_score, data index :", ind_score_max_median, score_max_list[ind_score_max_median], data_idx[ind_score_max_median])
print("score max index, max max_score, data index :", ind_score_max_max, score_max_list[ind_score_max_max], data_idx[ind_score_max_max])

plt.plot(score_median_list, 'r')
plt.plot(score_mean_list, 'b')
plt.show()
