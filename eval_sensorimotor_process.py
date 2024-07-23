#-*- coding: utf-8 -*-

#!/usr/bin/env python

# By Hyunki Seong.
# Email : hynkis@kaist.ac.kr

import argparse
import datetime
import time
from matplotlib.pyplot import contour

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# MoNet data
BASE_PATH        = "./eval_data/indoor_rccar/result_rollout"
# BASE_PATH        = "./dataset/rosbag_n1_3rd_5th/samples_bag/230607_n1_5th_MoNet"
# DATASET_PATH        = BASE_PATH + "/filtered"
DATA_PATH = BASE_PATH + '/MoNet_v01_add_LGC_rollout.csv'
# DATA_PATH = BASE_PATH + '/MoNet_v01_rollout.csv'

data = pd.read_csv(DATA_PATH, header=None).to_numpy()
print(data)
print("data :", data.shape)

# dataset_np:
# - latent_decision_list[:,:16] # (N, 16)
# - decision_prob_list[:,16:20] # (N, 4)
# - decision_ent_list[:,20:21] # (N, 1)
# - action_output_list[:,21:23] # (N, 2)
data_decision = data[:,:16]
data_prob = data[:,16:20]
data_entropy = data[:,20:21]
data_action = data[:,-2:]

data_decision_viz = pd.DataFrame(data_decision.T) # (dim, N)
data_entropy_viz = data_entropy # (N,) entropy
data_steer_viz = - data_action[:,0] # (N,) steering angle is flipped

plt.figure()
sns.heatmap(data_decision_viz, robust=True, cmap='binary', xticklabels=10,  yticklabels=False, cbar=True)

plt.figure()
max_entropy = np.ones_like(data_entropy_viz) * 2.0 
mid_entropy = np.ones_like(data_entropy_viz) * 1.0 
plt.plot(max_entropy, 'k--')
plt.plot(mid_entropy, 'r--')
plt.plot(data_entropy_viz, '-')
plt.xlim([0,600])
plt.ylim([0,2.2])

plt.figure()
plt.plot(data_steer_viz, '-')
plt.xlim([0,600])

# plt.grid()
plt.show()
