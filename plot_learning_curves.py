import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

MAX_EPISODE_NUM = 650000

DATA_PATH_LOSS_MONET_ADD = "./eval_data/indoor_rccar/result_learning_curves/monet_add/run-2023-05-19_20-42-58_MoNet_v01_add_N1_total_230519_log_-tag-loss_L1_valid.csv"
DATA_PATH_LOSS_MONET_ADD_LGC = "./eval_data/indoor_rccar/result_learning_curves/monet_add_lgc/run-2023-08-07_20-59-02_MoNet_v01_add_LGC_N1_total_230807_log_-tag-loss_L1_valid.csv"
DATA_PATH_LOSS_MONET = "./eval_data/indoor_rccar/result_learning_curves/monet/run-2023-05-26_19-32-35_MoNet_v01_N1_total_230526_log_-tag-loss_L1_valid.csv"
DATA_PATH_LOSS_MONET_NOLGC = "./eval_data/indoor_rccar/result_learning_curves/monet_nolgc/run-2023-05-25_13-36-15_MoNet_v01_N1_total_230525_log_-tag-loss_L1_valid.csv"
DATA_PATH_LOSS_MONET_NOPLAN_CONTROL = "./eval_data/indoor_rccar/result_learning_curves/monet_noplan_control/run-2023-10-30_16-49-50_MoNet_v01_NoPlan_N1_total_231030_log_-tag-loss_L1_valid.csv"

DATA_PATH_SCORE_MONET_ADD = "./eval_data/indoor_rccar/result_learning_curves/monet_add/2023-05-19_20-42-58_MoNet_v01_add_N1_total_230519_epoch_3656_iteration_500000.pth_rsm_curve.csv"
DATA_PATH_SCORE_MONET_ADD_LGC = "./eval_data/indoor_rccar/result_learning_curves/monet_add_lgc/2023-08-07_20-59-02_MoNet_v01_add_LGC_N1_total_230807_epoch_4827_iteration_660000.pth_rsm_curve.csv"
DATA_PATH_SCORE_MONET = "./eval_data/indoor_rccar/result_learning_curves/monet/2023-05-26_19-32-35_MoNet_v01_N1_total_230526_epoch_4754_iteration_650000.pth_rsm_curve.csv"
DATA_PATH_SCORE_MONET_NOLGC = "./eval_data/indoor_rccar/result_learning_curves/monet_nolgc/2023-05-25_13-36-15_MoNet_v01_N1_total_230525_epoch_4754_iteration_650000.pth_rsm_curve.csv"
DATA_PATH_SCORE_MONET_NOPLAN_CONTROL = "./eval_data/indoor_rccar/result_learning_curves/monet_noplan_control/2023-10-30_16-49-50_MoNet_v01_NoPlan_N1_total_231030_epoch_5265_iteration_720000.pth_rsm_curve.csv"
DATA_PATH_SCORE_MONET_NOPLAN_PERCEPT = "./eval_data/indoor_rccar/result_learning_curves/monet_noplan_percept/2023-10-30_16-49-50_MoNet_v01_NoPlan_N1_total_231030_epoch_5265_iteration_720000.pth_rsm_curve.csv"


# load data (data with column name)
dataset_loss_monet_add = pd.read_csv(DATA_PATH_LOSS_MONET_ADD)
dataset_loss_monet_add_lgc = pd.read_csv(DATA_PATH_LOSS_MONET_ADD_LGC)
dataset_loss_monet = pd.read_csv(DATA_PATH_LOSS_MONET)
dataset_loss_monet_nolgc = pd.read_csv(DATA_PATH_LOSS_MONET_NOLGC)
dataset_loss_monet_noplan_control = pd.read_csv(DATA_PATH_LOSS_MONET_NOPLAN_CONTROL)

# load data (data no column name)
with open(DATA_PATH_SCORE_MONET_ADD, 'r') as f:
    data_score_monet_add = list(csv.reader(f, delimiter=";"))
    data_score_monet_add = np.array(data_score_monet_add, dtype=np.float32) # (N,1)
    data_score_monet_add = np.squeeze(data_score_monet_add) # (N,)
with open(DATA_PATH_SCORE_MONET_ADD_LGC, 'r') as f:
    data_score_monet_add_lgc = list(csv.reader(f, delimiter=";"))
    data_score_monet_add_lgc = np.array(data_score_monet_add_lgc, dtype=np.float32) # (N,1)
    data_score_monet_add_lgc = np.squeeze(data_score_monet_add_lgc) # (N,)
with open(DATA_PATH_SCORE_MONET, 'r') as f:
    data_score_monet = list(csv.reader(f, delimiter=";"))
    data_score_monet = np.array(data_score_monet, dtype=np.float32) # (N,1)
    data_score_monet = np.squeeze(data_score_monet) # (N,)
with open(DATA_PATH_SCORE_MONET_NOLGC, 'r') as f:
    data_score_monet_nolgc = list(csv.reader(f, delimiter=";"))
    data_score_monet_nolgc = np.array(data_score_monet_nolgc, dtype=np.float32) # (N,1)
    data_score_monet_nolgc = np.squeeze(data_score_monet_nolgc) # (N,)
with open(DATA_PATH_SCORE_MONET_NOPLAN_CONTROL, 'r') as f:
    data_score_monet_noplan_control = list(csv.reader(f, delimiter=";"))
    data_score_monet_noplan_control = np.array(data_score_monet_noplan_control, dtype=np.float32) # (N,1)
    data_score_monet_noplan_control = np.squeeze(data_score_monet_noplan_control) # (N,)
with open(DATA_PATH_SCORE_MONET_NOPLAN_PERCEPT, 'r') as f:
    data_score_monet_noplan_percept = list(csv.reader(f, delimiter=";"))
    data_score_monet_noplan_percept = np.array(data_score_monet_noplan_percept, dtype=np.float32) # (N,1)
    data_score_monet_noplan_percept = np.squeeze(data_score_monet_noplan_percept) # (N,)

# make step data
step_data_650k = np.arange(65+1) * 1e+4 # [500, 10000, 20000, ..., 650000]
step_data_500k = np.arange(50+1) * 1e+4 # [500, 10000, 20000, ..., 500000]
step_data_650k[0] = 500 # [500, 10000, 20000, ..., 650000]
step_data_500k[0] = 500 # [500, 10000, 20000, ..., 500000]

# make dataframe with column names
dataset_score_monet_add = pd.DataFrame()
dataset_score_monet_add_lgc = pd.DataFrame()
dataset_score_monet = pd.DataFrame()
dataset_score_monet_nolgc = pd.DataFrame()
dataset_score_monet_noplan_control = pd.DataFrame()
dataset_score_monet_noplan_percept = pd.DataFrame()

dataset_score_monet_add['Step'] = step_data_500k
dataset_score_monet_add_lgc['Step'] = step_data_650k
dataset_score_monet['Step'] = step_data_650k
dataset_score_monet_nolgc['Step'] = step_data_650k
dataset_score_monet_noplan_control['Step'] = step_data_650k
dataset_score_monet_noplan_percept['Step'] = step_data_650k

dataset_score_monet_add['Value'] = data_score_monet_add
dataset_score_monet_add_lgc['Value'] = data_score_monet_add_lgc
dataset_score_monet['Value'] = data_score_monet
dataset_score_monet_nolgc['Value'] = data_score_monet_nolgc
dataset_score_monet_noplan_control['Value'] = data_score_monet_noplan_control
dataset_score_monet_noplan_percept['Value'] = data_score_monet_noplan_percept

print("dataset_score_monet :")
print(dataset_score_monet)

dataset_loss_monet_add_lgc['Policy'] = 'MoNet'
dataset_loss_monet_add['Policy'] = 'MoNet (w/o $L_{LGC}$)'
dataset_loss_monet_noplan_control['Policy'] = 'ViTNet'
dataset_loss_monet['Policy'] = 'MoNet-SOFT'
dataset_loss_monet_nolgc['Policy'] = 'MoNet-SOFT w/o LGC'

dataset_score_monet_add_lgc['Policy'] = 'MoNet' # 'MoNet ($h^d$)'
dataset_score_monet_add['Policy'] = 'MoNet (w/o $L_{LGC}$)' # ($h^d$, w/o $L_{LGC}$)'
dataset_score_monet_noplan_control['Policy'] = 'ViTNet ($z^c$)'
dataset_score_monet_noplan_percept['Policy'] = 'ViTNet ($z^p$)'
dataset_score_monet['Policy'] = 'MoNet-SOFT'
dataset_score_monet_nolgc['Policy'] = 'SOFT w/o LGC'

dataset_loss_monet_add = dataset_loss_monet_add[dataset_loss_monet_add['Step'] <= MAX_EPISODE_NUM]
dataset_loss_monet_add_lgc = dataset_loss_monet_add_lgc[dataset_loss_monet_add_lgc['Step'] <= MAX_EPISODE_NUM]
dataset_loss_monet = dataset_loss_monet[dataset_loss_monet['Step'] <= MAX_EPISODE_NUM]
dataset_loss_monet_nolgc = dataset_loss_monet_nolgc[dataset_loss_monet_nolgc['Step'] <= MAX_EPISODE_NUM]
dataset_loss_monet_noplan_control = dataset_loss_monet_noplan_control[dataset_loss_monet_noplan_control['Step'] <= MAX_EPISODE_NUM]

dataset_score_monet_add = dataset_score_monet_add[dataset_score_monet_add['Step'] <= MAX_EPISODE_NUM]
dataset_score_monet_add_lgc = dataset_score_monet_add_lgc[dataset_score_monet_add_lgc['Step'] <= MAX_EPISODE_NUM]
dataset_score_monet = dataset_score_monet[dataset_score_monet['Step'] <= MAX_EPISODE_NUM]
dataset_score_monet_nolgc = dataset_score_monet_nolgc[dataset_score_monet_nolgc['Step'] <= MAX_EPISODE_NUM]
dataset_score_monet_noplan_control = dataset_score_monet_noplan_control[dataset_score_monet_noplan_control['Step'] <= MAX_EPISODE_NUM]
dataset_score_monet_noplan_percept = dataset_score_monet_noplan_percept[dataset_score_monet_noplan_percept['Step'] <= MAX_EPISODE_NUM]

dataset_loss_list = [dataset_loss_monet_add_lgc, dataset_loss_monet_add, dataset_loss_monet_noplan_control]
dataset_score_list = [dataset_score_monet_add_lgc, dataset_score_monet_add, dataset_score_monet_noplan_percept, dataset_score_monet_noplan_control]
# dataset_loss_list = [dataset_loss_monet_add_lgc, dataset_loss_monet_add, dataset_loss_monet, dataset_loss_monet_nolgc, dataset_loss_monet_noplan_control]
# dataset_score_list = [dataset_score_monet_add_lgc, dataset_score_monet_add, dataset_score_monet, dataset_score_monet_nolgc, dataset_score_monet_noplan_control]
dataset_loss_combined = pd.concat(dataset_loss_list, ignore_index=True)
dataset_score_combined = pd.concat(dataset_score_list, ignore_index=True)
print(dataset_loss_combined)
print("dataset_loss_combined :", dataset_loss_combined.shape)
print(dataset_score_combined)
print("dataset_score_combined :", dataset_score_combined.shape)

# Create a figure and a set of subplots (first axes)
fig, ax1 = plt.subplots()

# # color palette: https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
# seaborn_plot1 = sns.lineplot(x='Step', y='Value', ax=ax1, data=dataset_loss_combined, hue="Policy", palette=sns.color_palette()[:4], linewidth=1)

# Set the palette to another palette, e.g., 'pastel'
sns.set_palette("pastel")
seaborn_plot1 = sns.lineplot(x='Step',
                             y='Value',
                             ax=ax1,
                             data=dataset_loss_combined,
                             hue="Policy",
                             markers=True,
                             linewidth=1)

# Create a second set of axes (second y-axis) sharing the same x-axis
ax2 = ax1.twinx()
# Set the palette to the default ('deep')
sns.set_palette("deep")
seaborn_plot2 = sns.lineplot(x='Step',
                             y='Value',
                             ax=ax2,
                             data=dataset_score_combined,
                             hue="Policy",
                             marker="o",
                             markersize=4,  # Set marker size
                             linestyle="-",  # Set line style
                            #  linestyle="-.",  # Set line style
                             linewidth=1
                             )

# # for removing title of legend
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(handles=handles[0:], labels=labels[0:], loc=2) # loc 1,2,3 : 'upper right', 'upper left', 'upper center' / loc 4,5,6 : 'lower right', 'lower left', 'lower center' / loc 7,8,9 : 'center right', 'center left', 'center center'


# Chage x/y label
ax1.set_xlabel('Update Step')
ax1.set_ylabel('L1 Loss')
# Getting the legend
legend_ax1 = ax1.legend()

ax2.set_ylabel('Similarity Score')
# ax2.set_ylim([0.95, 1.40])
# Getting the legend
legend_ax2 = ax2.legend()

# # Show the legend (this will combine the legends from both plots)
# plt.legend()

# Place the legend to the lower center outside of the plot
ax1.legend(title="L1 Loss", loc='lower center', bbox_to_anchor=(0.22, -0.3), ncol=2)
ax2.legend(title="Similarity Score", loc='lower center', bbox_to_anchor=(0.75, -0.3), ncol=2)
# plt.legend(loc='lower center', bbox_to_anchor=(1, 1))

# Adjust the plot so the legend does not cut off
plt.tight_layout()

# plt.xlabel('Update Step')
# plt.ylabel('Evaluation Loss')
# plt.xlim([0,3200])
plt.show()
