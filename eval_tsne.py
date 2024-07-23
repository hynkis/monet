# reference: https://gaussian37.github.io/ml-concept-t_sne/
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
from matplot_selector import SelectFromCollection

# # MNIST data
# data = load_digits()
# print("data.data :", data.data.shape)

# Path (result data)
MODEL_NAME = 'MoNet_v01_add_LGC'
# MODEL_NAME = 'MoNet_v01'
BASE_PATH = './baseline_model/' + MODEL_NAME + '/results/final_model_result'
DATA_PATH = BASE_PATH + '/' + MODEL_NAME + '_data.csv'
# BASE_PATH = './baseline_model/MoNet_v01/results/final_model_result'
# DATA_PATH = BASE_PATH + '/MoNet_v01_data.csv'

data = pd.read_csv(DATA_PATH, header=None)
print(data)
print("data :", data.shape)

# data_decision = data[:,:16]

# t-sne model
# for perp in [50, 100, 150, 200, 500, 1000]:

n_dim = 2

model = TSNE(
            random_state=0,
            n_components = n_dim, # 3; 2; embedding dimention
            perplexity=30, # best: 30; 45; 50; 5 ~ 30 ~ 50
            early_exaggeration=50, # best: 10; 1-; 50; 1 ~ 4 ~
            n_iter=1000, # 1000; 10 ~ 200 ~ 1000
            learning_rate='auto',
            init='random',
            )
            
# model = TSNE(
#             random_state=0,
#             n_components = n_dim, # 3; 2; embedding dimention
#             perplexity=30, # best: 30; 45; 50; 5 ~ 30 ~ 50
#             early_exaggeration=50, # best: 10; 1-; 50; 1 ~ 4 ~
#             n_iter=1000, # 1000; 10 ~ 200 ~ 1000
#             learning_rate='auto',
#             init='random',
#             )

# model = TSNE(
#             random_state=0,
#             n_components = n_dim, # 3; 2; embedding dimention
#             perplexity=30, # best: 30; 45; 50; 5 ~ 30 ~ 50
#             early_exaggeration=10, # best: 10; 1-; 50; 1 ~ 4 ~
#             n_iter=1000, # 1000; 10 ~ 200 ~ 1000
#             learning_rate='auto',
#             init='random',
#             )

# model = TSNE()
data_decision_np = data.to_numpy()[:,:16]
# data_decision_np = data_decision.to_numpy()[:900]
result = model.fit_transform(data_decision_np)
print("result :", result.shape)

fig, (ax1, ax2) = plt.subplots(1,2)

# 1-D
if n_dim == 1:
    data_index = list(range(len(result)))
    # pts = ax1.plot(result[:,0], result[:,1], 'o', color='cornflowerblue', alpha=0.5, label="result")
    pts1 = ax1.scatter(data_index, result[:,0])
    pts2 = ax2.scatter(data_index, result[:,0], alpha=0.2)
    # ax1.plot(result[:,0], 'o', color='cornflowerblue', alpha=0.5, label="result")
    ax1.set_xlabel("X")
    ax2.set_xlabel("X")

# 2-D
elif n_dim == 2:
    # pts = ax1.plot(result[:,0], result[:,1], 'o', color='cornflowerblue', alpha=0.5, label="result")
    # pts1 = ax1.scatter(result[:,0], result[:,1]) # total

    index_CA_left_end  = 31
    index_CA_right_end = 31+36
    index_LT_end       = 31+36+59
    index_RT_end       = 31+36+59+59
    index_ST_end       = 31+36+59+59+64
    index_ST_wall_end  = 31+36+59+59+64+318

    pts1 = ax1.scatter(result[:index_CA_left_end,0], result[:index_CA_left_end,1], alpha=0.5,  color='green') # CA-left
    pts1 = ax1.scatter(result[index_CA_left_end:index_CA_right_end,0], result[index_CA_left_end:index_CA_right_end,1], alpha=0.5,  color='green') # CA-right
    pts1 = ax1.scatter(result[index_CA_right_end:index_LT_end,0], result[index_CA_right_end:index_LT_end,1], alpha=0.5,  color='red') # LT
    pts1 = ax1.scatter(result[index_LT_end:index_RT_end,0], result[index_LT_end:index_RT_end,1], alpha=0.5,  color='blue') # RT
    pts1 = ax1.scatter(result[index_RT_end:index_ST_end,0], result[index_RT_end:index_ST_end,1], alpha=0.5,  color='orange') # ST
    pts1 = ax1.scatter(result[index_ST_end:index_ST_wall_end,0], result[index_ST_end:index_ST_wall_end,1], alpha=0.5,  color='brown') # ST-wall

    pts2 = ax2.scatter(result[:,0], result[:,1], alpha=0.2)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

# 3-D
elif n_dim == 3:
    # ax1 = fig.add_subplot(1,2,1, projection='3d')
    # ax2 = fig.add_subplot(1,2,2, projection='3d')
    # ax1.plot(result[:,0], result[:,1], result[:,2], 'o', color='cornflowerblue', alpha=0.5, label="result")
    pts1 = ax1.scatter(result[:,0], result[:,1], result[:,2])
    pts2 = ax2.scatter(result[:,0], result[:,1], result[:,2], alpha=0.2)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    
selector = SelectFromCollection(ax1, pts1)

def accept(event):
    if event.key == "enter":
        print("===========================================")
        print("Selected points:")
        # print(selector.xys[selector.ind])
        print(selector.ind)
        # print("data with index :")
        # print(data_decision_np[selector.ind])
        # selector.disconnect()
        # ax1.set_title("Choose data")
        # fig.canvas.draw()

fig.canvas.mpl_connect("key_press_event", accept)
# ax1.set_title("Press enter to accept selected points. (Left Mouse for reset)")


plt.show()