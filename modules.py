import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import torchvision.models as models
from torch.autograd import Variable

from einops import rearrange, repeat

"""
state: 
- s_bev (1,224,224)   = [front camera image (grayscale)]
- s_bev_route (3,64,64) = [BEV route image (color)]
- s_vehicle (2)     = [imu_accel_y, imu_yaw_rate]
- action = [norm_curv_steer, norm_des_vel]
"""
action_dim        = 2
state_vehicle_dim = 2

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape

def conv3x3(in_planes, out_planes, stride=1):
    """"
    From PyTorch Resnet implementation
    3x3 convolution with padding
    :param in_planes: (int)
    :param out_planes: (int)
    :param stride: (int)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

## ---------------------- Modular End-to-End Network ---------------------- ##

class MoNet(nn.Module):
    def __init__(self, args):
        super(MoNet, self).__init__()
        # Parameters
        self.device = torch.device("cuda:" + str(args.gpu_num))
        # self.device = torch.device("cuda")
        # - dimensions
        # -- for sensory input (front camera image, grayscale, agents)
        self.z_cnn_dim   = 64
        self.z_bev_dim   = 64+1 # 64+1,  # +1 for pos_embedding. final dim of bev_img

        # -- for sensory input (BEV route, color)
        self.z_route_cnn_dim  = 64
        self.z_bev_route_dim1 = 1*1*self.z_route_cnn_dim # (batch, T, 64, 1, 1) -> (batch, T, 64*1*1)
        self.z_bev_route_dim2 = 64

        self.z_merge_dim = self.z_bev_dim + self.z_bev_route_dim2 # z_bev(65) + z_route_bev(64)

        # - Spatial Attention
        self.prob_drop_sp = 0.3
        self.att_map_sp    = None
        self.n_heads_sp    = 1
        self.N_sp          = 6*6 # 36, N_sp: size of spatial attention map. (cH * cW)
        self.node_size_sp  = 64

        # - Contextual Attention
        self.prob_drop_ct = 0.3
        self.ct_feat_dim  = 64 # for (batch, N_ct, 1) to (batch, N_ct, ct_feat_dim)
        self.att_map_ct   = None
        self.n_heads_ct   = 1
        self.N_ct         = self.z_merge_dim # final dim of z_merge
        self.node_size_ct = 64 # 128 16
        # - decision output
        self.z_decision  = None
        self.d_layer_dim = 16 # 64

        # - control output
        self.c_hidden_dim1 = 256 # self.z_merge_dim(skip-connection) # 128 # 64 256 32
        self.c_hidden_dim2 = 256 # self.z_merge_dim(skip-connection) # 128 # 64 256 16
        self.action_dim = action_dim # [norm_curv_steer, norm_target_vel]

        # - standard deviation for Gaussian decision
        self.d_std = args.decision_std
        # - temperature factor for Boltzmann Distribution
        self.boltz_alpha = args.boltz_alpha # [default (=1) ~ smooth softmax (<1)] 
 
        # ============================== #
        # ----- Perception Network ----- #
        # ============================== #

        # ----- Spatial Feature Extractor ------ #
        # 1. feature extractor for front camera image (grayscale), (1,224,224)
        #   Inspired by ResNet:
        #       conv3x3 followed by BatchNorm2d
        #       channel increasing: 16 -> 32 -> 64 (21.12.24, seong)
        #       channel constant: 64 -> 64 -> 64 (21.12.27, seong)
        #       RELU was better than GELU (22.01.19, seong)
        
        self.perception_s_img = nn.Sequential(
            nn.Conv2d(1, self.z_cnn_dim, kernel_size=7, stride=2, padding=3, bias=False), # for (224,224) 
            nn.BatchNorm2d(self.z_cnn_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # (self.z_cnn_dim,56,56) for (224,224)

            conv3x3(in_planes=self.z_cnn_dim, out_planes=self.z_cnn_dim, stride=1),
            nn.BatchNorm2d(self.z_cnn_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (self.z_cnn_dim,27,27) for (224,224)

            conv3x3(in_planes=self.z_cnn_dim, out_planes=self.z_cnn_dim, stride=2),
            nn.BatchNorm2d(self.z_cnn_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # (self.z_cnn_dim,6,6) for (224,224) 
        )

        # - Spatial Attention network (inspired from Vision Transformer)
        # -- position embedding
        #   (inspired from https://github.com/lucidrains/vit-pytorch/blob/e52ac4195550faa9c3372533d325bf649f7354ad/vit_pytorch/vit.py#L96-L97)
        # -- no need class token
        self.pos_embedding_img = nn.Parameter(torch.randn(1, 1, 6, 6)) # for learning positioning feature, 1 for batch dim, 1 for pos embedding, 6, 6 for 6x6 feature.
        # -- layer normalization
        self.norm_att_sp1 = nn.LayerNorm([self.N_sp, self.z_bev_dim], elementwise_affine=True) # (batch, N_sp, z_bev_dim)
        # -- attention mechanism
        self.proj_shape_sp = (self.z_bev_dim, self.n_heads_sp * self.node_size_sp)
        self.k_proj_sp = nn.Linear(*self.proj_shape_sp)
        self.q_proj_sp = nn.Linear(*self.proj_shape_sp)
        self.v_proj_sp = nn.Linear(*self.proj_shape_sp)

        self.norm_att_head_sp = nn.LayerNorm([self.N_sp, self.n_heads_sp * self.node_size_sp], elementwise_affine=True) # (batch, N_sp, head*dim)
        self.linear_att_head_sp = nn.Linear(self.n_heads_sp * self.node_size_sp, self.z_bev_dim)
        # -- dropout
        self.dropout_sp1 = nn.Dropout(self.prob_drop_sp)

        # -- layer normalization
        self.norm_att_sp2 = nn.LayerNorm([self.N_sp, self.z_bev_dim], elementwise_affine=True) # (batch, N, z_bev_dim)
        # -- post feedforward
        self.linear_att_sp = nn.Linear(self.z_bev_dim, self.z_bev_dim)
        # -- dropout
        self.dropout_sp2 = nn.Dropout(self.prob_drop_sp)
        
        #   - Weight initialization (Kaiming normal for attention fc)
        nn.init.kaiming_normal_(self.linear_att_head_sp.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_att_sp.weight, mode='fan_in', nonlinearity='relu')

        # 2. feature extractor for BEV Route state (color), (3,224,224)
        self.perception_s_bev_route = nn.Sequential(
            nn.Conv2d(3, self.z_route_cnn_dim, kernel_size=7, stride=2, padding=3, bias=False), # for (224,224) 
            nn.BatchNorm2d(self.z_route_cnn_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            conv3x3(in_planes=self.z_route_cnn_dim, out_planes=self.z_route_cnn_dim, stride=1),
            nn.BatchNorm2d(self.z_route_cnn_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            conv3x3(in_planes=self.z_route_cnn_dim, out_planes=self.z_route_cnn_dim, stride=2),
            nn.BatchNorm2d(self.z_route_cnn_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Flatten(start_dim=1, end_dim=-1), # (batch*T, 64, 1, 1) -> (batch*T, 1*1*64)
            nn.Linear(self.z_bev_route_dim1, self.z_bev_route_dim2), # (batch*T, 1*1*64) -> (batch*T, z_bev_route_dim2)
        )

        # ----- Temporal Feature Extractor ------ #
        # - for LSTM/GRU
        self.lstm = nn.LSTM(self.z_merge_dim, self.z_merge_dim, batch_first=True) # (batch, z_merge_dim)

        # ============================ #
        # ----- Decision Network ----- #
        # ============================ #
        self.pre_linear_ct = nn.Linear(1, self.ct_feat_dim) # (batch, N_ct, 1) --> (batch, N_ct, ct_feat_dim)
        # -- position embedding (inspired from https://github.com/lucidrains/vit-pytorch/blob/e52ac4195550faa9c3372533d325bf649f7354ad/vit_pytorch/vit.py#L96-L97)
        self.pos_embedding_ct = nn.Parameter(torch.randn(1, self.N_ct, 1)) # for learning positioning feature, 1 for batch dim, N_ct for context, 1 for embedding feature.
        # -- layer normalization
        self.norm_att_ct1 = nn.LayerNorm([self.N_ct, self.ct_feat_dim+1], elementwise_affine=True) # (batch, N_ct, ct_feat_dim+1)
        # -- contextual Attention network
        #   (ct_feat_dim+1, head*dim). +1 for pos_embedding. span from ct_feat_dim+1 to head*dim
        self.proj_shape_ct = (self.ct_feat_dim+1, self.n_heads_ct * self.node_size_ct)
        self.k_proj_ct = nn.Linear(*self.proj_shape_ct)
        self.q_proj_ct = nn.Linear(*self.proj_shape_ct)
        self.v_proj_ct = nn.Linear(*self.proj_shape_ct)

        self.norm_att_head_ct   = nn.LayerNorm([self.N_ct, self.n_heads_ct * self.node_size_ct], elementwise_affine=True) # (batch, N_ct, head*dim)
        self.linear_att_head_ct = nn.Linear(self.n_heads_ct * self.node_size_ct, self.ct_feat_dim+1) # (batch, N_ct, head*dim) -> (batch, N_ct, ct_feat_dim+1)
        # -- dropout
        self.dropout_ct1 = nn.Dropout(self.prob_drop_ct)

        # -- layer normalization
        self.norm_att_ct2 = nn.LayerNorm([self.N_ct, self.ct_feat_dim+1], elementwise_affine=True) # (batch, N_ct, ct_feat_dim+1)
        # -- post feedforward
        self.linear_att_ct = nn.Linear(self.ct_feat_dim+1, self.ct_feat_dim+1) # (batch, N_ct, ct_feat_dim+1)
        # -- linear before boltzmann distribution
        self.linear_out_ct = nn.Linear(self.ct_feat_dim+1, self.d_layer_dim) # (batch, ct_feat_dim+1) -> (batch, d_layer_dim)
        # -- dropout
        self.dropout_ct2 = nn.Dropout(self.prob_drop_ct)

        # -- weight initialization (Kaiming normal for attention fc)
        nn.init.kaiming_normal_(self.linear_att_head_ct.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_att_ct.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_out_ct.weight, mode='fan_in', nonlinearity='relu')
        

        # =========================== #
        # ----- Control Network ----- #
        # =========================== #
        # --------- (concat with output of decision network) ---------- #
        self.control_layer1 = nn.Linear(self.N_ct, self.c_hidden_dim1) # (batch, z_merge_dim+1) -> (batch, c_hidden_dim1)
        self.control_layer2 = nn.Linear(self.c_hidden_dim1, self.c_hidden_dim2) # (batch, c_hidden_dim1) -> (batch, c_hidden_dim2)
        self.control_layer3 = nn.Linear(self.c_hidden_dim2, self.d_layer_dim) # (batch, c_hidden_dim2) -> (batch, d_layer_dim)
        self.control_layer4 = nn.Linear(self.d_layer_dim, self.d_layer_dim) # (batch, d_layer_dim) -> (batch, d_layer_dim)
        self.control_layer5 = nn.Linear(self.d_layer_dim, action_dim)
        # self.control_layer_norm1 = nn.LayerNorm([self.c_hidden_dim1], elementwise_affine=True)
        # self.control_layer_norm2 = nn.LayerNorm([self.c_hidden_dim2], elementwise_affine=True)

        # - weight initialization (Normal for control fc)
        torch.nn.init.xavier_uniform_(self.control_layer1.weight, gain=1)
        torch.nn.init.constant_(self.control_layer1.bias, 0)
        torch.nn.init.xavier_uniform_(self.control_layer2.weight, gain=1)
        torch.nn.init.constant_(self.control_layer2.bias, 0)
        torch.nn.init.xavier_uniform_(self.control_layer3.weight, gain=1)
        torch.nn.init.constant_(self.control_layer3.bias, 0)
        torch.nn.init.xavier_uniform_(self.control_layer4.weight, gain=1)
        torch.nn.init.constant_(self.control_layer4.bias, 0)
        torch.nn.init.xavier_uniform_(self.control_layer5.weight, gain=1)
        torch.nn.init.constant_(self.control_layer5.bias, 0)

        # ---------------------- Control Network (END) ---------------------- #

    def forward_spatial_perception(self, s_img, s_bev_route):
        """
        - state observations:
            - s_img       : (batch, C, H, W)
            - s_bev_route : (batch, H, W)
            - s_vehicle   : (batch, dim)
        - encoded feature (spatial):
            - z_merge : (batch, z_merge_dim)
        - encoded feature (spatio-temporal):
            - z_merge : (batch, z_merge_dim)
        """
        # Encode image input
        B, C, H, W = s_img.shape
        s_img = self.perception_s_img(s_img) # (batch, 64, 6, 6)

        # B, T, C, H, W = s_img.shape
        # # - Convert dimension (batch, sequence, C, H, W) -> (batch*sequence, C, H, W)
        # s_img = s_img.view(B*T, C, H, W)
        # # - Convert dimension (batch*sequence, 64, 6, 6) -> (batch, sequence(=T), 64, 6, 6)
        # s_img = s_img.view(B, T, self.z_cnn_dim, 6, 6)

        # Position embedding
        pos_embedding_img = repeat(self.pos_embedding_img, '() d h w -> batch d h w', batch = B)
        s_img = torch.cat([s_img, pos_embedding_img], dim=1) # (batch, 64+1, 6, 6); dim=1 for channel dim; +1 for pos embedding
        s_img = s_img.permute(0, 2, 3, 1) # (batch, H(6), W(6), C(64+1))
        s_img = s_img.flatten(1, 2) # (batch, H, W, C(64+1)) -> (batch, H*W(== N_sp), C(64+1)(=z_img_dim)). 

        # Spatial Attention
        # - pre layer normalization before attention
        s_img_norm = self.norm_att_sp1(s_img) # (batch, N_sp, z_bev_dim)
        # - (batch, N_sp, z_bev_dim) -> (batch, N_sp, (head*dim)) -> (batch, head, N_sp, dim)
        Q_sp = rearrange(self.q_proj_sp(s_img_norm), "batch N (head d) -> batch head N d", head=self.n_heads_sp) # -> (batch, head, N_sp, dim)
        K_sp = rearrange(self.k_proj_sp(s_img_norm), "batch N (head d) -> batch head N d", head=self.n_heads_sp)
        V_sp = rearrange(self.v_proj_sp(s_img_norm), "batch N (head d) -> batch head N d", head=self.n_heads_sp)
        A_sp = torch.einsum('bhfe,bhge->bhfg', Q_sp, K_sp) # (batch, head, N_sp, dim) -> (batch, head, N_sp, N_sp)
        A_sp = A_sp / np.sqrt(self.node_size_sp)
        A_sp = nn.functional.softmax(A_sp, dim=3)  # A_sp x V : softmax w.r.t. last dim
        with torch.no_grad():
            self.att_map_sp = A_sp.clone()
        # - (batch, head, N_sp, N_sp) * (batch head N_sp, dim) -> (batch head, N_sp, dim)
        E_sp = torch.einsum('bhfc,bhcd->bhfd', A_sp, V_sp)
        # - (batch, head, N_sp, dim) -> (batch, N_sp, head*dim)
        E_sp = rearrange(E_sp, 'batch head N d -> batch N (head d)')
        # - layer normalization
        E_sp = self.norm_att_head_sp(E_sp) # (batch, N_sp, head*dim)
        # - feedforward
        E_sp = self.linear_att_head_sp(E_sp) # (batch, N, head*dim) -> (batch, N, z_bev_dim)

        # Skip connection with feature (before pre layer norm)
        E_sp = E_sp + s_img # (batch, N, z_bev_dim)
        # - apply dropout
        E_sp = self.dropout_sp1(E_sp)

        # Post feedforward
        # - pre layer normalization before post feedforward
        E_sp_post = self.norm_att_sp2(E_sp) # (batch, N, z_bev_dim)
        # - post feedforward
        # -- RELU was better than GELU (22.01.19, seong)
        E_sp_post = self.linear_att_sp(E_sp_post)
        E_sp_post = torch.relu(E_sp_post)

        # - skip connection with feature (before pre layer norm)
        E_sp_post = E_sp_post + E_sp # (batch, N, z_bev_dim)
        # - apply dropout
        E_sp_post = self.dropout_sp2(E_sp_post)

        # Mean pool
        z_img = E_sp_post.mean(dim=1) # (batch, N, z_bev_dim) -> (batch, z_bev_dim);

        # - Extract BEV route feature
        z_bev_route = self.perception_s_bev_route(s_bev_route)

        # - Merge features
        z_merge = torch.cat([z_img, z_bev_route], dim=-1) # (batch, T, z_merge_dim)
        
        return z_merge

    # def forward_temporal_perception(self, z_merge):
    #     """
    #     Perception Network (Temporal Feature)
    #     """
    #     # LSTM
    #     self.lstm.flatten_parameters() # for multi GPU
    #     z_merge, _  = self.lstm(z_merge) # (batch, sequence(=N_tp), z_merge_dim)
    #     # - get last hidden output
    #     z_context = z_merge[:, -1, :]

    #     return z_context

    def forward_decision(self, z_context, is_stoch_decision=False):
        """
        Decision Network
        """
        # Position embedding
        B, _ = z_context.shape # (batch, z_merge_dim+1(=N_ct)); (+1 for temporal position embedding)
        # - encoding
        z_context = z_context.unsqueeze(dim=-1) # (batch, N_ct) -> (batch, N_ct, 1)
        z_context = self.pre_linear_ct(z_context) # (batch, N_ct, 1) -> (batch, N_ct, ct_feat_dim)
        # - add position embedding
        pos_embedding_ct = repeat(self.pos_embedding_ct, '() C d -> batch C d', batch = B)
        z_context = torch.cat([z_context, pos_embedding_ct], dim=2) # (batch, N_ct, ct_feat_dim+1(=dim))
        
        # - Multi-head Attention
        # - pre layer normalization before attention
        z_context_norm = self.norm_att_ct1(z_context)
        # -- (batch, N_ct, dim) -> (batch, N_ct, (head*dim)) -> (batch, head, N_ct, dim)
        Q = rearrange(self.q_proj_ct(z_context_norm), "batch C (head d) -> batch head C d", head=self.n_heads_ct) # -> (batch, head, N_ct, dim)
        K = rearrange(self.k_proj_ct(z_context_norm), "batch C (head d) -> batch head C d", head=self.n_heads_ct)
        V = rearrange(self.v_proj_ct(z_context_norm), "batch C (head d) -> batch head C d", head=self.n_heads_ct)
        A = torch.einsum('bhfe,bhge->bhfg', Q,K) # (batch, head, N_ct, dim) -> (batch, head, N_ct, N_ct)
        A = A / np.sqrt(self.node_size_ct)
        A = nn.functional.softmax(A, dim=3)  # A x V : softmax w.r.t. last dim
        with torch.no_grad():
            self.att_map_ct = A.clone()
        E = torch.einsum('bhfc,bhcd->bhfd', A, V) # (batch, head, N_ct, N_ct) * (batch head, N_ct, dim) -> (batch, head, N_ct, dim)
        E = rearrange(E, 'batch head C d -> batch C (head d)') # (batch, head, N_ct, dim) -> (batch, N_ct, head*dim)
        # - layer normalization
        E = self.norm_att_head_ct(E) # (batch, N_ct, head*dim)
        # - feedforward
        E = self.linear_att_head_ct(E) # (batch, N_ct, head*dim) -> (batch, N_ct, ct_feat_dim+1)
        
        # Skip connection
        E = E + z_context # (batch, N_ct, ct_feat_dim+1)
        # apply dropout
        E = self.dropout_ct1(E)

        # Post feedforward
        # -- RELU was better than GELU (22.01.19, seong)
        # - pre layer normalization before post feedforward
        E_post = self.norm_att_ct2(E) # (batch, N_ct, ct_feat_dim+1)
        # - post feedforward
        E_post = self.linear_att_ct(E_post) # (batch, N_ct, ct_feat_dim+1)
        E_post = torch.relu(E_post)

        # Skip connection
        E_post = E_post + E # (batch, N_ct, ct_feat_dim+1)
        # apply dropout
        E_post = self.dropout_ct2(E_post)

        # Mean pool
        d_mu = E_post.mean(dim=1) # (batch, N_ct, ct_feat_dim+1) -> (batch, ct_feat_dim+1)
        d_mu = self.linear_out_ct(d_mu) # (batch, ct_feat_dim+1) -> (batch, d_layer_dim)
        
        # Gaussian Decision
        if is_stoch_decision:
            z_normal = Normal(d_mu, self.d_std)
            z_decision = F.softmax(z_normal.rsample() / self.boltz_alpha, dim=-1) # Boltzmann Decision
            # print("Stochastic decision making")
        else:
            z_decision = F.softmax(d_mu / self.boltz_alpha, dim=-1) # Boltzmann Decision
            # print("Deterministic decision making")
        # # Soft Decision
        # z_decision = F.softmax(z_decision, dim=-1) # softDecision

        with torch.no_grad():
            self.z_decision = z_decision.clone()

        return z_decision

    def forward_control(self, z_context, z_decision):
        """
        Control Network
        """
        # Sensorimotor layer - 1
        z_control = self.control_layer1(z_context) # (batch z_merge_dim+1(=N_ct))
        z_control = torch.relu(z_control)
        z_control = self.control_layer2(z_control)
        z_control = torch.relu(z_control)

        # Applying Decision output (Matmul)
        z_control = self.control_layer3(z_control)
        z_control = torch.mul(z_control, z_decision) # (batch, d_dim)

        # Sensorimotor layer - 2 (Gaussian policy)
        z_control = self.control_layer4(z_control)
        z_control = torch.relu(z_control)
        z_control = self.control_layer5(z_control)
        control = torch.tanh(z_control)

        return control

    def forward(self, s_img, s_bev_route, is_stoch_decision=False):
        """
        - state observations: s_img, s_bev_route, (B, T, D) (th.Tensor)
        - control command vector: a (B, T, 2) (th.Tensor)
        """
        # - Perception Network: Encode states
        z_context = self.forward_spatial_perception(s_img, s_bev_route)
        # z_context = self.forward_temporal_perception(z_context)

        # - Decision Network: Apply multi-head self-attention
        z_decision = self.forward_decision(z_context, is_stoch_decision=is_stoch_decision)
        
        # - Control Network: feedforward attention output with concatenating Decision output.
        action_squashed = self.forward_control(z_context, z_decision)

        return action_squashed, z_decision, z_context

    def get_decision(self, s_img, s_bev_route, is_stoch_decision=False):
        """
        - state observations: s_img, s_bev_route, (B, T, D) (th.Tensor)
        - control command vector: a (B, T, 2) (th.Tensor)
        """
        # - Perception Network: Encode states
        z_context = self.forward_spatial_perception(s_img, s_bev_route)
        # z_context = self.forward_temporal_perception(z_context)

        # - Decision Network: Apply multi-head self-attention
        z_decision = self.forward_decision(z_context, is_stoch_decision)

        return z_decision