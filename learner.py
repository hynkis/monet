import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
from .utils import compute_WD, reorder_tensor, calc_entropy
from .modules import MoNet

class IL(object):
    """
    Self-Supervised and Interpretable Sensorimotor Learning via Latent Neural Modularity
    """
    def __init__(self, img_shape, args):
        # Image size
        self.img_height = img_shape[0]
        self.img_width  = img_shape[1]
        # Hyperparam
        self.batch_size = args.batch_size
        self.max_grad_norm = args.max_grad_norm
        #self.weight_WD  = args.w_wasser_loss

        self.w_dec = args.w_dec_loss
        self.w_ent = args.w_ent_loss

        # Device
        self.device = torch.device("cuda:" + str(args.gpu_num))
        if args.cpu:
            self.device = torch.device("cpu")

        # Network and Optimizer
        self.monet = MoNet(
            args=args,
        ).to(self.device)
        self.d_layer_dim = self.monet.d_layer_dim

        self.monet_optim = torch.optim.Adam(self.monet.parameters(), lr=args.lr)

        self.monet_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.monet_optim,
            lr_lambda=lambda epoch: args.lr_lambda**epoch, # lr_lambda: 0.99
            last_epoch=-1,
        )

    # Loss function
    # - Weighted L1 distance loss (WL1)
    def loss_function_L1(self, action, label):
        # MSE = F.mse_loss(action, label)
        weight = torch.tensor([1.0, 0.1]).to(self.device)
        # weight = torch.tensor([1.0, 0.5]).to(self.device)
        WL1 = torch.sum(weight * abs(action - label)) # squared sum
        WL1 = WL1 / action.shape[0] # mean
        return WL1

    # - Weighted Mean Squared Error (MSE)
    def loss_function_MSE(self, action, label):
        # MSE = F.mse_loss(action, label)
        weight = torch.tensor([1.0, 0.1]).to(self.device)
        # weight = torch.tensor([1.0, 0.5]).to(self.device)
        WMSE = torch.sum(weight * (action - label) ** 2) # squared sum
        WMSE = WMSE / action.shape[0] # mean
        return WMSE

    def loss_contrastive(self, z_p, h_d):
        """
            # L_dec = {1 - cos(h_i, h_j)},            if sign{cos(z_i, z_j) - 0.5} == +1
                      max(0, cos(h_i, h_j) - margin), if sign{cos(z_i, z_j) - 0.5} == -1
            : if perceptual saliency maps are similar (+1), decrease dissimilarity
            : if perceptual saliency maps are not similar (-1), increase dissimilarity
            
            # arguments
            - Att: bottom-up perceptual saliency map (batch, N, N)
            - h_d: top-down latent decision (batch, D)

        """
        # set two different batch data
        # : changing order of element in batch is enough to make them different
        # : x1 [0, 1, 2, 3]
        # : x2 [3, 0, 1, 2]

        # Set data1, data2
        z_i = z_p
        z_j = reorder_tensor(z_p)
        h_i = h_d
        h_j = reorder_tensor(h_d)
        
        # Define functions
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-08)
        criterion = nn.CosineEmbeddingLoss(reduction='none')
        
        # Check similarity sign
        batch = z_i.shape[0]
        with torch.no_grad():
            # cos(attention, attention) is always in [0, 1]
            # normalize to -1, +1
            z_i_flat = torch.reshape(z_i, (batch, -1))
            z_j_flat = torch.reshape(z_j, (batch, -1))
            # target_label = torch.sign(cos_sim(z_i_flat, z_j_flat))
            # target_weight = torch.abs(cos_sim(z_i_flat, z_j_flat))

            # Cosine similarity (-1 if cos value is larger thatn orthogonal (0.5) )
            target_label = torch.sign(2*cos_sim(z_i_flat, z_j_flat) - 1)
            # target_label = torch.sign(cos_sim(z_i_flat, z_j_flat))
        
        # Compute contrastive cosine loss
        # L_cos = torch.sum(target_weight * criterion(h_i, h_j, target_label)) / batch
        L_cos = torch.sum(criterion(h_i, h_j, target_label)) / batch
        # L_cos = torch.sum(torch.abs(2*cos_sim(z_i_flat, z_j_flat) - 1) * criterion(h_i, h_j, target_label))

        print("target_label :", torch.sum(target_label)/batch)
        # print("target_weight :", torch.sum(target_weight)/batch)

        return L_cos
    
    def loss_entropy(self, decision):
        # entropy w.r.t. batch decision 
        d_ent = calc_entropy(decision)

        # max entropy
        normal_decision = [1/self.d_layer_dim]*self.d_layer_dim # [1/16, ..., 1/16]
        normal_decision = torch.tensor(normal_decision).to(self.device)
        max_ent = calc_entropy(normal_decision)

        norm_ent = torch.sum(1 - (d_ent / max_ent))
        norm_ent = norm_ent/ decision.shape[0]
        
        return norm_ent

    def update(self, X, y, is_decision_loss=False, is_entropy_loss=False, is_stoch_decision=False):
        # Parsing inputs (image, bev)
        s_img, s_bev = X
        
        # Feedforward network
        control, decision, perception = self.monet(s_img, s_bev, is_stoch_decision)

        # Calculate lose
        # loss_MSE = self.loss_function_MSE(a, y)
        # loss = loss_MSE
        loss_L1 = self.loss_function_L1(control, y)
        loss = loss_L1

        if is_decision_loss:
            # get bottom-up saliency map
            #Att = self.monet.att_map_sp
            loss_dec = self.loss_contrastive(perception, decision)
            loss = loss + self.w_dec * loss_dec

        if is_entropy_loss:
            loss_ent = self.loss_entropy(decision)
            loss = loss + self.w_ent * loss_ent

        # # Compute entropy of distribution of the latent decision
        # d = self.monet.get_decision(s_img, s_bev, is_stoch_decision)
        # loss_ent = self.loss_entropy(d)

        # Update
        self.monet_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.monet.parameters(), self.max_grad_norm)
        self.monet_optim.step()

        # cosine similarity calc for recording (with no_grad)
        if is_decision_loss == False:
            with torch.no_grad():
                #Att = self.monet.att_map_sp
                loss_dec = self.loss_contrastive(perception, decision)

        # entropy calc for recording (with no_grad)
        if is_entropy_loss == False:
            with torch.no_grad():
                loss_ent = self.loss_entropy(decision)

        # for memory
        del control, decision, perception, s_img, s_bev, loss

        return loss_L1.detach().cpu().item(), loss_dec.detach().cpu().item(), loss_ent.detach().cpu().item()

    def calc_loss(self, X, y, is_stoch_decision=False):
        # Parsing inputs (image, bev)
        s_img, s_bev = X
        
        # Feedforward network
        control, decision, perception = self.monet(s_img, s_bev, is_stoch_decision)

        # Calculate lose
        loss_L1 = self.loss_function_L1(control, y)
        loss_dec = self.loss_contrastive(perception, decision)
        loss_ent = self.loss_entropy(decision)

        # for memory
        del control, decision, perception, s_img, s_bev
        
        return loss_L1.detach().cpu().item(), loss_dec.detach().cpu().item(), loss_ent.detach().cpu().item()

    def save_model(self, path):
        torch.save({
                    'monet' : self.monet.state_dict(),
                    'monet_optim' : self.monet_optim.state_dict(),
                    }, path)

    def load_model(self, path, is_eval=False):
        checkpoint = torch.load(path, map_location=self.device)
        self.monet.load_state_dict(checkpoint['monet'])
        self.monet_optim.load_state_dict(checkpoint['monet_optim'])
        self.monet.to(self.device)
        
        if is_eval is False:
            self.monet.train()
        else:
            self.monet.eval()

            
