import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

import yaml
from types import SimpleNamespace
from einops import rearrange
from utils.utils import denormalize, normalize, similarity_matrix
from models.loss import CLIPLoss


class Retrieval(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.music_encoder = Music_Encoder(input_size1=438, input_size2=1024)
        self.dance_encoder = Dance_Encoder(input_size1=3, input_size2=147)

        self.clipLoss = CLIPLoss()
    
        mean, std = torch.load('./Pretrained/mean.pt'), torch.load('./Pretrained/std.pt')
        self.smpl_trans_mean, self.smpl_poses_mean, self.smpl_root_vel_mean, \
        self.music_librosa_mean, self.music_mert_mean = \
            mean['smpl_trans_mean'].to(device).float(), mean['smpl_poses_mean'].to(device).float(), \
            mean['smpl_root_vel_mean'].to(device).float(), mean['music_librosa_mean'].to(device).float(), \
            mean['music_mert_mean'].to(device).float(),
        self.smpl_trans_std, self.smpl_poses_std, self.smpl_root_vel_std, \
        self.music_librosa_std, self.music_mert_std = \
            std['smpl_trans_std'].to(device).float(), std['smpl_poses_std'].to(device).float(), \
            std['smpl_root_vel_std'].to(device).float(), std['music_librosa_std'].to(device).float(), \
            std['music_mert_std'].to(device).float(),


    def forward(self, data, device):
        
        music_librosa = data['music_librosa'].to(device).float()
        music_mert = data['music_mert'].to(device).float()
        music_librosa = normalize(music_librosa, self.music_librosa_mean, self.music_librosa_std)
        music_mert = normalize(music_mert, self.music_mert_mean, self.music_mert_std)

        smpl_trans = data['smpl_trans'].to(device).float()
        smpl_poses = data['smpl_poses'].to(device).float()
        smpl_root_vel = data['smpl_root_vel'].to(device).float()
        smpl_trans = normalize(smpl_trans, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses = normalize(smpl_poses, self.smpl_poses_mean, self.smpl_poses_std)
        smpl_root_vel = normalize(smpl_root_vel, self.smpl_root_vel_mean, self.smpl_root_vel_std)

        music_features = self.music_encoder(music_librosa, music_mert)
        dance_features = self.dance_encoder(torch.cat([smpl_root_vel, smpl_poses], dim=3), smpl_trans)
        sim_mat = similarity_matrix(music_features, dance_features)

        result = {'music_features': music_features, 'dance_features': dance_features}
        loss = {'total': self.clipLoss(sim_mat)}

        return result, loss

    
    def inference(self, data, device):

        music_librosa = data['music_librosa'].to(device).float()
        music_mert = data['music_mert'].to(device).float()
        music_librosa = normalize(music_librosa, self.music_librosa_mean, self.music_librosa_std)
        music_mert = normalize(music_mert, self.music_mert_mean, self.music_mert_std)

        smpl_trans = data['smpl_trans'].to(device).float()
        smpl_poses = data['smpl_poses'].to(device).float()
        smpl_root_vel = data['smpl_root_vel'].to(device).float()

        smpl_trans = normalize(smpl_trans, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses = normalize(smpl_poses, self.smpl_poses_mean, self.smpl_poses_std)
        smpl_root_vel = normalize(smpl_root_vel, self.smpl_root_vel_mean, self.smpl_root_vel_std)

        music_features = self.music_encoder(music_librosa, music_mert)
        dance_features = self.dance_encoder(torch.cat([smpl_root_vel, smpl_poses], dim=3), smpl_trans)
        sim_mat = similarity_matrix(music_features, dance_features)

        result = {'music_features': music_features, 'dance_features': dance_features}
        loss = {'total': self.clipLoss(sim_mat)}

        return result, loss

    def music_encode(self, music_librosa, music_mert):
        music_feature = self.music_encoder(music_librosa, music_mert)
        # music_feature = rearrange(music_feature, 'b t c -> b (t c)')
        return music_feature

    def dance_encode(self, smpl_root_vel, smpl_poses, smpl_trans):

        smpl_trans = normalize(smpl_trans, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses = normalize(smpl_poses, self.smpl_poses_mean, self.smpl_poses_std)
        smpl_root_vel = normalize(smpl_root_vel, self.smpl_root_vel_mean, self.smpl_root_vel_std)

        dance_feature = self.dance_encoder(torch.cat([smpl_root_vel, smpl_poses], dim=3), smpl_trans)
        # dance_feature = rearrange(dance_feature, 'b t c -> b (t c)')
        return dance_feature


class Music_Encoder(nn.Module):
    def __init__(self, input_size1=438, input_size2=4800, hidden_size=256, num_heads=8, layer=6):
        super(Music_Encoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_size1, 512),
            nn.Linear(512, hidden_size)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_size2, 768),
            nn.Linear(768, hidden_size)
        )
        
        blocks = nn.Sequential()
        for i in range(2):
            blocks.add_module('Librosa_Former_{}'.format(i), TTransformer(hidden_size=hidden_size, num_heads=8))
            blocks.add_module('Librosa_DownSample_{}'.format(i), Music_DownSample(rate=2, hidden_size=hidden_size))
        self.librosa_encoder = blocks

        blocks = nn.Sequential()
        for i in range(2):
            blocks.add_module('MERT_Former_{}'.format(i), TTransformer(hidden_size=hidden_size, num_heads=8))
            blocks.add_module('MERT_DownSample_{}'.format(i), Music_DownSample(rate=2, hidden_size=hidden_size))
        self.mert_encoder = blocks

        blocks = nn.Sequential()
        for i in range(4):
            blocks.add_module('Fuse_Former_{}_B'.format(i), TTransformer(hidden_size=hidden_size, num_heads=8))
            blocks.add_module('Fuse_DownSample_{}'.format(i), Music_DownSample(rate=2, hidden_size=hidden_size))
            blocks.add_module('Fuse_Former_{}_A'.format(i), TTransformer(hidden_size=hidden_size, num_heads=8))
        self.fuse_encoder = blocks    

    def forward(self, music_librosa, music_mert):
        music_librosa = self.librosa_encoder(self.linear1(music_librosa))
        music_mert = self.mert_encoder(self.linear2(music_mert))
        x = self.fuse_encoder(music_librosa + music_mert)
        return x  


class TTransformer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super(TTransformer, self).__init__()
        Layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.layer = nn.TransformerEncoder(Layer, num_layers=1)

    def forward(self, x):
        x = rearrange(x, 'b t c -> t b c')
        x = self.layer(x)
        x = rearrange(x, 't b c -> b t c')
        return x


class Music_DownSample(nn.Module):
    def __init__(self, rate=2, hidden_size=512):
        super(Music_DownSample, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=rate, padding=1)

    def forward(self, x):
        t, b, _ = x.shape
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv1d(x)
        x = rearrange(x, 'b c t -> b t c')
        return x


class Dance_Encoder(nn.Module):
    def __init__(self, input_size1=3, input_size2=144, input_size3=72, hidden_size=256, layer=6, mode=None):
        super(Dance_Encoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_size1, 128),
            nn.Linear(128, hidden_size)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_size2, 128),
            nn.Linear(128, hidden_size)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(input_size3, 128),
            nn.Linear(128, hidden_size)
        )

        blocks = nn.Sequential()
        for i in range(2):
            blocks.add_module('SMPL_Former_{}'.format(i), TNTransformer(hidden_size=hidden_size))
            blocks.add_module('SMPL_DownSample_{}'.format(i), Dance_DownSample(rate=2, hidden_size=hidden_size))
        self.smpl_encoder = blocks

        blocks = nn.Sequential()
        for i in range(2):
            blocks.add_module('Trans_Former_{}'.format(i), TNTransformer(hidden_size=hidden_size))
            blocks.add_module('Trans_DownSample_{}'.format(i), Dance_DownSample(rate=2, hidden_size=hidden_size))
        self.trans_encoder = blocks

        blocks = nn.Sequential()
        for i in range(4):
            blocks.add_module('Fuse_Former_{}_B'.format(i), TNTransformer(hidden_size=hidden_size))
            blocks.add_module('Fuse_DownSample_{}'.format(i), Dance_DownSample(rate=2, hidden_size=hidden_size))
            blocks.add_module('Fuse_Former_{}_A'.format(i), TNTransformer(hidden_size=hidden_size))
        self.fuse_encoder = blocks


    def forward(self, smpl, trans):
        trans = self.trans_encoder(self.linear1(trans))
        smpl = self.smpl_encoder(self.linear2(smpl))
        x = self.fuse_encoder(trans + smpl)
        x = torch.mean(x, dim=1)
        return x  


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, adj, x):
        # Compute the degree matrix
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.  # Handle zero degree nodes

        # Normalize adjacency matrix
        adj_norm = adj * deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)

        # Apply graph convolution
        x = torch.matmul(adj_norm, x)  # Shape: (b, n, c)
        x = torch.matmul(x, self.weight)  # Shape: (b, n, out_features)
        x = x + self.bias
        return x

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, out_features)

    def forward(self, adj, x):
        x = F.relu(self.gcn1(adj, x))
        x = self.gcn2(adj, x)
        return x
        
class TNTransformer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super(TNTransformer, self).__init__()
        T_Layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        N_Layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.T_Layer = nn.TransformerEncoder(T_Layer, num_layers=1)
        self.N_Layer = nn.TransformerEncoder(N_Layer, num_layers=1)

    def forward(self, x):
        b, n, t, c = x.shape
        x = rearrange(x, 'b n t c -> t (b n) c')
        x = self.T_Layer(x)
        x = rearrange(x, 't (b n) c -> n (b t) c', b=b)
        x = self.N_Layer(x)
        x = rearrange(x, 'n (b t) c -> b n t c', b=b)
        return x


class Dance_DownSample(nn.Module):
    def __init__(self, rate=2, hidden_size=512):
        super(Dance_DownSample, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=rate, padding=1)

    def forward(self, x):
        b, n, t, _ = x.shape
        x = rearrange(x, 'b n t c -> (b n) c t')
        x = self.conv1d(x)
        x = rearrange(x, '(b n) c t -> b n t c', b=b)
        return x