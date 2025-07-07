import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
import math

class Motion_loss(nn.Module):
    def __init__(self, processor=None):
        super().__init__()
        self.processor = processor
        self.L1Loss = nn.L1Loss()
    
    def forward(self, dance_pred, dance_gt):
        smpl_trans_pred, smpl_poses_pred = dance_pred[:, :, :, :3], dance_pred[:, :, :, 3:]
        smpl_trans_gt, smpl_poses_gt = dance_gt[:, :, :, :3], dance_gt[:, :, :, 3:]
        smpl_trans_vel_pred = smpl_trans_pred[:, :, 1:, :] - smpl_trans_pred[:, :, :-1, :]
        smpl_trans_vel_gt = smpl_trans_gt[:, :, 1:, :] - smpl_trans_gt[:, :, :-1, :]     
        smpl_trans_loss = self.L1Loss(smpl_trans_pred, smpl_trans_gt)
        smpl_poses_loss = self.L1Loss(smpl_poses_pred, smpl_poses_gt)
        smpl_trans_vel_loss = self.L1Loss(smpl_trans_vel_pred, smpl_trans_vel_gt)
        loss = (smpl_trans_loss + smpl_poses_loss + smpl_trans_vel_loss)
        return loss

class Music2Dance(nn.Module):
    def __init__(self, device=None, processor=None):
        super().__init__()
        self.music_encoder = Music_Encoder()
        self.dance_decoder = Dance_Decoder()
        self.motion_loss =  Motion_loss(processor)
        self.processor = processor
        self.max_len = 120

    def forward(self, music_src, dance_src, dance_tgt, mask=None):
        b, n, t, _ = dance_src.shape
        mid_feature = self.music_encoder(music_src, None)
        music_src = rearrange(mid_feature, 'b t c -> b 1 t c').repeat(1, n, 1, 1)
        dance_pred = self.dance_decoder(music_src, dance_src, mask)
        loss = self.motion_loss(dance_pred, dance_tgt)
        return dance_pred, mid_feature, loss

    def inference(self, music, dance):
        _, n, _, _ = dance.shape
        _, t, _ = music.shape
        dance_preds, dance_src, music_src = dance[:, :, :1, :].clone(), dance[:, :, :1, :].clone(), music[:, :self.max_len, :].clone()
        for i in range(1, t):
            music_src = rearrange(self.music_encoder(music_src, None), 'b t c -> b 1 t c').repeat(1, n, 1, 1)
            dance_pred = self.dance_decoder(music_src, dance_src, None)[:, :, -1:, :]
            dance_preds = torch.cat([dance_preds, dance_pred], dim=2)
            if i < self.max_len:              
                dance_src = torch.cat([dance_src, dance_pred], dim=2)
                music_src = music[:, :self.max_len, :]
            else:
                dance_src = torch.cat([dance_src[:, :, -self.max_len+1:, :], dance_pred], dim=2)
                music_src = music[:, i-self.max_len:i, :]
        return dance_preds


class Dance_Decoder(nn.Module):
    def __init__(self, output_dim=147, hidden_dim=512, num_layers=4, nhead=8):
        super().__init__()
        self.output_l1 = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.Linear(256, hidden_dim)
        )
        self.output_l3 = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Linear(256, output_dim)
        )
        self.dance_pos_embed = DancePositionEmbedding(d_model=512, max_len=360)
        self.dance_decoder = nn.ModuleList([TNTransformer(hidden_size=512, num_heads=8) for _ in range(4)])

    def forward(self, music_src, smpl_src, mask):
        b, n, t, _ = smpl_src.shape
        smpl_src = self.output_l1(smpl_src)
        smpl_src = self.dance_pos_embed(smpl_src)
        for decoder in self.dance_decoder:
            music_src, smpl_src, mask = decoder(music_src, smpl_src, mask)
        smpl_pred = self.output_l3(smpl_src)
        return smpl_pred


class DancePositionEmbedding(nn.Module):
    def __init__(self, d_model=512, max_len=360):
        super(DancePositionEmbedding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(max_len, d_model)).cuda()
        nn.init.uniform_(self.position_embeddings, -0.1, 0.1)  # Initialize with a uniform distribution

    def forward(self, x):
        b, n, t, _ = x.shape
        pos_embed = rearrange(self.position_embeddings[:t, :], 't c -> 1 1 t c')
        pos_embed = pos_embed.repeat(b, n, 1, 1)
        return x + pos_embed


class TNTransformer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super(TNTransformer, self).__init__()
        T_Layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads)
        self.T_Layer = nn.TransformerDecoder(T_Layer, num_layers=1)
        N_Layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.N_Layer = nn.TransformerEncoder(N_Layer, num_layers=1)

    def forward(self, music_src, smpl_src, mask):
        b, n, t, c = smpl_src.shape
        smpl_src = rearrange(smpl_src, 'b n t c -> t (b n) c')
        music_src = rearrange(music_src, 'b n t c -> t (b n) c')
        smpl_src = self.T_Layer(smpl_src, music_src, tgt_mask=mask, memory_mask=None)
        smpl_src = rearrange(smpl_src, 't (b n) c -> n (b t) c', b=b)
        smpl_src = self.N_Layer(smpl_src)
        smpl_src = rearrange(smpl_src, 'n (b t) c -> b n t c', b=b)
        music_src = rearrange(music_src, 't (b n) c -> b n t c', b=b)
        return music_src, smpl_src, mask


class Music_Encoder(nn.Module):
    def __init__(self, input_size1=438, input_size2=0, hidden_size=512, num_heads=8, layer=6):
        super(Music_Encoder, self).__init__()
        self.music_linear = nn.Sequential(
            nn.Linear(input_size1+input_size2, 768),
            nn.Linear(768, hidden_size)
        )
        self.music_encoder = nn.ModuleList([TTransformer(hidden_size=512, num_heads=8) for _ in range(4)])
        self.music_pos_emb = MusicPositionEmbedding(d_model=512, max_len=360).cuda()

    def forward(self, music, mask):
        x = self.music_linear(music)
        x = self.music_pos_emb(x)
        for encoder in self.music_encoder:
            x, mask = encoder(x, mask)
        return x


class MusicPositionEmbedding(nn.Module):
    def __init__(self, d_model=512, max_len=360):
        super(MusicPositionEmbedding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.uniform_(self.position_embeddings, -0.1, 0.1)  # Initialize with a uniform distribution

    def forward(self, x):
        b, t, _ = x.shape
        pos_embed = rearrange(self.position_embeddings[:t, :], 't c -> 1 t c').repeat(b, 1, 1)
        return x + pos_embed


class TTransformer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super(TTransformer, self).__init__()
        Layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.layer = nn.TransformerEncoder(Layer, num_layers=1)

    def forward(self, x, mask):
        x = rearrange(x, 'b t c -> t b c')
        x = self.layer(x, mask=mask)
        x = rearrange(x, 't b c -> b t c')
        return x, mask