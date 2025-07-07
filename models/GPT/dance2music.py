import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
import math

class Dance2Music(nn.Module):
    def __init__(self, device=None, processor=None):
        super().__init__()
        self.dance_encoder = Dance_Encoder()
        self.music_decoder = Music_Decoder()
        self.L1Loss = nn.L1Loss()
        self.processor = processor
        self.max_len = 120

    def forward(self, dance_src, music_src, music_tgt, mask=None):
        dance_src = torch.mean(self.dance_encoder(dance_src, None), dim=1)
        mid_feature = dance_src.clone()
        music_pred = self.music_decoder(dance_src, music_src, mask)
        # librosa_loss, mert_loss = self.L1Loss(music_pred[:, :, :438], music_tgt[:, :, :438]), self.L1Loss(music_pred[:, :, 438:], music_tgt[:, :, 438:])
        loss = self.L1Loss(music_pred, music_tgt)        
        return music_pred, mid_feature, loss

    def inference(self, dance, music):
        b, t, _ = music.shape
        music_preds, music_src, dance_src = music[:, :1, :].clone(), music[:, :1, :].clone(), dance[:, :, :self.max_len, :].clone()
        for i in range(1, t):
            dance_src = torch.mean(self.dance_encoder(dance_src, None), dim=1)
            music_pred = self.music_decoder(dance_src, music_src, None)[:, -1:, :]
            music_preds = torch.cat([music_preds, music_pred], dim=1)
            if i < self.max_len:              
                music_src = torch.cat([music_src, music_pred], dim=1)
                dance_src = dance[:, :, :self.max_len, :]
            else:
                music_src = torch.cat([music_src[:, -self.max_len+1:, :], music_pred], dim=1)
                dance_src = dance[:, :, i-self.max_len:i, :]
        return music_preds

class Dance_Encoder(nn.Module):
    def __init__(self, output_dim=147, hidden_dim=512, num_layers=4, nhead=8):
        super().__init__()
        self.output_l1 = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.Linear(256, hidden_dim)
        )
        self.dance_pos_embed = DancePositionEmbedding(d_model=512, max_len=360)
        self.dance_encoder = nn.ModuleList([TNTransformer(hidden_size=512, num_heads=8) for _ in range(4)])

    def forward(self, dance_src, mask):
        b, n, t, _ = dance_src.shape
        dance_src = self.dance_pos_embed(self.output_l1(dance_src))
        for encoder in self.dance_encoder:
            dance_src, mask = encoder(dance_src, mask)
        return dance_src


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
        T_Layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.T_Layer = nn.TransformerEncoder(T_Layer, num_layers=1)
        N_Layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.N_Layer = nn.TransformerEncoder(N_Layer, num_layers=1)

    def forward(self, smpl_src, mask):
        b, n, t, c = smpl_src.shape
        smpl_src = rearrange(smpl_src, 'b n t c -> t (b n) c')
        smpl_src = self.T_Layer(smpl_src, mask=mask)
        smpl_src = rearrange(smpl_src, 't (b n) c -> n (b t) c', b=b)
        smpl_src = self.N_Layer(smpl_src)
        smpl_src = rearrange(smpl_src, 'n (b t) c -> b n t c', b=b)
        return smpl_src, mask


class Music_Decoder(nn.Module):
    def __init__(self, input_size1=438, input_size2=1024, hidden_size=512, num_heads=8, layer=6):
        super(Music_Decoder, self).__init__()
        self.input_l1 = nn.Sequential(
            nn.Linear(input_size1, 512),
            nn.Linear(512, hidden_size)
        )
        self.output_l1 = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.Linear(512, input_size1)
        )
        self.music_decoder = nn.ModuleList([TTransformer(hidden_size=512, num_heads=8) for _ in range(4)])
        self.music_pos_emb = MusicPositionEmbedding(d_model=512, max_len=360).cuda()

    def forward(self, dance_src, music_src, mask):
        music_src = self.music_pos_emb(self.input_l1(music_src))
        for decoder in self.music_decoder:
            dance_src, music_src, mask = decoder(dance_src, music_src, mask)
        music_pred = self.output_l1(music_src)
        return music_pred


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
        Layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads)
        self.layer = nn.TransformerDecoder(Layer, num_layers=1)

    def forward(self, dance, music, mask):
        b, t, c = dance.shape
        music = rearrange(music, 'b t c -> t b c')
        dance = rearrange(dance, 'b t c -> t b c')
        music = self.layer(music, dance, tgt_mask=mask, memory_mask=None)
        music = rearrange(music, 't b c -> b t c')
        dance = rearrange(dance, 't b c -> b t c')
        return dance, music, mask