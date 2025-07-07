import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
import math

class DanceDisc(nn.Module):
    def __init__(self, device=None, processor=None):
        super().__init__()
        self.dance_encoder = Dance_Encoder()
        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 2)
        )
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, dance, mask, label):
        dance = torch.mean(torch.mean(self.dance_encoder(dance, mask), dim=1), dim=1)
        pred = self.linear(dance)
        loss = self.cls_loss(pred, label)
        return loss


class MusicDisc(nn.Module):
    def __init__(self, device=None, processor=None):
        super().__init__()
        self.music_encoder = Music_Encoder()
        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 2)
        )
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, music, mask, label):
        music = self.music_encoder(music, mask)
        pred = torch.mean(self.linear(music), dim=1)
        loss = self.cls_loss(pred, label)
        return loss


class Music_Encoder(nn.Module):
    def __init__(self, input_size1=438, input_size2=0, hidden_size=512, num_heads=8, layer=6):
        super(Music_Encoder, self).__init__()
        self.music_linear = nn.Sequential(
            nn.Linear(input_size1+input_size2, 768),
            nn.Linear(768, hidden_size)
        )
        self.N = 7
        self.music_encoder = nn.ModuleList([TTransformer(hidden_size=512, num_heads=8) for _ in range(self.N)])
        self.music_dowmsample = nn.ModuleList([Music_DownSample(rate=2) for _ in range(self.N)])
        self.music_pos_emb = MusicPositionEmbedding(d_model=512, max_len=360).cuda()

    def forward(self, music, mask):
        x = self.music_linear(music)
        x = self.music_pos_emb(x)
        for i in range(self.N):
            x, mask = self.music_encoder[i](x, mask)
            x = self.music_dowmsample[i](x)
        return x

class Music_DownSample(nn.Module):
    def __init__(self, rate=2, hidden_size=512):
        super(Music_DownSample, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=rate, padding=1)

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv1d(x)
        x = rearrange(x, 'b c t -> b t c')
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



class Dance_Encoder(nn.Module):
    def __init__(self, output_dim=147, hidden_dim=512, num_layers=4, nhead=8):
        super().__init__()
        self.output_l1 = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.Linear(256, hidden_dim)
        )
        self.N = 7
        self.dance_pos_embed = DancePositionEmbedding(d_model=512, max_len=360)
        self.dance_encoder = nn.ModuleList([TNTransformer(hidden_size=512, num_heads=8) for _ in range(self.N)])
        self.dance_downsample = nn.ModuleList([Dance_DownSample(rate=2) for _ in range(self.N)])

    def forward(self, dance_src, mask):
        b, n, t, _ = dance_src.shape
        dance_src = self.dance_pos_embed(self.output_l1(dance_src))
        for i in range(self.N):
            dance_src, mask = self.dance_encoder[i](dance_src, mask)
            dance_src = self.dance_downsample[i](dance_src)
        return dance_src

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
