import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
import math
from models.GPT.music2dance import Music2Dance
from models.GPT.dance2music import Dance2Music
from models.GPT.discriminator import DanceDisc, MusicDisc
from models.GPT.adan import Adan
import torch.optim as optim

class CycleDancers(nn.Module):
    def __init__(self, device=None, processor=None):
        super().__init__()
        self.processor = processor
        self.mask = self.processor.get_mask(sz=360)
        self.L1Loss = nn.L1Loss()
        self.Music2Dance = Music2Dance(device, processor)
        self.Dance2Music = Dance2Music(device, processor)
        self.MusicDisc = MusicDisc(device, processor)
        self.DanceDisc = DanceDisc(device, processor)       
        self.optimizer_G_m2d = Adan(self.Music2Dance.parameters(), lr=4e-4, weight_decay=0.02)
        self.optimizer_G_d2m = Adan(self.Dance2Music.parameters(), lr=4e-4, weight_decay=0.02)
        self.optimizer_D_m = Adan(self.MusicDisc.parameters(), lr=4e-4, weight_decay=0.02)
        self.optimizer_D_d = Adan(self.DanceDisc.parameters(), lr=4e-4, weight_decay=0.02)
        
    def forward(self, music_librosa, music_mert, smpl_trans, smpl_root_vel, smpl_poses):
        b, n, t, _ = smpl_trans.shape
        music = music_librosa
        dance = torch.cat([smpl_trans, smpl_poses], dim=3)

        # Enhance Dis
        self.DanceDisc.train()
        self.MusicDisc.train()
        self.Music2Dance.eval()
        self.Dance2Music.eval()
        self.optimizer_D_d.zero_grad()
        self.optimizer_D_m.zero_grad()
        dance_pred, _, _ = self.Music2Dance(music[:, 1:, :], dance[:, :, :-1, :], dance[:, :, 1:, :], self.mask[:t-1, :t-1])
        music_pred, _, _ = self.Dance2Music(dance[:, :, 1:, :], music[:, :-1, :], music[:, 1:, :], self.mask[:t-1, :t-1])          
        dis_fake_d_loss = self.DanceDisc(dance_pred, None, torch.zeros((b), dtype=torch.long).to(dance_pred.device))
        dis_real_d_loss = self.DanceDisc(dance[:, :, 1:, :], None, torch.ones((b), dtype=torch.long).to(dance_pred.device))
        dis_fake_m_loss = self.MusicDisc(music_pred, None, torch.zeros((b), dtype=torch.long).to(music_pred.device))
        dis_real_m_loss = self.MusicDisc(music[:, 1:, :], None, torch.ones((b), dtype=torch.long).to(music_pred.device))
        dis_loss = dis_fake_d_loss + dis_real_d_loss + dis_fake_m_loss + dis_real_m_loss
        dis_loss.backward()
        self.optimizer_D_d.step()
        self.optimizer_D_m.step()

        # Enhance Gen
        self.Music2Dance.train()
        self.Dance2Music.train()
        self.DanceDisc.eval()
        self.MusicDisc.eval()
        self.optimizer_G_m2d.zero_grad()
        self.optimizer_G_d2m.zero_grad()
        dance_src = self.get_dance_src(dance[:, :, :-1, :], dance_pred[:, :, :-1, :], self.processor.epoch, self.processor.max_epoch)
        dance_pred, mid_feature_d, m2d_loss = self.Music2Dance(music[:, 1:, :], dance_src, dance[:, :, 1:, :], self.mask[:t-1, :t-1])
        music_pred, mid_feature_m, d2m_loss = self.Dance2Music(dance[:, :, 1:, :], music[:, :-1, :], music[:, 1:, :], self.mask[:t-1, :t-1])
        # al_loss = self.L1Loss(mid_feature_d, mid_feature_m) * 0.1
        al_loss = 0
        dance_pred, music_pred = torch.cat([dance[:, :, :1, :], dance_pred[:, :, :-1, :]], dim=2), torch.cat([music[:, :1, :], music_pred[:, :-1, :]], dim=1)
        _, _, m2d2m_loss = self.Dance2Music(dance_pred, music[:, :-1, :], music[:, 1:, :, ], self.mask[:t-1, :t-1])
        _, _, d2m2d_loss = self.Music2Dance(music_pred, dance[:, :, :-1, :], dance[:, :, 1:, :], self.mask[:t-1, :t-1])    
        gen_fake_d_loss = self.DanceDisc(dance_pred, None, torch.ones((b), dtype=torch.long).to(dance_pred.device))
        gen_fake_m_loss = self.MusicDisc(music_pred, None, torch.ones((b), dtype=torch.long).to(music_pred.device))
        gen_loss = gen_fake_d_loss + gen_fake_m_loss + m2d_loss + d2m_loss + m2d2m_loss + d2m2d_loss + al_loss
        gen_loss.backward()
        self.optimizer_G_m2d.step()
        self.optimizer_G_d2m.step()

        loss = {'total': gen_loss+dis_loss, 'dis_fake_d_loss': dis_fake_d_loss, 'dis_real_d_loss': dis_real_d_loss, 'dis_fake_m_loss': dis_fake_m_loss, 'dis_real_m_loss': dis_real_m_loss, \
                'm2d_loss': m2d_loss, 'd2m_loss': d2m_loss, 'm2d2m_loss': m2d2m_loss, 'd2m2d_loss': d2m2d_loss, 'al_loss': al_loss, 'gen_fake_d_loss': gen_fake_d_loss, 'gen_fake_m_loss': gen_fake_m_loss}
        return _, loss

    def demo(self, music_librosa, smpl_trans, smpl_root_vel, smpl_poses):
        b, n, t, _ = smpl_trans.shape
        music = music_librosa
        dance = torch.cat([smpl_trans, smpl_poses], dim=3)
        dance_pred = self.Music2Dance.inference(music[:, 1:, :], dance)
        return dance_pred

    def val(self, music_librosa, music_mert, smpl_trans, smpl_root_vel, smpl_poses):
        b, n, t, _ = smpl_trans.shape
        music = music_librosa
        dance = torch.cat([smpl_trans, smpl_poses], dim=3)
        dance_pred = self.Music2Dance.inference(music[:, 1:, :], dance[:, :, :-1, :])
        return dance_pred

    def test(self, music_librosa, music_mert, smpl_trans, smpl_root_vel, smpl_poses):
        b, n, t, _ = smpl_trans.shape
        music = music_librosa
        dance = torch.cat([smpl_trans, smpl_poses], dim=3)
        dance_pred = self.Music2Dance.inference(music[:, 1:, :], dance[:, :, :-1, :])
        music_pred = self.Dance2Music.inference(dance[:, :, 1:, :], music[:, :-1, :])
        # print(music_pred.shape, music_mert.shape)
        music_pred = torch.cat([music_pred, music_mert[:, 1:, :]], dim=2)
        return dance_pred, music_pred

    def get_dance_src(self, dance_gt, dance_pred, cur_epoch, tol_epoch):
        if cur_epoch <= 300:
            p = 1.0
        else:
            p = 1.0 - ((cur_epoch-300) / (tol_epoch-300)) * 0.7
        dance_pred = torch.cat([dance_gt[:, :, :1, :], dance_pred.detach().clone()], dim=2)
        mask = torch.rand_like(dance_gt) <= p
        dance_src = torch.where(mask, dance_gt, dance_pred)
        return dance_src