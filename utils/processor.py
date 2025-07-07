import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

import yaml
from types import SimpleNamespace
from einops import rearrange
from utils.utils import denormalize, normalize, root_postprocess, root_preprocess, \
                        keypoint_from_smpl, rotation_6d_to_angle_axis
from utils.load_model import load_model
from models.Retrieval.retrieval import Retrieval
from smplx import SMPL

class Processor(nn.Module):
    def __init__(self, device=None, mode=None):
        super().__init__()

        self.device = device
        self.mode = mode
        self.epoch = 0
        self.max_epoch = 1000
        self.rt_model = Retrieval(device=device).eval()
        self.rt_model = load_model(self.rt_model, 'MD-Retrieval', 60)
       
        self.smpl = SMPL('./Pretrained/SMPL/SMPL_FEMALE.pkl', batch_size=1).to(device)
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
            std['music_mert_std'].to(device).float()

    def update_epoch(self):
        self.epoch += 1

    def get_mask(self, sz=360):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(self.device)

    def get_music_dance_feature(self, music_librosa_pred, music_mert_pred, music_librosa_gt, music_mert_gt, \
                                     smpl_poses_pred, smpl_trans_pred, smpl_poses_gt, smpl_trans_gt):
        _, smpl_root_vel_pred  = root_preprocess(smpl_trans_pred)
        _, smpl_root_vel_gt = root_preprocess(smpl_trans_gt)
        music_gt_feature = self.rt_model.music_encode(music_librosa_gt, music_mert_gt)
        dance_pred_feature = self.rt_model.dance_encode(smpl_root_vel_pred, smpl_poses_pred, smpl_trans_pred)
        dance_gt_feature = self.rt_model.dance_encode(smpl_root_vel_gt, smpl_poses_gt, smpl_trans_gt)
        if self.mode == 'test':   
            music_pred_feature = self.rt_model.music_encode(music_librosa_pred, music_mert_pred)
        else:
            music_pred_feature = None
        return music_pred_feature, music_gt_feature, dance_pred_feature, dance_gt_feature

    def postprocess(self, smpl, smpl_root_init):
        smpl_trans, smpl_poses = smpl[:, :, :, :3].clone(), smpl[:, :, :, 3:].clone()
        smpl_trans = denormalize(smpl_trans, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses = denormalize(smpl_poses, self.smpl_poses_mean, self.smpl_poses_std)
        keypoints = keypoint_from_smpl(smpl_trans, rotation_6d_to_angle_axis(smpl_poses.clone()), self.smpl)
        return smpl_trans, smpl_poses, keypoints

    def preprocess(self, data, device):
        music_librosa = data['music_librosa'].to(device).float()
        music_mert = data['music_mert'].to(device).float()
        music_librosa = normalize(music_librosa, self.music_librosa_mean, self.music_librosa_std)
        music_beat = torch.where(music_librosa[:, :, 53] < 0, torch.tensor(0, device=self.device), torch.tensor(1, device=self.device))
        music_mert = normalize(music_mert, self.music_mert_mean, self.music_mert_std)

        smpl_trans_gt = data['smpl_trans'].to(device).float()
        smpl_poses_gt = data['smpl_poses'].to(device).float()
        smpl_root_vel_gt = data['smpl_root_vel'].to(device).float()
        smpl_root_init = data['smpl_root_init'].to(device).float()

        smpl_trans_gt = normalize(smpl_trans_gt, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses_gt = normalize(smpl_poses_gt, self.smpl_poses_mean, self.smpl_poses_std)
        smpl_root_vel_gt = normalize(smpl_root_vel_gt, self.smpl_root_vel_mean, self.smpl_root_vel_std)
        return music_librosa, music_mert, music_beat, \
            smpl_trans_gt, smpl_poses_gt, smpl_root_vel_gt, smpl_root_init

    def demo_preprocess(self, data, device):
        music_librosa = data['music_librosa'].to(device).float()
        music_librosa = normalize(music_librosa, self.music_librosa_mean, self.music_librosa_std)

        smpl_trans_gt = data['smpl_trans'].to(device).float()
        smpl_poses_gt = data['smpl_poses'].to(device).float()
        smpl_root_vel_gt = data['smpl_root_vel'].to(device).float()
        smpl_root_init = data['smpl_root_init'].to(device).float()

        smpl_trans_gt = normalize(smpl_trans_gt, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses_gt = normalize(smpl_poses_gt, self.smpl_poses_mean, self.smpl_poses_std)
        smpl_root_vel_gt = normalize(smpl_root_vel_gt, self.smpl_root_vel_mean, self.smpl_root_vel_std)
        return music_librosa, smpl_trans_gt, smpl_poses_gt, smpl_root_vel_gt, smpl_root_init
        
    def demo_postprocess(self, smpl, smpl_root_init):
        smpl_trans, smpl_poses = smpl[:, :, :, :3].clone(), smpl[:, :, :, 3:].clone()
        smpl_trans = denormalize(smpl_trans, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses = denormalize(smpl_poses, self.smpl_poses_mean, self.smpl_poses_std)
        smpl_poses = rotation_6d_to_angle_axis(smpl_poses.clone())
        keypoints = keypoint_from_smpl(smpl_trans, smpl_poses, self.smpl)
        return smpl_trans, smpl_poses, keypoints