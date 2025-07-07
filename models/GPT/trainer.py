import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import yaml
from types import SimpleNamespace
from einops import rearrange
from utils.utils import denormalize, normalize, root_postprocess, keypoint_from_smpl, rotation_6d_to_angle_axis
from utils.processor import Processor
from utils.load_model import load_model
from models.GPT.model import CycleDancers
import pickle
import json
import pickle
import json

class Trainer(nn.Module):
    def __init__(self, device=None, processor=None):
        super().__init__()
        self.L1Loss = nn.L1Loss()
        self.device = device
        self.processor = processor
        self.model = CycleDancers(self.device, self.processor)

    def forward(self, data, device):
        music_librosa, music_mert, music_beat, \
            smpl_trans_gt, smpl_poses_gt, smpl_root_vel_gt, _ = self.processor.preprocess(data, device)
        _, loss = self.model(music_librosa, music_mert, smpl_trans_gt, smpl_root_vel_gt, smpl_poses_gt)
        return _, loss

    def demo(self, data, device):
        music_librosa_gt, smpl_trans_gt, smpl_poses_gt, smpl_root_vel_gt, smpl_root_init = self.processor.demo_preprocess(data, device)
        dance_gts = torch.cat([smpl_trans_gt, smpl_poses_gt], dim=3).clone()
        dance_preds = self.model.demo(music_librosa_gt, smpl_trans_gt, smpl_root_vel_gt, smpl_poses_gt)
        smpl_trans_pred, smpl_poses_pred, keypoints_pred = self.processor.demo_postprocess(dance_preds, smpl_root_init)
        with open(os.path.join(data['root_dir'][0], 'keypoint', data['file_name'][0].replace('.mp3', '.json')), 'w') as f:
            json.dump({'keypoints': keypoints_pred[0].cpu().detach().numpy().tolist()}, f, indent=4)     
        with open(os.path.join(data['root_dir'][0], 'smpl', data['file_name'][0].replace('.mp3', '.pkl')), 'wb') as f:
            pickle.dump({'smpl_trans': smpl_trans_pred[0].cpu().detach().numpy(), 'smpl_poses': smpl_poses_pred[0].cpu().detach().numpy()}, f)  

    def val(self, data, device):
        music_librosa_gt, music_mert_gt, music_beat, smpl_trans_gt, \
            smpl_poses_gt, smpl_root_vel_gt, smpl_root_init = self.processor.preprocess(data, device)
        dance_gts = torch.cat([smpl_trans_gt, smpl_poses_gt], dim=3).clone()
        dance_preds = self.model.val(music_librosa_gt, music_mert_gt, smpl_trans_gt, smpl_root_vel_gt, smpl_poses_gt)
        music_librosa_pred, music_mert_pred =  music_librosa_gt, music_mert_gt

        smpl_trans_pred, smpl_poses_pred, keypoints_pred = self.processor.postprocess(dance_preds, smpl_root_init)
        smpl_trans_gt, smpl_poses_gt, keypoints_gt = self.processor.postprocess(dance_gts, smpl_root_init)
        
        music_pred_feature, music_gt_feature, dance_pred_feature, dance_gt_feature = self.processor.get_music_dance_feature(
            music_librosa_pred, music_mert_pred, music_librosa_gt, music_mert_gt, \
            smpl_poses_pred, smpl_trans_pred, smpl_poses_gt, smpl_trans_gt, 
        )
        result = {
            'smpl_poses_pred': smpl_poses_pred, 'smpl_trans_pred': smpl_trans_pred, 'keypoints_pred': keypoints_pred, 'dance_pred_feature': dance_pred_feature, \
            'smpl_poses_gt': smpl_poses_gt, 'smpl_trans_gt': smpl_trans_gt, 'keypoints_gt': keypoints_gt, 'dance_gt_feature': dance_gt_feature, \
            'music_gt_feature': music_gt_feature, 'music_pred_feature': music_pred_feature,  'music_beat': music_beat, 'file_name': data['file_name'], \
        }
        loss = {
            'total': 0
        }

        return result, loss


    def test(self, data, device):
        
        music_librosa_gt, music_mert_gt, music_beat, smpl_trans_gt, \
            smpl_poses_gt, smpl_root_vel_gt, smpl_root_init = self.processor.preprocess(data, device)

        dance_gts = torch.cat([smpl_trans_gt, smpl_poses_gt], dim=3).clone()
        dance_preds, music_pred = self.model.test(music_librosa_gt, music_mert_gt, smpl_trans_gt, smpl_root_vel_gt, smpl_poses_gt)
        music_librosa_pred, music_mert_pred = music_pred[:, :, :438], music_pred[:, :, 438:]

        smpl_trans_pred, smpl_poses_pred, keypoints_pred = self.processor.postprocess(dance_preds, smpl_root_init)
        smpl_trans_gt, smpl_poses_gt, keypoints_gt = self.processor.postprocess(dance_gts, smpl_root_init)
        
        music_pred_feature, music_gt_feature, dance_pred_feature, dance_gt_feature = self.processor.get_music_dance_feature(
            music_librosa_pred, music_mert_pred, music_librosa_gt, music_mert_gt, \
            smpl_poses_pred, smpl_trans_pred, smpl_poses_gt, smpl_trans_gt, 
        )

        result = {
            'smpl_poses_pred': smpl_poses_pred, 'smpl_trans_pred': smpl_trans_pred, 'keypoints_pred': keypoints_pred, 'dance_pred_feature': dance_pred_feature, \
            'smpl_poses_gt': smpl_poses_gt, 'smpl_trans_gt': smpl_trans_gt, 'keypoints_gt': keypoints_gt, 'dance_gt_feature': dance_gt_feature, \
            'music_gt_feature': music_gt_feature, 'music_pred_feature': music_pred_feature,  'music_beat': music_beat, 'file_name': data['file_name'], \
        }
        loss = {
            'total': 0
        }

        return result, loss