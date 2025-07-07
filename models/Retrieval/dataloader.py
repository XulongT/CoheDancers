import torch
from torch.utils.data import Dataset
import os
import pickle as pkl
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
from utils.utils import root_preprocess, root_postprocess, normalize, denormalize, \
                        rotation_matrix_to_rotation_6d, rotation_6d_to_angle_axis
from utils.utils import keypoint_from_smpl
import torch.nn.functional as F
from einops import rearrange

def createTrainDataset(root_dir='./data', batch_size=32, stride=120, sample_len=300, start=15, end=-15):
    with open(os.path.join(root_dir, 'train.txt'), 'r') as file:
        file_names = [line.strip() for line in file]
    smpl_poses, smpl_trans, music_librosa, music_mert = [], [], [], []

    for file_name in tqdm(file_names):
        motion_file_path = os.path.join(root_dir, 'motion', file_name)
        with open(motion_file_path, 'rb') as file:
            data = pickle.load(file)
            smpl_pose, smpl_tran = data['smpl_poses'], data['smpl_trans']
            end1 = smpl_tran.shape[1] + end
            for i in range(start, end1-sample_len+1, stride):
                smpl_poses.append(smpl_pose[:, i:i+sample_len, :, :])
                smpl_trans.append(smpl_tran[:, i:i+sample_len, :])
                
        librosa_file_path = os.path.join(root_dir, 'librosa', file_name.replace('_smpl.pkl', '.pkl'))
        with open(librosa_file_path, 'rb') as file:
            music = pickle.load(file)['music']
            for i in range(start, end1-sample_len+1, stride):
                music_librosa.append(music[i:i+sample_len, :])

        mert_file_path = os.path.join(root_dir, 'mert', file_name.replace('_smpl.pkl', '.pkl'))
        with open(mert_file_path, 'rb') as file:
            music = pickle.load(file)['music']
            for i in range(start, end1-sample_len+1, stride):
                music_mert.append(music[i:i+sample_len, :])

    smpl_poses, smpl_trans, music_librosa, music_mert = np.array(smpl_poses), np.array(smpl_trans), \
                                                        np.array(music_librosa), np.array(music_mert)
    print(smpl_poses.shape, smpl_trans.shape, music_librosa.shape, music_mert.shape)
    print('Train Dataset len: ', smpl_poses.shape[0])
    train_dataloader = DataLoader(TrainDataset(smpl_poses, smpl_trans, music_librosa, music_mert, file_names), batch_size=batch_size, shuffle=True)
    return train_dataloader


def createEvalDataset(root_dir='./data', batch_size=32, stride=120, sample_len=300, start=15, end=-15):
    with open(os.path.join(root_dir, 'test.txt'), 'r') as file:
        file_names = [line.strip() for line in file]
    smpl_poses, smpl_trans, music_librosa, music_mert = [], [], [], []

    file_names_plus = []
    for file_name in tqdm(file_names):
        motion_file_path = os.path.join(root_dir, 'motion', file_name)
        with open(motion_file_path, 'rb') as file:
            data = pickle.load(file)
            smpl_pose, smpl_tran = data['smpl_poses'], data['smpl_trans']
            end1 = smpl_tran.shape[1] + end
            for i in range(start, end1-sample_len+1, stride):
                smpl_poses.append(smpl_pose[:, i:i+sample_len, :, :])
                smpl_trans.append(smpl_tran[:, i:i+sample_len, :])
                file_names_plus.append(file_name)

        librosa_file_path = os.path.join(root_dir, 'librosa', file_name.replace('_smpl.pkl', '.pkl'))
        with open(librosa_file_path, 'rb') as file:
            music = pickle.load(file)['music']
            for i in range(start, end1-sample_len+1, stride):
                music_librosa.append(music[i:i+sample_len, :])

        mert_file_path = os.path.join(root_dir, 'mert', file_name.replace('_smpl.pkl', '.pkl'))
        with open(mert_file_path, 'rb') as file:
            music = pickle.load(file)['music']
            for i in range(start, end1-sample_len+1, stride):
                music_mert.append(music[i:i+sample_len, :])

    smpl_poses, smpl_trans, music_librosa, music_mert = np.array(smpl_poses), np.array(smpl_trans), \
                                                        np.array(music_librosa), np.array(music_mert)
    print(smpl_poses.shape, smpl_trans.shape, music_librosa.shape, music_mert.shape)
    print('Test Dataset len: ', smpl_poses.shape[0])
    eval_dataloader = DataLoader(EvalDataset(smpl_poses, smpl_trans, music_librosa, music_mert, file_names_plus), batch_size=batch_size, shuffle=False)
    return eval_dataloader



class TrainDataset(Dataset):
    def __init__(self, smpl_poses, smpl_trans, music_librosa, music_mert, file_names):

        music_librosa = torch.from_numpy(music_librosa)
        music_mert = torch.from_numpy(music_mert)
        smpl_trans = torch.from_numpy(smpl_trans)
        _, smpl_root_vel = root_preprocess(smpl_trans)
        smpl_poses = rotation_matrix_to_rotation_6d(torch.from_numpy(smpl_poses))
        
        music_librosa_mean, music_librosa_std = torch.mean(music_librosa, dim=(0, 1)), torch.std(music_librosa, dim=(0, 1))
        music_mert_mean, music_mert_std = torch.mean(music_mert, dim=(0, 1)), torch.std(music_mert, dim=(0, 1))

        smpl_trans_mean, smpl_trans_std = torch.mean(smpl_trans, dim=(0, 1, 2)), torch.std(smpl_trans, dim=(0, 1, 2))
        smpl_poses_mean, smpl_poses_std = torch.mean(smpl_poses, dim=(0, 1, 2)), torch.std(smpl_poses, dim=(0, 1, 2))
        smpl_root_vel_mean, smpl_root_vel_std = torch.mean(smpl_root_vel, dim=(0, 1, 2)), torch.std(smpl_root_vel, dim=(0, 1, 2))

        torch.save({
            'music_librosa_mean': music_librosa_mean,
            'music_mert_mean': music_mert_mean,
            'smpl_trans_mean': smpl_trans_mean,
            'smpl_poses_mean': smpl_poses_mean,
            'smpl_root_vel_mean': smpl_root_vel_mean,
        }, './Pretrained/mean.pt')
        torch.save({
            'music_librosa_std': music_librosa_std,
            'music_mert_std': music_mert_std,
            'smpl_trans_std': smpl_trans_std,
            'smpl_poses_std': smpl_poses_std,
            'smpl_root_vel_std': smpl_root_vel_std,
        }, './Pretrained/std.pt')

        self.music_librosa = music_librosa
        self.music_mert = music_mert
        self.smpl_trans = smpl_trans
        self.smpl_poses = smpl_poses
        self.smpl_root_vel = smpl_root_vel


    def __len__(self):
        return len(self.smpl_poses)
    
    def __getitem__(self, idx):

        n, t, _ =  self.smpl_trans[idx].shape
        shuffled_index = torch.randperm(n)
        smpl_trans = self.smpl_trans[idx][shuffled_index, :, :]
        smpl_poses = self.smpl_poses[idx][shuffled_index, :, :]
        smpl_root_vel = self.smpl_root_vel[idx][shuffled_index, :, :]
        music_librosa = self.music_librosa[idx]
        music_mert = self.music_mert[idx]

        return {'smpl_trans': smpl_trans, 'smpl_poses': smpl_poses, 'smpl_root_vel': smpl_root_vel, \
                'music_librosa': music_librosa, 'music_mert': music_mert}


class EvalDataset(Dataset):
    def __init__(self, smpl_poses, smpl_trans, music_librosa, music_mert, file_names):
        
        music_librosa = torch.from_numpy(music_librosa)
        music_mert = torch.from_numpy(music_mert)
        smpl_root_init, smpl_root_vel = root_preprocess(torch.from_numpy(smpl_trans))

        smpl_poses = torch.from_numpy(smpl_poses)
        smpl_poses = rotation_matrix_to_rotation_6d(smpl_poses)

        self.music_librosa = music_librosa
        self.music_mert = music_mert
        self.smpl_trans = smpl_trans
        self.smpl_poses = smpl_poses
        self.smpl_root_vel = smpl_root_vel
        self.smpl_root_init = smpl_root_init

        self.file_names = file_names
        

    def __len__(self):
        return len(self.smpl_poses)

    def __getitem__(self, idx):
        return {'smpl_trans': self.smpl_trans[idx], 'smpl_poses': self.smpl_poses[idx],  \
                'smpl_root_vel': self.smpl_root_vel[idx], 'smpl_root_init': self.smpl_root_init[idx], \
                'music_librosa': self.music_librosa[idx], 'music_mert': self.music_mert[idx], \
                'file_name': self.file_names[idx]}
