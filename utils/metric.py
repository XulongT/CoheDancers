import torch
import torch.nn as nn
import os
from smplx import SMPL
import torch
import numpy as np
import pickle as pkl
from utils.features.kinetic import extract_kinetic_features
from utils.features.manual import extract_manual_features
from utils.utils import similarity_matrix, compute_rank
from scipy import linalg
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from einops import rearrange
from tqdm import tqdm


class TestMetric():
    def __init__(self, root_dir='./data'):
        
        self.MAE_Loss = torch.nn.L1Loss()
        self.MSE_Loss = nn.MSELoss()

        self.smpl_poses_gt = []
        self.smpl_root_vel_gt = []
        self.smpl_poses_pred = []
        self.smpl_root_vel_pred = []

        self.keypoints_pred = []
        self.keypoints_gt = []

        self.dance_gt_feature = []
        self.dance_pred_feature = []

        self.music_gt_feature = []
        self.music_pred_feature = []
        self.music_beat = []

    def update(self, result):
        self.smpl_poses_pred.append(result['smpl_poses_pred'].detach().cpu())
        self.smpl_poses_gt.append(result['smpl_poses_gt'].detach().cpu())
        # self.smpl_root_vel_pred.append(result['smpl_root_vel_pred'].detach().cpu())
        # self.smpl_root_vel_gt.append(result['smpl_root_vel_gt'].detach().cpu())
        self.keypoints_pred.append(result['keypoints_pred'])
        self.keypoints_gt.append(result['keypoints_gt'])
        self.dance_pred_feature.append(result['dance_pred_feature'].detach().cpu())
        self.dance_gt_feature.append(result['dance_gt_feature'].detach().cpu())
        self.music_pred_feature.append(result['music_pred_feature'].detach().cpu())
        self.music_gt_feature.append(result['music_gt_feature'].detach().cpu())
        self.music_beat.append(result['music_beat'].detach().cpu())
        

    def result(self):
        smpl_poses_gt, smpl_poses_pred = torch.cat(self.smpl_poses_gt, dim=0), torch.cat(self.smpl_poses_pred, dim=0)
        # smpl_root_vel_gt, smpl_root_vel_pred = torch.cat(self.smpl_root_vel_gt, dim=0), torch.cat(self.smpl_root_vel_pred, dim=0)
        keypoints_gt, keypoints_pred = torch.cat(self.keypoints_gt, dim=0), torch.cat(self.keypoints_pred, dim=0)
        dance_gt_feature, dance_pred_feature = torch.cat(self.dance_gt_feature, dim=0), torch.cat(self.dance_pred_feature, dim=0)
        music_gt_feature, music_pred_feature, music_beat = torch.cat(self.music_gt_feature, dim=0), torch.cat(self.music_pred_feature, dim=0), torch.cat(self.music_beat, dim=0)

        print('*------Music2Dance Task------*')
        sim_mat = similarity_matrix(music_gt_feature, dance_pred_feature)
        print('MultiModal Distance of Pred ↓: ', multimodal_distance(music_gt_feature, dance_pred_feature))
        print('Dance Diversity of Pred ↑: ', calculate_avg_distance(dance_pred_feature))
        print('Group MD Beat Similarity of Pred ↑: ', calculate_MD_beat_similarity(music_beat, keypoints_pred))
        print('Group Dance Beat Similarity of Pred ↑: ', calculate_DD_beat_similarity(keypoints_pred))

        print('*------Ground Truth:------*')
        sim_mat = similarity_matrix(music_gt_feature, dance_gt_feature)
        print('MultiModal Distance of GT ↓: ', multimodal_distance(music_gt_feature, dance_gt_feature))
        print('Dance Diversity of GT ↑: ', calculate_avg_distance(dance_gt_feature))
        print('Group MD Beat Similarity of GT ↑: ', calculate_MD_beat_similarity(music_beat, keypoints_gt))
        print('Group Dance Beat Similarity of GT ↑: ', calculate_DD_beat_similarity(keypoints_gt))

        print('*------Groun Truth and Pred:------*')
        print('Dance FID of Pred and GT ↓: ', calc_fid(dance_pred_feature, dance_gt_feature))
        print('Dance Feature Distance between GT and Pred ↓: ', feature_distance(dance_gt_feature, dance_pred_feature))


def calculate_DD_beat_similarity(keypoints):
    keypoints = np.array(keypoints)
    b, p, t, _, _ = keypoints.shape
    dd_scores = []
    for k in range(b):
        for i in range(p):
            for j in range(i+1, p, 1):
                db1 = get_db(keypoints[k][i])
                db2 = get_db(keypoints[k][j])
                dd_scores.append(BA(db1, db2))
    return np.mean(dd_scores)


def calculate_MD_beat_similarity(music_beat, keypoints):
    music_beat, keypoints = np.array(music_beat), np.array(keypoints)
    b, p, t, _, _ = keypoints.shape
    ba_score = []
    for i in range(b):
        mb = get_mb(music_beat[i])
        max_ba = 0
        for j in range(p):
            db = get_db(keypoints[i][j])
            max_ba = max(max_ba, BA(mb, db))
        ba_score.append(max_ba)
    return np.mean(ba_score)


def BA(music_beats, motion_beats):
    if len(music_beats) == 0 or len(motion_beats) == 0:
        return 0
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats - bb)**2) / 2 / 9)
    mb = ba / len(music_beats)

    ba = 0
    for bb in motion_beats:
        ba +=  np.exp(-np.min((music_beats - bb)**2) / 2 / 9)
    db = ba / len(motion_beats)

    return (mb + db) / 2

def get_mb(music_beats):
    t = music_beats.shape[0]
    beats = music_beats.astype(bool)
    beat_axis = np.arange(t)
    beat_axis = beat_axis[beats]
    return beat_axis

def get_db(keypoints):
    t, _, _ = keypoints.shape
    keypoints = keypoints.reshape(t, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)[0]
    return motion_beats


def multimodality(music_feat, dance_feat):
    # music_feat 和 dance_feat 都应是 b x t x c 维
    music_feat, dance_feat = np.array(music_feat), np.array(dance_feat)
    b, t, c = music_feat.shape

    music_feat, dance_feat = music_feat[:, np.newaxis, :, :], dance_feat[np.newaxis, :, :, :]
    distances = np.sqrt(np.sum(np.square(music_feat - dance_feat), axis=3))

    mask = np.ones((b, b, t), dtype=bool)
    range_indices = np.arange(b)
    mask[range_indices, range_indices, :] = False
    valid_distances = np.where(mask, distances, np.nan)

    overall_average = np.nanmean(valid_distances)
    return overall_average


def multimodal_distance(music_feat, dance_feat):
    music_feat, dance_feat = np.array(music_feat), np.array(dance_feat)
    distances = np.sqrt(np.sum(np.square(music_feat - dance_feat), axis=2))
    distances = np.mean(distances)
    return distances

def feature_distance(pred, gt):
    pred, gt = np.array(pred), np.array(gt)
    distances = np.sqrt(np.sum(np.square(pred - gt), axis=2))
    distances = np.mean(distances)
    return distances

def calculate_avg_distance(feat):
    feat = np.array(feat)
    b, t, c = feat.shape
    dist = 0
    for k in range(t):
        for i in range(b):
            for j in range(i + 1, b):
                dist += np.linalg.norm(feat[i, k] - feat[j, k])
    dist /= (b * b - b) / 2
    dist /= t
    return dist

# def normal(kps_gen, kps_gt):
#     mean, std = np.mean(kps_gt, axis=0), np.std(kps_gt, axis=0)
#     kps_gt = (kps_gt - mean) / std
#     kps_gen = (kps_gen - mean) / std
#     return kps_gen, kps_gt

def calc_fid(kps_gen, kps_gt):
    kps_gen, kps_gt = np.array(kps_gen), np.array(kps_gt)
    b, t, _ = kps_gen.shape
    fid = 0
    for i in range(t):
        fid += calculate_fid_per_timestep(kps_gen[:, i, :], kps_gt[:, i, :])
    return fid / t

def calculate_fid_per_timestep(kps_gen, kps_gt):

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1, mu2, sigma1, sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


class RetrievalMetric():
    def __init__(self, root_dir='./data'):
        
        self.MAE_Loss = torch.nn.L1Loss()
        self.MSE_Loss = nn.MSELoss()
        self.music_features = []
        self.dance_features = []

    def update(self, result):
        self.music_features.append(result['music_features'].detach().cpu())
        self.dance_features.append(result['dance_features'].detach().cpu())    

    def result(self):
        music_features, dance_features = torch.cat(self.music_features, dim=0), torch.cat(self.dance_features, dim=0)
        sim_mat = similarity_matrix(music_features, dance_features)
        print("Music-Dance Result: ", compute_rank(sim_mat))
        print("Dance-Music Result: ", compute_rank(sim_mat.t()))


from sklearn.metrics import accuracy_score, precision_score, recall_score
class ClassificationMetric():
    def __init__(self, root_dir='./data'):
        self.pred_labels = []
        self.gt_labels = []

    def update(self, result):
        self.pred_labels.append(result['dance_pred'].detach().cpu())
        self.gt_labels.append(result['dance_gt'].detach().cpu())    

    def result(self):
        pred_labels = torch.cat(self.pred_labels, dim=0).squeeze().numpy()
        gt_labels = torch.cat(self.gt_labels, dim=0).squeeze().numpy()
        print(f'Accuracy: {accuracy_score(pred_labels, gt_labels)}')
        precision = precision_score(pred_labels, gt_labels, average='macro')
        print(f'Precision: {precision}')
        recall = recall_score(pred_labels, gt_labels, average='macro')
        print(f'Recall: {recall}')


if __name__ =='__main__':
    # metric = Metric()

    with open('./data1/gt/kinetic_features.pkl', 'rb') as file:
        gt_features_k = pkl.load(file)['kinetic_features']
    
    gt_features_k = np.repeat(gt_features_k, 10, 0)

    pred_features_k = np.random.permutation(gt_features_k)
    
    gt_features_k, pred_features_k = normalize(gt_features_k, pred_features_k)
    fid_k = calc_fid(pred_features_k, gt_features_k)
    print(fid_k)