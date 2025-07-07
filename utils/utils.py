from smplx import SMPL
import torch
import torch as t
from torch.nn import functional as F
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
# from pytorch3d.transforms import (matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix)


def keypoint_from_smpl(smpl_trans, smpl_pred, smpl):
    b, p, t, _ = smpl_trans.shape
    keypoints_r = torch.zeros((b, p, t, 45, 3))
    J, stride = 23, 16
    for i in range(0, b, stride):
        upper_bound = min(i+stride, b)
        smpl_trans_slice = smpl_trans[i:upper_bound].reshape(-1, 3).cuda()
        global_orient = smpl_pred[i:upper_bound, :, :, :3].reshape(-1, 3).cuda()
        body_pose = smpl_pred[i:upper_bound, :, :, 3:].reshape(-1, 3*J).cuda()
        keypoints = smpl.forward(
            global_orient=global_orient.float(),
            body_pose=body_pose.float(),
            transl=smpl_trans_slice.float(),
        ).joints.reshape(upper_bound - i, p, t, -1, 3)
        keypoints_r[i:upper_bound] = keypoints.detach().cpu()
    return keypoints_r[:, :, :, :24, :]


def root_preprocess(x):
    b, n, t, _ = x.shape
    root_init = x[:, :, 0, :3].clone()
    root_vel = torch.zeros((b, n, t, 3)).to(x.device)
    root_vel[:, :, :-1, :3] = x[:, :, 1:, :3] - x[:, :, :-1, :3] # Root相对位置
    return root_init, root_vel

def root_postprocess(root_init, x):
    b, n, t, _ = x.shape
    global_pos = torch.zeros((b, n, t, 3)).to(x.device)
    global_pos[:, :, 0, :3] = root_init.clone()
    for i in range(1, t):
        global_pos[:, :, i, :3] = global_pos[:, :, i-1, :3] + x[:, :, i-1, :3].clone()
    return global_pos

def normalize(data, mean=0, std=1):
    std[std < 1e-5] = 1e-5
    return (data - mean) / std

def denormalize(data, mean=0, std=1):
    std[std < 1e-5] = 1e-5
    return data * std + mean 

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def batch_rotation_matrix_to_angle_axis(rotation_matrices):
    # 确定矩阵的轨迹，适应不同的输入形状
    traces = torch.diagonal(rotation_matrices, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (traces - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)
    angles = torch.acos(cos_theta)
    small_angles = angles < 1e-6
    
    # 初始化输出数组，确保处理任意形状的输入
    omegas = torch.empty(rotation_matrices.shape[:-2] + (3,), device=rotation_matrices.device, dtype=rotation_matrices.dtype)
    omegas[..., 0] = rotation_matrices[..., 2, 1] - rotation_matrices[..., 1, 2]
    omegas[..., 1] = rotation_matrices[..., 0, 2] - rotation_matrices[..., 2, 0]
    omegas[..., 2] = rotation_matrices[..., 1, 0] - rotation_matrices[..., 0, 1]
    
    sin_angles = torch.sin(angles)
    with torch.no_grad():  # 使用 no_grad 避免构建不必要的计算图
        omegas /= (2 * sin_angles.unsqueeze(-1))
        omegas[small_angles] = 0.0  # 对非常小的角度设置旋转向量为0
    
    angle_axis = angles.unsqueeze(-1) * omegas
    angle_axis[small_angles] = 0.0  # 对非常小的角度设置角轴向量为0
    
    return angle_axis
    traces = np.trace(rotation_matrices, axis1=3, axis2=4)
    cos_theta = (traces - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    angles = np.arccos(cos_theta)
    small_angles = angles < 1e-6
    
    omegas = np.empty(rotation_matrices.shape[:-1])
    omegas[..., 0] = rotation_matrices[..., 2, 1] - rotation_matrices[..., 1, 2]
    omegas[..., 1] = rotation_matrices[..., 0, 2] - rotation_matrices[..., 2, 0]
    omegas[..., 2] = rotation_matrices[..., 1, 0] - rotation_matrices[..., 0, 1]
    
    sin_angles = np.sin(angles)
    with np.errstate(divide='ignore', invalid='ignore'):
        omegas /= (2 * sin_angles[..., np.newaxis])
        omegas[small_angles] = 0  # 对非常小的角度设置旋转向量为0
    
    angle_axis = angles[..., np.newaxis] * omegas
    angle_axis[small_angles] = 0  # 对非常小的角度设置角轴向量为0
    
    return angle_axis


def rotation_matrix_to_rotation_6d(smpl_poses):
    J = 24
    b, n, t, j, _, _ = smpl_poses.shape
    smpl_poses = smpl_poses.reshape(-1, 3, 3)
    smpl_poses = matrix_to_rotation_6d(smpl_poses).reshape(b, n, t, -1)
    return smpl_poses


def rotation_6d_to_angle_axis(smpl_poses):
    J = 24
    b, n, t, _ = smpl_poses.shape
    smpl_poses = smpl_poses.reshape(-1, 6)
    smpl_poses = rotation_6d_to_matrix(smpl_poses).reshape(-1, 3, 3)
    smpl_poses = batch_rotation_matrix_to_angle_axis(smpl_poses).reshape(b, n, t, -1)
    return smpl_poses

def similarity_matrix(music_features, dance_features):

    music_features = rearrange(music_features, 'b t c -> b (t c)')
    dance_features = rearrange(dance_features, 'b t c -> b (t c)')
    music_norm = F.normalize(music_features, p=2, dim=1)
    dance_norm = F.normalize(dance_features, p=2, dim=1)
    cos_sim = F.cosine_similarity(music_norm.unsqueeze(1), dance_norm.unsqueeze(0), dim=2)
    return cos_sim


def similarity_matrix(music_features, dance_features):
    """
    计算给定音乐和舞蹈特征的余弦相似度矩阵。
    
    参数:
    music_features -- 张量，形状为 (b, t, c)，其中b是批次大小，t是时间步数，c是特征数
    dance_features -- 张量，形状为 (b, t, c)，与music_features相同
    
    返回:
    similarity_matrix -- 张量，形状为 (t, b, b)，表示每个时间步的相似度矩阵
    """
    # 标准化向量
    music_norm = music_features / music_features.norm(dim=2, keepdim=True)
    dance_norm = dance_features / dance_features.norm(dim=2, keepdim=True)
    
    # 调整张量形状以便批量计算
    music_norm = music_norm.permute(1, 0, 2)  # 从 (b, t, c) 转换为 (t, b, c)
    dance_norm = dance_norm.permute(1, 0, 2)  # 同上
    
    # 批量计算所有时间步的余弦相似度
    similarity_matrix = torch.bmm(music_norm, dance_norm.transpose(1, 2))
    similarity_matrix = torch.mean(similarity_matrix, dim=0)
    
    return similarity_matrix
    

def compute_rank(sims):

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)
    lst = torch.diag(sims_sort_2)

    metrics = {}
    metrics["R1"] = (100 * float(torch.sum(lst == 0)) / len(lst))
    metrics["R5"] = (100 * float(torch.sum(lst < 5)) / len(lst))
    metrics["R10"] = (100 * float(torch.sum(lst < 10)) / len(lst))
    metrics["MedR"] = float(torch.median(lst.float()) + 1)
    metrics["MeanR"] = float(torch.mean(lst.float()) + 1)

    return metrics


import torch
import numpy as np
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False