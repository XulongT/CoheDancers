import os
import numpy as np
import pickle as pkl
import matplotlib
import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from mpl_toolkits.mplot3d import Axes3D
from einops import rearrange
import subprocess

def adjust_pose(joint):
    np_dance_trans = np.zeros([3, 25]).copy()
    joint = np.transpose(joint)

    # head
    np_dance_trans[:, 0] = joint[:, 15]
    
    #neck
    np_dance_trans[:, 1] = joint[:, 12]
    
    # left up
    np_dance_trans[:, 2] = joint[:, 16]
    np_dance_trans[:, 3] = joint[:, 18]
    np_dance_trans[:, 4] = joint[:, 20]

    # right up
    np_dance_trans[:, 5] = joint[:, 17]
    np_dance_trans[:, 6] = joint[:, 19]
    np_dance_trans[:, 7] = joint[:, 21]

    
    np_dance_trans[:, 8] = joint[:, 0]
    
    np_dance_trans[:, 9] = joint[:, 1]
    np_dance_trans[:, 10] = joint[:, 4]
    np_dance_trans[:, 11] = joint[:, 7]

    np_dance_trans[:, 12] = joint[:, 2]
    np_dance_trans[:, 13] = joint[:, 5]
    np_dance_trans[:, 14] = joint[:, 8]

    np_dance_trans[:, 15] = joint[:, 15]
    np_dance_trans[:, 16] = joint[:, 15]
    np_dance_trans[:, 17] = joint[:, 15]
    np_dance_trans[:, 18] = joint[:, 15]

    np_dance_trans[:, 19] = joint[:, 11]
    np_dance_trans[:, 20] = joint[:, 11]
    np_dance_trans[:, 21] = joint[:, 8]

    np_dance_trans[:, 22] = joint[:, 10]
    np_dance_trans[:, 23] = joint[:, 10]
    np_dance_trans[:, 24] = joint[:, 7]

    np_dance_trans = np.transpose(np_dance_trans)

    return np_dance_trans

pose_edge_list = [        
    [ 0,  1], [ 1,  8],                                         # body
    [ 1,  2], [ 2,  3], [ 3,  4],                               # right arm
    [ 1,  5], [ 5,  6], [ 6,  7],                               # left arm
    [ 8,  9], [ 9, 10], [10, 11], [11, 24], [11, 22], [22, 23], # right leg
    [ 8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20]  # left leg
]
pose_color_list = [
    [153,  0, 51], [153,  0,  0],
    [153, 51,  0], [153,102,  0], [153,153,  0],
    [102,153,  0], [ 51,153,  0], [  0,153,  0],
    [  0,153, 51], [  0,153,102], [  0,153,153], [  0,153,153], [  0,153,153], [  0,153,153],
    [  0,102,153], [  0, 51,153], [  0,  0,153], [  0,  0,153], [  0,  0,153], [  0,  0,153]
]
def plot_line(joint, ax):
    for i, e in enumerate(pose_edge_list):
        ax.plot([joint[e[0]][0], joint[e[1]][0]], [joint[e[0]][1], joint[e[1]][1]], [joint[e[0]][2], joint[e[1]][2]], \
                    color=(pose_color_list[i][0]/255, pose_color_list[i][1]/255, pose_color_list[i][2]/255))

def swap(joint):
    tmp = np.zeros_like(joint)
    tmp[:, :, :, 0] = joint[:, :, :, 0]
    tmp[:, :, :, 1] = joint[:, :, :, 2]
    tmp[:, :, :, 2] = joint[:, :, :, 1]
    return tmp

def calculate_coordinate_range(joints):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for joint in joints:
        x = joint[:, 0]
        y = joint[:, 1]
        min_x = min(min_x, np.min(x))
        max_x = max(max_x, np.max(x))
        min_y = min(min_y, np.min(y))
        max_y = max(max_y, np.max(y))

    print('min_x, max_x, min_y, max_y', min_x, max_x, min_y, max_y)
    return min_x, max_x, min_y, max_y

def save_img(k, all_joints3d, image_path):

    # 设置视角
    elev_main, azim_main = 0, 90
    elev_side, azim_side = 0, 0
    elev_top, azim_top = 90, 180

    # 获取关节数据范围
    min_lin, max_lin = np.min(all_joints3d[:, :, :, :].reshape(-1, 3), axis=0), np.max(all_joints3d[:, :, :, :].reshape(-1, 3), axis=0)

    # 创建一个新的图形，包含三个子图
    fig = plt.figure(figsize=(12, 6))

    # 主视图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Main View')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    joints3d = all_joints3d[:, k]
    for joint in joints3d:
        joint = adjust_pose(joint[:, :3])
        ax1.scatter(joint[:, 0], joint[:, 1], joint[:, 2], color='black', s=10)
        plot_line(joint, ax1)
    ax1.set_xlim(min_lin[0], max_lin[0])
    ax1.set_ylim(min_lin[1], max_lin[1])
    ax1.set_zlim(min_lin[2], max_lin[2])
    ax1.view_init(elev=elev_main, azim=azim_main)

    # 侧视图
    # ax2 = fig.add_subplot(132, projection='3d')
    # ax2.set_title('Side View')
    # ax2.set_xlabel('X axis')
    # ax2.set_ylabel('Y axis')
    # ax2.set_zlabel('Z axis')
    # joints3d = all_joints3d[:, k]
    # for joint in joints3d:
    #     joint = adjust_pose(joint[:, :3])
    #     ax2.scatter(joint[:, 0], joint[:, 1], joint[:, 2], color='black', s=10)
    #     plot_line(joint, ax2)
    # ax2.set_xlim(min_lin[0], max_lin[0])
    # ax2.set_ylim(min_lin[1], max_lin[1])
    # ax2.set_zlim(min_lin[2], max_lin[2])
    # ax2.view_init(elev=elev_side, azim=azim_side)

    # 俯视图
    ax3 = fig.add_subplot(122, projection='3d')
    ax3.set_title('Top View')
    ax3.set_xlabel('X axis')
    ax3.set_ylabel('Y axis')
    ax3.set_zlabel('Z axis')
    joints3d = all_joints3d[:, k]
    for joint in joints3d:
        joint = adjust_pose(joint[:, :3])
        ax3.scatter(joint[:, 0], joint[:, 1], joint[:, 2], color='black', s=10)
        plot_line(joint, ax3)
    ax3.set_xlim(min_lin[0], max_lin[0])
    ax3.set_ylim(min_lin[1], max_lin[1])
    ax3.set_zlim(min_lin[2], max_lin[2])
    ax3.view_init(elev=elev_top, azim=azim_top)

    # 保存大图
    output_image_path = os.path.join(image_path, '{}.png'.format(k))
    plt.savefig(output_image_path)
    plt.close()


def vis(keypoints_path, music_path, video_path, image_path):
    
    for keypoints_file in tqdm(sorted(os.listdir(keypoints_path))):

        
        print(os.path.join(keypoints_path, keypoints_file))
        all_joints3d = np.array(json.load(open(os.path.join(keypoints_path, keypoints_file), "rb"))['keypoints'])
        all_joints3d = swap(all_joints3d)

        timestep = list(range(all_joints3d.shape[1]))
        with ProcessPoolExecutor(max_workers=8) as executor:
            executor.map(save_img, timestep, [all_joints3d]*len(timestep), [image_path]*len(timestep))
        
        video_file = video_path + '/{}'.format(keypoints_file.replace('.json', '.mp4'))
        cmd = f"ffmpeg -r 30 -i {image_path}/%d.png -vb 20M -vcodec mpeg4 -y {video_file}"
        os.system(cmd)

        music_file = music_path + '/{}'.format(keypoints_file.replace('.json', '.mp3'))
        video_file_new = video_file.replace('.mp4', '_audio.mp4')
        cmd_audio = f"ffmpeg -i {video_file} -i {music_file} -map 0:v -map 1:a -c:v copy -shortest -y {video_file_new} -loglevel quiet"
        os.system(cmd_audio)

        # if os.path.exists(image_path):
        #     shutil.rmtree(image_path)
        # if os.path.exists(video_file):
        #     os.remove(video_file)
        # if os.path.exists(music_file_new):
        #     os.remove(music_file_new)

def visual(exp_name='hhh', epoch=120):
    root_dir = './output'

    keypoints_path = os.path.join(root_dir, '{}/{}/Keypoints'.format(exp_name, epoch))
    image_path = os.path.join(root_dir, '{}/{}/Image'.format(exp_name, epoch))
    video_path = os.path.join(root_dir, '{}/{}/Video'.format(exp_name, epoch))
    music_path = os.path.join(root_dir, '{}/{}/Music'.format(exp_name, epoch))
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(music_path, exist_ok=True)
    vis(keypoints_path, music_path, video_path, image_path)

def visual1(root_dir='./demo1'):

    music_src_path = os.path.join(root_dir, 'music')
    keypoints_path = os.path.join(root_dir, 'keypoint')
    image_path = os.path.join(root_dir, 'image')
    video_path = os.path.join(root_dir, 'video')
    music_path = os.path.join(root_dir, 'music')
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(music_path, exist_ok=True)
    vis(keypoints_path, music_path, video_path, image_path)

if __name__ =='__main__':
    # visual(exp_name='group_gpt', epoch=460)
    visual1(root_dir='./demo')



