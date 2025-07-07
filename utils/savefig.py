import torch
import os
import torch
import pickle
import json
from utils.utils import root_postprocess, keypoint_from_smpl, rotation_6d_to_angle_axis

output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)
def savefig(result, epoch, exp_name):

    smpl_trans, smpl_poses, keypoint = result['smpl_trans_pred'], result['smpl_poses_pred'], result['keypoints_pred']
    smpl_poses = rotation_6d_to_angle_axis(smpl_poses)
    file_name = result['file_name']
    output_keypoints_file = os.path.join(output_dir, exp_name, str(epoch), 'Keypoints')
    os.makedirs(output_keypoints_file, exist_ok=True)
    output_smpls_file = os.path.join(output_dir, exp_name, str(epoch), 'SMPLs')
    os.makedirs(output_smpls_file, exist_ok=True)

    for i in range(len(file_name)):
        with open(os.path.join(output_keypoints_file, file_name[i].replace('.pkl', '.json')), 'w') as f:
            json.dump({'keypoints': keypoint[i].cpu().detach().numpy().tolist()}, f, indent=4)
        # with open(os.path.join(output_smpls_file, file_name[i].replace('.pkl', '.json')), 'w') as f:
        #     json.dump({'smpl_trans': smpl_trans[i].cpu().detach().numpy().tolist(), \
        #                 'smpl_poses': smpl_poses[i].cpu().detach().numpy().tolist()}, f, indent=4)
        with open(os.path.join(output_smpls_file, file_name[i]), 'wb') as f:
            pickle.dump({'smpl_trans': smpl_trans[i].cpu().detach().numpy(),
                            'smpl_poses': smpl_poses[i].cpu().detach().numpy()}, f)
                        
def save_model(model, epoch, exp_name):
    output_model_file = os.path.join(output_dir, exp_name, str(epoch), 'Model')
    os.makedirs(output_model_file, exist_ok=True)
    torch.save({'model': model.state_dict()}, os.path.join(output_model_file, 'model.pth'))



