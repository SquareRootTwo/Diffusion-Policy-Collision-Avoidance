import numpy as np
import torch

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

# add root to path
import sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

# custom imports
from src.utils.panda_kinematics import PandaKinematics

pk = PandaKinematics()

def compute_loss_fk(noisy_actions, naction, target):
    # target.shape = (bs, 7)
    final_joint_state = naction[:, -1, :7]
    start_joint_state = naction[:, 0, :7]

    pred_final_joint_state = noisy_actions[:, -1, :7]
    pred_start_joint_state = noisy_actions[:, 0, :7]
    
    loss_final_joint_state = torch.mean(torch.sum(torch.absolute(final_joint_state - pred_final_joint_state), dim=1))
    loss_start_joint_state = torch.mean(torch.sum(torch.absolute(start_joint_state - pred_start_joint_state), dim=1))

    pred_fk_final_ee_pos, pred_fk_final_ee_rot = pk.get_ee_pose(pred_final_joint_state)
    target_position = target[:, :3]
    target_rot = target[:, 3:]

    # convert from w, i, j, k to i, j, k, w
    pred_fk_final_ee_rot_scipy = torch.cat([pred_fk_final_ee_rot[:, 1:], pred_fk_final_ee_rot[:, :1]], dim=1)
    target_rot_scipy = torch.cat([target_rot[:, 1:], target_rot[:, :1]], dim=1)
    

    # normalise quaternion vectors 
    final_rotation_scipy = F.normalize(target_rot_scipy, p=2, dim=1)
    pred_final_rotation_scipy = F.normalize(pred_fk_final_ee_rot_scipy, p=2, dim=1)

    final_quat = R.from_quat(final_rotation_scipy.to("cpu").numpy())
    pred_final_quat = R.from_quat(pred_final_rotation_scipy.to("cpu").numpy())

    relative_q_final = final_quat.inv() * pred_final_quat

    rot_vec_final = relative_q_final.as_rotvec(degrees=False)
    angle_final = np.linalg.norm(rot_vec_final, axis=1)
    angle_final = np.where(angle_final > np.pi, 2 * np.pi - angle_final, angle_final)

    loss_angle_final = np.mean(angle_final)

    loss_final_position = torch.mean(torch.norm(target_position - pred_fk_final_ee_pos, dim=-1))

    # smoothness loss
    velocity = torch.diff(noisy_actions, dim=1)

    acceleration = torch.diff(velocity, dim=1)

    jerk = torch.diff(acceleration, dim=1)

    loss_vel = torch.mean(torch.absolute(velocity))

    loss_acc = torch.mean(torch.absolute(acceleration))

    loss_jerk = torch.mean(torch.absolute(jerk))

    loss_dict = {
        "loss_final_joint_state": loss_final_joint_state,
        "loss_start_joint_state": loss_start_joint_state,
        "loss_vel": loss_vel,
        "loss_acc": loss_acc,
        "loss_jerk": loss_jerk,
        "loss_angle_final": loss_angle_final,
        "loss_final_position": loss_final_position
    }

    return loss_dict