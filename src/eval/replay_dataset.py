import pybullet as p
import time
import pybullet_data
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import json
import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)


# import diffusion model
from src.dataset.dataset_loader import Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader
from src.utils.pybullet_replay_utils import *

blue = np.array([0, 0, 1, 0.7])
green = np.array([0, 1, 0, 0.7])

blue_panda = np.array([0, 0, 1, 0.0])
green_panda = np.array([0, 1, 0, 0.0])

yellow = np.array([1, 1, 0, 0.7])
red = np.array([1, 0, 0, 0.7])

panda_collision_sphere_colors = [
    i / 8.0 * blue_panda + (8.0 - i) / 8.0 * green_panda for i in range(8)
]

collision_sphere_colors = [
    i / 8.0 * blue + (8.0 - i) / 8.0 * green for i in range(61)
]


def get_obs_dict(obs: torch.Tensor):
    obs_dim = obs.shape[1]
    obs_dict = {}

    for i in range(obs_dim):
        obs_dict[f"obs_{i}"] = obs[0, i].item()

    return obs_dict


def main():
    # set gui size
    p.connect(p.GUI, options='--width=3000 --height=2000')

    # set initial camera pose
    p.resetDebugVisualizerCamera(
        cameraDistance=1.1, 
        cameraYaw=0, 
        cameraPitch=-45, 
        cameraTargetPosition=[0.5,-0.1,0.3])
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
    p.setGravity(0,0,-9.81)
    p.setRealTimeSimulation(0)

    retract_pose = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]

    #  Robot at [0,0,0]
    #  collision robot will be at [1,0,0] (mirror y axis)
    #                   x
    #                   |   
    #          [ Collision Robot ]   
    #       __________  |  __________ 
    #      |          | | |          |
    #      |  pick    | | |  place   |
    #      |          | | |          |
    #      |__________| | |__________|
    # y --------------[0,0]--------------
    #                   |
    #                   |
    #                   |
    #                   |
    #                   |
    # 

    # load dataset
    dataset = Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader(
        df_path = "/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_robot_collision_dataset.parquet",
    )

    eval_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # load panda robot
    startPos = [0,0,0]
    startOrientation = p.getQuaternionFromEuler([0,0,0])

    robot_id = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=1)
    # add ground plane
    p.loadURDF("plane.urdf")

    # set panda pose to retract pose
    reset_panda(robot_id, retract_pose)

    # add a custom sphere with radius r and pos [x,y,z]
    # get second item from eval_dataloader
    data = dataset.__getitem__(2, no_augmentation=True)

    for k in data.keys():
        if torch.is_tensor(data[k]):
            data[k] = data[k].unsqueeze(0)

    obs = data["obs"].clone()
    target = data["target"].clone().squeeze(0)
    target_pos = target[: ,0:3].clone()
    target_rot = target[: ,3:].clone()

    obstacles_flattend = obs[:, :, 21:21+244]
    robot_collision_spheres_flattend = obs[:, :, 21+244:]
    
    # init collision spheres
    collision_sphere_list = init_collision_spheres(ghost=True)
    panda_collision_sphere_list = init_collision_spheres(ghost=False)

    # convert target_rot to pybullet format
    target_rot_scipy = torch.cat([target_rot[:,1:], target_rot[:,0:1]], dim=1)
    target_id = init_target(target_pos.squeeze(0), target_rot_scipy.squeeze(0))

    # add trajectory points

    yellow = np.array([1.0, 1.0, 0.0, 0.7])
    red = np.array([1.0, 0.0, 0.0, 0.7])
    blue = np.array([0.0, 0.0, 1.0, 0.7])
    green = np.array([0.0, 1.0, 0.0, 0.7]) 

    color_ee_gt = []
    for i in range(34):
        c = (1 - i / 34) * blue + (i / 34) * green
        color_ee_gt.append(c.tolist())

    color_ee_pred = []
    for i in range(34):
        c = (1 - i / 34) * yellow + (i / 34) * red
        color_ee_pred.append(c.tolist())

    trajectory_points_gt = init_trajectory_points(obs, 34, color_ee_gt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    faulty_episodes = [50, 58, 63, 80, 90, 99, 117, 119, 139, 154, 162, 175, 188, 218, 238, 245, 284, 291, 297, 298, 316, 335, 345,352, 361, 414, 499,]
    init_pick_place_area()

    for i, data in enumerate(eval_dataloader):
    
        obs = data["obs"].clone().to(device)
        action = data["action"].clone().to(device)
        step = data["step"]

        if step == 0:
            for i in range(10000):
                p.stepSimulation()
            p.removeAllUserDebugItems()
            draw_pick_and_place_area()

        target = obs[:, 33, 14:21].clone()


        target_pos = target[: ,0:3].clone()
        target_rot = target[: ,3:].clone()

    
        next_pose = action[0, step, :7]
    
        ee_pos_gt = action[:, :, 7:10]
        ee_rot_gt = action[:, :, 10:14]
        ee_pos_gt = ee_pos_gt.squeeze(0)
        ee_rot_gt = ee_rot_gt.squeeze(0)
        # convert ee_rot_gt quaternion to scipy format
        ee_rot_gt_scipy = torch.cat([ee_rot_gt[step, 1:], ee_rot_gt[step, 0:1]], dim=1).squeeze(0).to("cpu").numpy()
        curr_ee_pos = ee_pos_gt[step].squeeze(0).to("cpu").numpy()

        draw_orientation(curr_ee_pos, ee_rot_gt_scipy, 0.2)
        draw_trajectory(ee_pos_gt, trajectory_points_gt)

        # set panda pose 
        joint_pos_pred = next_pose.squeeze(0).to("cpu").numpy()

        p.setJointMotorControlArray(
            robot_id,
            np.arange(7),
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_pos_pred[:7]
        )


        for i in range(200):
            p.stepSimulation()

        link_state = p.getLinkState(robot_id, 7, computeForwardKinematics=True)
        ee_pos = link_state[0]
        link_state = p.getLinkState(robot_id, 8, computeForwardKinematics=True)
        ee_rot_scipy = link_state[1]

        ee_rot_scipy = np.array(ee_rot_scipy)
        if ee_rot_scipy[3] < 0:
            ee_rot_scipy = -1 * ee_rot_scipy

        # update panda collision sphere positions

        curr_obs = obs[:, step, :].cpu().squeeze(0)
        update_collision_spheres(panda_collision_sphere_list, curr_obs, ghost=False)
        update_collision_spheres(collision_sphere_list, curr_obs, ghost=True)
        # update target position and orientation
        target_rot_scipy = torch.cat([target_rot[:,1:], target_rot[:,0:1]], dim=1)
        update_target_pose(target_id, target_pos.squeeze(0), target_rot_scipy.squeeze(0))

        draw_orientation(ee_pos, ee_rot_scipy, 0.1)



    p.disconnect()

if __name__ == "__main__":
    main()