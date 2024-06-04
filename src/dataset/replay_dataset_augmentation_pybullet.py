import pybullet as p
import time
import pybullet_data
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import json
import os, sys, yaml
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
import time


# import diffusion model
from src.models.unet_diffusion_policy import ConditionalUnet1D
from src.dataset.dataset_loader import Panda_Diffusion_Policy_Dataset_Loader
from diffusers.training_utils import EMAModel
blue = np.array([0, 0, 1, 0.7])
green = np.array([0, 1, 0, 0.7])

blue_panda = np.array([0, 0, 1, 0.0])
green_panda = np.array([0, 1, 0, 0.0])

yellow = np.array([1, 1, 0, 0.7])
red = np.array([1, 0, 0, 0.7])

panda_collision_sphere_colors = [
    i / 8.0 * yellow + (8.0 - i) / 8.0 * red for i in range(8)
]

collision_sphere_colors = [
    i / 8.0 * blue + (8.0 - i) / 8.0 * green for i in range(61)
]


def init_collision_spheres(p, obstacles_flattend: np.ndarray, sphere_id_to_joint_id: dict, collision_sphere_colors: list):
    """
    
    """

    collision_objects = []
    nr_collision_objects = 61

    for i in range(nr_collision_objects):
        # robot collision spheres in range 
        radius = obstacles_flattend[0, 0, 4*i + 3]
        if radius == 0:
            radius = 0.01
        
        position = [
            obstacles_flattend[0, 0, 4*i],
            obstacles_flattend[0, 0, 4*i + 1],
            obstacles_flattend[0, 0, 4*i + 2]
        ]

        joint_id = int(sphere_id_to_joint_id[str(i)])
        
        sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=collision_sphere_colors[joint_id].tolist()) # [1,1,1,1]
        sphereBodyId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, basePosition=position)

        # disable gravity and dynamics for the sphere
        p.changeDynamics(sphereBodyId, -1, linearDamping=0, angularDamping=0)
        p.changeDynamics(sphereBodyId, -1, mass=0, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0, contactStiffness=1e6, contactDamping=1e6, contactProcessingThreshold=0.001)

        collision_objects.append(sphereBodyId)

    return collision_objects


def update_collision_sphere_list_positions(p, collision_sphere_list, obstacles_flattend):
    """
    
    """

    assert obstacles_flattend.shape == (244,)

    for i in range(len(collision_sphere_list)):
        position = [
            obstacles_flattend[4*i],
            obstacles_flattend[4*i + 1],
            obstacles_flattend[4*i + 2]
        ]
        p.resetBasePositionAndOrientation(collision_sphere_list[i], position, [0,0,0,1])


def init_trajectory_points(p, obs, pred_horizon, colors):
    """
    
    """

    trajectory_points = []
    for i in range(pred_horizon):
        position = [
            obs[0, 0, 4*i],
            obs[0, 0, 4*i + 1],
            obs[0, 0, 4*i + 2]
        ]
        sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=0.0075, rgbaColor=colors[i])
        sphereBodyId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, basePosition=position)
        trajectory_points.append(sphereBodyId)
    return trajectory_points


def draw_trajectory(p, ee_pos, trajectory_point_ids, pred_horizon):
    """
    
    """

    for i in range(pred_horizon):
        p.resetBasePositionAndOrientation(trajectory_point_ids[i], ee_pos[i], [0,0,0,1])


def draw_pick_and_place_area(pick_area, place_area):
    """
    
    """

    # draw the area as rectangle in pybullet
    p.addUserDebugLine(pick_area[0], pick_area[1], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(pick_area[1], pick_area[3], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(pick_area[3], pick_area[2], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(pick_area[2], pick_area[0], lineColorRGB=[1, 0, 0], lineWidth=2)

    p.addUserDebugLine(place_area[0], place_area[1], lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(place_area[1], place_area[3], lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(place_area[3], place_area[2], lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(place_area[2], place_area[0], lineColorRGB=[0, 1, 0], lineWidth=2)

    
def get_observation(robot_id, target_pos, target_rot, collision_sphere_list, panda_collision_sphere_list, device):
    """
    
    """
    # get robot ee position and rotation
    link_state = p.getLinkState(robot_id, 7, computeForwardKinematics=True)
    ee_pos = torch.tensor(link_state[0]).unsqueeze(0)
    ee_rot = torch.tensor(link_state[1]).unsqueeze(0)
    # convert ee_rot quaternion to NN format: from pybullet format: i, j, k, w to NN format: w, i, j, k
    ee_rot = torch.cat([ee_rot[:, 3:], ee_rot[:, :3]], dim=1)

    assert ee_pos.shape == (1, 3)
    assert ee_rot.shape == (1, 4)

    assert target_pos.shape == (1, 3)
    assert target_rot.shape == (1, 4) # currently target rotation is in NN format

    obstacles_flattend = torch.zeros((1, 244))
    for i in range(len(collision_sphere_list)):
        position, _ = p.getBasePositionAndOrientation(collision_sphere_list[i])
        obstacles_flattend[0, (4*i):(4*i) + 3] = torch.tensor(position)

        # get sphere radius
        visual_shape_data = p.getVisualShapeData(collision_sphere_list[i])
        radius = visual_shape_data[0][3][0]
        obstacles_flattend[0, 4*i+3] = radius


    robot_collision_spheres_flattend = torch.zeros((1, 244))
    for i in range(len(panda_collision_sphere_list)):
        position, _ = p.getBasePositionAndOrientation(panda_collision_sphere_list[i])
        robot_collision_spheres_flattend[0, (4*i):(4*i) + 3] = torch.tensor(position)

        # get sphere radius
        visual_shape_data = p.getVisualShapeData(panda_collision_sphere_list[i])
        radius = visual_shape_data[0][3][0]
        robot_collision_spheres_flattend[0, 4*i+3] = radius

    ee_pos = ee_pos.to(device)
    ee_rot = ee_rot.to(device)
    target_pos = target_pos.to(device)
    target_rot = target_rot.to(device)
    obstacles_flattend = obstacles_flattend.to(device)
    robot_collision_spheres_flattend = robot_collision_spheres_flattend.to(device)

    obs_ik = torch.cat([ 
        ee_pos, 
        ee_rot, 
        target_pos, 
        target_rot, 
        obstacles_flattend, 
        robot_collision_spheres_flattend], 
        dim=1
    )

    return obs_ik


def init_target(target_position, target_orientation):
    """
    
    """
    # add a visual cube
    target_visual_shape = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=[0.025, 0.025, 0.025], 
        rgbaColor=[1, 0, 0, 1]
    )

    # set target position and orientation
    target_id = p.createMultiBody(
        baseMass=0,
        baseInertialFramePosition=[0, 0, 0],
        baseVisualShapeIndex=target_visual_shape,
        basePosition=target_position,
        baseOrientation=target_orientation
    )

    return target_id


def draw_ee_pos_and_rot(p, ee_pos, ee_rot):
    """
    
    """

    # clear debug lines
    p.removeAllUserDebugItems()
    # draw debug lines from ee_pos in x y and z direction given ee_rot
    x = np.array([0.1, 0, 0])
    y = np.array([0, 0.1, 0])
    z = np.array([0, 0, 0.1])

    # convert ee_rot w, i, j, k to scipy rotation format
    ee_rot = [ee_rot[1], ee_rot[2], ee_rot[3], ee_rot[0]]

    x_rot = R.from_quat(ee_rot).apply(x)
    y_rot = R.from_quat(ee_rot).apply(y)
    z_rot = R.from_quat(ee_rot).apply(z)

    p.addUserDebugLine(ee_pos, ee_pos + x_rot, lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(ee_pos, ee_pos + y_rot, lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(ee_pos, ee_pos + z_rot, lineColorRGB=[0, 0, 1], lineWidth=2)


def update_target_pose(target_id, target_position, target_orientation):
    """
    
    """
    p.resetBasePositionAndOrientation(target_id, target_position, target_orientation)


def main():
    """
    
    """

    # set gui size
    physicsClient = p.connect(p.GUI, options='--width=3000 --height=2000')


    # set initial camera pose
    p.resetDebugVisualizerCamera(
        cameraDistance=1.4, 
        cameraYaw=0, 
        cameraPitch=-50, 
        cameraTargetPosition=[0.0,-0.3,0.4])
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
    p.setGravity(0,0,-9.81)
    p.setRealTimeSimulation(0)

    # retract_pose = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    retract_pose = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]

    # pick and place area
    pick_y_min = -0.6 # -0.8
    pick_y_max = -0.125 # -0.2
    pick_x_min = 0.3 # 0.2
    pick_x_max = 0.7 # 0.8

    pick_area = [
        (pick_x_max, pick_y_min, 0.01),
        (pick_x_max, pick_y_max, 0.01),
        (pick_x_min, pick_y_min, 0.01),
        (pick_x_min, pick_y_max, 0.01),
    ]

    # for collision dataset this will be the pick area
    place_y_min = 0.125 # 0.2
    place_y_max = 0.6 # 0.8
    place_x_min = 0.3 # 0.2
    place_x_max = 0.7 # 0.8

    place_area = [
        (place_x_max, place_y_min, 0.01),
        (place_x_max, place_y_max, 0.01),
        (place_x_min, place_y_min, 0.01),
        (place_x_min, place_y_max, 0.01),
    ]

    draw_pick_and_place_area(pick_area, place_area)

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

    # NN input dimensions
    # parameters
    # Obs:      [] -> context
    # Action:   [] -> state
    #|o|o|o|o|                         observations: 4
    #|     |a|a|a|a|a|a|               actions executed: 8 (starts at step 4)
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    pred_horizon: int = 16 # nr steps predicted
    obs_horizon: int = 4 # nr steps observed
    action_horizon: int = 8 # nr steps executed
    action_dim: int = 7
    obs_dim: int = 502 
    num_diffusion_iters: int = 100
    train_fraction = 0.8 # here used as fraction to get the evaluation dataset


    # set GUI window size to 1000x1000
    planeId = p.loadURDF("plane.urdf")
    
    df_path = os.path.join(root_path, f"src/data/curobo_panda_pick_and_place_robot_collision_dataset_diffusion_policy_ik_only.parquet")


    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    video_out_path = os.path.join(root_path, f"src/data/dataset_augmentation_video_{timestamp}.mp4")

    # load dataset
    eval_dataset = Panda_Diffusion_Policy_Dataset_Loader(
        df_path = df_path,
        pred_horizon = pred_horizon,
        obs_horizon = obs_horizon,
        fraction_to_use = train_fraction,
        eval = False, # uses episodes > fraction_to_use * nr_episodes
        augment_data = True # if True, noise is added to the observation ee positions and ee rotations
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=True,
    )


    # load json file with sphere id to joint id mapping
    sphere_id_to_joint_id = json.load(open(os.path.join(root_path, "src/data/sphere_id_to_joint_id.json")))

    # add a custom sphere with radius r and pos [x,y,z]
    data = next(iter(eval_dataloader))
    obs = data["obs"].clone()
    obstacles_flattend = obs[:, :, 14:14+244]
    robot_collision_spheres_flattend = obs[:, :, 14+244:]
    
    print(f"obstacles_flattend.shape: {obstacles_flattend.shape}")  
    assert obstacles_flattend.shape == (1, 4, 244)

    collision_sphere_list = init_collision_spheres(p, obstacles_flattend, sphere_id_to_joint_id, collision_sphere_colors)
    panda_collision_sphere_list = init_collision_spheres(p, robot_collision_spheres_flattend, sphere_id_to_joint_id, panda_collision_sphere_colors)

    # add target
    target_pos = obs[:, :, 7:10].clone()
    target_rot = obs[:, :, 10:14].clone()

    # convert target_rot to pybullet format
    target_rot = torch.cat([target_rot[:, :, 1:], target_rot[:, :, 0:1]], dim=2)
    target_id = init_target(target_pos, target_rot)

    # add trajectory points

    yellow = np.array([1.0, 1.0, 0.0, 0.7])
    red = np.array([1.0, 0.0, 0.0, 0.7])
    blue = np.array([0.0, 0.0, 1.0, 0.7])
    green = np.array([0.0, 1.0, 0.0, 0.7]) 

    color_ee_gt = []
    for i in range(pred_horizon):
        c = (1 - i / pred_horizon) * blue + (i / pred_horizon) * green
        color_ee_gt.append(c.tolist())

    color_ee_pred = []
    for i in range(pred_horizon):
        c = (1 - i / pred_horizon) * yellow + (i / pred_horizon) * red
        color_ee_pred.append(c.tolist())

    trajectory_points_gt = init_trajectory_points(p, obs, pred_horizon, color_ee_gt)

    # start video recording
    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_out_path)

    try:
        for i, data in enumerate(eval_dataloader):
            if i > 20:
                # stop video recording
                p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
                break

            obs = data["obs"].clone()
            action = data["action"].clone()

            ee_pos = obs[:, :, :3]
            # recorded data is in quaternion format: w, i, j, k
            # pybullet uses quaternion format: i, j, k, w
            ee_rot = obs[:, :, 3:7]


            target_pos = obs[:, :, 7:10]
            target_rot = obs[:, :, 10:14]
            
            obstacles_flattend = obs[:, :, 14:258]
            robot_collision_spheres_flattend = obs[:, :, 258:]

            # update target
            current_target_pos = target_pos[0, 0]
            current_target_rot = target_rot[0, 0]
            target_rot_converted = torch.cat([current_target_rot[1:], current_target_rot[0:1]], dim=0)
            update_target_pose(target_id, current_target_pos, target_rot_converted)

            draw_trajectory(p, action[0, :, 0:3], trajectory_points_gt, pred_horizon)
            for i in range(obs_horizon):
                draw_ee_pos_and_rot(p, ee_pos[0, i], ee_rot[0, i])
                # draw_pick_and_place_area(pick_area, place_area)

                # update panda collision sphere positions and ghost collision sphere positions
                update_collision_sphere_list_positions(p, panda_collision_sphere_list, robot_collision_spheres_flattend[0, i])
                update_collision_sphere_list_positions(p, collision_sphere_list, obstacles_flattend[0, i])
                
                for i in range(2):
                    p.stepSimulation()
                    time.sleep(0.1)

            time.sleep(0.75)

    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()