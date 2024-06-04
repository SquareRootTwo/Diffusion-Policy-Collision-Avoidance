import pybullet as p
import time
import pybullet_data
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image

from dataclasses import dataclass
from typing import Sequence
import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)


# import diffusion model
from src.models.diffusion_transformer import TransformerForDiffusion
from src.models.unet_diffusion_policy import ConditionalUnet1D
from src.dataset.dataset_loader import Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader
from src.utils.normalisation import normalize_data, unnormalize_data
from src.utils.panda_kinematics import PandaKinematics

# constants 

blue = np.array([0, 0, 1, 1.0])
green = np.array([0, 1, 0, 1.0])

blue_panda = np.array([0, 0, 1, 0.0])
green_panda = np.array([0, 1, 0, 0.0])

yellow = np.array([1, 1, 0, 1.0])
red = np.array([1, 0, 0, 1.0])

panda_collision_sphere_colors = [
    i / 8.0 * blue_panda + (8.0 - i) / 8.0 * green_panda for i in range(8)
]

collision_sphere_colors = [
    i / 8.0 * blue + (8.0 - i) / 8.0 * green for i in range(61)
]

# pick and place area
pick_y_min = -0.6
pick_y_max = -0.125
pick_x_min = 0.3 
pick_x_max = 0.7 

pick_area = [
    (pick_x_max, pick_y_min, 0.01),
    (pick_x_max, pick_y_max, 0.01),
    (pick_x_min, pick_y_min, 0.01),
    (pick_x_min, pick_y_max, 0.01),
]

# for collision dataset this will be the pick area
place_y_min = 0.125
place_y_max = 0.6
place_x_min = 0.3
place_x_max = 0.7

place_area = [
    (place_x_max, place_y_min, 0.01),
    (place_x_max, place_y_max, 0.01),
    (place_x_min, place_y_min, 0.01),
    (place_x_min, place_y_max, 0.01),
]

pk = PandaKinematics()

# data collection scripts

def capture_image(out_path, step, resolution=(1920, 1080)):
    camera_target_position = [0.5, -0.1, 0.3]
    camera_pitch=-45
    camera_yaw=0
    camera_distance=1.2

    # Calculate camera position based on distance, yaw, and pitch
    camera_pos = [
        camera_target_position[0] - camera_distance * np.sin(np.radians(camera_yaw)),
        camera_target_position[1] + camera_distance * np.cos(np.radians(camera_yaw)) * np.sin(np.radians(camera_pitch)),
        camera_target_position[2] + camera_distance * np.cos(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch))
    ]
    
    # Compute view matrix and projection matrix
    view_matrix = p.computeViewMatrix(cameraEyePosition=camera_pos, cameraTargetPosition=camera_target_position, cameraUpVector=[0, 0, 1])
    projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=resolution[0] / resolution[1], nearVal=0.1, farVal=100.0)
    
    # Capture image
    img_arr = p.getCameraImage(
        width=resolution[0], 
        height=resolution[1], 
        viewMatrix=view_matrix, 
        shadow=1,
        projectionMatrix=projection_matrix, 
        renderer=p.ER_BULLET_HARDWARE_OPENGL, 
    )
    img = np.reshape(img_arr[2], (resolution[1], resolution[0], 4))[:, :, :3]  # Remove alpha channel
    
    img_path = os.path.join(out_path, f"frame_{int(step):05d}.png")
    Image.fromarray(np.uint8(img)).save(img_path)


# Initialisation scripts for scene objects

def init_pick_place_area():
    pick_x = (pick_x_min + pick_x_max) / 2
    pick_y = (pick_y_min + pick_y_max) / 2
    place_x = (place_x_min + place_x_max) / 2
    place_y = (place_y_min + place_y_max) / 2
    pick_width = pick_x_max - pick_x_min
    pick_depth = pick_y_max - pick_y_min

    blockVisualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[pick_width/2, pick_depth/2, 0.001], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray color
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=blockVisualShapeId, basePosition=[pick_x, pick_y, 0])

    blockVisualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[pick_width/2, pick_depth/2, 0.001], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray color
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=blockVisualShapeId, basePosition=[place_x, place_y, 0])



def init_target(target_position, target_orientation):
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


def init_trajectory_points(obs, pred_horizon, colors):
    trajectory_points = []
    for i in range(pred_horizon):
        position = [
            obs[0, i, 7],
            obs[0, i, 8],
            obs[0, i, 9]
        ]
        sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=0.0075, rgbaColor=colors[i])
        sphereBodyId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, basePosition=position)
        trajectory_points.append(sphereBodyId)
    return trajectory_points


def init_collision_spheres(ghost=False):
    collision_objects = []

    rot = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], device=pk.device)
    
    offset = torch.tensor([1.0, 0.0, 0.0], device=pk.device)

    collision_sphere_positions = pk.get_panda_collision_spheres(pk.retract_pose.clone().unsqueeze(0).to(pk.device)).squeeze(0)

    for i in range(61):
        # robot collision spheres in range 
        radius = pk.collision_sphere_radius[i]
        position = collision_sphere_positions[i]

        if ghost:
            position = torch.matmul(position, rot) + offset

        joint_id = int(pk.sphere_id_to_joint_id[i])

        position = position.cpu().numpy()
        
        if ghost:
            sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=collision_sphere_colors[joint_id].tolist()) # [1,1,1,1]
        else:
            sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=panda_collision_sphere_colors[joint_id].tolist()) # [1,1,1,1]
        sphereBodyId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, basePosition=position)

        # disable gravity and dynamics for the sphere
        p.changeDynamics(sphereBodyId, -1, linearDamping=0, angularDamping=0)
        p.changeDynamics(sphereBodyId, -1, mass=0, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0, contactStiffness=1e6, contactDamping=1e6, contactProcessingThreshold=0.001)

        collision_objects.append(sphereBodyId)

    return collision_objects


# Update/Obesravation scripts

def update_target_pose(target_id, target_position, target_orientation):
    p.resetBasePositionAndOrientation(target_id, target_position, target_orientation)



def check_collision(panda_collision_spheres, ghost_collision_spheres):

    # check collision between panda and ghost
    for i in range(len(panda_collision_spheres)):
        sphere_i_pos, _ = p.getBasePositionAndOrientation(panda_collision_spheres[i])
        sphere_i_pos = np.array(sphere_i_pos)

        # get sphere radius
        visual_shape_data = p.getVisualShapeData(panda_collision_spheres[i])
        sphere_i_radius = visual_shape_data[0][3][0]

        for j in range(len(ghost_collision_spheres)):
            sphere_j_pos, _ = p.getBasePositionAndOrientation(ghost_collision_spheres[j])
            sphere_j_pos = np.array(sphere_j_pos)

            # get sphere radius
            visual_shape_data = p.getVisualShapeData(ghost_collision_spheres[j])
            sphere_j_radius = visual_shape_data[0][3][0]

            if np.linalg.norm(sphere_i_pos - sphere_j_pos) < sphere_i_radius + sphere_j_radius:
                print(f"Collision between panda and ghost sphere: {i} and {j}")
                return True
            
    return False


def reset_panda(robot_id, retract_pose):
    p.setJointMotorControlArray(
        robot_id,
        np.arange(7),
        controlMode=p.POSITION_CONTROL,
        targetPositions=retract_pose[:7]
    )

    for i in range(20):
        p.stepSimulation()


def get_gt_dict(obs: torch.Tensor, action: torch.Tensor, step: int):

    gt_action_dim = 14
    gt_obs_dim = 509

    gt_dict = {}

    for i in range(gt_action_dim):
        gt_dict[f"gt_action_{i}"] = action[0, step, i].item()

    for i in range(gt_obs_dim):
        gt_dict[f"gt_obs_{i}"] = obs[0, step, i].item()

    return gt_dict


def update_2_robot_collision_spheres(sphere_list, curr_obs, ghost=False):
    rot = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    if ghost:
        collision_spheres = curr_obs[1, 0, 21+244:21+244+244].clone().cpu()
    else:
        collision_spheres = curr_obs[0, 0, 21+244:21+244+244].clone().cpu()

    for i in range(len(sphere_list)):
        # position = collision_spheres[i].to("cpu").numpy()
        if ghost:
            position = (torch.matmul(
                collision_spheres[4*i:4*i+3], 
                rot
            ) + torch.tensor([1, 0, 0])).cpu().numpy()
        else:
            position = collision_spheres[4*i:4*i+3].cpu().numpy()
        p.resetBasePositionAndOrientation(sphere_list[i], position, [0,0,0,1])


def update_collision_spheres(sphere_list, curr_obs: torch.tensor, ghost=False):
    if ghost:
        collision_spheres = curr_obs[0, 21:21+244].clone().cpu()
    else:
        collision_spheres = curr_obs[0, 21+244:21+244+244].clone().cpu()

    for i in range(len(sphere_list)):
        position = collision_spheres[4*i:4*i+3].numpy()
        p.resetBasePositionAndOrientation(sphere_list[i], position, [0,0,0,1])



def get_obs_dict(obs: torch.Tensor):
    obs_dim = obs.shape[1]
    obs_dict = {}

    for i in range(obs_dim):
        obs_dict[f"obs_{i}"] = obs[0, i].item()

    return obs_dict


def get_observation(robot_id, target_pose, collision_sphere_list, device):
    # get joint states
    joint_states = p.getJointStates(robot_id, range(7))
    joint_positions = [joint_states[i][0] for i in range(7)]
    joint_positions = torch.tensor(joint_positions).unsqueeze(0)
    joint_positions = joint_positions.to(device)

    assert joint_positions.shape == (1, 7)

    # get robot ee position and rotation
    link_state = p.getLinkState(robot_id, 7, computeForwardKinematics=True)
    ee_pos = torch.tensor(link_state[0]).unsqueeze(0)
    link_state = p.getLinkState(robot_id, 8, computeForwardKinematics=True)
    ee_rot_scipy = torch.tensor(link_state[1]).unsqueeze(0)

    # convert ee_rot quaternion to NN format: from pybullet format: i, j, k, w to NN format: w, i, j, k
    if ee_rot_scipy[:, 3] < 0:
        ee_rot = torch.cat([-ee_rot_scipy[:, 3:], -ee_rot_scipy[:, :3]], dim=1)
    else:
        ee_rot = torch.cat([ee_rot_scipy[:, 3:], ee_rot_scipy[:, :3]], dim=1)
        
    assert ee_pos.shape == (1, 3)
    assert ee_rot.shape == (1, 4)

    robot_collision_spheres = pk.get_panda_collision_spheres(joint_positions)
    robot_collision_spheres = robot_collision_spheres.squeeze(0)
    robot_collision_spheres_flattend = torch.zeros((1, 244)).to(device)

    obstacles_flattend = torch.zeros((1, 244))
    for i in range(len(collision_sphere_list)):
        position, _ = p.getBasePositionAndOrientation(collision_sphere_list[i])
        obstacles_flattend[0, (4*i):(4*i) + 3] = torch.tensor(position)

        # get sphere radius
        visual_shape_data = p.getVisualShapeData(collision_sphere_list[i])
        radius = visual_shape_data[0][3][0]
        obstacles_flattend[0, 4*i+3] = radius

        robot_collision_spheres_flattend[0, (4*i):(4*i) + 3] = robot_collision_spheres[i, :]
        robot_collision_spheres_flattend[0, 4*i+3] = radius


    ee_pos = ee_pos.to(device)
    ee_rot = ee_rot.to(device)
    obstacles_flattend = obstacles_flattend.to(device)
    robot_collision_spheres_flattend = robot_collision_spheres_flattend.to(device)
    target_pose = target_pose.to(device)

    obs = torch.cat([ 
        joint_positions,
        ee_pos, 
        ee_rot,
        target_pose,
        obstacles_flattend, 
        robot_collision_spheres_flattend], 
        dim=1
    )

    obs = obs.to(torch.float32)

    assert obs.shape == (1, 509)

    return obs

# Debug draw 

def draw_pick_and_place_area():

    # draw the area as rectangle in pybullet
    p.addUserDebugLine(pick_area[0], pick_area[1], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(pick_area[1], pick_area[3], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(pick_area[3], pick_area[2], lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(pick_area[2], pick_area[0], lineColorRGB=[1, 0, 0], lineWidth=2)

    p.addUserDebugLine(place_area[0], place_area[1], lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(place_area[1], place_area[3], lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(place_area[3], place_area[2], lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(place_area[2], place_area[0], lineColorRGB=[0, 1, 0], lineWidth=2)


def draw_trajectory(ee_pos, trajectory_point_ids):
    for i in range(34):
        p.resetBasePositionAndOrientation(trajectory_point_ids[i], ee_pos[i], [0,0,0,1])


def draw_orientation(pos, rot, length=0.1):

    # normalize rotation
    rot = rot / np.linalg.norm(rot)
    # get rotation matrix
    r = R.from_quat(rot)

    rot_matrix = r.as_matrix()

    # get x, y, z axis
    x_axis = rot_matrix[:, 0]
    y_axis = rot_matrix[:, 1]
    z_axis = rot_matrix[:, 2]

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    pos_x_offset = pos + length * x_axis
    pos_y_offset = pos + length * y_axis
    pos_z_offset = pos + length * z_axis


    # draw x, y, z axis
    p.addUserDebugLine(pos, pos_x_offset, lineColorRGB=[1, 0, 0], lineWidth=2)
    p.addUserDebugLine(pos, pos_y_offset, lineColorRGB=[0, 1, 0], lineWidth=2)
    p.addUserDebugLine(pos, pos_z_offset, lineColorRGB=[0, 0, 1], lineWidth=2)