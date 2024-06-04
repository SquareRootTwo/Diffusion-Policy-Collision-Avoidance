import pybullet as p
import time
import pybullet_data
import numpy as np
from PIL import Image
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from dataclasses import dataclass
from typing import Sequence
import json
import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)


# import diffusion model
from src.models.diffusion_transformer import TransformerForDiffusion
from src.models.unet_diffusion_policy import ConditionalUnet1D
from src.dataset.dataset_loader import Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader
from src.utils.normalisation import normalize_data, unnormalize_data
from src.utils.panda_kinematics import PandaKinematics

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

pick_y_min = -0.9
pick_y_max = -0.1
pick_x_min = 0.1
pick_x_max = 0.9

pick_area = [
    (pick_x_min, pick_y_min, 0.0),
    (pick_x_max, pick_y_max, 0.0),
    (pick_x_min, pick_y_max, 0.0),
    (pick_x_max, pick_y_min, 0.0),
]

# for collision dataset this will be the pick area
place_y_min = 0.1
place_y_max = 0.9
place_x_min = 0.1
place_x_max = 0.9

place_area = [
    (place_x_min, place_y_min, 0.0),
    (place_x_max, place_y_max, 0.0),
    (place_x_min, place_y_max, 0.0),
    (place_x_max, place_y_min, 0.0),
]


pk = PandaKinematics()

def init_panda_collision_sphere(p, robot_id, robot_collision_spheres_flattend, sphere_id_to_joint_id):
    collision_objects = []
    nr_collision_objects = 61

    relative_position_map = {}

    for i in range(nr_collision_objects):
        # robot collision spheres in range 

        joint_id = int(sphere_id_to_joint_id[str(i)])

        radius = robot_collision_spheres_flattend[0, 0, 4*i + 3]
        position = [
            robot_collision_spheres_flattend[0, 0, 4*i],
            robot_collision_spheres_flattend[0, 0, 4*i + 1],
            robot_collision_spheres_flattend[0, 0, 4*i + 2]
        ]

        if joint_id == 0:
            relative_position = [
                position[0],
                position[1],
                position[2]
            ]
        else:
            link_state = p.getLinkState(robot_id, (joint_id-1))
            joint_position = link_state[0]
            joint_orientation = link_state[1]

            relative_position = [
                position[0] - joint_position[0],
                position[1] - joint_position[1],
                position[2] - joint_position[2]
            ]

            # rotate relative position to world frame
            r = R.from_quat(joint_orientation)
            r_inv = r.inv()
            relative_position = r_inv.apply(relative_position)

        relative_position_map[str(i)] = relative_position
        
        sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=panda_collision_sphere_colors[joint_id].tolist()) # [1,1,1,1]
        sphereBodyId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, basePosition=position)

        # disable gravity and dynamics for the sphere
        p.changeDynamics(sphereBodyId, -1, linearDamping=0, angularDamping=0)
        p.changeDynamics(sphereBodyId, -1, mass=0, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0, contactStiffness=1e6, contactDamping=1e6, contactProcessingThreshold=0.001)

        collision_objects.append(sphereBodyId)

    return collision_objects, relative_position_map


def update_panda_collision_sphere_positions(p, panda_collision_sphere_list, sphere_id_to_joint_id, robot_id, relative_position_map):
    # get all joint world states
    joint_world_state = []
    for i in range(7):
        joint_world_state.append(p.getLinkState(robot_id, i))


    for i in range(len(panda_collision_sphere_list)):
        joint_id = int(sphere_id_to_joint_id[str(i)])

        relative_position = relative_position_map[str(i)]

        if joint_id == 0:
            position = [
                relative_position[0],
                relative_position[1],
                relative_position[2]
            ]
        else:
            link_state = joint_world_state[(joint_id-1)]
            joint_position = link_state[0]
            joint_orientation = link_state[1]
            # rotate relative position to world frame
            r = R.from_quat(joint_orientation)
            relative_position = r.apply(relative_position)

            position = [
                joint_position[0] + relative_position[0],
                joint_position[1] + relative_position[1],
                joint_position[2] + relative_position[2]
            ]

        p.resetBasePositionAndOrientation(panda_collision_sphere_list[i], position, [0,0,0,1])

    pass


def init_collision_spheres(p, obstacles_flattend: np.ndarray, sphere_id_to_joint_id: dict):

    collision_objects = []
    nr_collision_objects = 61

    for i in range(nr_collision_objects):
        # robot collision spheres in range 
        radius = obstacles_flattend[0, 0, 4*i + 3]
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


def update_ghost_collision_sphere_list_positions(p, collision_sphere_list, obs, step):
    obstacles_flattend = obs[0, step, 21:21+244] 
    assert obstacles_flattend.shape == (244,)

    for i in range(len(collision_sphere_list)):
        position = [
            obstacles_flattend[4*i],
            obstacles_flattend[4*i + 1],
            obstacles_flattend[4*i + 2]
        ]
        p.resetBasePositionAndOrientation(collision_sphere_list[i], position, [0,0,0,1])


def init_trajectory_points(p, obs, pred_horizon, colors):
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


def draw_trajectory(p, ee_pos, trajectory_point_ids):
    for i in range(34):
        p.resetBasePositionAndOrientation(trajectory_point_ids[i], ee_pos[i], [0,0,0,1])


def draw_pick_and_place_area():
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


def update_target_pose(target_id, target_position, target_orientation):
    """
    
    """
    p.resetBasePositionAndOrientation(target_id, target_position, target_orientation)


def check_collision(p, panda_collision_spheres, ghost_collision_spheres):
    """

    """

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

            if np.linalg.norm(sphere_i_pos - sphere_j_pos) < sphere_i_radius + sphere_j_radius + 5e-3:
                print(f"Collision between panda and ghost sphere: {i} and {j}")
                return True
            
    return False


def reset_panda(p, robot_id, retract_pose):
    """
    
    """
    p.setJointMotorControlArray(
        robot_id,
        np.arange(7),
        controlMode=p.POSITION_CONTROL,
        targetPositions=retract_pose[:7]
    )

    for i in range(20):
        p.stepSimulation()


def get_gt_dict(obs: torch.Tensor, action: torch.Tensor, step: int):
    """
    
    """

    gt_action_dim = 14
    gt_obs_dim = 509

    gt_dict = {}

    for i in range(gt_action_dim):
        gt_dict[f"gt_action_{i}"] = action[0, step, i].item()

    for i in range(gt_obs_dim):
        gt_dict[f"gt_obs_{i}"] = obs[0, step, i].item()

    return gt_dict

def get_obs_dict(obs: torch.Tensor):
    """
    
    """
    obs_dim = obs.shape[1]
    obs_dict = {}

    for i in range(obs_dim):
        obs_dict[f"obs_{i}"] = obs[0, i].item()

    return obs_dict


def draw_orientation(p, pos, rot, length=0.1):
    """
    
    """

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


@dataclass
class TrainConfig:
    pred_horizon: int = 34
    obs_horizon: int = 34
    action_horizon: int = 8

    num_workers: int = 5
    train_fraction: float = 0.8

    # dataset parameters
    augment_data: bool = False
    dtype = torch.float32

    model_type = "transformer" # unet

    # Transformer nn parameters
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768
    causal_attn: bool = True 
    n_cond_layers: int = 0

    # Unet nn parameters
    down_dims: Sequence[int] = (256, 512, 1024, 1024) 
    kernel_size: int = 3 # 3
    diffusion_step_embed_dim: int = 256 # 64
    n_groups: int = 8 # 8
    num_diffusion_iters: int = 100
    conditional_dim: int = 509
    action_dim: int = 7

    # training parameters
    batch_size: int = 256
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-6
    jerk_loss_alpha: float = 0.2
    acc_loss_alpha: float = 0.2
    vel_loss_alpha: float = 0.2

    # action_dim: int = 14
    target_dim: int = 7

    # logging parameters
    wandb_model_log_freq: int = 30
    wandb_model_save_freq: int = 100

    # evaluation
    eval_step_interval: int = 100


def capture_image(out_path, resolution=(1920, 1080)):
    camera_target_position = [0.5, -0.1, 0.3]
    camera_pitch=-45
    camera_yaw=0
    camera_distance=1.4

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
    
    img_path = os.path.join(out_path, f"pick_and_place_setup.png")
    Image.fromarray(np.uint8(img)).save(img_path)


def main():
    """
    
    """
    config = TrainConfig()

    # set gui size
    physicsClient = p.connect(p.GUI, options='--width=3000 --height=2000')

    # set initial camera pose
    p.resetDebugVisualizerCamera(
        cameraDistance=1.4, 
        cameraYaw=0, 
        cameraPitch=-45, 
        cameraTargetPosition=[0.5,-0.1,0.3])
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
    p.setGravity(0,0,-9.81)
    p.setRealTimeSimulation(0)

    retract_pose = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]

    obstacle_pick_place_depth = 0.8 # x
    obstacle_pick_place_width = 0.05 # y
    obstacle_pick_place_height = 0.15 # z

    obstacle_pick_place_position_x = (pick_x_max + pick_x_min) / 2.0
    obstacle_pick_place_position_y = (pick_y_min + place_y_max) / 2.0
    obstacle_pick_place_position_z = obstacle_pick_place_height / 2.0

    # add bar to the scene

    bar = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=np.array([
            obstacle_pick_place_depth, 
            obstacle_pick_place_width, 
            obstacle_pick_place_height
        ])/2, 
        rgbaColor=[0.9, 0.9, 0.9, 1]
    )

    # set target position and orientation
    bar_id = p.createMultiBody(
        baseMass=0,
        baseInertialFramePosition=[0, 0, 0],
        baseVisualShapeIndex=bar,
        basePosition=[obstacle_pick_place_position_x, obstacle_pick_place_position_y, obstacle_pick_place_position_z],
        baseOrientation=[1,0,0,0]
    )


    # set sphere obstacle
    obstacle_position_x_min = 0.2
    obstacle_position_x_max = 0.8
    obstacle_position_y_min = -0.1
    obstacle_position_y_max = 0.1
    obstacle_position_z_min = 0.1
    obstacle_position_z_max = 1.0
    obstacle_radius = 0.09

    obstacle_position_x = np.random.uniform(obstacle_position_x_min, obstacle_position_x_max)
    obstacle_position_y = np.random.uniform(obstacle_position_y_min, obstacle_position_y_max)
    obstacle_position_z = np.random.uniform(obstacle_position_z_min, obstacle_position_z_max)

    # add random collision sphere to the scene
    sphereId = p.createVisualShape(p.GEOM_SPHERE, 
                    radius=obstacle_radius, 
                    rgbaColor=[0.9,0.9,0.9,1])
    sphereBodyId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, basePosition=np.array([
        obstacle_position_x,
        obstacle_position_y,
        obstacle_position_z
    ]))


    draw_pick_and_place_area()

    # set GUI window size to 1000x1000
    planeId = p.loadURDF("plane.urdf")

    # load panda robot
    startPos = [0,0,0]
    startOrientation = p.getQuaternionFromEuler([0,0,0])

    robot_id = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=1)

    # set panda pose to retract pose
    reset_panda(p, robot_id, retract_pose)

    target_rot = np.array([1,0,0,0])
    target_pos = np.array([0.3, -0.4,0.1])

    # convert target_rot to scipy format
    target_id = init_target(target_pos, target_rot)

    capture_image(os.path.join(root_path, "src/plot_scripts"))

    while True:
        pass


    p.disconnect()

if __name__ == "__main__":
    main()