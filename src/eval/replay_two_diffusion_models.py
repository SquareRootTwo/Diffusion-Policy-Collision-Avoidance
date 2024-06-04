import pybullet as p
import time
import pybullet_data
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from pytorch_kinematics.transforms import matrix_to_quaternion, quaternion_to_matrix
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
from src.utils.draw_info import draw_trajectory, draw_pick_and_place_area
from src.utils.pybullet_replay_utils import capture_image, init_pick_place_area

blue = np.array([0, 0, 1, 0.7])
green = np.array([0, 1, 0, 0.7])

blue_panda = np.array([0, 0, 1, 0.0])
green_panda = np.array([0, 1, 0, 0.0])

yellow = np.array([1, 1, 0, 0.7])
red = np.array([1, 0, 0, 0.7])

panda_yellow = np.array([1, 1, 0, 0.0])
panda_red = np.array([1, 0, 0, 0.0])

panda_1_collision_sphere_colors = [
    i / 8.0 * blue_panda + (8.0 - i) / 8.0 * green_panda for i in range(61)
]

panda_2_collision_sphere_colors = [
    i / 8.0 * panda_yellow + (8.0 - i) / 8.0 * panda_red for i in range(61)
]

color_ee_gt = []
for i in range(34):
    c = (1 - i / 34) * green + (i / 34) * blue
    color_ee_gt.append(c.tolist())

color_ee_pred = []
for i in range(34):
    c = (1 - i / 34) * yellow + (i / 34) * red
    color_ee_pred.append(c.tolist())

pk = PandaKinematics()


def update_collision_spheres(sphere_list, curr_obs, ghost=False):

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


def init_collision_spheres(device, ghost=False):
    collision_objects = []

    rot = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], device=device)
    
    offset = torch.tensor([1.0, 0.0, 0.0], device=device)

    collision_sphere_positions = pk.get_panda_collision_spheres(pk.retract_pose.clone().unsqueeze(0).to(pk.device)).squeeze(0)

    if ghost:
        color_list = panda_2_collision_sphere_colors
    else:
        color_list = panda_1_collision_sphere_colors

    for i in range(61):
        # robot collision spheres in range 
        radius = pk.collision_sphere_radius[i]
        position = collision_sphere_positions[i]

        if ghost:
            position = torch.matmul(position, rot) + offset

        joint_id = int(pk.sphere_id_to_joint_id[i])
        
        sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color_list[joint_id].tolist()) # [1,1,1,1]
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


def init_trajectory_points(pred_horizon, colors):
    trajectory_points = []
    for i in range(pred_horizon):
        position = [0, 0, i]
        sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=0.0075, rgbaColor=colors[i])
        sphereBodyId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, basePosition=position)
        trajectory_points.append(sphereBodyId)
    return trajectory_points


def get_panda_joint_state(panda_id):
    joint_states = p.getJointStates(panda_id, range(7))
    joint_positions = [joint_states[i][0] for i in range(7)]
    joint_positions = torch.tensor(joint_positions).unsqueeze(0)
    return joint_positions

def update_target_pose(target_id, target_position, target_orientation, ghost=False):

    if ghost:
        panda_2_transform_rotation_matrix = torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0,  0, 1]
        ], dtype=torch.float32)
        panda_2_transform_offset = torch.tensor([1, 0, 0], dtype=torch.float32)
        # rotate target 2 to be oriented local to panda 2
        target_2_rot_mat = quaternion_to_matrix(torch.tensor(target_orientation, dtype=torch.float32))
        target_2_rot_mat_transformed = torch.matmul(target_2_rot_mat, panda_2_transform_rotation_matrix)
        target_orientation = matrix_to_quaternion(target_2_rot_mat_transformed).numpy()

        target_position = torch.matmul(torch.tensor(target_position, dtype=torch.float32), panda_2_transform_rotation_matrix) + panda_2_transform_offset
        target_position = target_position.numpy()

    # convert quaternion to scipy format
    target_orientation_scipy = [target_orientation[1], target_orientation[2], target_orientation[3], target_orientation[0]]
    p.resetBasePositionAndOrientation(target_id, target_position, target_orientation_scipy)


def init_target(target_position, target_orientation, ghost):
    
    if ghost:
        color = [1, 0, 0, 1]
    else:
        color = [0, 0, 1, 1]

    
    # add a visual cube
    target_visual_shape = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=[0.025, 0.025, 0.025], 
        rgbaColor=color
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


def reset_panda(robot_id, retract_pose):

    for i in range(7):
        p.resetJointState(robot_id, i, retract_pose[i])

    # reset fingers
    p.resetJointState(robot_id, 7, 0.04)
    p.resetJointState(robot_id, 8, 0.04)

    # reset joint velocities
    p.setJointMotorControlArray(
        robot_id,
        np.arange(7),
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=[0,0,0,0,0,0,0]
    )

    for i in range(4):
        p.stepSimulation()

    time.sleep(0.05)


def get_gt_dict(obs: torch.Tensor, action: torch.Tensor, step: int):

    gt_action_dim = 14
    gt_obs_dim = 509

    gt_dict = {}

    for i in range(gt_action_dim):
        gt_dict[f"gt_action_{i}"] = action[0, step, i].item()

    for i in range(gt_obs_dim):
        gt_dict[f"gt_obs_{i}"] = obs[0, step, i].item()

    return gt_dict


def get_obs_dict(obs: torch.Tensor):
    assert obs.shape[1] == 509
    obs_dim = obs.shape[1]
    obs_dict = {}

    for i in range(obs_dim):
        obs_dict[f"obs_{i}"] = obs[0, i].item()

    return obs_dict


def draw_orientation(p, pos, rot, length=0.1):

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


def sample_target_position(x_min, x_max, y_min, y_max, sample_condition=None):
    if sample_condition is None:
        pos = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), 0.15])

        random_z_rot = np.random.uniform(0, 2*np.pi)
        rot = p.getQuaternionFromEuler([0, 0, random_z_rot])

    else:
        pos = sample_condition
        while np.linalg.norm(pos - sample_condition) < 0.3:
            pos = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), 0.15])

        random_z_rot = np.random.uniform(0, 2*np.pi)
        rot = p.getQuaternionFromEuler([0, 0, random_z_rot])

    # convert quaternion to NN format (xyzw -> wxyz)
    # ensure quaternion w is positive
    if rot[3] < 0:
        rot = [-rot[3], -rot[0], -rot[1], -rot[2]]
    else: 
        rot = [rot[3], rot[0], rot[1], rot[2]]

    return pos, rot


@dataclass
class TrainConfig:
    pred_horizon: int = 34
    obs_horizon: int = 34
    action_horizon: int = 8
    nr_episodes: int = 10

    dtype = torch.float32
    collect_data: bool = True

    # simulation parameters
    nr_sim_steps: int = 20
    # unet_exec_time: float = 1.0136377716064453 / nr_sim_steps
    unet_exec_time: float = 0.09885454177856445 / nr_sim_steps
    # transformer_exec_time: float = 0.7119650936126709 / nr_sim_steps
    transformer_exec_time: float = 0.07048392295837402 / nr_sim_steps

    # model_type: str = "transformer"
    model_type: str =  "unet"

    unet_timestamp: str = "2024-04-18_00-49-51"
    transformer_timestamp: str = "2024-04-22_00-04-23" 

    # Transformer nn parameters
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768
    causal_attn: bool = True 
    n_cond_layers: int = 0

    # Unet nn parameters
    down_dims: Sequence[int] = (256, 512, 1024, 1024) 
    kernel_size: int = 3
    diffusion_step_embed_dim: int = 256
    n_groups: int = 8

    
    num_diffusion_iters: int = 100
    num_eval_iters: int = 10
    
    conditional_dim: int = 509
    action_dim: int = 7

def main():
    config = TrainConfig()

    # set gui size
    physicsClient = p.connect(p.GUI, options='--width=3000 --height=2000')

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
    init_pick_place_area()

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

    device = torch.device("cuda:0")
    
    # load normalisation stats, same as dataset uses but here just for ik version
   
    obs_norm_path = os.path.join(root_path, 'src/data/normalisation_obs.csv')
    action_norm_path = os.path.join(root_path, 'src/data/normalisation_action.csv')
    
    action_stats = torch.from_numpy(
        np.genfromtxt(action_norm_path, delimiter=',')
    ).to(device=device)

    obs_stats = torch.from_numpy(
        np.genfromtxt(obs_norm_path, delimiter=',')
    ).to(device=device)

    # set GUI window size to 1000x1000
    planeId = p.loadURDF("plane.urdf")

    if config.model_type == "unet":
        p.setTimeStep(config.unet_exec_time)
    else:
        p.setTimeStep(config.transformer_exec_time)


    if config.model_type == "transformer":
        logging_out_base_path = os.path.join(root_path, f"src/data/thesis_eval/{config.transformer_timestamp}_2_diffusion_model_setup")
        os.makedirs(logging_out_base_path, exist_ok=True)
        ema_model_path = os.path.join(root_path, f"src/logs/diffusion_policy/{config.model_type}/{config.transformer_timestamp}_run", f"{config.transformer_timestamp}_ema_weights.pt")
        
        noise_pred_net = TransformerForDiffusion(
            input_dim=config.action_dim,
            output_dim=config.action_dim,
            horizon=config.pred_horizon,
            cond_dim=config.conditional_dim,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_emb = config.n_emb,
            causal_attn=config.causal_attn,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=config.n_cond_layers
        )
    else:
        logging_out_base_path = os.path.join(root_path, f"src/data/thesis_eval/{config.unet_timestamp}_2_diffusion_model_setup")
        os.makedirs(logging_out_base_path, exist_ok=True)
        ema_model_path = os.path.join(root_path, f"src/logs/diffusion_policy/{config.model_type}/{config.unet_timestamp}_run", f"{config.unet_timestamp}_ema_weights.pt")
        
        noise_pred_net = ConditionalUnet1D(
            input_dim=config.action_dim,
            cond_dim=config.conditional_dim*config.obs_horizon,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            n_groups=config.n_groups,
        )

    noise_pred_net = torch.optim.swa_utils.AveragedModel(
        noise_pred_net,
    )

    noise_pred_net.load_state_dict(torch.load(ema_model_path))

    noise_pred_net = noise_pred_net.to(device)
    noise_pred_net.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    noise_scheduler.set_timesteps(config.num_eval_iters)


    # load panda robot
    startPos_panda_1 = [0,0,0]
    startPos_panda_2 = [1,0,0]

    startOrientation_panda_1 = p.getQuaternionFromEuler([0,0,0])
    startOrientation_panda_2 = p.getQuaternionFromEuler([0,0,np.pi])

    panda_1 = p.loadURDF("franka_panda/panda.urdf", startPos_panda_1, startOrientation_panda_1, useFixedBase=1)
    panda_2 = p.loadURDF("franka_panda/panda.urdf", startPos_panda_2, startOrientation_panda_2, useFixedBase=1)

    # set panda pose to retract pose
    reset_panda(panda_1, retract_pose)
    reset_panda(panda_2, retract_pose)

    # init target cubes
    panda_1_target_id = init_target(np.array([0.5, 0.5, 0.1]), np.array([1, 0, 0, 0]), ghost=False)
    panda_2_target_id = init_target(np.array([0.5, -0.5, 0.1]), np.array([1, 0, 0, 0]), ghost=True)

    # init collision spheres
    panda_1_collision_spheres = init_collision_spheres(device=device, ghost=False)
    panda_2_collision_spheres =  init_collision_spheres(device=device, ghost=True)

    trajectory_points_panda_1 = init_trajectory_points(config.pred_horizon, color_ee_gt)
    trajectory_points_panda_2 = init_trajectory_points(config.pred_horizon, color_ee_pred)

    panda_retract_pos, panda_retract_rot = pk.get_ee_pose(pose=torch.tensor(retract_pose).unsqueeze(0).to(pk.device), return_quat=True)
    
    rot = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], device=device)
    offset = torch.tensor([1.0, 0.0, 0.0], device=device)

    panda_retract_pos = panda_retract_pos.squeeze(0).to("cpu").numpy()
    panda_retract_rot = panda_retract_rot.squeeze(0).to("cpu").numpy()

    if panda_retract_rot[0] < 0:
        panda_retract_rot = -panda_retract_rot

    assert panda_retract_rot.shape == (4,)
    assert panda_retract_pos.shape == (3,)

    def get_trajectory(obs, step, prev_action):
        if step % config.action_horizon == 0:
            # recompute trajectory
            with torch.no_grad():
                nobs = normalize_data(obs.clone().to(device), obs_stats)
                if config.model_type == "unet":
                    obs_cond = nobs.flatten(start_dim=1).to(config.dtype)
                else: 
                    obs_cond = nobs

                obs_cond = obs_cond.to(dtype=config.dtype)

                # initialize action from Guassian noise
                naction = torch.randn((1, 34, 7), device=device, dtype=config.dtype)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = noise_pred_net(
                        sample=naction,
                        timestep=k,
                        cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample


                naction = torch.cat([naction, torch.zeros_like(naction, device=device)], dim=2)
                action_pred = unnormalize_data(naction, stats=action_stats).squeeze(0)

                action =  action_pred[:, :7].cpu()
                return action
        else:
            return prev_action.cpu()


    for e in range(config.nr_episodes):
        draw_trajectory(p, torch.zeros(34,3).numpy(), trajectory_points_panda_1)
        draw_trajectory(p, torch.zeros(34,3).numpy(), trajectory_points_panda_2)

        video_frame_step = 0
        curr_episode_video_out_path = os.path.join(logging_out_base_path, f"episode_{int(e):03d}")
        os.makedirs(curr_episode_video_out_path, exist_ok=True)

        reset_panda(panda_1, retract_pose=pk.retract_pose)
        reset_panda(panda_2, retract_pose=pk.retract_pose)

        # sample new targets
        panda_1_pick_target_pos, panda_1_pick_target_rot = sample_target_position(
            x_min=pick_x_min,
            x_max=pick_x_max,
            y_min=pick_y_min,
            y_max=pick_y_max
        )
        
        panda_1_place_target_pos, panda_1_place_target_rot = sample_target_position(
            x_min=place_x_min,
            x_max=place_x_max,
            y_min=place_y_min,
            y_max=place_y_max
        )

        panda_2_pick_cond = panda_1_place_target_pos.copy()
        panda_2_pick_cond[0] *= -1 
        panda_2_pick_cond[0] += 1
        panda_2_pick_cond[1] *= -1

        panda_2_pick_target_pos, panda_2_pick_target_rot = sample_target_position(
            x_min=pick_x_min,
            x_max=pick_x_max,
            y_min=pick_y_min,
            y_max=pick_y_max,
            sample_condition=panda_2_pick_cond
        )

        panda_2_place_cond = panda_1_pick_target_pos.copy()
        panda_2_place_cond[0] *= -1 
        panda_2_place_cond[0] += 1
        panda_2_place_cond[1] *= -1

        panda_2_place_target_pos, panda_2_place_target_rot = sample_target_position(
            x_min=place_x_min,
            x_max=place_x_max,
            y_min=place_y_min,
            y_max=place_y_max,
            sample_condition=panda_2_place_cond
        )

        panda_1_phase = 0
        panda_2_phase = 0

        panda_1_step = 0
        panda_2_step = np.random.randint(-20, 1)
        
        panda_1_target = torch.cat([
            torch.tensor(panda_1_pick_target_pos),
            torch.tensor(panda_1_pick_target_rot)
        ], dim=0)

        update_target_pose(
            target_id=panda_1_target_id,
            target_position=panda_1_target[:3],
            target_orientation=panda_1_target[3:],
            ghost=False
        )

        panda_2_target = torch.cat([
                torch.tensor(panda_2_pick_target_pos),
                torch.tensor(panda_2_pick_target_rot)
            ], dim=0)
        
        update_target_pose(
            target_id=panda_2_target_id,
            target_position=panda_2_target[:3],
            target_orientation=panda_2_target[3:],
            ghost=True
        )


        panda_1_obs = torch.zeros(1, 34, 509).to(device=device)
        panda_1_action = torch.zeros(34, 7).to(device=device)
        panda_1_obs[0, :, 14:21] = panda_1_target.to(device=device).unsqueeze(0).repeat(34, 1)
        
        panda_2_obs = torch.zeros(1, 34, 509).to(device=device)
        panda_2_action = torch.zeros(34, 7).to(device=device)    
        panda_2_obs[0, :, 14:21] = panda_2_target.to(device=device).unsqueeze(0).repeat(34, 1)


        while (panda_1_phase < 3) or (panda_2_phase < 3):
            # set current panda target
            if panda_1_step == 34:
                panda_1_step = 0
                panda_1_phase += 1

                if panda_1_phase == 1:
                    panda_1_target = torch.cat([
                        torch.tensor(panda_1_place_target_pos),
                        torch.tensor(panda_1_place_target_rot)
                    ], dim=0)
                else:
                    panda_1_target = torch.cat([
                        torch.tensor(panda_retract_pos),
                        torch.tensor(panda_retract_rot)
                    ], dim=0)


                panda_1_obs = torch.zeros(1, 34, 509).to(device=device)
                panda_1_action = torch.zeros(34, 7).to(device=device)
                panda_1_obs[0, :, 14:21] = panda_1_target.to(device=device).unsqueeze(0).repeat(34, 1)
               
                update_target_pose(
                    target_id=panda_1_target_id,
                    target_position=panda_1_target[:3],
                    target_orientation=panda_1_target[3:],
                    ghost=False
                )

            if panda_2_step == 34:
                panda_2_step = 0
                panda_2_phase += 1

                if panda_2_phase == 1:
                    panda_2_target = torch.cat([
                        torch.tensor(panda_2_place_target_pos),
                        torch.tensor(panda_2_place_target_rot)
                    ], dim=0)
                else:
                    panda_2_target = torch.cat([
                        torch.tensor(panda_retract_pos),
                        torch.tensor(panda_retract_rot)
                    ], dim=0)


                update_target_pose(
                    target_id=panda_2_target_id,
                    target_position=panda_2_target[:3],
                    target_orientation=panda_2_target[3:],
                    ghost=True
                )

                panda_2_obs = torch.zeros(1, 34, 509).to(device=device)
                panda_2_action = torch.zeros(34, 7).to(device=device)    
                panda_2_obs[0, :, 14:21] = panda_2_target.to(device=device).unsqueeze(0).repeat(34, 1)

            # get new obs
            # shape = (1, 7)
            panda_1_th = get_panda_joint_state(panda_1)
            # shape = (1, 7)
            panda_2_th = get_panda_joint_state(panda_2)
            
            # shape = (2, 7)
            panda_targets = torch.stack([panda_1_target, panda_2_target], dim=0)

            current_obs = pk.get_2_panda_obs(
                th=torch.cat([panda_1_th, panda_2_th], dim=0),
                target=panda_targets
            )

            if panda_1_phase < 3:
                panda_1_obs[0, min(33,panda_1_step), :] = current_obs[0, 0, :]
                panda_1_action = get_trajectory(panda_1_obs, panda_1_step, panda_1_action)
                ee_pos_pred, _ = pk.get_ee_pose(panda_1_action.to(pk.device))
                draw_trajectory(p, ee_pos_pred.cpu().numpy(), trajectory_points_panda_1)
                update_collision_spheres(panda_1_collision_spheres, current_obs, ghost=False)

                # set joint positions
                p.setJointMotorControlArray(
                    panda_1,
                    np.arange(7),
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=panda_1_action[panda_1_step]
                )
            
            if panda_2_step >= 0:
                panda_2_obs[0, min(33,panda_2_step), :] = current_obs[1, 0, :]
                
                panda_2_action = get_trajectory(panda_2_obs, panda_2_step, panda_2_action)
                ee_pos_pred, _ = pk.get_ee_pose(panda_2_action.to(pk.device))

                ee_pos_pred = (torch.matmul(ee_pos_pred, rot) + offset).cpu().numpy()
                draw_trajectory(p, ee_pos_pred, trajectory_points_panda_2)
                update_collision_spheres(panda_2_collision_spheres, current_obs, ghost=True)

                p.setJointMotorControlArray(
                    panda_2,
                    np.arange(7),
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=panda_2_action[panda_2_step]
                )

            for si in range(config.nr_sim_steps):
                if si % 10 == 0:
                    capture_image(curr_episode_video_out_path, video_frame_step)
                    video_frame_step += 1

                p.stepSimulation()

            # update indices
            panda_1_step += 1
            panda_2_step += 1
       
    p.disconnect()

if __name__ == "__main__":
    main()