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
from src.utils.pybullet_replay_utils import *


@dataclass
class EvalConfig:
    pred_horizon: int = 34
    obs_horizon: int = 34
    action_horizon: int = 8
    dtype = torch.float32

    train_fraction: float = 0.8

    model_type = "unet" 
    # run_timestamp = "2024-04-18_00-49-51" # unet
    run_timestamp = "2024-05-10_23-40-20" # unet

    # model_type = "transformer"
    # run_timestamp = "2024-04-22_00-04-23" # transformer
    
    nr_sim_steps: int = 20
    unet_exec_time: float = 0.09885454177856445 / nr_sim_steps
    transformer_exec_time: float = 0.07048392295837402 / nr_sim_steps

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

    collect_data: bool = True


def main():
    """
    
    #  Robot at [0,0,0]
    #  collision robot will be at [1,0,0] (mirror y axis)
    #                   x
    #                   |   
    #          [ Collision Robot ]   
    #       __________  |  __________ 
    #      |          | | |          |
    #      |  place   | | |  pick    |
    #      |          | | |          |
    #      |__________| | |__________|
    # y --------------[0,0]--------------
    #                   |
    #                   |
    #                   |
    #                   |
    #                   |
    # 

    """
    config = EvalConfig()

    # set gui size
    physicsClient = p.connect(p.GUI, options='--width=3000 --height=2000')

    if config.model_type == "unet":
        p.setTimeStep(config.unet_exec_time)
    else:
        p.setTimeStep(config.transformer_exec_time)
    
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

    logging_out_base_path = os.path.join(root_path, f"src/data/thesis_eval/{config.run_timestamp}_{config.model_type}_joint_space")
    os.makedirs(logging_out_base_path, exist_ok=True)

    load_ema = True

    model_step = "99"

    ema_model_path = os.path.join(root_path, f"src/logs/diffusion_policy/{config.model_type}/{config.run_timestamp}_run", f"{config.run_timestamp}_ema_weights.pt")
    model_path = os.path.join(root_path, f"src/logs/diffusion_policy/{config.model_type}/{config.run_timestamp}_run", f"{config.run_timestamp}_epoch_{model_step}_weights.pt")

    if config.model_type == "transformer":
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
        noise_pred_net = ConditionalUnet1D(
            input_dim=config.action_dim,
            cond_dim=config.conditional_dim*config.obs_horizon,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            n_groups=config.n_groups,
        )

    if load_ema:
        noise_pred_net = torch.optim.swa_utils.AveragedModel(
            noise_pred_net,
        )

        noise_pred_net.load_state_dict(torch.load(ema_model_path))
    else: 
        noise_pred_net.load_state_dict(torch.load(model_path))

    noise_pred_net = noise_pred_net.to(device)
    noise_pred_net.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    noise_scheduler.set_timesteps(config.num_eval_iters)

    # load dataset
    eval_dataset = Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader(
        fraction_to_use=config.train_fraction,
        eval=True,
        augment_data=False
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
    )

    # load panda robot
    startPos = [0,0,0]
    startOrientation = p.getQuaternionFromEuler([0,0,0])

    robot_id = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=1)

    # set panda pose to retract pose
    reset_panda(robot_id, retract_pose)

    # add a custom sphere with radius r and pos [x,y,z]
    # get second item from eval_dataloader
    data = eval_dataset.__getitem__(2, no_augmentation=True)

    for k in data.keys():
        if torch.is_tensor(data[k]):
            data[k] = data[k].unsqueeze(0)

    obs = data["obs"].clone()
    target = data["target"].clone().squeeze(0)
    target_pos = target[: ,0:3].clone()
    target_rot = target[: ,3:].clone()
    
    collision_sphere_list = init_collision_spheres(ghost=True)
    panda_collision_sphere_list = init_collision_spheres(ghost=False)

    # convert target_rot to scipy format
    target_rot_scipy = torch.cat([target_rot[:,1:], target_rot[:,0:1]], dim=1)
    target_id = init_target(target_pos.squeeze(0), target_rot_scipy.squeeze(0))

    # add trajectory points
    yellow = np.array([1.0, 1.0, 0.0, 0.7])
    red = np.array([1.0, 0.0, 0.0, 0.7])
    blue = np.array([0.0, 0.0, 1.0, 0.7])
    green = np.array([0.0, 1.0, 0.0, 0.7]) 

    color_ee_gt = []
    for i in range(config.pred_horizon):
        c = (1 - i / config.pred_horizon) * blue + (i / config.pred_horizon) * green
        color_ee_gt.append(c.tolist())

    color_ee_pred = []
    for i in range(config.pred_horizon):
        c = (1 - i / config.pred_horizon) * yellow + (i / config.pred_horizon) * red
        color_ee_pred.append(c.tolist())

    trajectory_points_gt = init_trajectory_points(obs, config.pred_horizon, color_ee_gt)
    trajectory_points_pred = init_trajectory_points(obs, config.pred_horizon, color_ee_pred)

    init_pick_place_area()

    prev_episode = -1
    curr_episode = -1

    dataset_iterator = iter(eval_dataloader)

    failed_convergence = 0
    total_collisions = 0

    for episode_i in range(101):
        if config.collect_data:
            # start new pybullet MP4 video recording
            # Note: Pybullet vidoe recording needs Bkank Screen to be disabled in the settings (otherwise it will record too fast videos)
            video_out_path = os.path.join(logging_out_base_path, f"episode_{int(curr_episode+1):03d}_video/")
            os.makedirs(video_out_path, exist_ok=True)
            
        episode_df_data = []
        reset_panda(robot_id, retract_pose)
        # counter for video frames
        frame_counter = 0

        for phase in range(1, 4):
            init_phase = True

            # clear debug lines
            p.removeAllUserDebugItems()
            
            # draw pick and place area
            draw_pick_and_place_area()

            for step in range(34):
                current_df = {}
                data = next(dataset_iterator)

                # only need to perform this at the first step in each phase
                if init_phase:
                    init_phase = False

                    target = data["target"].clone().squeeze(0)
                    target_pos = target[: ,0:3].clone()
                    target_rot = target[: ,3:].clone()
                    
                    # update target position and orientation
                    target_rot_scipy = torch.cat([target_rot[:,1:], target_rot[:,0:1]], dim=1)
                    update_target_pose(target_id, target_pos.squeeze(0), target_rot_scipy.squeeze(0))
                    target_rot_scipy = target_rot_scipy.clone().squeeze(0).cpu().numpy()

                    sim_obs = torch.zeros((1, 34, 509), dtype=torch.float32).to(device)
                    sim_obs[0, :, 14:17] = target_pos.clone().repeat(34, 1)
                    sim_obs[0, :, 17:21] = target_rot.clone().repeat(34, 1)

                curr_episode = data["episode"]
                obs = data["obs"].clone().to(device)
                target = data["target"].clone().squeeze(0)
                action = data["action"].clone().to(device)
                gt_obs = data["gt_obs"].clone().cpu()

                target_pos = target[: ,0:3].clone()
                target_rot = target[: ,3:].clone()

                # need to be updated before get_observation is called
                update_collision_spheres(collision_sphere_list, gt_obs[:, step], ghost=True)

                # shape = (1, 509)
                current_obs = get_observation(
                    robot_id, 
                    target,
                    collision_sphere_list,
                    device
                )

                # update panda collision sphere positions
                update_collision_spheres(panda_collision_sphere_list, current_obs, ghost=False)

                sim_obs[0, step, :] = current_obs.clone()
                
                if step % config.action_horizon == 0:
                    
                    # replan trajectory
                    with torch.no_grad():
                        obs = sim_obs.clone().to(device)

                        nobs = normalize_data(obs, obs_stats)

                        if config.model_type == "unet":
                            obs_cond = nobs.flatten(start_dim=1).to(torch.float32)
                        else: 
                            obs_cond = nobs

                        obs_cond = obs_cond.to(dtype=config.dtype)
                        B = 1

                        # initialize action from Guassian noise
                        noisy_action = torch.randn(
                            (B, config.pred_horizon, config.action_dim), device=device)
                        naction = noisy_action.to(dtype=config.dtype)

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

                    pre_action_joints =  action_pred[:, :7].cpu()

                    ee_pos_pred, ee_rot_pred = pk.get_ee_pose(pre_action_joints.to(pk.device))
                    draw_trajectory(ee_pos_pred, trajectory_points_pred)


                ee_pos_gt = gt_obs[0, :, 7:10].clone().cpu()
                draw_trajectory(ee_pos_gt, trajectory_points_gt)

                predicted_joint_angles_dict = {}

                pre_action_step = pre_action_joints[step, :]

                for i in range(7):
                    predicted_joint_angles_dict[f"pred_panda_joint{i+1}"] = pre_action_step[i]

                p.setJointMotorControlArray(
                    robot_id,
                    np.arange(7),
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=pre_action_step[:7]
                )

                for i in range(config.nr_sim_steps):
                    if config.collect_data and i % 10 == 0:
                        capture_image(
                            out_path = video_out_path,
                            step = frame_counter
                        )
                        frame_counter += 1
                    p.stepSimulation()

                link_state = p.getLinkState(robot_id, 7, computeForwardKinematics=True)
                ee_pos = link_state[0]
                link_state = p.getLinkState(robot_id, 8, computeForwardKinematics=True)
                ee_rot_scipy = link_state[1]

                ee_rot_scipy = np.array(ee_rot_scipy)
                if ee_rot_scipy[3] < 0:
                    ee_rot_scipy = -1 * ee_rot_scipy

                draw_orientation(ee_pos, ee_rot_scipy, 0.1)
                draw_orientation(target_pos.clone().squeeze(0).cpu(), target_rot_scipy, 0.1)

                prev_episode = curr_episode

                if config.collect_data:
                    # convergence metrics
                    ee_pos_dist = np.linalg.norm(np.array(ee_pos) - np.array(target_pos.squeeze(0).cpu()))
                    ee_rot_dist = np.linalg.norm(ee_rot_scipy - target_rot_scipy)

                    q_ee_rot_scipy = R.from_quat(ee_rot_scipy)
                    q_target_rot_scipy = R.from_quat(target_rot_scipy)

                    quat_vec = q_ee_rot_scipy.as_quat()
                    quat_vec = quat_vec / np.linalg.norm(quat_vec)
                    q_ee_rot_scipy = R.from_quat(quat_vec)

                    quat_vec = q_target_rot_scipy.as_quat()
                    quat_vec = quat_vec / np.linalg.norm(quat_vec)
                    q_target_rot_scipy = R.from_quat(quat_vec)

                    relative_q = q_target_rot_scipy.inv() * q_ee_rot_scipy

                    # as rotation vector and angle
                    rot_vec = relative_q.as_rotvec()
                    angle = np.linalg.norm(rot_vec)

                    if angle > np.pi:
                        angle_dist = 2 * np.pi - angle
                    else:
                        angle_dist = angle

                    x_angle_dist, y_angle_dist, z_angle_dist = relative_q.as_euler('xyz', degrees=True)


                    for i in range(7):
                        joint_state = p.getJointState(robot_id, i)
                        current_df[f"pred_panda_joint{i+1}"] = pre_action_step[i].item()
                        current_df[f"panda_joint{i+1}"] = joint_state[0]

                    current_df["phase"] = data["phase"].item()
                    current_df["episode"] = data["episode"].item()

                    if step == config.pred_horizon:
                        if ee_pos_dist < 0.05:
                            print("Panda has converged to the target position")
                            current_df["convergence_failed"] = 0
                        else:
                            print(f"Panda has failed to converge in episode {data['episode'].item()}, total failed convergence: {failed_convergence}")
                            current_df["convergence_failed"] = 1
                    else:
                        current_df["convergence_failed"] = 0

                    # check if collision occured
                    if check_collision(panda_collision_sphere_list, collision_sphere_list):
                        current_df["collision"] = 1
                        total_collisions += 1
                        print(f"total collisions occured: {total_collisions}")
                    else:
                        current_df["collision"] = 0

                    # update episode df data
                    current_df["ee_pos_dist"] = ee_pos_dist
                    current_df["ee_rot_dist"] = ee_rot_dist
                    current_df["x_angle_dist"] = x_angle_dist
                    current_df["y_angle_dist"] = y_angle_dist
                    current_df["z_angle_dist"] = z_angle_dist
                    current_df["angle_dist"] = angle_dist
                    current_df["step"] = step

                    obs_dict = get_obs_dict(current_obs.clone())

                    for k, v in obs_dict.items():
                        current_df[k] = v

                    gt_dict = get_gt_dict(gt_obs, action.clone().cpu(), step)

                    for k, v in gt_dict.items():
                        current_df[k] = v

                    episode_df_data.append(current_df)

        if config.collect_data:
            capture_image(
                out_path = video_out_path,
                step = frame_counter
            )
            # store episode data
            df_out_path = os.path.join(logging_out_base_path, f"episode_{prev_episode.item()}_data.parquet")
            episode_df = pd.DataFrame(episode_df_data)

            print(f"Saving episode data to: {df_out_path}")
            print(f"episode_df.shape: {episode_df.shape}")
            print(f"episode_df.head(): {episode_df.head()}")

            episode_df.to_parquet(
                df_out_path, 
                index=False
            )
            episode_df_data = []
    
    p.disconnect()

if __name__ == "__main__":
    main()