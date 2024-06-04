import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import pandas as pd
import numpy as np
import math
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Dict, Callable
import os
from scipy.spatial.transform import Rotation as R

import sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
from src.utils.panda_kinematics import PandaKinematics

from tqdm import tqdm

pk = PandaKinematics(device="cpu")

class Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader(Dataset):
    def __init__(self, 
                df_path: str = "/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_robot_collision_dataset_fixed.parquet", 
                fraction_to_use=1.0, 
                eval=False, 
                augment_data=False,
                load_dataset_to_memory=False,
                inject_noise_to_obs=False,
                # filter_faulty_episodes=False
                augment_panda_joint_1_lower_limit = -(5.0/180.0) * np.pi,
                augment_panda_joint_1_upper_limit = (5.0/180.0) * np.pi
        ):
        """
        args:
            df_path: path to the parquet file containing the dataset
            timehorizon: number of timesteps to include in the batch for the model to predict
        """
        self.dataset_path = df_path
        self.df = pd.read_parquet(df_path)
        self.df = self.df.sort_values(by=['episode', 'phase', 'step'])

        self.panda_joint_1_lower_limit = augment_panda_joint_1_lower_limit
        self.panda_joint_1_upper_limit = augment_panda_joint_1_upper_limit

        # if filter_faulty_episodes:
        #     # faulty_episodes = [50, 58, 63, 80, 90, 99, 117, 119, 139, 154, 162, 175, 188, 218, 238, 245, 284, 291, 297, 298, 316, 335, 345,352, 361, 414, 499,]
        #     faulty_episodes = [63, 99, 117, 162, 218, 284, 291, 298]
        #     # only get the faulty episodes
        #     self.df = self.df[self.df['episode'].isin(faulty_episodes)]

        #     # reset episode counter to start from 0
        #     for i, ep in enumerate(faulty_episodes):
        #         self.df.loc[self.df['episode'] == ep, 'episode'] = i


        # fraction_to_use: fraction of the dataset to use for training (if not training on the entire dataset)
        if fraction_to_use < 1.0 and not eval:
            total_df_len = self.df['episode'].nunique()
            episode_limit = int(total_df_len * fraction_to_use)
            self.df = self.df[self.df['episode'] < episode_limit]
            print(f"episodes to train on: { self.df['episode'].unique()}" )

        # use only the last fraction_to_use of the dataset for evaluation
        elif fraction_to_use < 1.0 and eval:
            total_df_len = self.df['episode'].nunique()
            episode_limit = int(total_df_len * fraction_to_use)
            self.df = self.df[self.df['episode'] >= episode_limit]

            print(f"episodes to evaluate on: { self.df['episode'].unique()}" )
            # reset episode counter to start from 0
            self.df['episode'] = self.df['episode'] - episode_limit

        self.augment_data = augment_data
        self.inject_noise_to_obs = inject_noise_to_obs

        self.action_dim = len(self.df.filter(regex='^action_\d+$', axis=1).columns)
        self.obs_dim = len(self.df.filter(regex='^obs_\d+$', axis=1).columns)
        self.num_episodes = self.df['episode'].nunique()

        self.pred_horizon = 34
        self.obs_horizon = 34

        # 3 = nr of phases
        self.len = self.num_episodes * 3 * self.pred_horizon

        self.stats = dict()

        self.memory_dataset = []

        self.eval = eval

        self.load_dataset_to_memory = load_dataset_to_memory

        if load_dataset_to_memory:
            print(f"DatasetLoader: Loading dataset to memory", flush=True)
            self.__load_dataset_to_memory__()

        print(f"DatasetLoader: Loaded dataset from {df_path.split('/')[-1]}", flush=True)
        print(f"DatasetLoader: dataset length: {self.len}", flush=True)
        print(f"DatasetLoader: Action space dim: {self.action_dim}", flush=True)
        print(f"DatasetLoader: Observation space dim: {self.obs_dim}", flush=True)
        print(f"DatasetLoader: df head:    {self.df.head()}", flush=True)

        # print columns containing NaN
        print(f"DatasetLoader: Columns containing NaN: {self.df.columns[self.df.isna().any()].tolist()}", flush=True)

        expl = self.__getitem__(0)
        print(f"DatasetLoader: Sample data obs:    {expl['obs'].shape}", flush=True)
        print(f"DatasetLoader: Sample data action: {expl['action'].shape}", flush=True)

    
    def __len__(self) -> int:
        return self.len
    

    def __load_dataset_to_memory__(self):
        desc = "Loading Train Dataset" if not self.eval else "Loading Eval Dataset"
        for idx in tqdm(range(self.len), desc=desc):
            self.memory_dataset.append(self.get_item(idx))

    def get_weights(self):
        phase_1_weight = 1.0
        phase_2_weight = 1.0
        phase_3_weight = 0.02

        phase_1_uniform = np.ones(34) * phase_1_weight
        phase_2_uniform = np.ones(34) * phase_2_weight
        phase_3_uniform = np.ones(34) * phase_3_weight

        episode_weights = np.concatenate((phase_1_uniform, phase_2_uniform, phase_3_uniform))
        weights = np.zeros(self.num_episodes * 3 * 34)
        # replicate num_episodes times
        for i in range(self.num_episodes):
            weights[i * 3 * 34:(i+1) * 3 * 34] = episode_weights.copy()

        # normalize weights
        weights /= np.sum(weights)

        assert np.isclose(np.sum(weights), 1.0)

        return weights

        # old weighting scheme

        # num_distributions = self.num_episodes * 3

        # phase_length = self.pred_horizon

        # weights = np.zeros(num_distributions * phase_length)
        # indices = np.linspace(0, 1, phase_length)
        # indices[0] += 1e-8
        # indices[-1] -= 1e-8
        # probs = 1.0 / (np.pi * np.sqrt(indices * (1 - indices)))

        # probs_sum = np.sum(probs)

        # probs /= probs_sum

        # # arcsine distribution for each phase (special case of beta distribution: alpha = beta = 0.5)
        # for i in range(num_distributions):
        #     weights[i * phase_length:(i+1) * phase_length] = probs.copy()

        # weights /= num_distributions

        # return weights


    def get_item(self, idx):
        episode = int(idx // (3 * self.pred_horizon))
         # 1: pick, 2: place, 3: retract
        episode_start_index = episode * 3 * self.pred_horizon
        phase = int((idx - episode_start_index) // (self.pred_horizon)) + 1
        step = int(idx - episode_start_index - (phase-1) * (self.pred_horizon))

        df_episode = self.df[(self.df['episode'] == episode)]
        df_episode_phase = df_episode[(df_episode['phase'] == phase)]

        # obstacles_flattend: dim = [obs_horizon, num_obstacles * 4]
        obstacles_flattend = torch.zeros((self.pred_horizon, 61 * 4))
        for i in range(61):
            obstacles_flattend[:, i * 4 + 0] = torch.tensor(df_episode_phase[f"sphere_{i}_x"].values)
            obstacles_flattend[:, i * 4 + 1] = torch.tensor(df_episode_phase[f"sphere_{i}_y"].values)
            obstacles_flattend[:, i * 4 + 2] = torch.tensor(df_episode_phase[f"sphere_{i}_z"].values)
            obstacles_flattend[:, i * 4 + 3] = torch.tensor(df_episode_phase[f"sphere_{i}_radius"].values)

        # robot_collision_spheres_flattend: dim = [obs_horizon, num_robot_collision_sphere * 4]
        robot_collision_spheres_flattend = torch.zeros((self.pred_horizon, 61 * 4))
        for i in range(61):
            robot_collision_spheres_flattend[:, i * 4 + 0] = torch.tensor(df_episode_phase[f"robot_sphere_{i}_x"].values)
            robot_collision_spheres_flattend[:, i * 4 + 1] = torch.tensor(df_episode_phase[f"robot_sphere_{i}_y"].values)
            robot_collision_spheres_flattend[:, i * 4 + 2] = torch.tensor(df_episode_phase[f"robot_sphere_{i}_z"].values)
            robot_collision_spheres_flattend[:, i * 4 + 3] = torch.tensor(df_episode_phase[f"robot_sphere_{i}_radius"].values)


        # joint_positions: dim = [pred_horizon, 7]
        joint_positions = torch.zeros((self.pred_horizon, 7))
        # panda_joint1,panda_joint2,panda_joint3,panda_joint4,panda_joint5,panda_joint6,panda_joint7,panda_finger_joint1,panda_finger_joint2
        joint_positions[:, 0] = torch.tensor(df_episode_phase["panda_joint1"].values)
        joint_positions[:, 1] = torch.tensor(df_episode_phase["panda_joint2"].values)
        joint_positions[:, 2] = torch.tensor(df_episode_phase["panda_joint3"].values)
        joint_positions[:, 3] = torch.tensor(df_episode_phase["panda_joint4"].values)
        joint_positions[:, 4] = torch.tensor(df_episode_phase["panda_joint5"].values)
        joint_positions[:, 5] = torch.tensor(df_episode_phase["panda_joint6"].values)
        joint_positions[:, 6] = torch.tensor(df_episode_phase["panda_joint7"].values)

        # end_effector: dim = [pred_horizon, 7]
        # ee positions
        end_effector = torch.zeros((self.pred_horizon, 7))
        end_effector[:, 0] = torch.tensor(df_episode_phase["ee_position_x"].values)
        end_effector[:, 1] = torch.tensor(df_episode_phase["ee_position_y"].values)
        end_effector[:, 2] = torch.tensor(df_episode_phase["ee_position_z"].values)

        # ee rotations
        ee_w_values = torch.tensor(df_episode_phase["ee_rotation_w"].values)
        ee_i_values = torch.tensor(df_episode_phase["ee_rotation_i"].values)
        ee_j_values = torch.tensor(df_episode_phase["ee_rotation_j"].values)
        ee_k_values = torch.tensor(df_episode_phase["ee_rotation_k"].values)

        ee_cnd = ee_w_values < 0
        end_effector[:, 3] = torch.where(ee_cnd, -1.0 * ee_w_values, ee_w_values)
        end_effector[:, 4] = torch.where(ee_cnd, -1.0 * ee_i_values, ee_i_values)
        end_effector[:, 5] = torch.where(ee_cnd, -1.0 * ee_j_values, ee_j_values)
        end_effector[:, 6] = torch.where(ee_cnd, -1.0 * ee_k_values, ee_k_values)

        # target: dim = [pred_horizon, 7]
        target = torch.zeros((self.pred_horizon, 7))
        # target positions 
        # just take the first element -> the target is the same for all steps in the phase of an episode
        target[:, 0] = torch.tensor(df_episode_phase["target_position_x"].values)
        target[:, 1] = torch.tensor(df_episode_phase["target_position_y"].values)
        target[:, 2] = torch.tensor(df_episode_phase["target_position_z"].values)

        # target rotations
        target_w_values = torch.tensor(df_episode_phase["target_rotation_w"].values)
        target_i_values = torch.tensor(df_episode_phase["target_rotation_i"].values)
        target_j_values = torch.tensor(df_episode_phase["target_rotation_j"].values)
        target_k_values = torch.tensor(df_episode_phase["target_rotation_k"].values)

        target_cnd = target_w_values < 0
        target[:, 3] = torch.where(target_cnd, -1.0 * target_w_values, target_w_values)
        target[:, 4] = torch.where(target_cnd, -1.0 * target_i_values, target_i_values)
        target[:, 5] = torch.where(target_cnd, -1.0 * target_j_values, target_j_values)
        target[:, 6] = torch.where(target_cnd, -1.0 * target_k_values, target_k_values)

        # action element: [joint positions (radians), end effector position (meters), end effector rotation (quaternion)]
        # action: dim = [pred_horizon, 7 + 7] = [pred_horizon, 14]
        action = torch.cat((joint_positions, end_effector), dim=1)
        
        observation = torch.cat((
            joint_positions, # 0:7
            end_effector, # 7:14
            target, # 14:21
            obstacles_flattend, # 21:21+244
            robot_collision_spheres_flattend # 21+244:21+244+244
        ), dim=1)

        # mask all observations from the current step
        gt_observation = observation.clone()

        # let the model only see the target from the next step to the end for timesteps > step
        # set all values > step to 0 -> avoid that model has empty predictions 
        observation[(step+1):, :14] = 0.0
        observation[(step+1):, 21:] = 0.0

        return {
            'action': action.to(torch.float32), 
            'obs': observation.to(torch.float32),
            'target': target.to(torch.float32)[:1, :],
            'gt_obs': gt_observation.to(torch.float32),
            'episode': episode,
            'phase': phase,
            'step': step
        }

    
    def __getitem__(self, idx, no_augmentation=False):
        if self.load_dataset_to_memory:
            ret_dict = self.memory_dataset[idx]
        else:
            ret_dict = self.get_item(idx)

        if no_augmentation:
            return ret_dict

        if self.augment_data:
            collision_free = False

            while not collision_free:
                obs = ret_dict['gt_obs'].clone()
                action = ret_dict['action'].clone()
                target = ret_dict['target'].clone()
                episode = ret_dict['episode']
                phase = ret_dict['phase']
                step = ret_dict['step']

                angle = np.random.uniform(self.panda_joint_1_lower_limit, self.panda_joint_1_upper_limit)

                # from euler angles ( 0 , 0 , angle )
                # random_z_rot = R.from_euler('xyz', [0, 0, angle]).as_quat()

                # Augment action data
                # augment angle to the base joint
                action[:, 0] += angle

                ee_pos, ee_rot = pk.get_ee_pose(action[:, 0:7])

                action[:, 7:10] = ee_pos
                action[:, 10:14] = ee_rot

                obs[:, 0] += angle
                obs[:, 7:10] = ee_pos[:, :]
                obs[:, 10:14] = ee_rot[:, :]
                obs[:, 14:17] = ee_pos[-1, :].repeat(self.pred_horizon, 1)
                obs[:, 17:21] = ee_rot[-1, :].repeat(self.pred_horizon, 1)
                target = torch.cat([ee_pos[-1, :], ee_rot[-1, :]], dim=0).unsqueeze(0)

                # shape (34, 61, 3)
                panda_collision_spheres = pk.get_panda_collision_spheres(action[:, 0:7])
                
                # only skip sphere 0 and 1 -> base of robot
                for j in range(2, 61):
                    start_j = 265 + j * 4
                    obs[:, start_j:start_j+3] = panda_collision_spheres[:, j, :].squeeze(1)

                # check if the new observation is collision free
                collisions = 0
                for si in range(10, 61):
                    for sj in range(10, 61):
                        panda_start_si = 265 + si * 4
                        ghost_start_sj = 21 + sj * 4

                        panda_collision_spheres = obs[:, panda_start_si:panda_start_si+3]
                        panda_collision_radius = obs[:, panda_start_si+3]

                        ghost_collision_spheres = obs[:, ghost_start_sj:ghost_start_sj+3]
                        ghost_collision_radius = obs[:, ghost_start_sj + 3]

                        dist = torch.norm(panda_collision_spheres - ghost_collision_spheres, dim=1)
                        
                        collisions += torch.sum(dist < panda_collision_radius + ghost_collision_radius + 0.005)


                if collisions == 0:
                    collision_free = True

                    # set obs > step to 0
                    obs[(step+1):, :14] = 0.0
                    obs[(step+1):, 21:] = 0.0

                    ret_dict = {
                        'action': action.to(torch.float32), 
                        'obs': obs.to(torch.float32),
                        'target': target.to(torch.float32),
                        'episode': episode,
                        'phase': phase,
                        'step': step
                    }

        if self.inject_noise_to_obs:
            obs = ret_dict["obs"]
            observation_shape = obs.shape
            mean = 0
            std_dev = 0.01
            step = ret_dict["step"]

            noise = torch.tensor(np.random.normal(mean, std_dev, observation_shape))

            # dont add noise to target
            noise[:, 14:21] = 0.0

            # dont add noise to future steps that are 0
            noise[(step+1):, :] = 0.0

            obs = obs + noise

            ret_dict["obs"] = obs.to(torch.float32)
            
        return ret_dict
