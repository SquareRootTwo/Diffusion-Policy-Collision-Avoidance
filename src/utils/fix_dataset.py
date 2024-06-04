import pandas as pd
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# add root to path
import sys, os
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from src.utils.panda_kinematics import PandaKinematics

pk = PandaKinematics()

df_path = "/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_robot_collision_dataset.parquet"

df = pd.read_parquet(df_path)


# first check if all joint angles correspond to the ee pose
# joint pose range
dist_pos_list = []
dist_rot_list = []
angle_rot_list = []

faulty_episodes = [50, 58, 63, 80, 90, 99, 117, 119, 139, 154, 162, 175, 188, 218, 238, 245, 284, 291, 297, 298, 316, 335, 345,352, 361, 414, 499,]

new_df = pd.DataFrame(columns=df.columns)

for e in df["episode"].unique():
    e_df = df[df["episode"] == e]
    
    # in the end we only need to replace the ee pose entries (last ee pos plus shifting the others)
    # copy the dataframe (deep copy)
    current_new_df = e_df.copy()

    # e_df = e_df[e_df["phase"] == 1]
        
    # print(f"joint poses: {e_df.columns[16:23].values}")
    jp = e_df.iloc[:-1, 16:23].values

    last_jp = e_df.iloc[-1, 16:23].values
    last_jp = torch.tensor(last_jp).unsqueeze(0).to(pk.device)
    last_ee_pos, last_ee_rot = pk.get_ee_pose(last_jp)

    last_ee_rot = last_ee_rot[0]
    if last_ee_rot[0] < 0:
        last_ee_rot = -last_ee_rot

    last_ee_rot = last_ee_rot.unsqueeze(0)

    # print(f"ee_pos: {e_df.columns[9:12].values}")
    ee_pos = torch.tensor(e_df.iloc[1:, 9:12].values)

    assert len(jp) == len(ee_pos)

    # print(f"ee_rot: {e_df.columns[12:16].values}")
    ee_rot = torch.tensor(e_df.iloc[1:, 12:16].values)

    cond = ee_rot[:, 0] < 0
    ee_rot[:, 0] = torch.where(cond, -ee_rot[:, 0], ee_rot[:, 0])
    ee_rot[:, 1] = torch.where(cond, -ee_rot[:, 1], ee_rot[:, 1])
    ee_rot[:, 2] = torch.where(cond, -ee_rot[:, 2], ee_rot[:, 2])
    ee_rot[:, 3] = torch.where(cond, -ee_rot[:, 3], ee_rot[:, 3])

    # append the last ee pose
    ee_pos = torch.cat([ee_pos, last_ee_pos.to("cpu")])
    ee_rot = torch.cat([ee_rot, last_ee_rot.to("cpu")])

    # add back to new dataframe
    current_new_df.iloc[:, 9:12] = ee_pos.numpy()
    current_new_df.iloc[:, 12:16] = ee_rot.numpy()

    # shift panda collision spheres
    panda_collision_spheres = e_df.iloc[1:, 25:269].values
    last_spheres = pk.get_panda_collision_spheres(last_jp)
    last_spheres = last_spheres.to("cpu")
    last_spheres_full_vec = torch.zeros((1, 244))

    # take first sphere 3 (4 entries) from old dataframe -> small little sphere at the ground of the panda that shouldnt move
    last_spheres_full_vec[0, 0:4*3] = torch.tensor(panda_collision_spheres[0, 0:4*3].copy())

    # 2,3 -> not change, 1 is only one to change
    for si in range(2, 61):
        si_pos = last_spheres[si]
        si_pos = si_pos.flatten()
        # add x -> si*4, y -> si*4+1, z -> si*4+2
        last_spheres_full_vec[0, si*4:si*4+3] = si_pos

        # radius -> si*4+3
        radius_si = panda_collision_spheres[0, si*4+3]
        last_spheres_full_vec[0, si*4+3] = torch.tensor(radius_si)

    panda_collision_spheres = np.concatenate([panda_collision_spheres, last_spheres_full_vec.to("cpu").numpy()], axis=0)

    current_new_df.iloc[:, 25:269] = panda_collision_spheres

    new_df = pd.concat([new_df, current_new_df], ignore_index=True)


for faulty_e in faulty_episodes:
    faulty_e_df = new_df[new_df["episode"] == faulty_e]

    ghost_collision_spheres = faulty_e_df.iloc[:, 277:521].values

    # get max jerk index of the ghost collision spheres
    # p[i+1] - p[i] = v[i]
    vel = np.diff(ghost_collision_spheres, axis=0)
    # v[i+1] - v[i] = a[i] = (p[i+2] - p[i+1]) - (p[i+1] - p[i]) = p[i+2] - 2*p[i+1] + p[i]
    # -> at max(v[i]) -> i+1
    acc = np.diff(vel, axis=0)
    # a[i+1] - a[i] = j[i] = (v[i+2] - v[i+1]) - (v[i+1] - v[i]) 
    # = (p[i+4] - 2*p[i+3] + p[i+2]) - 2*(p[i+3] - 2*p[i+2] + p[i+1]) + (p[i+2] - 2*p[i+1] + p[i])
    # = p[i+4] - 2*p[i+3] + p[i+2] - 2*p[i+3] + 4*p[i+2] - 2*p[i+1] + p[i+2] - 2*p[i+1] + p[i]
    # = p[i+4] - 4*p[i+3] + 6*p[i+2] - 4*p[i+1] + p[i]
    # -> at max(a[i]) -> i+2
    jerk = np.diff(acc, axis=0)

    jerk = np.linalg.norm(jerk, axis=1)
    max_index = np.argmax(jerk) + 2 # -> max is at i+2

    interpolated_ghost_collision_spheres = (ghost_collision_spheres[max_index-1] + ghost_collision_spheres[max_index+1]) / 2

    ghost_collision_spheres[max_index] = interpolated_ghost_collision_spheres

    # add back to the dataframe
    new_df.iloc[faulty_e_df.index, 277:521] = ghost_collision_spheres

# another round to fix the faulty episodes
faulty_episodes = [63, 99, 117, 162, 218, 284, 291, 298]

for _ in range(10):
    for faulty_e in faulty_episodes:
        faulty_e_df = new_df[new_df["episode"] == faulty_e]

        ghost_collision_spheres = faulty_e_df.iloc[:, 277:521].values

        # get max jerk index of the ghost collision spheres
        # p[i+1] - p[i] = v[i]
        vel = np.diff(ghost_collision_spheres, axis=0)
        # v[i+1] - v[i] = a[i] = (p[i+2] - p[i+1]) - (p[i+1] - p[i]) = p[i+2] - 2*p[i+1] + p[i]
        # -> at max(v[i]) -> i+1
        acc = np.diff(vel, axis=0)
        # a[i+1] - a[i] = j[i] = (v[i+2] - v[i+1]) - (v[i+1] - v[i]) 
        # = (p[i+4] - 2*p[i+3] + p[i+2]) - 2*(p[i+3] - 2*p[i+2] + p[i+1]) + (p[i+2] - 2*p[i+1] + p[i])
        # = p[i+4] - 2*p[i+3] + p[i+2] - 2*p[i+3] + 4*p[i+2] - 2*p[i+1] + p[i+2] - 2*p[i+1] + p[i]
        # = p[i+4] - 4*p[i+3] + 6*p[i+2] - 4*p[i+1] + p[i]
        # -> at max(a[i]) -> i+2
        jerk = np.diff(acc, axis=0)

        jerk = np.linalg.norm(jerk, axis=1)
        max_index = np.argmax(jerk) + 2 # -> max is at i+2

        # interpolate between i-2 and i+2
        interpolated_ghost_collision_spheres = (ghost_collision_spheres[max_index-1] + ghost_collision_spheres[max_index+1]) / 2

        ghost_collision_spheres[max_index] = interpolated_ghost_collision_spheres

        # add back to the dataframe
        new_df.iloc[faulty_e_df.index, 277:521] = ghost_collision_spheres


# now with the updated collision spheres [449, 479] have a collision 
# TODO: fix this


# save the new dataframe
new_df.to_parquet(df_path.replace(".parquet", "_fixed.parquet"))