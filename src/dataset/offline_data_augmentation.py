import pandas as pd
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# add root to path
import sys, os
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from src.utils.panda_kinematics import PandaKinematics
from src.dataset.dataset_loader import Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader

def offline_dataset_augmentation():
    nr_iterations = 10

    df_path = "/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_robot_collision_dataset_fixed.parquet"

    dataset = Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader(
        df_path,
        fraction_to_use=0.8, # only augment 80% of the dataset -> 20 % for validation (no augmentation)
        eval=False,
        augment_data=True,
    )

    df = pd.read_parquet(df_path)

    pick_area = df.iloc[0:34, 269:273]
    place_area = df.iloc[0:34, 273:277]

    new_df = pd.DataFrame(df.columns)

    data_frames = []

    print(f"len(dataset) / 34: {len(dataset) / 34}")
    nr_iters = int(len(dataset) / 34)

    base = 0
    nr_episodes_in_dataset = len(dataset) / (34*3)

    print(f"nr_episodes_in_dataset: {nr_episodes_in_dataset}, max: {df['episode'].max()}, min: {df['episode'].min()}")

    for r in tqdm(range(nr_iterations), desc="Nr iteration", leave=True, position=0, total=nr_iterations):
        for i in tqdm(range(1, (nr_iters+1)), desc=f"Augmenting dataset {r}", leave=True, position=1, total=nr_iters):
            data = dataset.__getitem__((34*i-1))
            assert data["step"] == 33

            # shape (34, 509)
            obs = data["obs"].numpy()

            # episode (34, 1)
            episode = (base + data["episode"]) * np.ones((34, 1), dtype=int)

            # episode (34, 1)
            step = np.arange(0, 34, 1, dtype=int).reshape(-1, 1)
            
            # phase (34, 1)
            phase = data["phase"] * np.ones((34, 1), dtype=int)

            # shape (34, 7,)
            target = np.ones((34, 7)) * obs[-1, 14:21]

            # end effector (34, 7)
            ee_pos = obs[:, 7:14]

            # joint positions (34, 7)
            joint_pos = obs[:, 0:7]

            panda_finger_joint = 0.04 * np.ones((34, 2))

            # panda collision spheres spahe: (34, 244)
            panda_collision_spheres = obs[:, 265:509]

            # ghost collision spheres shape: (34, 244)
            ghost_collision_spheres = obs[:, 21:265]

            df_data = np.concatenate(
                # episode, step, target pose, ee pose, panda joint pose (+ panda finger joint), panda collisions spheres,
                # pick area, place area, ghost robot collision spheres, phase
                (episode, step, target, ee_pos, joint_pos, panda_finger_joint, panda_collision_spheres, pick_area, place_area, ghost_collision_spheres, phase),
                axis=1
            )

            current_new_df = pd.DataFrame(
                columns=df.columns,
                data=df_data.copy()
            )

            data_frames.append(current_new_df)

        base += nr_episodes_in_dataset


    new_df = pd.concat(data_frames, axis=0)

    print(new_df.shape)
    print(new_df.head())

    new_df_path = "/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_robot_collision_dataset_fixed_augmented.parquet"
    new_df.to_parquet(new_df_path)

if __name__ == "__main__":
    offline_dataset_augmentation()
