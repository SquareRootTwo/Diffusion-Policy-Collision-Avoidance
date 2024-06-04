import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.spatial.transform import Rotation as R

# Add the path to the parent directory to augment search for module
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from src.dataset.dataset_loader import Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader

df_path: str = os.path.join(root_path, "src/data/curobo_panda_pick_and_place_robot_collision_dataset_fixed.parquet")

sns.set_theme(style="dark")

df = pd.read_parquet(df_path)
total_df_len = df['episode'].nunique()
episode_limit = int(total_df_len * 0.8)
train_df = df[df['episode'] < episode_limit]
train_df = train_df[train_df["step"] == 33]

eval_df = df[df['episode'] >= episode_limit]
eval_df = eval_df[eval_df["step"] == 33]

train_pick_x_points = (train_df[train_df["phase"] == 1]).iloc[:, 2].values
train_pick_y_points = (train_df[train_df["phase"] == 1]).iloc[:, 3].values
train_place_x_points = (train_df[train_df["phase"] == 2]).iloc[:, 2].values
train_place_y_points = (train_df[train_df["phase"] == 2]).iloc[:, 3].values

eval_pick_x_points = (eval_df[eval_df["phase"] == 1]).iloc[:, 2].values
eval_pick_y_points = (eval_df[eval_df["phase"] == 1]).iloc[:, 3].values
eval_place_x_points = (eval_df[eval_df["phase"] == 2]).iloc[:, 2].values
eval_place_y_points = (eval_df[eval_df["phase"] == 2]).iloc[:, 3].values


pick_y_min = -0.6
pick_y_max = -0.125
pick_x_min = 0.3
pick_x_max = 0.7

place_y_min = 0.125
place_y_max = 0.6
place_x_min = 0.3
place_x_max = 0.7

grid_shape = (1, 2)

fig = plt.figure()
ax1 = plt.subplot2grid(grid_shape, (0,0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid(grid_shape, (0,1), rowspan=1, colspan=1)

ax1.set_xlim(pick_x_min, pick_x_max)
ax1.set_ylim(pick_y_min, pick_y_max)

ax2.set_xlim(place_x_min, place_x_max)
ax2.set_ylim(place_y_min, place_y_max)

ax1.set_aspect('equal')
ax2.set_aspect('equal')

# plot train points
ax1.scatter(train_pick_x_points, train_pick_y_points, c="coral")
ax1.scatter(train_place_x_points, train_place_y_points, c="coral")

# plot test points
ax2.scatter(eval_pick_x_points, eval_pick_y_points, c="steelblue")
ax2.scatter(eval_place_x_points, eval_place_y_points, c="steelblue")

# save fig
plt.savefig("./eval/train_eval_split_area.png", dpi=300)