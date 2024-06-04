import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import seaborn as sns
import os
import sys
from glob import glob
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Add the path to the parent directory to augment search for module
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from src.utils.panda_kinematics import PandaKinematics

base_path = os.path.join(root_path, "src/data/thesis_eval/")

pk = PandaKinematics()

# list all directories in the base path
dirs = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
dirs = [
    os.path.join(base_path, "2024-04-18_00-49-51_unet_joint_space"),
    os.path.join(base_path, "2024-04-18_19-46-27_unet_cartesian_space"),
    os.path.join(base_path, "2024-04-22_00-04-23_transformer_joint_space"),
    os.path.join(base_path, "2024-04-26_02-07-12_transformer_cartesian_space"),
]

nr_eval_dirs = len(dirs)
nr_steps = 34
nr_episodes = 101

target_pos_indices = [f"obs_{i}" for i in range(14, 17)]
target_rot_indices = [f"obs_{i}" for i in range(17, 21)]
ee_pos_indices = [f"obs_{i}" for i in range(7, 10)]
ee_rot_indices = [f"obs_{i}" for i in range(10, 14)]

y_lables = []

# categorize target rotations into 4 categories: [0, 90), [90, 180), [180, 270), [270, 360)
pick_linear_dist_stats = [[] for _ in range(nr_eval_dirs)]
pick_angular_dist_stats = [[] for _ in range(nr_eval_dirs)]
pick_final_angle = [[] for _ in range(nr_eval_dirs)]
pick_target_x = [[] for _ in range(nr_eval_dirs)]
pick_target_y = [[] for _ in range(nr_eval_dirs)]

place_linear_dist_stats = [[] for _ in range(nr_eval_dirs)]
place_angular_dist_stats = [[] for _ in range(nr_eval_dirs)]
place_final_angle = [[] for _ in range(nr_eval_dirs)]
place_target_x = [[] for _ in range(nr_eval_dirs)]
place_target_y = [[] for _ in range(nr_eval_dirs)]

pick_z_rot = [[] for _ in range(nr_eval_dirs)]
place_z_rot = [[] for _ in range(nr_eval_dirs)]


df_path: str = os.path.join(root_path, "/data/curobo_panda_pick_and_place_robot_collision_dataset_fixed.parquet")

df_path = "/Users/jonasspieler/Documents/Code/curobodataset/src/data/curobo_panda_pick_and_place_robot_collision_dataset_fixed.parquet"
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

for i, eval_dir in tqdm(enumerate(dirs), desc="Evaluating directories", total=nr_eval_dirs, position=0, leave=True):
    
    if "2024" not in eval_dir:
        continue

    # y_label = eval_dir.split("_")[-2] + "_" + eval_dir.split("_")[-1]
    # y_label = eval_dir.split("_")[-3].capitalize() + " " +eval_dir.split("_")[-2].capitalize() + " " + (eval_dir.split("_")[-1]).capitalize()
    y_label = eval_dir.split("_")[-2].capitalize() + "-" + (eval_dir.split("_")[-1]).capitalize() + " " +  eval_dir.split("_")[-3].capitalize()
    y_lables.append(y_label)

    for eval_episode in tqdm(range(nr_episodes), desc="Evaluating episodes", total=nr_episodes, position=1, leave=False):
        eval_episode_path = os.path.join(base_path, eval_dir, f"episode_{eval_episode}_data.parquet")
        if os.path.exists(eval_episode_path):
            df = pd.read_parquet(eval_episode_path)

            # pick phase
            target_position = df[target_pos_indices].values[33]
            target_rotation = df[target_rot_indices].values[33]
            target_rot_scipy = np.array([target_rotation[3], target_rotation[0], target_rotation[1], target_rotation[2]])
            target_z_rot = R.from_quat(target_rot_scipy).as_euler('zyx')[0]
            pick_z_rot[i].append(target_z_rot)

            # category = get_category(target_z_rot)

            angle_dist = df["angle_dist"].values[33]
            linear_dist = df["ee_pos_dist"].values[33]

            final_ee_rotation = df[ee_rot_indices].values[33]
            final_ee_rot_scipy = np.array([final_ee_rotation[3], final_ee_rotation[0], final_ee_rotation[1], final_ee_rotation[2]])
            final_ee_z_rot = R.from_quat(final_ee_rot_scipy).as_euler('zyx')[0]

            pick_final_angle[i].append(final_ee_z_rot)

            pick_linear_dist_stats[i].append(linear_dist)
            pick_angular_dist_stats[i].append(angle_dist)
            x_pos = target_position[0]
            y_pos = target_position[1]
            pick_target_x[i].append(x_pos)
            pick_target_y[i].append(y_pos)

            # place phase
            target_position = df[target_pos_indices].values[67]
            target_rotation = df[target_rot_indices].values[67]
            target_rot_scipy = np.array([target_rotation[3], target_rotation[0], target_rotation[1], target_rotation[2]])
            target_z_rot = R.from_quat(target_rot_scipy).as_euler('zyx')[0]
            place_z_rot[i].append(target_z_rot)
            
            # category = get_category(target_z_rot)

            final_ee_rotation = df[ee_rot_indices].values[67]
            final_ee_rot_scipy = np.array([final_ee_rotation[3], final_ee_rotation[0], final_ee_rotation[1], final_ee_rotation[2]])
            final_ee_z_rot = R.from_quat(final_ee_rot_scipy).as_euler('zyx')[0]

            place_final_angle[i].append(final_ee_z_rot)

            angle_dist = df["angle_dist"].values[67]
            linear_dist = df["ee_pos_dist"].values[67]
            x_pos = target_position[0]
            y_pos = target_position[1]

            place_linear_dist_stats[i].append(linear_dist)
            place_angular_dist_stats[i].append(angle_dist)
            place_target_x[i].append(x_pos)
            place_target_y[i].append(y_pos)


# nr_eval_dirs plots
fig, axs = plt.subplots(nr_eval_dirs, 1, figsize=(14, 5*nr_eval_dirs))

# plot train datapoints 
for i in range(nr_eval_dirs):

    # add vertical label left to the plot
    axs[i].text(-0.08, 0.5, y_lables[i], fontsize=18, ha='center', va='center', rotation='vertical', transform=axs[i].transAxes)

    sns.scatterplot(
        y=train_pick_x_points, 
        x=train_pick_y_points,
        color='white',
        s=120,
        ax=axs[i],
        marker="."
    )

    sns.scatterplot(
        y=train_place_x_points, 
        x=train_place_y_points,
        color='white',
        s=120,
        ax=axs[i],
        marker=".",
    )

    # mark the pick and place area as lines on plot
    axs[i].plot([pick_y_min, pick_y_min], [pick_x_min, pick_x_max], color='white', linewidth=1.5)
    axs[i].plot([pick_y_max, pick_y_max], [pick_x_min, pick_x_max], color='white', linewidth=1.5)
    axs[i].plot([pick_y_min, pick_y_max], [pick_x_min, pick_x_min], color='white', linewidth=1.5)
    axs[i].plot([pick_y_min, pick_y_max], [pick_x_max, pick_x_max], color='white', linewidth=1.5)

    axs[i].plot([place_y_min, place_y_min], [place_x_min, place_x_max], color='white', linewidth=1.5)
    axs[i].plot([place_y_max, place_y_max], [place_x_min, place_x_max], color='white', linewidth=1.5)
    axs[i].plot([place_y_min, place_y_max], [place_x_min, place_x_min], color='white', linewidth=1.5)
    axs[i].plot([place_y_min, place_y_max], [place_x_max, place_x_max], color='white', linewidth=1.5)


# Compute the global minimum and maximum
global_min = np.min([np.min(x) for x in [pick_linear_dist_stats, place_linear_dist_stats]])
global_max = np.max([np.max(x) for x in [pick_linear_dist_stats, place_linear_dist_stats]])
# Create a normalization object
from matplotlib.colors import Normalize
norm = Normalize(vmin=global_min, vmax=global_max)

# iterate over categories
palette = sns.color_palette("magma", n_colors=50)
# palette = sns.color_palette("ch:s=-.2,r=.6", n_colors=20)
# invert the palette
palette = palette[::-1]
cmap = mcolors.LinearSegmentedColormap.from_list("", palette)

for i in range(nr_eval_dirs):
    if len(pick_linear_dist_stats)>0:
        # pick area linear distance
        sns.scatterplot(
            y=pick_target_x[i], 
            x=pick_target_y[i],
            hue=pick_linear_dist_stats[i],
            s=350,
            linewidth=0,
            hue_norm=norm,
            palette=cmap,
            ax=axs[i],
            legend=False
        )

        # final ee z rotation
        u = 0.015*np.cos(pick_final_angle[i])
        v = 0.015*np.sin(pick_final_angle[i])
        axs[i].plot([pick_target_y[i], pick_target_y[i] + u], [pick_target_x[i], pick_target_x[i] + v], 
                    color='springgreen', linewidth=2)

        # target z rotation
        u = 0.015*np.cos(pick_z_rot[i])
        v = 0.015*np.sin(pick_z_rot[i])
        axs[i].plot([pick_target_y[i], pick_target_y[i] + u], [pick_target_x[i], pick_target_x[i] + v], 
                    color='midnightblue', linewidth=2)


    if len(place_linear_dist_stats)>0:
        # place area linear distance
        sns.scatterplot(
            y=place_target_x[i], 
            x=place_target_y[i],
            hue=place_linear_dist_stats[i],
            linewidth=0,
            s=350,
            hue_norm=norm,
            palette=cmap,
            ax=axs[i],
            legend=False
        )

        # final ee z rotation
        u = 0.015*np.cos(place_final_angle[i]) 
        v = 0.015*np.sin(place_final_angle[i])
        axs[i].plot([place_target_y[i], place_target_y[i] + u], [place_target_x[i], place_target_x[i] + v], 
                    color='springgreen', linewidth=2)
        # target z rotation
        u = 0.015*np.cos(place_z_rot[i])
        v = 0.015*np.sin(place_z_rot[i])
        axs[i].plot([place_target_y[i], place_target_y[i] + u], [place_target_x[i], place_target_x[i] + v], 
                    color='midnightblue', linewidth=2)

# set aspect ratio
for i in range(nr_eval_dirs):
    axs[i].set_aspect('equal', adjustable='box')

# add one legend with color map for the overall figure
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, orientation='horizontal',  
                    label='Final EE Distance', shrink=0.5, pad=40
)
# set cbar abs position
cbar.ax.set_position([0.4, 0.025, 0.2, 0.05])
cbar.set_label('Final EE Distance', labelpad=20, fontsize=24)

# set figure title
fig.suptitle('Final EE Distance per Target Position', fontsize=32, y=0.95)

# add legend for final ee rotation and target rotation

fig.tight_layout()
plt.subplots_adjust(left=0.025, right=0.975, top=0.9, bottom=0.1)


legend_elements = [
    Line2D([0], [0], marker=None, color="springgreen", label='Final EE Z Rotation'),
    Line2D([0], [0], marker=None, color="midnightblue", label='Target Z Rotation'),
    Line2D([0], [0], marker='o', color="w", linewidth=0, label='Train Data Points', markerfacecolor='w', markersize=10) 
]

# Add the legend to the figure
plt.legend(handles=legend_elements, loc=(0.7, -0.4), fontsize=14)


out_path = os.path.join(base_path, "eval_area_precision.png")

plt.savefig(out_path, dpi=300)
