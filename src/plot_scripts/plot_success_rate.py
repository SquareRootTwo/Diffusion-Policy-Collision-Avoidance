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

# Add the path to the parent directory to augment search for module
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from src.utils.panda_kinematics import PandaKinematics

base_path = os.path.join(root_path, "src/data/thesis_eval/")

pk = PandaKinematics()

# list all directories in the base path
# dirs = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

dirs = [
    os.path.join(base_path, "2024-04-26_02-07-12_transformer_cartesian_space"),
    os.path.join(base_path, "2024-04-22_00-04-23_transformer_joint_space"),
    os.path.join(base_path, "2024-04-18_19-46-27_unet_cartesian_space"),
    os.path.join(base_path, "2024-04-18_00-49-51_unet_joint_space"),

]

nr_eval_dirs = len(dirs)
nr_steps = 34
nr_episodes = 101

heatmap = np.zeros((nr_eval_dirs, nr_steps))

target_pos_indices = [f"obs_{i}" for i in range(14, 17)]
target_rot_indices = [f"obs_{i}" for i in range(17, 21)]
ee_pos_indices = [f"obs_{i}" for i in range(7, 10)]
ee_rot_indices = [f"obs_{i}" for i in range(10, 14)]

y_lables = []

# plot

# 3 x 2 subplots
fig, ax = plt.subplots(2, 1, figsize=(15, 10))
sns.set_theme(style="ticks")
palette = sns.color_palette("tab10", n_colors=nr_eval_dirs)
# color_map = ["gold","darkorange","lightblue","seagreen",]
# color_map = ["darkorange","seagreen"]
# palette = sns.color_palette(color_map)

# all model data
all_model_data = []

for i, eval_dir in tqdm(enumerate(dirs), desc="Evaluating directories", total=nr_eval_dirs, position=0, leave=True):
    
    # y_label = eval_dir.split("_")[-3] + "_" + eval_dir.split("_")[-2] + "_" + eval_dir.split("_")[-1]

    if "2024" not in eval_dir:
        continue

    y_label = eval_dir.split("_")[-2].capitalize() + "-" + (eval_dir.split("_")[-1]).capitalize() + " " + eval_dir.split("_")[-3].capitalize()
    y_lables.append(y_label)

    fig.clf()
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))

    # add horizontal 

    model_df = []

    final_dist = [[], [], []]
    final_dist_ang = [[], [], []]
    for eval_episode in tqdm(range(nr_episodes), desc="Evaluating episodes", total=nr_episodes, position=1, leave=False):
        eval_episode_path = os.path.join(base_path, eval_dir, f"episode_{eval_episode}_data.parquet")
        if os.path.exists(eval_episode_path):


            df = pd.read_parquet(eval_episode_path)
            for phase in range(1,4):
                phase_df = df[df["phase"] == phase]

                angle_dist = np.absolute(phase_df["angle_dist"].values)
                linear_dist = np.absolute(phase_df["ee_pos_dist"].values)

                final_dist[phase-1].append(linear_dist[-1])
                final_dist_ang[phase-1].append(angle_dist[-1])

                steps = np.arange(0, 34)
                phase_array = int(phase) * np.ones(34, dtype=int)

                model_df.append(pd.DataFrame({
                    "Angular Distance": np.rad2deg(angle_dist), 
                    "Linear Distance": linear_dist, 
                    "Steps": steps, 
                    "Phase": phase_array})
                )

    model_df = pd.concat(model_df)
    print(model_df.head())

    pick_linear_succ = np.array(final_dist[0]) < 0.01
    pick_angular_succ = np.rad2deg(np.array(final_dist_ang[0])) < 15

    pick_combined_succ = np.logical_and(pick_linear_succ, pick_angular_succ)

    place_linear_succ = np.array(final_dist[1]) < 0.01
    place_angular_succ = np.rad2deg(np.array(final_dist_ang[1])) < 15

    place_combined_succ = np.logical_and(place_linear_succ, place_angular_succ)

    retract_linear_succ = np.array(final_dist[2]) < 0.01
    retract_angular_succ = np.rad2deg(np.array(final_dist_ang[2])) < 15

    retract_combined_succ = np.logical_and(retract_linear_succ, retract_angular_succ)

    pick_nr_distance_below_thresh = np.sum(pick_combined_succ == True)
    place_nr_distance_below_thresh = np.sum(place_combined_succ == True)
    retract_nr_distance_below_thresh = np.sum(retract_combined_succ == True)

    pick_success_rate = pick_nr_distance_below_thresh / nr_episodes
    place_success_rate = place_nr_distance_below_thresh / nr_episodes
    retract_success_rate = retract_nr_distance_below_thresh / nr_episodes

    model = eval_dir.split("_")[-3] + "_" + eval_dir.split("_")[-2] + "_" + eval_dir.split("_")[-1]
    print(f"{model} Pick s rate: {pick_success_rate:.4f}, Place s rate: {place_success_rate:.4f}, Retract s rate: {retract_success_rate:.4f}")

