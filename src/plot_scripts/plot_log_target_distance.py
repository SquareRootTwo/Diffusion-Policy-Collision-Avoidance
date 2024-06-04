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

    print(len(model_df))

    # plot linear distance
    sns.lineplot(
        data=model_df, 
        x="Steps", 
        y="Linear Distance", 
        hue="Phase", 
        ax=ax[0],
        palette=palette,
        errorbar="sd"
    )

    print("Linear dist plotted")

    # plot angle distance
    sns.lineplot(
        data=model_df, 
        x="Steps", 
        y="Angular Distance", 
        hue="Phase", 
        ax=ax[1],
        palette=palette,
        errorbar="sd"
    )

    print("Angular dist plotted")

    # set log y axis
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    # add y axis grid
    ax[0].grid(axis='y')
    ax[1].grid(axis='y')

    # remove legend
    ax[0].get_legend().remove()
    ax[1].get_legend().remove()

    # manually add one legend for both subplots
    handles, labels = ax[0].get_legend_handles_labels()

    # cange labels
    labels = [f"Pick Phase (final: {100*np.mean(final_dist[0]):.2f} cm)", f"Place Phase (final: {100*np.mean(final_dist[1]):.2f} cm)", f"Retract Phase (final: {100*np.mean(final_dist[2]):.2f} cm)"]
    ax[0].legend(handles, labels, loc=(0.05, 0.05))

    labels = [f"Pick Phase (final: {np.rad2deg(np.mean(final_dist_ang[0])):.2f}$\\degree$)", f"Place Phase (final: {np.rad2deg(np.mean(final_dist_ang[1])):.2f}$\\degree$)", f"Retract Phase (final: {np.rad2deg(np.mean(final_dist_ang[2])):.2f}$\\degree$)"]
    ax[1].legend(handles, labels, loc=(0.05, 0.05))

    # set title
    # model = eval_dir.split("_")[-3].capitalize() + " " + eval_dir.split("_")[-2].capitalize() + " " + (eval_dir.split("_")[-1]).capitalize()
    model = eval_dir.split("_")[-2].capitalize() + "-" + (eval_dir.split("_")[-1]).capitalize() + " " +  eval_dir.split("_")[-3].capitalize()
    fig.suptitle(f"End-Effector Distance to Target\n{model}", fontsize=20, y=0.95)

    # move y axis labels to the right
    ax[0].yaxis.tick_right()
    ax[1].yaxis.tick_right()

    # move y axis description to the right
    ax[0].yaxis.set_label_position("right")
    ax[1].yaxis.set_label_position("right")

    # rotate y axis description
    ax[0].set_ylabel("Distance [m]", rotation=270, labelpad=20)
    ax[1].set_ylabel("Angular Distance [deg]", rotation=270, labelpad=20)


    # remove border lines on top on right
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['left'].set_visible(False)

    fig.tight_layout()
    
    # add margin
    fig.subplots_adjust(
        top=0.85,
        bottom=0.1,
        left=0.12,
        right=0.88,    
    )

    # save plot
    model = eval_dir.split("_")[-3] + "_" + eval_dir.split("_")[-2] + "_" + eval_dir.split("_")[-1]
    # model = eval_dir.split("_")[-2].capitalize() + "-" + (eval_dir.split("_")[-1]).capitalize() + " " +  eval_dir.split("_")[-3].capitalize()
    out_path = os.path.join(base_path, f"log_distance_{model}.png")
    print(out_path)
    fig.savefig(out_path, dpi=300)

    model_nr = np.ones(len(model_df), dtype=int) * int(i)

    model_df["Model"] = model_nr

    all_model_data.append(model_df)


all_model_data = pd.concat(all_model_data)
fig.clf()
fig, ax = plt.subplots(3, 2, figsize=(7, 9))
palette = sns.color_palette("colorblind", n_colors=nr_eval_dirs)
palette = palette[::-1]
# color_map = ["seagreen","lightblue","darkorange","gold"]
# palette = sns.color_palette(color_map)


all_model_pick_df = all_model_data[all_model_data["Phase"] == 1]
all_model_place_df = all_model_data[all_model_data["Phase"] == 2]
all_model_retract_df = all_model_data[all_model_data["Phase"] == 3]

# plot linear distance
sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Linear Distance", 
    hue="Model", 
    ax=ax[0, 0],
    palette=palette,
    errorbar=("sd", 0)
)

ax[0, 0].set_title("Pick Phase", pad=.3)

# plot angle distance
sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Angular Distance", 
    hue="Model", 
    ax=ax[0, 1],
    palette=palette,
    errorbar=("sd", 0)
)

ax[0, 1].set_title("Pick Phase", pad=.3)

# plot linear distance
sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Linear Distance", 
    hue="Model", 
    ax=ax[1, 0],
    palette=palette,
    errorbar=("sd", 0)
)

ax[1, 0].set_title("Place Phase", pad=.3)

# plot angle distance
sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Angular Distance", 
    hue="Model", 
    ax=ax[1, 1],
    palette=palette,
    errorbar=("sd", 0)
)

ax[1, 1].set_title("Place Phase", pad=.3)

# plot linear distance
sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Linear Distance", 
    hue="Model",
    ax=ax[2, 0],
    palette=palette,
    errorbar=("sd", 0)
)

ax[2, 0].set_title("Retract Phase", pad=.3)

# plot angle distance
sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Angular Distance", 
    hue="Model", 
    ax=ax[2, 1],
    palette=palette,
    errorbar=("sd", 0)
)

ax[2, 1].set_title("Retract Phase", pad=.3)

# set log y axis
for i in range(3):
    for j in range(2):
        ax[i, j].set_yscale("log")
        ax[i, j].grid(axis='y')
        ax[i, j].yaxis.tick_right()
        ax[i, j].spines['top'].set_visible(False)
        ax[i, j].spines['left'].set_visible(False)
        ax[i, j].spines['top'].set_visible(False)
        ax[i, j].spines['left'].set_visible(False)

        # add padding
        ax[i, j].yaxis.labelpad = 20

        # remove legend
        ax[i, j].get_legend().remove()

ax[0, 1].yaxis.set_label_position("right")
ax[1, 1].yaxis.set_label_position("right")
ax[2, 1].yaxis.set_label_position("right")

# rotate y axis description
ax[0, 0].set_ylabel("Distance [m]", rotation=90, labelpad=20)
ax[0, 1].set_ylabel("Angular Distance [deg]", rotation=270, labelpad=20)
ax[1, 0].set_ylabel("Distance [m]", rotation=90, labelpad=20)
ax[1, 1].set_ylabel("Angular Distance [deg]", rotation=270, labelpad=20)
ax[2, 0].set_ylabel("Distance [m]", rotation=90, labelpad=20)
ax[2, 1].set_ylabel("Angular Distance [deg]", rotation=270, labelpad=20)

# reduce number of x ticks displayed
for i in range(3):
    for j in range(2):
        ax[i, j].set_xticks(np.arange(0, 34, 5))

linear_y_min = all_model_place_df['Linear Distance'].min()
linear_y_min = min(all_model_pick_df['Linear Distance'].min(), linear_y_min)
linear_y_min = min(all_model_retract_df['Linear Distance'].min(), linear_y_min)

linear_y_max = all_model_place_df['Linear Distance'].max()
linear_y_max = max(all_model_pick_df['Linear Distance'].max(), linear_y_max)
linear_y_max = max(all_model_retract_df['Linear Distance'].max(), linear_y_max)

angular_y_min = all_model_place_df['Angular Distance'].min()
angular_y_min = min(all_model_pick_df['Angular Distance'].min(), angular_y_min)
angular_y_min = min(all_model_retract_df['Angular Distance'].min(), angular_y_min)

angular_y_max = all_model_place_df['Angular Distance'].max()
angular_y_max = max(all_model_pick_df['Angular Distance'].max(), angular_y_max)
angular_y_max = max(all_model_retract_df['Angular Distance'].max(), angular_y_max)

print("linear min, linear_y_min")
print(linear_y_min, linear_y_max)
print("angular min, angular_y_max")
print(angular_y_min, angular_y_max)

for i in range(3):
    ax[i, 0].set_ylim(linear_y_min, linear_y_max)

for i in range(3):
    ax[i, 1].set_ylim(angular_y_min, angular_y_max)

# manually add one legend for both subplots
handles, labels = ax[0, 0].get_legend_handles_labels()

ax[0,0].set_xlabel("")
ax[0,1].set_xlabel("")
ax[1,0].set_xlabel("")
ax[1,1].set_xlabel("")


# cange labels
labels = y_lables

fig.legend(handles, labels, loc="lower center", ncols=2)
# tighten layout
fig.tight_layout()

# add margin
fig.subplots_adjust(
    top=0.85,
    bottom=0.125,
    left=0.15,
    right=0.85,    
)


# set title
fig.suptitle(f"End-Effector Distance to Target: All Models", fontsize=22, y=0.94)

# save plot
out_path = os.path.join(base_path, f"log_distance_all_models.png")
print(out_path)
fig.savefig(out_path, dpi=300)
