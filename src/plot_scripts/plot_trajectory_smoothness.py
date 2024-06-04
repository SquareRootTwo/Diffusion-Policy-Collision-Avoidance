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
from matplotlib.lines import Line2D

# Add the path to the parent directory to augment search for module
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from src.utils.panda_kinematics import PandaKinematics

base_path = os.path.join(root_path, "src/data/thesis_eval/")

pk = PandaKinematics()

# list all directories in the base path
# dirs = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
dirs = [
    # os.path.join(base_path, "2024-04-18_00-49-51_unet_joint_space"),
    # os.path.join(base_path, "2024-04-18_19-46-27_unet_cartesian_space"),
    # os.path.join(base_path, "2024-04-26_02-07-12_transformer_cartesian_space"),
    # os.path.join(base_path, "2024-04-22_00-04-23_transformer_joint_space"),

    os.path.join(base_path, "2024-04-18_19-46-27_unet_cartesian_space"),
    os.path.join(base_path, "2024-04-26_02-07-12_transformer_cartesian_space"),

    # os.path.join(base_path, "2024-04-18_00-49-51_unet_joint_space"),
    # os.path.join(base_path, "2024-04-22_00-04-23_transformer_joint_space"),

    # os.path.join(base_path, "2024-04-22_00-04-23_transformer_joint_space"),
    # os.path.join(base_path, "2024-04-22_00-04-23_transformer_trajectory_interpolation"),
]
nr_eval_dirs = len(dirs)
nr_steps = 34
nr_episodes = 101

heatmap = np.zeros((nr_eval_dirs, nr_steps))

# target_pos_indices = [f"obs_{i}" for i in range(14, 17)]
# target_rot_indices = [f"obs_{i}" for i in range(17, 21)]
# ee_pos_indices = [f"obs_{i}" for i in range(7, 10)]
# ee_rot_indices = [f"obs_{i}" for i in range(10, 14)]
joint_indices = [f"obs_{i}" for i in range(0, 7)]
gt_joint_indices = [f"gt_obs_{i}" for i in range(0, 7)]

joint_limits = {
    "Joint 1": (-2.8973, 2.8973),
    "Joint 2": (-1.7628, 1.7628),
    "Joint 3": (-2.8973, 2.8973),
    "Joint 4": (-3.0718, -0.0698),
    "Joint 5": (-2.8973, 2.8973),
    "Joint 6": (-0.0175, 3.7525),
    "Joint 7": (-2.8973, 2.8973),
}

joint_vel_limit = {
    "Joint 1": 2.1750,
    "Joint 2": 2.1750,
    "Joint 3": 2.1750,
    "Joint 4": 2.1750,
    "Joint 5": 2.61,
    "Joint 6": 2.61,
    "Joint 7": 2.61,
}


joint_acc_limit = {
    "Joint 1": 15,
    "Joint 2": 7.5,
    "Joint 3": 10,
    "Joint 4": 12.5,
    "Joint 5": 15,
    "Joint 6": 20,
    "Joint 7": 20,
}

joint_jerk_limit = {
    "Joint 1": 7500,
    "Joint 2": 3750,
    "Joint 3": 5000,
    "Joint 4": 6250,
    "Joint 5": 7500,
    "Joint 6": 10000,
    "Joint 7": 10000
}


y_lables = []
model_names = []

# plot

# assume trajectory execution time is 2 seconds
unet_exec_time: float = 0.099 # total time: 9.7865996361 s
transformer_exec_time: float = 0.070 # total time: 6.9779083729 s


# dt = 2.0 / 34.0
# 3 x 2 subplots
fig, ax = plt.subplots(2, 1, figsize=(15, 10))
sns.set_theme(style="ticks")
palette = sns.color_palette("colorblind", n_colors=len(dirs))
gt_palette = sns.color_palette("viridis", n_colors=5)

palette = sns.color_palette("husl", n_colors=6)
# palette = sns.cubehelix_palette(start=.5, rot=-.75, n_colors=100)

# all model data
all_model_data = []

for i, eval_dir in tqdm(enumerate(dirs), desc="Evaluating directories", total=nr_eval_dirs, position=0, leave=True):
    
    model_name = eval_dir.split("_")[-2].capitalize() + "-" + eval_dir.split("_")[-1].capitalize() + " " + eval_dir.split("_")[-3].capitalize() 
    model_names.append(model_name)
    
    y_label = eval_dir.split("_")[-3].capitalize() + "_" + eval_dir.split("_")[-2].capitalize() + "_" + eval_dir.split("_")[-1].capitalize()
    
    y_lables.append(y_label)

    if "unet" in y_label.lower():
        print(f"evaluate unet: {y_label}")
        dt = unet_exec_time
    else:
        print(f"evaluate transformer: {y_label}")
        dt = transformer_exec_time 

    # add horizontal 


    model_df = []
    if "2024" not in eval_dir:
        continue
    
    for eval_episode in tqdm(range(nr_episodes), desc="Evaluating episodes", total=nr_episodes, position=1, leave=False):
        eval_episode_path = os.path.join(base_path, eval_dir, f"episode_{eval_episode}_data.parquet")

        if os.path.exists(eval_episode_path):
            df = pd.read_parquet(eval_episode_path)
            for phase in range(1,4):
                phase_df = df[df["phase"] == phase]

                joint_positions = phase_df[joint_indices].values

                vel = np.diff(joint_positions, axis=0) / dt
                vel = np.pad(vel, ((1, 0), (0, 0)), mode='edge')
                
                acc = np.diff(vel, axis=0) / dt
                acc = np.pad(acc, ((1, 0), (0, 0)), mode='edge')
                
                jerk = np.diff(acc, axis=0) / dt
                jerk = np.pad(jerk, ((1, 0), (0, 0)), mode='edge')

                vel_mean = np.mean(vel, axis=1)
                acc_mean = np.mean(acc, axis=1)
                jerk_mean = np.mean(jerk, axis=1)

                vel_max_abs = np.max(np.abs(vel),  axis=1)
                acc_max_abs = np.max(np.abs(acc), axis=1)
                jerk_max_abs = np.max(np.abs(jerk), axis=1)

                gt_vel = np.diff(phase_df[gt_joint_indices].values, axis=0) / dt
                gt_vel = np.pad(gt_vel, ((1, 0), (0, 0)), mode='edge')

                gt_acc = np.diff(gt_vel, axis=0) / dt
                gt_acc = np.pad(gt_acc, ((1, 0), (0, 0)), mode='edge')

                gt_jerk = np.diff(gt_acc, axis=0) / dt
                gt_jerk = np.pad(gt_jerk, ((1, 0), (0, 0)), mode='edge')

                gt_vel_mean = np.mean(gt_vel, axis=1)
                gt_acc_mean = np.mean(gt_acc, axis=1)
                gt_jerk_mean = np.mean(gt_jerk, axis=1)

                gt_vel_max_abs = np.max(np.abs(gt_vel), axis=1)
                gt_acc_max_abs = np.max(np.abs(gt_acc), axis=1)
                gt_jerk_max_abs = np.max(np.abs(gt_jerk), axis=1)



                steps = np.arange(0, 34)
                phase_array = int(phase) * np.ones(34, dtype=int)

                model_df.append(pd.DataFrame({
                    "Jerk": jerk_mean,
                    "Jerk Max Abs": jerk_max_abs,
                    "Jerk Joint 1": jerk[:, 0],
                    "Jerk Joint 2": jerk[:, 1],
                    "Jerk Joint 3": jerk[:, 2],
                    "Jerk Joint 4": jerk[:, 3],
                    "Jerk Joint 5": jerk[:, 4],
                    "Jerk Joint 6": jerk[:, 5],
                    "Jerk Joint 7": jerk[:, 6],
                    "Velocity": vel_mean,
                    "Velocity Max Abs": vel_max_abs,
                    "Velocity Joint 1": vel[:, 0],
                    "Velocity Joint 2": vel[:, 1],
                    "Velocity Joint 3": vel[:, 2],
                    "Velocity Joint 4": vel[:, 3],
                    "Velocity Joint 5": vel[:, 4],
                    "Velocity Joint 6": vel[:, 5],
                    "Velocity Joint 7": vel[:, 6],
                    "Acceleration": acc_mean,
                    "Acceleration Max Abs": acc_max_abs,
                    "Acceleration Joint 1": acc[:, 0],
                    "Acceleration Joint 2": acc[:, 1],
                    "Acceleration Joint 3": acc[:, 2],
                    "Acceleration Joint 4": acc[:, 3],
                    "Acceleration Joint 5": acc[:, 4],
                    "Acceleration Joint 6": acc[:, 5],
                    "Acceleration Joint 7": acc[:, 6],
                    "Ground Truth Jerk": gt_jerk_mean,
                    "Ground Truth Jerk Max Abs": gt_jerk_max_abs,
                    "Ground Truth Jerk Joint 1": gt_jerk[:, 0],
                    "Ground Truth Jerk Joint 2": gt_jerk[:, 1],
                    "Ground Truth Jerk Joint 3": gt_jerk[:, 2],
                    "Ground Truth Jerk Joint 4": gt_jerk[:, 3],
                    "Ground Truth Jerk Joint 5": gt_jerk[:, 4],
                    "Ground Truth Jerk Joint 6": gt_jerk[:, 5],
                    "Ground Truth Jerk Joint 7": gt_jerk[:, 6],
                    "Ground Truth Velocity": gt_vel_mean,
                    "Ground Truth Velocity Max Abs": gt_vel_max_abs,
                    "Ground Truth Velocity Joint 1": gt_vel[:, 0],
                    "Ground Truth Velocity Joint 2": gt_vel[:, 1],
                    "Ground Truth Velocity Joint 3": gt_vel[:, 2],
                    "Ground Truth Velocity Joint 4": gt_vel[:, 3],
                    "Ground Truth Velocity Joint 5": gt_vel[:, 4],
                    "Ground Truth Velocity Joint 6": gt_vel[:, 5],
                    "Ground Truth Velocity Joint 7": gt_vel[:, 6],
                    "Ground Truth Acceleration": gt_acc_mean,
                    "Ground Truth Acceleration Max Abs": gt_acc_max_abs,
                    "Ground Truth Acceleration Joint 1": gt_acc[:, 0],
                    "Ground Truth Acceleration Joint 2": gt_acc[:, 1],
                    "Ground Truth Acceleration Joint 3": gt_acc[:, 2],
                    "Ground Truth Acceleration Joint 4": gt_acc[:, 3],
                    "Ground Truth Acceleration Joint 5": gt_acc[:, 4],
                    "Ground Truth Acceleration Joint 6": gt_acc[:, 5],
                    "Ground Truth Acceleration Joint 7": gt_acc[:, 6],
                    "Steps": steps, 
                    "Phase": phase_array})
                )


    model_df = pd.concat(model_df)
    print(model_df.head())

    pick_df = model_df[model_df["Phase"] == 1]
    place_df = model_df[model_df["Phase"] == 2]
    retract_df = model_df[model_df["Phase"] == 3]

    print(len(model_df))

    # cange labels
    legend_elements = [
        Line2D([0], [0], marker=None, color="steelblue", label='Ground Truth'),
        Line2D([0], [0], marker=None, color="darkorange", label='Model Trajectory'),
    ]

    postfix = ["", " Max Abs", " Joint 1", " Joint 2", " Joint 3", " Joint 4", " Joint 5", " Joint 6", " Joint 7"]
    curr_model_base_path = os.path.join(base_path, f"{y_label}_smoothness")
    os.makedirs(curr_model_base_path, exist_ok=True)


    for pi in postfix:
        fig.clf()
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        # Add the legend to the figure
        # hack to get the legend below the grid plot
        ax[2, 1].legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.7), ncols=2)

        if "Joint" in pi:
            # plot joint limits for vel
            joint_key = pi[1:]
            print(f"plot joint limits for {joint_key}")
            ax[0, 0].axhline(y=joint_vel_limit[joint_key], color="tab:red", linestyle="--")
            ax[0, 0].axhline(y=-joint_vel_limit[joint_key], color="tab:red", linestyle="--")
            ax[1, 0].axhline(y=joint_vel_limit[joint_key], color="tab:red", linestyle="--")
            ax[1, 0].axhline(y=-joint_vel_limit[joint_key], color="tab:red", linestyle="--")
            ax[2, 0].axhline(y=joint_vel_limit[joint_key], color="tab:red", linestyle="--")
            ax[2, 0].axhline(y=-joint_vel_limit[joint_key], color="tab:red", linestyle="--")

            # plot joint limits for acc
            ax[0, 1].axhline(y=joint_acc_limit[joint_key], color="tab:red", linestyle="--")
            ax[0, 1].axhline(y=-joint_acc_limit[joint_key], color="tab:red", linestyle="--")
            ax[1, 1].axhline(y=joint_acc_limit[joint_key], color="tab:red", linestyle="--")
            ax[1, 1].axhline(y=-joint_acc_limit[joint_key], color="tab:red", linestyle="--")
            ax[2, 1].axhline(y=joint_acc_limit[joint_key], color="tab:red", linestyle="--")
            ax[2, 1].axhline(y=-joint_acc_limit[joint_key], color="tab:red", linestyle="--")

            # plot joint limits for jerk
            ax[0, 2].axhline(y=joint_jerk_limit[joint_key], color="tab:red", linestyle="--")
            ax[0, 2].axhline(y=-joint_jerk_limit[joint_key], color="tab:red", linestyle="--")
            ax[1, 2].axhline(y=joint_jerk_limit[joint_key], color="tab:red", linestyle="--")
            ax[1, 2].axhline(y=-joint_jerk_limit[joint_key], color="tab:red", linestyle="--")
            ax[2, 2].axhline(y=joint_jerk_limit[joint_key], color="tab:red", linestyle="--")
            ax[2, 2].axhline(y=-joint_jerk_limit[joint_key], color="tab:red", linestyle="--")

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Ground Truth Velocity" + pi, 
            # hue="Phase", 
            ax=ax[0,0],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Ground Truth Acceleration" + pi, 
            # hue="Phase", 
            ax=ax[0, 1],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Ground Truth Jerk" + pi, 
            # hue="Phase", 
            ax=ax[0,2],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Velocity" + pi, 
            # hue="Phase", 
            ax=ax[0,0],
            color="darkorange",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Acceleration" + pi, 
            # hue="Phase", 
            ax=ax[0,1],
            color="darkorange",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Jerk" + pi, 
            # hue="Phase", 
            ax=ax[0,2],
            color="darkorange",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=place_df, 
            x="Steps", 
            y="Ground Truth Velocity" + pi, 
            # hue="Phase", 
            ax=ax[1,0],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=place_df, 
            x="Steps", 
            y="Ground Truth Acceleration" + pi, 
            # hue="Phase", 
            ax=ax[1,1],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=place_df, 
            x="Steps", 
            y="Ground Truth Jerk" + pi, 
            # hue="Phase", 
            ax=ax[1,2],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=place_df, 
            x="Steps", 
            y="Velocity" + pi, 
            # hue="Phase", 
            ax=ax[1,0],
            color="darkorange",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=place_df, 
            x="Steps", 
            y="Acceleration" + pi, 
            # hue="Phase", 
            ax=ax[1,1],
            color="darkorange",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=place_df, 
            x="Steps", 
            y="Jerk" + pi, 
            # hue="Phase", 
            ax=ax[1,2],
            color="darkorange",
            errorbar=("pi", 100)
        )


        sns.lineplot(
            data=retract_df, 
            x="Steps", 
            y="Ground Truth Velocity" + pi, 
            # hue="Phase", 
            ax=ax[2,0],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=retract_df, 
            x="Steps", 
            y="Ground Truth Acceleration" + pi, 
            # hue="Phase", 
            ax=ax[2,1],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=retract_df, 
            x="Steps", 
            y="Ground Truth Jerk" + pi, 
            # hue="Phase", 
            ax=ax[2,2],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=retract_df, 
            x="Steps", 
            y="Velocity" + pi, 
            # hue="Phase", 
            ax=ax[2,0],
            color="darkorange",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=retract_df, 
            x="Steps", 
            y="Acceleration" + pi, 
            # hue="Phase", 
            ax=ax[2,1],
            color="darkorange",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=retract_df, 
            x="Steps", 
            y="Jerk" + pi, 
            # hue="Phase", 
            ax=ax[2,2],
            color="darkorange",
            errorbar=("pi", 100)
        )

        # add y axis grid
        ax[0, 0].grid(axis='y')
        ax[0, 1].grid(axis='y')
        ax[0, 2].grid(axis='y')
        ax[1, 0].grid(axis='y')
        ax[1, 1].grid(axis='y')
        ax[1, 2].grid(axis='y')
        ax[2, 0].grid(axis='y')
        ax[2, 1].grid(axis='y')
        ax[2, 2].grid(axis='y')

        # set title
        model = eval_dir.split("_")[-2].capitalize() + "-" + (eval_dir.split("_")[-1]).capitalize() + " " +  eval_dir.split("_")[-3].capitalize()
        fig.suptitle(f"Trajectory Smoothness\n{model},{pi} (dt = {dt} s)", fontsize=24, y=0.94)

        # rotate y axis description
        ax[0,0].set_ylabel(None)
        ax[1,0].set_ylabel(None)
        ax[2,0].set_ylabel(None)
        ax[0,1].set_ylabel(None)
        ax[1,1].set_ylabel(None)
        ax[2,1].set_ylabel(None)
        ax[0,2].set_ylabel(None)
        ax[1,2].set_ylabel(None)
        ax[2,2].set_ylabel(None)

        ax[0,0].set_xlabel(None)
        ax[0,1].set_xlabel(None)
        ax[0,2].set_xlabel(None)
        ax[1,0].set_xlabel(None)
        ax[1,1].set_xlabel(None)
        ax[1,2].set_xlabel(None)

        ax[0,0].set_xlim(0,34)
        ax[1,0].set_xlim(0,34)
        ax[2,0].set_xlim(0,34)
        ax[0,1].set_xlim(0,34)
        ax[1,1].set_xlim(0,34)
        ax[2,1].set_xlim(0,34)
        ax[0,2].set_xlim(0,34)
        ax[1,2].set_xlim(0,34)
        ax[2,2].set_xlim(0,34)

        ax[0,0].set_title('Velocity $ \\left[ \\frac{rad}{s} \\right] $ ')
        ax[0,1].set_title('Acceleration $ \\left[ \\frac{rad}{s^{2}} \\right] $ ')
        ax[0,2].set_title('Jerk  $\\left[ \\frac{rad}{s^{3}} \\right] $')

        ax[0,0].set_ylabel("Pick Phase")
        ax[1,0].set_ylabel("Place Phase")
        ax[2,0].set_ylabel("Retract Phase")

        # remove border lines on top on right
        ax[0,0].spines['top'].set_visible(False)
        ax[1,0].spines['top'].set_visible(False)
        ax[2,0].spines['top'].set_visible(False)
        ax[0,0].spines['right'].set_visible(False)
        ax[1,0].spines['right'].set_visible(False)
        ax[2,0].spines['right'].set_visible(False)
        ax[0,1].spines['top'].set_visible(False)
        ax[1,1].spines['top'].set_visible(False)
        ax[2,1].spines['top'].set_visible(False)
        ax[0,1].spines['right'].set_visible(False)
        ax[1,1].spines['right'].set_visible(False)
        ax[2,1].spines['right'].set_visible(False)
        ax[0,2].spines['top'].set_visible(False)
        ax[1,2].spines['top'].set_visible(False)
        ax[2,2].spines['top'].set_visible(False)
        ax[0,2].spines['right'].set_visible(False)
        ax[1,2].spines['right'].set_visible(False)
        ax[2,2].spines['right'].set_visible(False)

        fig.tight_layout()
        
        # add margin
        fig.subplots_adjust(
            top=0.8,
            bottom=0.2,
            left=0.1,
            right=0.9,    
        )

        # save plot
        model = eval_dir.split("_")[-3] + "_" + eval_dir.split("_")[-2] + "_" + eval_dir.split("_")[-1]
        out_path = os.path.join(curr_model_base_path, f"trajectory_smoothness_{model}{pi.replace(' ', '_')}.png")
        print(out_path)
        fig.savefig(out_path, dpi=300)


        # plot only pick data
        fig.clf()
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        # Add the legend to the figure
        # hack to get the legend below the grid plot
        ax[2].legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.7), ncols=2)

        if "Joint" in pi:
            # plot joint limits for vel
            joint_key = pi[1:]
            print(f"plot joint limits for {joint_key}")
            ax[0].axhline(y=joint_vel_limit[joint_key], color="tab:red", linestyle="--")
            ax[0].axhline(y=-joint_vel_limit[joint_key], color="tab:red", linestyle="--")

            # plot joint limits for acc
            ax[1].axhline(y=joint_acc_limit[joint_key], color="tab:red", linestyle="--")
            ax[1].axhline(y=-joint_acc_limit[joint_key], color="tab:red", linestyle="--")

            # plot joint limits for jerk
            ax[2].axhline(y=joint_jerk_limit[joint_key], color="tab:red", linestyle="--")
            ax[2].axhline(y=-joint_jerk_limit[joint_key], color="tab:red", linestyle="--")

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Ground Truth Velocity" + pi, 
            # hue="Phase", 
            ax=ax[0],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Ground Truth Acceleration" + pi, 
            # hue="Phase", 
            ax=ax[1],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Ground Truth Jerk" + pi, 
            # hue="Phase", 
            ax=ax[2],
            color="steelblue",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Velocity" + pi, 
            # hue="Phase", 
            ax=ax[0],
            color="darkorange",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Acceleration" + pi, 
            # hue="Phase", 
            ax=ax[1],
            color="darkorange",
            errorbar=("pi", 100)
        )

        sns.lineplot(
            data=pick_df, 
            x="Steps", 
            y="Jerk" + pi, 
            # hue="Phase", 
            ax=ax[2],
            color="darkorange",
            errorbar=("pi", 100)
        )

        # add y axis grid
        ax[0].grid(axis='y')
        ax[1].grid(axis='y')
        ax[2].grid(axis='y')


        # set title
        model = eval_dir.split("_")[-2].capitalize() + "-" + (eval_dir.split("_")[-1]).capitalize() + " " +  eval_dir.split("_")[-3].capitalize()
        fig.suptitle(f"Trajectory Smoothness\n{model}, Pick Phase,{pi} (dt = {dt} s)", fontsize=16, y=0.94)

        # rotate y axis description
        ax[0].set_ylabel(None)
        ax[1].set_ylabel(None)
        ax[2].set_ylabel(None)


        ax[0].set_xlabel(None)
        ax[1].set_xlabel("Steps")
        ax[2].set_xlabel(None)

        ax[0].set_xlim(0,34)
        ax[1].set_xlim(0,34)
        ax[2].set_xlim(0,34)


        ax[0].set_title('Velocity $ \\left[ \\frac{rad}{s} \\right] $ ')
        ax[1].set_title('Acceleration $ \\left[ \\frac{rad}{s^{2}} \\right] $ ')
        ax[2].set_title('Jerk  $\\left[ \\frac{rad}{s^{3}} \\right] $')

        # remove border lines on top on right
        ax[0].spines['top'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[2].spines['top'].set_visible(False)

        fig.tight_layout()
        
        # add margin
        fig.subplots_adjust(
            top=0.75,
            bottom=0.2,
            left=0.1,
            right=0.9,    
        )

        # save plot
        model = eval_dir.split("_")[-3] + "_" + eval_dir.split("_")[-2] + "_" + eval_dir.split("_")[-1]
        out_path = os.path.join(curr_model_base_path, f"trajectory_smoothness_{model}_pick_phase{pi.replace(' ', '_')}.png")
        print(out_path)
        fig.savefig(out_path, dpi=300)


    model = eval_dir.split("_")[-2].capitalize() + "-" + (eval_dir.split("_")[-1]).capitalize() + " " +  eval_dir.split("_")[-3].capitalize()

    model_nr = np.ones(len(model_df), dtype=int) * int(i)

    # model_df["Model"] = model_nr
    # set "Model" to model name
    model_df["Model"] = model

    all_model_data.append(model_df)


all_model_data = pd.concat(all_model_data)
fig.clf()
fig, ax = plt.subplots(3, 3, figsize=(10, 10))


# color_map = ["gold","darkorange","lightblue","seagreen",]
color_map = ["darkorange","seagreen"]

palette = sns.color_palette(color_map)

all_model_pick_df = all_model_data[all_model_data["Phase"] == 1]
all_model_place_df = all_model_data[all_model_data["Phase"] == 2]
all_model_retract_df = all_model_data[all_model_data["Phase"] == 3]

# ground truth data

sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Ground Truth Jerk", 
    # hue="Model", 
    ax=ax[2, 2],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Velocity Max Abs", 
    # hue="Model", 
    ax=ax[0, 0],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Ground Truth Velocity Max Abs", 
    # hue="Model", 
    ax=ax[1, 0],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Jerk Max Abs", 
    # hue="Model", 
    ax=ax[0, 2],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Acceleration Max Abs", 
    # hue="Model", 
    ax=ax[0, 1],
    palette=palette,
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Ground Truth Jerk Max Abs", 
    # hue="Model", 
    ax=ax[1, 2],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 100),
)
sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Ground Truth Velocity Max Abs", 
    # hue="Model",
    ax=ax[2, 0],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 100),
)
sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Ground Truth Acceleration Max Abs", 
    # hue="Model", 
    ax=ax[2, 1],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Ground Truth Acceleration Max Abs", 
    # hue="Model", 
    ax=ax[1, 1],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 100),
)

# predicted data

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Velocity Max Abs", 
    hue="Model", 
    ax=ax[0, 0],
    palette=palette,
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Acceleration Max Abs", 
    hue="Model", 
    ax=ax[0, 1],
    palette=palette,
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Jerk Max Abs", 
    hue="Model", 
    ax=ax[0, 2],
    palette=palette,
    errorbar=("pi", 100),
)


sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Velocity Max Abs", 
    hue="Model", 
    ax=ax[1, 0],
    palette=palette,
    errorbar=("pi", 100),
)


sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Acceleration Max Abs", 
    hue="Model", 
    ax=ax[1, 1],
    palette=palette,
    errorbar=("pi", 100),
)


sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Jerk Max Abs", 
    hue="Model", 
    ax=ax[1, 2],
    palette=palette,
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Velocity Max Abs", 
    hue="Model",
    ax=ax[2, 0],
    palette=palette,
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Acceleration Max Abs", 
    hue="Model", 
    ax=ax[2, 1],
    palette=palette,
    errorbar=("pi", 100),
)


sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Jerk Max Abs", 
    hue="Model", 
    ax=ax[2, 2],
    palette=palette,
    errorbar=("pi", 100),
)

for i in range(3):
    for j in range(3):
        ax[i, j].grid(axis='y')
        ax[i, j].spines['top'].set_visible(False)
        ax[i, j].spines['right'].set_visible(False)
        ax[i, j].spines['top'].set_visible(False)
        ax[i, j].spines['right'].set_visible(False)

        # add padding
        ax[i, j].yaxis.labelpad = 20

        # remove legend
        ax[i, j].get_legend().remove()

        ax[i, j].set_ylabel("")


ax[0, 0].set_title("Velocity $ \\left[ \\frac{rad}{s} \\right] $")
ax[0, 1].set_title("Acceleration $ \\left[ \\frac{rad}{s^{2}} \\right] $")
ax[0, 2].set_title("Jerk $\\left[ \\frac{rad}{s^{3}} \\right] $")

# rotate y axis description
ax[0, 0].set_ylabel("Pick Phase", rotation=90, labelpad=20)
ax[1, 0].set_ylabel("Place Phase", rotation=90, labelpad=20)
ax[2, 0].set_ylabel("Retract Phase", rotation=90, labelpad=20)

ax[0, 0].set_xticks(ax[0, 0].get_xticks()[::5])
ax[0, 1].set_xticks(ax[0, 1].get_xticks()[::5])
ax[0, 2].set_xticks(ax[0, 2].get_xticks()[::5])

ax[1, 0].set_xticks(ax[1, 0].get_xticks()[::5])
ax[1, 1].set_xticks(ax[1, 1].get_xticks()[::5])
ax[1, 2].set_xticks(ax[1, 2].get_xticks()[::5])

ax[2, 0].set_xticks(ax[2, 0].get_xticks()[::5])
ax[2, 1].set_xticks(ax[2, 1].get_xticks()[::5])
ax[2, 2].set_xticks(ax[2, 2].get_xticks()[::5])

ax[0, 0].set_xlim(0,34)
ax[1, 0].set_xlim(0,34)
ax[2, 0].set_xlim(0,34)
ax[0, 1].set_xlim(0,34)
ax[1, 1].set_xlim(0,34)
ax[2, 1].set_xlim(0,34)
ax[0, 2].set_xlim(0,34)
ax[1, 2].set_xlim(0,34)
ax[2, 2].set_xlim(0,34)

# manually add one legend for both subplots
handles, labels = ax[0, 0].get_legend_handles_labels()

# add gt label
# gt_patch = matplotlib.patches.Patch(color='steelblue', label='Ground Truth')
# handles.append(gt_patch)

# tighten layout
fig.tight_layout()

# add margin
fig.subplots_adjust(
    top=0.85,
    bottom=0.1,
    left=0.1,
    right=0.9,    
)

# cange labels
labels = model_names
labels.append("Ground Truth")
handles.append(
    Line2D([0], [0], marker=None, color="steelblue", label='Ground Truth'),
)

# fig.legend(loc="lower center", ncols = 3)
fig.legend(handles, labels, loc="lower center", ncols = 3)

# set title
# fig.suptitle(f"Trajectory Smoothness: Joint-Space Transformer\nwith and without Interpolation (pi: 100)", fontsize=22, y=0.95)
fig.suptitle(f"Trajectory Smoothness: Max Abs\nCartesian-Space Models, (pi: 100)", fontsize=22, y=0.95)
# fig.suptitle(f"Trajectory Smoothness: Max Abs\nAll Models, (pi: 100)", fontsize=22, y=0.95)

# save plot
# out_path = os.path.join(base_path, f"trajectory_smoothness_interpolation_pi100.png")
out_path = os.path.join(base_path, f"trajectory_smoothness_cartesian_space_pi100.png")
# out_path = os.path.join(base_path, f"trajectory_smoothness_all_pi100.png")
print(out_path)
fig.savefig(out_path, dpi=300)


fig.clf()
fig, ax = plt.subplots(3, 3, figsize=(10, 10))


# ground truth data

sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Ground Truth Jerk", 
    # hue="Model", 
    ax=ax[2, 2],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Velocity Max Abs", 
    # hue="Model", 
    ax=ax[0, 0],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Ground Truth Velocity Max Abs", 
    # hue="Model", 
    ax=ax[1, 0],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Jerk Max Abs", 
    # hue="Model", 
    ax=ax[0, 2],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Acceleration Max Abs", 
    # hue="Model", 
    ax=ax[0, 1],
    palette=palette,
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Ground Truth Jerk Max Abs", 
    # hue="Model", 
    ax=ax[1, 2],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 75),
)
sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Ground Truth Velocity Max Abs", 
    # hue="Model",
    ax=ax[2, 0],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 75),
)
sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Ground Truth Acceleration Max Abs", 
    # hue="Model", 
    ax=ax[2, 1],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Ground Truth Acceleration Max Abs", 
    # hue="Model", 
    ax=ax[1, 1],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 75),
)

# predicted data

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Velocity Max Abs", 
    hue="Model", 
    ax=ax[0, 0],
    palette=palette,
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Acceleration Max Abs", 
    hue="Model", 
    ax=ax[0, 1],
    palette=palette,
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Jerk Max Abs", 
    hue="Model", 
    ax=ax[0, 2],
    palette=palette,
    errorbar=("pi", 75),
)


sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Velocity Max Abs", 
    hue="Model", 
    ax=ax[1, 0],
    palette=palette,
    errorbar=("pi", 75),
)


sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Acceleration Max Abs", 
    hue="Model", 
    ax=ax[1, 1],
    palette=palette,
    errorbar=("pi", 75),
)


sns.lineplot(
    data=all_model_place_df, 
    x="Steps", 
    y="Jerk Max Abs", 
    hue="Model", 
    ax=ax[1, 2],
    palette=palette,
    errorbar=("pi", 75),
)



sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Velocity Max Abs", 
    hue="Model",
    ax=ax[2, 0],
    palette=palette,
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Acceleration Max Abs", 
    hue="Model", 
    ax=ax[2, 1],
    palette=palette,
    errorbar=("pi", 75),
)


sns.lineplot(
    data=all_model_retract_df, 
    x="Steps", 
    y="Jerk Max Abs", 
    hue="Model", 
    ax=ax[2, 2],
    palette=palette,
    errorbar=("pi", 75),
)

for i in range(3):
    for j in range(3):
        ax[i, j].grid(axis='y')
        ax[i, j].spines['top'].set_visible(False)
        ax[i, j].spines['right'].set_visible(False)
        ax[i, j].spines['top'].set_visible(False)
        ax[i, j].spines['right'].set_visible(False)

        # add padding
        ax[i, j].yaxis.labelpad = 20

        # remove legend
        ax[i, j].get_legend().remove()

        ax[i, j].set_ylabel("")


ax[0, 0].set_title("Velocity $ \\left[ \\frac{rad}{s} \\right] $")
ax[0, 1].set_title("Acceleration $ \\left[ \\frac{rad}{s^{2}} \\right] $")
ax[0, 2].set_title("Jerk $\\left[ \\frac{rad}{s^{3}} \\right] $")

# rotate y axis description
ax[0, 0].set_ylabel("Pick Phase", rotation=90, labelpad=20)
ax[1, 0].set_ylabel("Place Phase", rotation=90, labelpad=20)
ax[2, 0].set_ylabel("Retract Phase", rotation=90, labelpad=20)

ax[0, 0].set_xticks(ax[0, 0].get_xticks()[::5])
ax[0, 1].set_xticks(ax[0, 1].get_xticks()[::5])
ax[0, 2].set_xticks(ax[0, 2].get_xticks()[::5])

ax[1, 0].set_xticks(ax[1, 0].get_xticks()[::5])
ax[1, 1].set_xticks(ax[1, 1].get_xticks()[::5])
ax[1, 2].set_xticks(ax[1, 2].get_xticks()[::5])

ax[2, 0].set_xticks(ax[2, 0].get_xticks()[::5])
ax[2, 1].set_xticks(ax[2, 1].get_xticks()[::5])
ax[2, 2].set_xticks(ax[2, 2].get_xticks()[::5])

ax[0, 0].set_xlim(0,34)
ax[1, 0].set_xlim(0,34)
ax[2, 0].set_xlim(0,34)
ax[0, 1].set_xlim(0,34)
ax[1, 1].set_xlim(0,34)
ax[2, 1].set_xlim(0,34)
ax[0, 2].set_xlim(0,34)
ax[1, 2].set_xlim(0,34)
ax[2, 2].set_xlim(0,34)

# manually add one legend for both subplots
handles, labels = ax[0, 0].get_legend_handles_labels()

# add gt label
# gt_patch = matplotlib.patches.Patch(color='steelblue', label='Ground Truth')
# handles.append(gt_patch)

# tighten layout
fig.tight_layout()

# add margin
fig.subplots_adjust(
    top=0.85,
    bottom=0.1,
    left=0.1,
    right=0.9,    
)

# cange labels
labels = model_names
labels.append("Ground Truth")
handles.append(
    Line2D([0], [0], marker=None, color="steelblue", label='Ground Truth'),
)

# fig.legend(loc="lower center", ncols = 3)
fig.legend(handles, labels, loc="lower center", ncols = 3)

# set title
# fig.suptitle(f"Trajectory Smoothness: Joint-Space Transformer\nwith and without Interpolation (pi: 75)", fontsize=22, y=0.95)
fig.suptitle(f"Trajectory Smoothness: Max Abs\nCartesian-Space Models, (pi: 75)", fontsize=22, y=0.95)
# fig.suptitle(f"Trajectory Smoothness: Max Abs\nAll Models, (pi: 75)", fontsize=22, y=0.95)

# save plot
# out_path = os.path.join(base_path, f"trajectory_smoothness_interpolation_pi75.png")
out_path = os.path.join(base_path, f"trajectory_smoothness_cartesian_space_pi75.png")
# out_path = os.path.join(base_path, f"trajectory_smoothness_all_pi75.png")
print(out_path)
fig.savefig(out_path, dpi=300)




# Plot only pick phase

fig.clf()
fig, ax = plt.subplots(1, 3, figsize=(10, 5))

all_model_max_abs_per_model = all_model_data[all_model_data["Phase"] == 1]
all_model_max_abs_per_model = all_model_max_abs_per_model.groupby(['Steps', 'Model']).max()

print(all_model_max_abs_per_model.head())

# ground truth data

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Velocity Max Abs", 
    # hue="Model", 
    ax=ax[0],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Jerk Max Abs", 
    # hue="Model", 
    ax=ax[2],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Acceleration Max Abs", 
    # hue="Model", 
    ax=ax[1],
    palette=palette,
    errorbar=("pi", 100),
)


# predicted data

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Velocity Max Abs", 
    hue="Model", 
    ax=ax[0],
    palette=palette,
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Acceleration Max Abs", 
    hue="Model", 
    ax=ax[1],
    palette=palette,
    errorbar=("pi", 100),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Jerk Max Abs", 
    hue="Model", 
    ax=ax[2],
    palette=palette,
    errorbar=("pi", 100),
)


for i in range(3):
    ax[i].grid(axis='y')
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

    # add padding
    ax[i].yaxis.labelpad = 20

    # remove legend
    ax[i].get_legend().remove()

    ax[i].set_ylabel("")


ax[0].set_title("Velocity $ \\left[ \\frac{rad}{s} \\right] $")
ax[1].set_title("Acceleration $ \\left[ \\frac{rad}{s^{2}} \\right] $")
ax[2].set_title("Jerk $\\left[ \\frac{rad}{s^{3}} \\right] $")


# ax[0].set_xticks(ax[0].get_xticks()[::5])
# ax[1].set_xticks(ax[1].get_xticks()[::5])
# ax[2].set_xticks(ax[2].get_xticks()[::5])


ax[0].set_xlim(0,34)
ax[1].set_xlim(0,34)
ax[2].set_xlim(0,34)


# manually add one legend for both subplots
handles, labels = ax[0].get_legend_handles_labels()

# tighten layout
fig.tight_layout()

# add margin
fig.subplots_adjust(
    top=0.75,
    bottom=0.2,
    left=0.1,
    right=0.9,    
)

# cange labels
labels = model_names
labels.append("Ground Truth")
handles.append(
    Line2D([0], [0], marker=None, color="steelblue", label='Ground Truth'),
)

# fig.legend(loc="lower center", ncols = 5)
fig.legend(handles, labels, loc="lower center", ncols = 5)

# set title
# fig.suptitle(f"Trajectory Smoothness: Joint-Space Transformer\nPick Phase, with and without Interpolation (pi: 100)", fontsize=18, y=0.95)
fig.suptitle(f"Trajectory Smoothness: Max Abs\nCartesian-Space Models, Pick Phase (pi: 100)", fontsize=18, y=0.95)
# fig.suptitle(f"Trajectory Smoothness: Max Abs\nAll Models, Pick Phase (pi: 100)", fontsize=18, y=0.95)

# save plot
# out_path = os.path.join(base_path, f"trajectory_smoothness_interpolation_pick_phase_pi100.png")
out_path = os.path.join(base_path, f"trajectory_smoothness_cartesian_space_pick_phase_pi100.png")
# out_path = os.path.join(base_path, f"trajectory_smoothness_all_models_pick_phase_pi100.png")
print(out_path)
fig.savefig(out_path, dpi=300)


fig.clf()
fig, ax = plt.subplots(1, 3, figsize=(10, 5))

# ground truth data

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Velocity Max Abs", 
    # hue="Model", 
    ax=ax[0],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Jerk Max Abs", 
    # hue="Model", 
    ax=ax[2],
    # palette=palette,
    color="steelblue",
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Ground Truth Acceleration Max Abs", 
    # hue="Model", 
    ax=ax[1],
    palette=palette,
    errorbar=("pi", 75),
)



# predicted data

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Velocity Max Abs", 
    hue="Model", 
    ax=ax[0],
    palette=palette,
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Acceleration Max Abs", 
    hue="Model", 
    ax=ax[1],
    palette=palette,
    errorbar=("pi", 75),
)

sns.lineplot(
    data=all_model_pick_df, 
    x="Steps", 
    y="Jerk Max Abs", 
    hue="Model", 
    ax=ax[2],
    palette=palette,
    errorbar=("pi", 75),
)



for i in range(3):
    ax[i].grid(axis='y')
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

    # add padding
    ax[i].yaxis.labelpad = 20

    # remove legend
    ax[i].get_legend().remove()

    ax[i].set_ylabel("")


ax[0].set_title("Velocity $ \\left[ \\frac{rad}{s} \\right] $")
ax[1].set_title("Acceleration $ \\left[ \\frac{rad}{s^{2}} \\right] $")
ax[2].set_title("Jerk $\\left[ \\frac{rad}{s^{3}} \\right] $")


# ax[0].set_xticks(ax[0].get_xticks()[::5])
# ax[1].set_xticks(ax[1].get_xticks()[::5])
# ax[2].set_xticks(ax[2].get_xticks()[::5])


ax[0].set_xlim(0,34)
ax[1].set_xlim(0,34)
ax[2].set_xlim(0,34)


# manually add one legend for both subplots
handles, labels = ax[0].get_legend_handles_labels()

# tighten layout
fig.tight_layout()

# add margin
fig.subplots_adjust(
    top=0.75,
    bottom=0.2,
    left=0.1,
    right=0.9,    
)

# cange labels
labels = model_names
labels.append("Ground Truth")
handles.append(
    Line2D([0], [0], marker=None, color="steelblue", label='Ground Truth'),
)

fig.legend(handles, labels, loc="lower center", ncols = 5, )
# fig.legend(loc="lower center", ncols = 5, )

# set title
# fig.suptitle(f"Trajectory Smoothness: Joint-Space Transformer\nPick Phase, with and without Interpolation (pi: 75)", fontsize=18, y=0.95)
fig.suptitle(f"Trajectory Smoothness: Max Abs\nCartesian-Space Models, Pick Phase (pi: 75)", fontsize=18, y=0.95)
# fig.suptitle(f"Trajectory Smoothness: Max Abs\nAll Models, Pick Phase (pi: 75)", fontsize=18, y=0.95)

# save plot
# out_path = os.path.join(base_path, f"trajectory_smoothness_interpolation_pick_phase_pi75.png")
out_path = os.path.join(base_path, f"trajectory_smoothness_cartesian_space_pick_phase_pi75.png")
# out_path = os.path.join(base_path, f"trajectory_smoothness_all_models_pick_phase_pi75.png")
print(out_path)
fig.savefig(out_path, dpi=300)
