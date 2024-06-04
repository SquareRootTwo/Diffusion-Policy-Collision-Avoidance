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
    os.path.join(base_path, "2024-04-18_00-49-51_unet_joint_space"),
    os.path.join(base_path, "2024-04-18_19-46-27_unet_cartesian_space"),
    os.path.join(base_path, "2024-04-22_00-04-23_transformer_joint_space"),
    os.path.join(base_path, "2024-04-26_02-07-12_transformer_cartesian_space"),
]
nr_eval_dirs = len(dirs)
nr_episodes = 101

heatmap = np.zeros((nr_eval_dirs, nr_episodes))

collision_obstacle_indices = [f"obs_{i}" for i in range(21, 21+244)]
joint_indices = [f"obs_{i}" for i in range(0, 7)]

y_lables = []

model_names = []

for i, eval_dir in tqdm(enumerate(dirs), desc="Evaluating directories", total=nr_eval_dirs, position=0, leave=True):
    
    if "2024" not in eval_dir:
        continue

    y_label = eval_dir.split("_")[-3] + "_" + eval_dir.split("_")[-2] + "_" + eval_dir.split("_")[-1]
    y_lables.append(y_label)
    
    model = eval_dir.split("_")[-2].capitalize() + "-" + (eval_dir.split("_")[-1]).capitalize() + " " +  eval_dir.split("_")[-3].capitalize()
    model_names.append(model)

    for eval_episode in tqdm(range(nr_episodes), desc="Evaluating episodes", total=nr_episodes, position=1, leave=False):
        eval_episode_path = os.path.join(base_path, eval_dir, f"episode_{eval_episode}_data.parquet")
        if os.path.exists(eval_episode_path):
            df = pd.read_parquet(eval_episode_path)

            joint_trajectory = df[joint_indices].values

            # shape = (102, 61*4)
            collision_trajectory = df[collision_obstacle_indices].values

            # shape = (102, 61*4)
            joint_trajectory = torch.tensor(joint_trajectory)

            panda_spheres = pk.get_panda_collision_spheres(joint_trajectory, return_flattened=True)
            
            pick_collision = 0
            for step in range(0, 34):
                curr_obstacles = collision_trajectory[step]
                curr_panda_spheres = panda_spheres[step].numpy()
                for si in range(61):
                    for sj in range(61):
                        obstacle_sphere = curr_obstacles[si*4:si*4+4]
                        panda_sphere = curr_panda_spheres[sj*4:sj*4+4]

                        # check for collision
                        if np.linalg.norm(obstacle_sphere[:3] - panda_sphere[:3]) < obstacle_sphere[3] + panda_sphere[3]:
                            pick_collision = 1

            place_collision = 0
            for step in range(34, 68):
                curr_obstacles = collision_trajectory[step]
                curr_panda_spheres = panda_spheres[step].numpy()
                for si in range(61):
                    for sj in range(61):
                        obstacle_sphere = curr_obstacles[si*4:si*4+4]
                        panda_sphere = curr_panda_spheres[sj*4:sj*4+4]

                        # check for collision
                        if np.linalg.norm(obstacle_sphere[:3] - panda_sphere[:3]) < obstacle_sphere[3] + panda_sphere[3]:
                            place_collision = 1
            
            retract_collision = 0
            for step in range(68, 102):
                curr_obstacles = collision_trajectory[step]
                curr_panda_spheres = panda_spheres[step].numpy()
                for si in range(61):
                    for sj in range(61):
                        obstacle_sphere = curr_obstacles[si*4:si*4+4]
                        panda_sphere = curr_panda_spheres[sj*4:sj*4+4]

                        # check for collision
                        if np.linalg.norm(obstacle_sphere[:3] - panda_sphere[:3]) < obstacle_sphere[3] + panda_sphere[3]:
                            retract_collision = 1

            heatmap[i, eval_episode] = pick_collision + place_collision + retract_collision


print(heatmap)
plt.figure(figsize=(17,4))

# total sum per model
sum_per_model = np.sum(heatmap, axis=1)
print(sum_per_model)

y_lables_with_total = [f"{m} (total: {int(sum_per_model[i])})" for i, m in enumerate(model_names)]

x_range = np.arange(0, nr_episodes)

cmap = ['#ffeee6', '#ffaa80', '#ff7633', '#b33b00']

palette = sns.color_palette("YlOrBr", n_colors=10, )
# invert the palette
# palette = palette[::-1]

# red colored heatmap
# set heatmap sns size
ax = sns.heatmap(
    heatmap, 
    cmap=palette,
    yticklabels=y_lables_with_total,
    xticklabels=x_range,
    vmin=0,
    vmax=3,
    # cbar_kws={'label': 'Nr. of Collisions'},
    linewidths=0,
    linecolor=[(0.4, 0.4, 0.4, 1)],
    fmt='d',
    # square=True,
    cbar=False
    # cbar=True
)
# add y lables horizontally
plt.yticks(rotation=0, ha='right')
# Set the linewidth of the border
# Rotate the x-axis labels
plt.xticks(rotation=45)

# set square height manually
ax.set_aspect(4)

# Increase the font size of the x-ticks and y-ticks
ax.tick_params(axis='both', which='major', labelsize=18)

for i, label in enumerate(ax.xaxis.get_ticklabels()):
    if i % 4 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)


# add more spacing for the boxes
plt.tight_layout()

# Define the colormap
cmap = mcolors.ListedColormap(palette)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# add padding
plt.subplots_adjust(left=0.32, right=0.95, top=0.95, bottom=0.05)

cbar = plt.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    ticks=[0, 1, 2, 3],
    orientation='vertical',
    shrink=0.5,
    aspect=20,
    fraction=0.05,
    pad=0.05,
)

cbar.set_label('Nr. of Collisions', rotation=270, labelpad=32, fontsize=18)
cbar.ax.tick_params(labelsize=23)

plt.xlabel("Episode", fontdict={'fontsize': 18}, labelpad=10)
# set cbar font size

plt.title("Collision Heatmap", fontdict={'fontsize': 32}, pad=18, y=1.0)

out_path = os.path.join(base_path, "collision_heatmap.png")

plt.savefig(out_path, dpi=300)
