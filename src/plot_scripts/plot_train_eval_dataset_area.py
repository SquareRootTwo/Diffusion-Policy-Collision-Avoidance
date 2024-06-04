import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the path to the parent directory to augment search for module
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

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

fig = plt.figure()

# mark the pick and place area as lines on plot
plt.plot([pick_y_min, pick_y_min], [pick_x_min, pick_x_max], color='darkorange', label="Pick Area", linewidth=0.7)
plt.plot([pick_y_max, pick_y_max], [pick_x_min, pick_x_max], color='darkorange', label="Pick Area", linewidth=0.7)
plt.plot([pick_y_min, pick_y_max], [pick_x_min, pick_x_min], color='darkorange', label="Pick Area", linewidth=0.7)
plt.plot([pick_y_min, pick_y_max], [pick_x_max, pick_x_max], color='darkorange', label="Pick Area", linewidth=0.7)

plt.plot([place_y_min, place_y_min], [place_x_min, place_x_max], color='olivedrab', label="Place Area", linewidth=0.7)
plt.plot([place_y_max, place_y_max], [place_x_min, place_x_max], color='olivedrab', label="Place Area", linewidth=0.7)
plt.plot([place_y_min, place_y_max], [place_x_min, place_x_min], color='olivedrab', label="Place Area", linewidth=0.7)
plt.plot([place_y_min, place_y_max], [place_x_max, place_x_max], color='olivedrab', label="Place Area", linewidth=0.7)

plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='center', bbox_to_anchor=(0.5, 1.2), ncol=2, facecolor='white')


# set aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')


# scatter plot train points
plt.scatter(train_pick_y_points, train_pick_x_points, color='steelblue', label='pick', s=5)
plt.scatter(train_place_y_points, train_place_x_points, color='steelblue', label='place', s=5)

# plot test points
plt.scatter(eval_pick_y_points, eval_pick_x_points, color='coral', label='pick', s=5)
plt.scatter(eval_place_y_points, eval_place_x_points, color='coral', label='place', s=5)


# white background
plt.gca().set_facecolor('white')

# save fig
plt.savefig("./eval/train_eval_split_area.png", dpi=300)