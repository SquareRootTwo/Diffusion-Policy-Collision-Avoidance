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

base_path = os.path.join(root_path, "src/data/wandb_training_data/")
out_path = os.path.join(root_path, "src/data/thesis_eval/train_plots")

os.makedirs(out_path, exist_ok=True)


dirs = [
    os.path.join(base_path, "last_ee_position_mean.csv"),
    os.path.join(base_path, "last_ee_position_std.csv"),
    
    os.path.join(base_path, "last_ee_rotation_mean.csv"),
    os.path.join(base_path, "last_ee_rotation_std.csv"),
    
    os.path.join(base_path, "loss_acc_mean.csv"),
    os.path.join(base_path, "loss_acc_std.csv"),
    
    os.path.join(base_path, "loss_vel_mean.csv"),
    os.path.join(base_path, "loss_vel_std.csv"),
    
    os.path.join(base_path, "loss_jerk_mean.csv"),
    os.path.join(base_path, "loss_jerk_std.csv"),
    
    os.path.join(base_path, "loss_mse_mean.csv"),
    os.path.join(base_path, "loss_mse_std.csv"),

    os.path.join(base_path, "loss_mse_train_epoch.csv"),
]

nr_eval_steps = 75

ee_position_cols = [
  "Group: Diffusion_Policy_Transformer_IK - eval_mean/loss_last_position_mean",
  "Group: Diffusion_Policy_Transformer_FK - eval_mean/loss_last_position_mean",
  "Group: Diffusion_Policy_Unet_IK - eval_mean/loss_last_position_mean",
  "Group: Diffusion_Policy_Unet_FK - eval_mean/loss_last_position_mean"
]

ee_position_std_cols = [
  "Group: Diffusion_Policy_Transformer_IK - eval_std/loss_last_position_std",
  "Group: Diffusion_Policy_Transformer_FK - eval_std/loss_last_position_std",
  "Group: Diffusion_Policy_Unet_IK - eval_std/loss_last_position_std",
  "Group: Diffusion_Policy_Unet_FK - eval_std/loss_last_position_std"
]

df = pd.read_csv(dirs[0])
df_std = pd.read_csv(dirs[1])

transformer_ik_mean = pd.to_numeric(df[ee_position_cols[0]], errors='coerce').values
transformer_ik_std = pd.to_numeric(df_std[ee_position_std_cols[0]], errors='coerce').values
transformer_fk_mean = pd.to_numeric(df[ee_position_cols[1]], errors='coerce').values
transformer_fk_std = pd.to_numeric(df_std[ee_position_std_cols[1]], errors='coerce').values
unet_ik_mean = pd.to_numeric(df[ee_position_cols[2]], errors='coerce').values
unet_ik_std = pd.to_numeric(df_std[ee_position_std_cols[2]], errors='coerce').values
unet_fk_mean = pd.to_numeric(df[ee_position_cols[3]], errors='coerce').values
unet_fk_std = pd.to_numeric(df_std[ee_position_std_cols[3]], errors='coerce').values

transformer_ik_mean = transformer_ik_mean[~np.isnan(transformer_ik_mean)]
transformer_fk_mean = transformer_fk_mean[~np.isnan(transformer_fk_mean)]
unet_ik_mean = unet_ik_mean[~np.isnan(unet_ik_mean)]
unet_fk_mean = unet_fk_mean[~np.isnan(unet_fk_mean)]

transformer_ik_std = transformer_ik_std[~np.isnan(transformer_ik_std)]
transformer_fk_std = transformer_fk_std[~np.isnan(transformer_fk_std)]
unet_ik_std = unet_ik_std[~np.isnan(unet_ik_std)]
unet_fk_std = unet_fk_std[~np.isnan(unet_fk_std)]


fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].errorbar(
  x=np.arange(75),
  y=unet_fk_mean,
  yerr=unet_fk_std,
  color="gold",
  # elinewidth=1.0,
  errorevery=(0,4),
  capsize=1.5
)
ax[0].errorbar(
  x=np.arange(75),
  y=unet_ik_mean,
  yerr=unet_ik_std,
  color="darkorange",
  # elinewidth=1.0,
  errorevery=(1,4),
  capsize=1.5
)
ax[0].errorbar(
  x=np.arange(75),
  y=transformer_ik_mean,
  yerr=transformer_ik_std,
  color="lightblue",
  # elinewidth=1.0,
  errorevery=(2,4),
  capsize=1.5
)
ax[0].errorbar(
  x=np.arange(75),
  y=transformer_fk_mean,
  yerr=transformer_fk_std,
  color="seagreen",
  # elinewidth=1.0,
  errorevery=(3,4),
  capsize=1.5
)


# sns.lineplot(
#   x=np.arange(75),
#   y=transformer_ik_mean,
#   color="lightblue",
#   ax=ax[0],
# )

# sns.lineplot(
#   x=np.arange(75),
#   y=transformer_fk_mean,
#   color="seagreen",
#   ax=ax[0],
# )

# sns.lineplot(
#   x=np.arange(75),
#   y=unet_ik_mean,
#   color="darkorange",
#   ax=ax[0],
# )

# sns.lineplot(
#   x=np.arange(75),
#   y=unet_fk_mean,
#   color="gold",
#   ax=ax[0],
# )



# fig.tight_layout()
fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.3)

fig.suptitle("Evaluation Progress on End Effector Distance", y=0.94,  fontsize=18,)

legend_elements = [
    Line2D([0], [0], marker=None, color="lightblue", label=f'Transformer Cartesian Space (final: {transformer_ik_mean[-1]:.3f} m)'),
    Line2D([0], [0], marker=None, color="seagreen", label=f'Transformer Joint Space (final: {transformer_fk_mean[-1]:.3f} m)'),
    Line2D([0], [0], marker=None, color="darkorange", label=f'Unet Cartesian Space (final: {unet_ik_mean[-1]:.3f} m)'),
    Line2D([0], [0], marker=None, color="gold", label=f'Unet Joint Space (final: {unet_fk_mean[-1]:.3f} m)'),
]
ax[0].legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5,-.55),fontsize=10)

# Remove the upper and right spines
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

ax[0].set_xlabel("Evaluation Iterations")
ax[1].set_xlabel("Evaluation Iterations")
ax[0].set_ylabel("EE Distance [m]")
ax[1].set_ylabel("EE Angular Distance [rad]")



ee_angle_cols = [
  "Group: Diffusion_Policy_Transformer_IK - eval_mean/loss_last_angle_mean",
  "Group: Diffusion_Policy_Transformer_FK - eval_mean/loss_last_angle_mean",
  "Group: Diffusion_Policy_Unet_IK - eval_mean/loss_last_angle_mean",
  "Group: Diffusion_Policy_Unet_FK - eval_mean/loss_last_angle_mean"
]

ee_angle_std_cols = [
  "Group: Diffusion_Policy_Transformer_IK - eval_std/loss_last_angle_std",
  "Group: Diffusion_Policy_Transformer_FK - eval_std/loss_last_angle_std",
  "Group: Diffusion_Policy_Unet_IK - eval_std/loss_last_angle_std",
  "Group: Diffusion_Policy_Unet_FK - eval_std/loss_last_angle_std"
]

df = pd.read_csv(dirs[2])
df_std = pd.read_csv(dirs[3])

transformer_ik_mean = pd.to_numeric(df[ee_angle_cols[0]], errors='coerce').values
transformer_ik_std = pd.to_numeric(df_std[ee_angle_std_cols[0]], errors='coerce').values
transformer_fk_mean = pd.to_numeric(df[ee_angle_cols[1]], errors='coerce').values
transformer_fk_std = pd.to_numeric(df_std[ee_angle_std_cols[1]], errors='coerce').values
unet_ik_mean = pd.to_numeric(df[ee_angle_cols[2]], errors='coerce').values
unet_ik_std = pd.to_numeric(df_std[ee_angle_std_cols[2]], errors='coerce').values
unet_fk_mean = pd.to_numeric(df[ee_angle_cols[3]], errors='coerce').values
unet_fk_std = pd.to_numeric(df_std[ee_angle_std_cols[3]], errors='coerce').values

transformer_ik_mean = transformer_ik_mean[~np.isnan(transformer_ik_mean)]
transformer_fk_mean = transformer_fk_mean[~np.isnan(transformer_fk_mean)]
unet_ik_mean = unet_ik_mean[~np.isnan(unet_ik_mean)]
unet_fk_mean = unet_fk_mean[~np.isnan(unet_fk_mean)]

transformer_ik_std = transformer_ik_std[~np.isnan(transformer_ik_std)]
transformer_fk_std = transformer_fk_std[~np.isnan(transformer_fk_std)]
unet_ik_std = unet_ik_std[~np.isnan(unet_ik_std)]
unet_fk_std = unet_fk_std[~np.isnan(unet_fk_std)]

ax[1].errorbar(
  x=np.arange(75),
  y=unet_fk_mean,
  yerr=unet_fk_std,
  color="gold",
  # elinewidth=1.0,
  errorevery=(0,4),
  capsize=1.5
)
ax[1].errorbar(
  x=np.arange(75),
  y=unet_ik_mean,
  yerr=unet_ik_std,
  color="darkorange",
  # elinewidth=1.0,
  errorevery=(1,4),
  capsize=1.5
)
ax[1].errorbar(
  x=np.arange(75),
  y=transformer_ik_mean,
  yerr=transformer_ik_std,
  color="lightblue",
  # elinewidth=1.0,
  errorevery=(2,4),
  capsize=1.5
)
ax[1].errorbar(
  x=np.arange(75),
  y=transformer_fk_mean,
  yerr=transformer_fk_std,
  color="seagreen",
  # elinewidth=1.0,
  errorevery=(3,4),
  capsize=1.5
)

legend_elements = [
    Line2D([0], [0], marker=None, color="lightblue", label=f'Cartesian-Space Transformer (final: {transformer_ik_mean[-1]:.3f} rad)'),
    Line2D([0], [0], marker=None, color="seagreen", label=f'Joint-Space Transformer (final: {transformer_fk_mean[-1]:.3f} rad)'),
    Line2D([0], [0], marker=None, color="darkorange", label=f'Cartesian-Space Unet (final: {unet_ik_mean[-1]:.3f} rad)'),
    Line2D([0], [0], marker=None, color="gold", label=f'Joint-Space Unet (final: {unet_fk_mean[-1]:.3f} rad)'),
]
ax[1].legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5,-.55),fontsize=10)


plt.savefig(
  os.path.join(out_path, "train_eval_ee_distance_plot.png"), dpi=300
)

plt.clf()


ee_acc_cols = [
  "Group: Diffusion_Policy_Transformer_IK - eval_mean/loss_acc_mean",
  "Group: Diffusion_Policy_Transformer_FK - eval_mean/loss_acc_mean",
  "Group: Diffusion_Policy_Unet_IK - eval_mean/loss_acc_mean",
  "Group: Diffusion_Policy_Unet_FK - eval_mean/loss_acc_mean"
]

ee_acc_std_cols = [
  "Group: Diffusion_Policy_Transformer_IK - eval_std/loss_acc_std",
  "Group: Diffusion_Policy_Transformer_FK - eval_std/loss_acc_std",
  "Group: Diffusion_Policy_Unet_IK - eval_std/loss_acc_std",
  "Group: Diffusion_Policy_Unet_FK - eval_std/loss_acc_std"
]


df = pd.read_csv(dirs[4])
df_std = pd.read_csv(dirs[5])

transformer_ik_mean = pd.to_numeric(df[ee_acc_cols[0]], errors='coerce').values
transformer_ik_std = pd.to_numeric(df_std[ee_acc_std_cols[0]], errors='coerce').values
transformer_fk_mean = pd.to_numeric(df[ee_acc_cols[1]], errors='coerce').values
transformer_fk_std = pd.to_numeric(df_std[ee_acc_std_cols[1]], errors='coerce').values
unet_ik_mean = pd.to_numeric(df[ee_acc_cols[2]], errors='coerce').values
unet_ik_std = pd.to_numeric(df_std[ee_acc_std_cols[2]], errors='coerce').values
unet_fk_mean = pd.to_numeric(df[ee_acc_cols[3]], errors='coerce').values
unet_fk_std = pd.to_numeric(df_std[ee_acc_std_cols[3]], errors='coerce').values

transformer_ik_mean = transformer_ik_mean[~np.isnan(transformer_ik_mean)]
transformer_fk_mean = transformer_fk_mean[~np.isnan(transformer_fk_mean)]
unet_ik_mean = unet_ik_mean[~np.isnan(unet_ik_mean)]
unet_fk_mean = unet_fk_mean[~np.isnan(unet_fk_mean)]

transformer_ik_std = transformer_ik_std[~np.isnan(transformer_ik_std)]
transformer_fk_std = transformer_fk_std[~np.isnan(transformer_fk_std)]
unet_ik_std = unet_ik_std[~np.isnan(unet_ik_std)]
unet_fk_std = unet_fk_std[~np.isnan(unet_fk_std)]


fig, ax = plt.subplots(1, 3, figsize=(10,4))

ax[0].errorbar(
  x=np.arange(75),
  y=unet_fk_mean,
  yerr=unet_fk_std,
  color="gold",
  # elinewidth=1.0,
  errorevery=(0,4),
  capsize=1.5
)
ax[0].errorbar(
  x=np.arange(75),
  y=unet_ik_mean,
  yerr=unet_ik_std,
  color="darkorange",
  # elinewidth=1.0,
  errorevery=(1,4),
  capsize=1.5
)
ax[0].errorbar(
  x=np.arange(75),
  y=transformer_ik_mean,
  yerr=transformer_ik_std,
  color="lightblue",
  # elinewidth=1.0,
  errorevery=(2,4),
  capsize=1.5
)
ax[0].errorbar(
  x=np.arange(75),
  y=transformer_fk_mean,
  yerr=transformer_fk_std,
  color="seagreen",
  # elinewidth=1.0,
  errorevery=(3,4),
  capsize=1.5
)

fig.tight_layout()
fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.22)

fig.suptitle("Evaluation Progress on Trajectory Smoothness", y=0.94,  fontsize=18,)

legend_elements = [
    Line2D([0], [0], marker=None, color="lightblue", label=f'Cartesian-Space Transformer'),
    Line2D([0], [0], marker=None, color="seagreen", label=f'Joint-Space Transformer'),
    Line2D([0], [0], marker=None, color="darkorange", label=f'Cartesian-Space Unet'),
    Line2D([0], [0], marker=None, color="gold", label=f'Joint-Space Unet'),
]
fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.), fontsize=10, ncols=4)


# Remove the upper and right spines
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)

ax[0].set_xlabel("Evaluation Iterations")
ax[1].set_xlabel("Evaluation Iterations")
ax[2].set_xlabel("Evaluation Iterations")
ax[0].set_ylabel("$\\frac{d}{dt}$ (vel)", fontsize=12)
ax[1].set_ylabel("$\\frac{d^2}{dt^2}$ (acc)", fontsize=12)
ax[2].set_ylabel("$\\frac{d^3}{dt^3}$ (jerk)", fontsize=12)


ee_vel_cols = [
  "Group: Diffusion_Policy_Transformer_IK - eval_mean/loss_vel_mean",
  "Group: Diffusion_Policy_Transformer_FK - eval_mean/loss_vel_mean",
  "Group: Diffusion_Policy_Unet_IK - eval_mean/loss_vel_mean",
  "Group: Diffusion_Policy_Unet_FK - eval_mean/loss_vel_mean"
]

ee_vel_std_cols = [
  "Group: Diffusion_Policy_Transformer_IK - eval_std/loss_vel_std",
  "Group: Diffusion_Policy_Transformer_FK - eval_std/loss_vel_std",
  "Group: Diffusion_Policy_Unet_IK - eval_std/loss_vel_std",
  "Group: Diffusion_Policy_Unet_FK - eval_std/loss_vel_std"
]

df = pd.read_csv(dirs[6])
df_std = pd.read_csv(dirs[7])

transformer_ik_mean = pd.to_numeric(df[ee_vel_cols[0]], errors='coerce').values
transformer_ik_std = pd.to_numeric(df_std[ee_vel_std_cols[0]], errors='coerce').values
transformer_fk_mean = pd.to_numeric(df[ee_vel_cols[1]], errors='coerce').values
transformer_fk_std = pd.to_numeric(df_std[ee_vel_std_cols[1]], errors='coerce').values
unet_ik_mean = pd.to_numeric(df[ee_vel_cols[2]], errors='coerce').values
unet_ik_std = pd.to_numeric(df_std[ee_vel_std_cols[2]], errors='coerce').values
unet_fk_mean = pd.to_numeric(df[ee_vel_cols[3]], errors='coerce').values
unet_fk_std = pd.to_numeric(df_std[ee_vel_std_cols[3]], errors='coerce').values

transformer_ik_mean = transformer_ik_mean[~np.isnan(transformer_ik_mean)]
transformer_fk_mean = transformer_fk_mean[~np.isnan(transformer_fk_mean)]
unet_ik_mean = unet_ik_mean[~np.isnan(unet_ik_mean)]
unet_fk_mean = unet_fk_mean[~np.isnan(unet_fk_mean)]

transformer_ik_std = transformer_ik_std[~np.isnan(transformer_ik_std)]
transformer_fk_std = transformer_fk_std[~np.isnan(transformer_fk_std)]
unet_ik_std = unet_ik_std[~np.isnan(unet_ik_std)]
unet_fk_std = unet_fk_std[~np.isnan(unet_fk_std)]


ax[1].errorbar(
  x=np.arange(75),
  y=unet_fk_mean,
  yerr=unet_fk_std,
  color="gold",
  # elinewidth=1.0,
  errorevery=(0,4),
  capsize=1.5
)
ax[1].errorbar(
  x=np.arange(75),
  y=unet_ik_mean,
  yerr=unet_ik_std,
  color="darkorange",
  # elinewidth=1.0,
  errorevery=(1,4),
  capsize=1.5
)
ax[1].errorbar(
  x=np.arange(75),
  y=transformer_ik_mean,
  yerr=transformer_ik_std,
  color="lightblue",
  # elinewidth=1.0,
  errorevery=(2,4),
  capsize=1.5
)
ax[1].errorbar(
  x=np.arange(75),
  y=transformer_fk_mean,
  yerr=transformer_fk_std,
  color="seagreen",
  # elinewidth=1.0,
  errorevery=(3,4),
  capsize=1.5
)

ee_jerk_cols = [
  "Group: Diffusion_Policy_Transformer_IK - eval_mean/loss_jerk_mean",
  "Group: Diffusion_Policy_Transformer_FK - eval_mean/loss_jerk_mean",
  "Group: Diffusion_Policy_Unet_IK - eval_mean/loss_jerk_mean",
  "Group: Diffusion_Policy_Unet_FK - eval_mean/loss_jerk_mean"
]

ee_jerk_std_cols = [
  "Group: Diffusion_Policy_Transformer_IK - eval_std/loss_jerk_std",
  "Group: Diffusion_Policy_Transformer_FK - eval_std/loss_jerk_std",
  "Group: Diffusion_Policy_Unet_IK - eval_std/loss_jerk_std",
  "Group: Diffusion_Policy_Unet_FK - eval_std/loss_jerk_std"
]

df = pd.read_csv(dirs[8])
df_std = pd.read_csv(dirs[9])

transformer_ik_mean = pd.to_numeric(df[ee_jerk_cols[0]], errors='coerce').values
transformer_ik_std = pd.to_numeric(df_std[ee_jerk_std_cols[0]], errors='coerce').values
transformer_fk_mean = pd.to_numeric(df[ee_jerk_cols[1]], errors='coerce').values
transformer_fk_std = pd.to_numeric(df_std[ee_jerk_std_cols[1]], errors='coerce').values
unet_ik_mean = pd.to_numeric(df[ee_jerk_cols[2]], errors='coerce').values
unet_ik_std = pd.to_numeric(df_std[ee_jerk_std_cols[2]], errors='coerce').values
unet_fk_mean = pd.to_numeric(df[ee_jerk_cols[3]], errors='coerce').values
unet_fk_std = pd.to_numeric(df_std[ee_jerk_std_cols[3]], errors='coerce').values

transformer_ik_mean = transformer_ik_mean[~np.isnan(transformer_ik_mean)]
transformer_fk_mean = transformer_fk_mean[~np.isnan(transformer_fk_mean)]
unet_ik_mean = unet_ik_mean[~np.isnan(unet_ik_mean)]
unet_fk_mean = unet_fk_mean[~np.isnan(unet_fk_mean)]

transformer_ik_std = transformer_ik_std[~np.isnan(transformer_ik_std)]
transformer_fk_std = transformer_fk_std[~np.isnan(transformer_fk_std)]
unet_ik_std = unet_ik_std[~np.isnan(unet_ik_std)]
unet_fk_std = unet_fk_std[~np.isnan(unet_fk_std)]


ax[2].errorbar(
  x=np.arange(75),
  y=unet_fk_mean,
  yerr=unet_fk_std,
  color="gold",
  # elinewidth=1.0,
  errorevery=(0,4),
  capsize=1.5
)
ax[2].errorbar(
  x=np.arange(75),
  y=unet_ik_mean,
  yerr=unet_ik_std,
  color="darkorange",
  # elinewidth=1.0,
  errorevery=(1,4),
  capsize=1.5
)
ax[2].errorbar(
  x=np.arange(75),
  y=transformer_ik_mean,
  yerr=transformer_ik_std,
  color="lightblue",
  # elinewidth=1.0,
  errorevery=(2,4),
  capsize=1.5
)
ax[2].errorbar(
  x=np.arange(75),
  y=transformer_fk_mean,
  yerr=transformer_fk_std,
  color="seagreen",
  # elinewidth=1.0,
  errorevery=(3,4),
  capsize=1.5
)


plt.savefig(os.path.join(out_path, "train_eval_trajectory_smoothness.png"), dpi=300)

plt.clf()

train_loss = [
  "Group: Diffusion_Policy_Transformer_IK - train/epoch_loss",
  "Group: Diffusion_Policy_Transformer_FK - train/epoch_loss",
  "Group: Diffusion_Policy_Unet_IK - train/epoch_loss",
  "Group: Diffusion_Policy_Unet_FK - train/epoch_loss"
]


eval_loss_mse_mean = [
  "Group: Diffusion_Policy_Transformer_IK - eval_mean/loss_mse_mean",
  "Group: Diffusion_Policy_Transformer_FK - eval_mean/loss_mse_mean",
  "Group: Diffusion_Policy_Unet_IK - eval_mean/loss_mse_mean",
  "Group: Diffusion_Policy_Unet_FK - eval_mean/loss_mse_mean"
]

eval_loss_mse_std = [
  "Group: Diffusion_Policy_Transformer_IK - eval_std/loss_mse_std",
  "Group: Diffusion_Policy_Transformer_FK - eval_std/loss_mse_std",
  "Group: Diffusion_Policy_Unet_IK - eval_std/loss_mse_std",
  "Group: Diffusion_Policy_Unet_FK - eval_std/loss_mse_std"
]

train_df = pd.read_csv(dirs[12])
df = pd.read_csv(dirs[10])
df_std = pd.read_csv(dirs[11])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

transformer_ik_mean = pd.to_numeric(df[eval_loss_mse_mean[0]], errors='coerce').values
transformer_ik_std = pd.to_numeric(df_std[eval_loss_mse_std[0]], errors='coerce').values
transformer_fk_mean = pd.to_numeric(df[eval_loss_mse_mean[1]], errors='coerce').values
transformer_fk_std = pd.to_numeric(df_std[eval_loss_mse_std[1]], errors='coerce').values
unet_ik_mean = pd.to_numeric(df[eval_loss_mse_mean[2]], errors='coerce').values
unet_ik_std = pd.to_numeric(df_std[eval_loss_mse_std[2]], errors='coerce').values
unet_fk_mean = pd.to_numeric(df[eval_loss_mse_mean[3]], errors='coerce').values
unet_fk_std = pd.to_numeric(df_std[eval_loss_mse_std[3]], errors='coerce').values

transformer_ik_mean = transformer_ik_mean[~np.isnan(transformer_ik_mean)]
transformer_fk_mean = transformer_fk_mean[~np.isnan(transformer_fk_mean)]
unet_ik_mean = unet_ik_mean[~np.isnan(unet_ik_mean)]
unet_fk_mean = unet_fk_mean[~np.isnan(unet_fk_mean)]

transformer_ik_std = transformer_ik_std[~np.isnan(transformer_ik_std)]
transformer_fk_std = transformer_fk_std[~np.isnan(transformer_fk_std)]
unet_ik_std = unet_ik_std[~np.isnan(unet_ik_std)]
unet_fk_std = unet_fk_std[~np.isnan(unet_fk_std)]


ax[1].errorbar(
    x=np.arange(75),
    y=unet_fk_mean,
    yerr=unet_fk_std,
    color="gold",
    # elinewidth=1.0,
    capsize=1.5,
    errorevery=(2,3),

)
ax[1].errorbar(
    x=np.arange(75),
    y=unet_ik_mean,
    yerr=unet_ik_std,
    color="darkorange",
    # elinewidth=1.0,
    capsize=1.5,
)
ax[1].errorbar(
    x=np.arange(75),
    y=transformer_ik_mean,
    yerr=transformer_ik_std,
    color="lightblue",
    # elinewidth=1.0,
    capsize=1.5,
    errorevery=(1,3),
)
ax[1].errorbar(
    x=np.arange(75),
    y=transformer_fk_mean,
    yerr=transformer_fk_std,
    color="seagreen",
    # elinewidth=1.0,
    capsize=1.5,
    errorevery=(0,3),

)

transformer_ik_mean = pd.to_numeric(train_df[train_loss[0]], errors='coerce').values
transformer_fk_mean = pd.to_numeric(train_df[train_loss[1]], errors='coerce').values
unet_ik_mean = pd.to_numeric(train_df[train_loss[2]], errors='coerce').values
unet_fk_mean = pd.to_numeric(train_df[train_loss[3]], errors='coerce').values

transformer_ik_mean = transformer_ik_mean[~np.isnan(transformer_ik_mean)]
transformer_fk_mean = transformer_fk_mean[~np.isnan(transformer_fk_mean)]
unet_ik_mean = unet_ik_mean[~np.isnan(unet_ik_mean)]
unet_fk_mean = unet_fk_mean[~np.isnan(unet_fk_mean)]

ax[0].errorbar(
  x=np.arange(1500),
  y=unet_fk_mean,
  color="gold",
)
ax[0].errorbar(
  x=np.arange(1500),
  y=unet_ik_mean,
  color="darkorange",
)
ax[0].errorbar(
  x=np.arange(1500),
  y=transformer_ik_mean,
  color="lightblue",
)
ax[0].errorbar(
  x=np.arange(1500),
  y=transformer_fk_mean,
  color="seagreen",
)

ax[0].set_yscale('log')

# Remove the upper and right spines
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

ax[0].set_xlabel("Epochs")
ax[1].set_xlabel("Evaluation Iterations")
ax[0].set_ylabel("MSE Loss", fontsize=12)
ax[1].set_ylabel("MSE Loss", fontsize=12)

# fig.tight_layout()
fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.22)

fig.suptitle("Train and Eval MSE Loss", y=0.94,  fontsize=18,)

legend_elements = [
    Line2D([0], [0], marker=None, color="lightblue", label=f'Cartesian-Space Transformer'),
    Line2D([0], [0], marker=None, color="seagreen", label=f'Joint-Space Transformer'),
    Line2D([0], [0], marker=None, color="darkorange", label=f'Cartesian-Space Unet'),
    Line2D([0], [0], marker=None, color="gold", label=f'Joint-Space Unet'),
]
fig.legend(handles=legend_elements, loc="lower center", fontsize=10, ncols=4)

plt.savefig(os.path.join(out_path, "train_eval_loss.png"), dpi=300)
