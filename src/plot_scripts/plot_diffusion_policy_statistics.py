import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.transform import Rotation as R

# Add the path to the parent directory to augment search for module
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

def compute_angle_between(q1, q2):
    q_rel = R.from_quat(q2).inv() * R.from_quat(q1)
    rotation_vector = q_rel.as_rotvec()
    angle = np.linalg.norm(rotation_vector)

    return angle

def plot_episode(episode_df, episode_id, out_path, timestamp):
    mean_color_gt = [1.0, 0.6, 0.2]
    mean_color_pred = [0.2, 0.6, 1.0]
    # nr_phases x nr_metrics
    fig, ax = plt.subplots(3, 3, figsize=(15, 10))
    # add space between subplots
    plt.subplots_adjust(wspace=1.6, hspace=1.0)
    # add space between subplots and the edge of the figure
    plt.tight_layout(pad=9.0)
    # add space from top
    plt.subplots_adjust(top=0.8)

    phase_names = {
        1: "Pick",
        2: "Place",
        3: "Retract",
    }

    collision = False

    # rms stats

    rms_acc = []
    rms_jerk = []
    rms_vel = []

    rms_acc_gt = []
    rms_jerk_gt = []
    rms_vel_gt = []

    rms_acc_ang = []
    rms_jerk_ang = []
    rms_vel_ang = []

    rms_acc_gt_ang = []
    rms_jerk_gt_ang = []
    rms_vel_gt_ang = []

    # target distance stats

    target_distance = []
    target_angular_distance = []

    for phase_i in episode_df['phase'].unique():

        phase_df = episode_df[episode_df['phase'] == phase_i]
        # check if collision occured and get the timesteps

        collision_timesteps = phase_df[phase_df['collision'] == 1]['step'].values
        # filter collision timesteps for unique values
        collision_timesteps = np.unique(collision_timesteps)
        print(f"episode: {episode_id} phase {phase_i}: collision_timesteps: {collision_timesteps}")

        # get number of steps
        num_steps = phase_df['step'].count()
        print(f"episode: {episode_id}: num_steps: {num_steps}")

        target_distance.append(
            np.absolute(phase_df['ee_pos_dist'].values.copy())
        )

        target_angular_distance.append([np.absolute(phase_df['angle_dist'].values.copy())])

        # get subplots for each of the metrics
        sub_vel = ax[(phase_i-1), 0]
        sub_acc = ax[(phase_i-1), 1]
        sub_jerk = ax[(phase_i-1), 2]

        # set x axis to be the step number
        sub_vel.set_xlabel('Step')
        sub_acc.set_xlabel('Step')
        sub_jerk.set_xlabel('Step')

        # set x axis range
        sub_vel.set_xlim(0, num_steps)
        sub_acc.set_xlim(0, num_steps)
        sub_jerk.set_xlim(0, num_steps)

        # set y axis to be the metric value
        sub_vel.set_ylabel('Velocity')
        sub_acc.set_ylabel('Acceleration')
        sub_jerk.set_ylabel('Jerk')

        t = np.arange(0, num_steps, 1)

        ee_obs_x = phase_df["obs_7"]
        ee_obs_y = phase_df["obs_8"]
        ee_obs_z = phase_df["obs_9"]

        ee_rot_obs_w = phase_df["obs_10"].values
        ee_rot_obs_i = phase_df["obs_11"].values
        ee_rot_obs_j = phase_df["obs_12"].values
        ee_rot_obs_k = phase_df["obs_13"].values

        ee_rot_obs_angular_vel = np.empty(len(ee_rot_obs_w))
        for i in range(len(ee_rot_obs_w)-1):
            q1 = np.array([ee_rot_obs_i[i], ee_rot_obs_j[i], ee_rot_obs_k[i], ee_rot_obs_w[i]])
            q2 = np.array([ee_rot_obs_i[i+1], ee_rot_obs_j[i+1], ee_rot_obs_k[i+1], ee_rot_obs_w[i+1]])
            ee_rot_obs_angular_vel[i+1] = compute_angle_between(q1, q2)

        ee_rot_obs_angular_vel[0] = ee_rot_obs_angular_vel[1]

        ee_rot_obs_angular_acc = np.gradient(ee_rot_obs_angular_vel, t)
        ee_rot_obs_angular_jerk = np.gradient(ee_rot_obs_angular_acc, t)

        ee_gt_x = phase_df["gt_obs_7"]
        ee_gt_y = phase_df["gt_obs_8"]
        ee_gt_z = phase_df["gt_obs_9"]

        ee_rot_gt_w = phase_df["gt_obs_10"].values
        ee_rot_gt_i = phase_df["gt_obs_11"].values
        ee_rot_gt_j = phase_df["gt_obs_12"].values
        ee_rot_gt_k = phase_df["gt_obs_13"].values

        ee_rot_gt_angular_vel = np.empty(len(ee_rot_gt_w))
        for i in range(len(ee_rot_gt_w)-1):
            q1 = np.array([ee_rot_gt_i[i], ee_rot_gt_j[i], ee_rot_gt_k[i], ee_rot_gt_w[i]])
            q2 = np.array([ee_rot_gt_i[i+1], ee_rot_gt_j[i+1], ee_rot_gt_k[i+1], ee_rot_gt_w[i+1]])
            ee_rot_gt_angular_vel[i+1] = compute_angle_between(q1, q2)

        ee_rot_gt_angular_vel[0] = ee_rot_gt_angular_vel[1]

        ee_rot_gt_angular_acc = np.gradient(ee_rot_gt_angular_vel, t)
        ee_rot_gt_angular_jerk = np.gradient(ee_rot_gt_angular_acc, t)

        vel_obs_x = np.gradient(ee_obs_x, t, axis=0)
        vel_obs_y = np.gradient(ee_obs_y, t, axis=0)
        vel_obs_z = np.gradient(ee_obs_z, t, axis=0)

        vel_obs = np.stack([vel_obs_x, vel_obs_y, vel_obs_z], axis=0)

        acc_obs_x = np.gradient(vel_obs_x, t, axis=0)
        acc_obs_y = np.gradient(vel_obs_y, t, axis=0)
        acc_obs_z = np.gradient(vel_obs_z, t, axis=0)

        acc_obs = np.stack([acc_obs_x, acc_obs_y, acc_obs_z], axis=0)

        jerk_obs_x = np.gradient(acc_obs_x, t, axis=0)
        jerk_obs_y = np.gradient(acc_obs_y, t, axis=0)
        jerk_obs_z = np.gradient(acc_obs_z, t, axis=0)

        jerk_obs = np.stack([jerk_obs_x, jerk_obs_y, jerk_obs_z], axis=0)

        vel_gt_x = np.gradient(ee_gt_x, t, axis=0)
        vel_gt_y = np.gradient(ee_gt_y, t, axis=0)
        vel_gt_z = np.gradient(ee_gt_z, t, axis=0)

        vel_gt = np.stack([vel_gt_x, vel_gt_y, vel_gt_z], axis=0)

        acc_gt_x = np.gradient(vel_gt_x, t, axis=0)
        acc_gt_y = np.gradient(vel_gt_y, t, axis=0)
        acc_gt_z = np.gradient(vel_gt_z, t, axis=0)

        acc_gt = np.stack([acc_gt_x, acc_gt_y, acc_gt_z], axis=0)

        jerk_gt_x = np.gradient(acc_gt_x, t, axis=0)
        jerk_gt_y = np.gradient(acc_gt_y, t, axis=0)
        jerk_gt_z = np.gradient(acc_gt_z, t, axis=0)

        jerk_gt = np.stack([jerk_gt_x, jerk_gt_y, jerk_gt_z], axis=0)

        vel_obs_norm = np.linalg.norm(vel_obs, axis=0)
        acc_obs_norm = np.linalg.norm(acc_obs, axis=0)
        jerk_obs_norm = np.linalg.norm(jerk_obs, axis=0)

        vel_gt_norm = np.linalg.norm(vel_gt, axis=0)
        acc_gt_norm = np.linalg.norm(acc_gt, axis=0)
        jerk_gt_norm = np.linalg.norm(jerk_gt, axis=0)


        # plot mean end effector values

        # Normal distances
        # Predictions
        # plot velocity
        sub_vel.plot(t, vel_obs_norm, label='prediction', color=mean_color_pred)

        # plot acceleration
        sub_acc.plot(t, acc_obs_norm, label='prediction', color=mean_color_pred)

        # plot jerk
        sub_jerk.plot(t, jerk_obs_norm, label='prediction', color=mean_color_pred)


        # Ground truth
        # plot velocity
        sub_vel.plot(t, vel_gt_norm, label='ground truth', color=mean_color_gt)

        # plot acceleration
        sub_acc.plot(t, acc_gt_norm, label='ground truth', color=mean_color_gt)

        # plot jerk
        sub_jerk.plot(t, jerk_gt_norm, label='ground truth', color=mean_color_gt)


        # plot mean end effector values
        sub_vel_ang = sub_vel.twinx() 
        sub_vel_ang.set_ylabel('Angular Velocity')
        sub_acc_ang = sub_acc.twinx()
        sub_acc_ang.set_ylabel('Angular Acceleration')
        sub_jerk_ang = sub_jerk.twinx()
        sub_jerk_ang.set_ylabel('Angular Jerk')

        # Angular distances
        # Predictions
        # plot angular velocity
        sub_vel_ang.plot(t, ee_rot_obs_angular_vel, label='prediction (angular)', linestyle=':', color=mean_color_pred)

        # plot angular acceleration
        sub_acc_ang.plot(t, ee_rot_obs_angular_acc, label='prediction (angular)', linestyle=':', color=mean_color_pred)

        # plot angular jerk
        sub_jerk_ang.plot(t, ee_rot_obs_angular_jerk, label='prediction (angular)', linestyle=':', color=mean_color_pred)

        # Ground truth
        # plot angular velocity
        sub_vel_ang.plot(t, ee_rot_gt_angular_vel, label='ground truth (angular)', linestyle=':', color=mean_color_gt)

        # plot angular acceleration
        sub_acc_ang.plot(t, ee_rot_gt_angular_acc, label='ground truth (angular)', linestyle=':', color=mean_color_gt)

        # plot angular jerk
        sub_jerk_ang.plot(t, ee_rot_gt_angular_jerk, label='ground truth (angular)', linestyle=':', color=mean_color_gt)

        # plot collision timesteps
        for collision_step in collision_timesteps:
            if collision_step > num_steps:
                continue

            collision = True
            sub_vel.axvline(x=collision_step, color=[0.6, 0.0, 0.0], linestyle='--', label="collision")
            sub_acc.axvline(x=collision_step, color=[0.6, 0.0, 0.0], linestyle='--', label="collision")
            sub_jerk.axvline(x=collision_step, color=[0.6, 0.0, 0.0], linestyle='--', label="collision")

        # add title to the subplots
        sub_vel.set_title(f'{phase_names[phase_i]} Phase - Velocity', fontsize=12) 
        sub_acc.set_title(f'{phase_names[phase_i]} Phase - Acceleration', fontsize=12)
        sub_jerk.set_title(f'{phase_names[phase_i]} Phase - Jerk', fontsize=12)


        # collect stats
        rms_vel.append(vel_obs_norm.copy())
        rms_acc.append(acc_obs_norm.copy())
        rms_jerk.append(jerk_obs_norm.copy())

        rms_vel_gt.append(vel_gt_norm.copy())
        rms_acc_gt.append(acc_gt_norm.copy())
        rms_jerk_gt.append(jerk_gt_norm.copy())

        rms_vel_ang.append(ee_rot_obs_angular_vel.copy())
        rms_acc_ang.append(ee_rot_obs_angular_acc.copy())
        rms_jerk_ang.append(ee_rot_obs_angular_jerk.copy())

        rms_vel_gt_ang.append(ee_rot_gt_angular_vel.copy())
        rms_acc_gt_ang.append(ee_rot_gt_angular_acc.copy())
        rms_jerk_gt_ang.append(ee_rot_gt_angular_jerk.copy())


    # handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = [], []

    for ax in plt.gcf().axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    by_label = dict(zip(labels, handles))
    # sort dict by labels
    by_label = dict(sorted(by_label.items(), key=lambda item: item[0]))

    if collision:
        ncols = 5
    else:
        ncols = 4

    fig.legend(
        by_label.values(), 
        by_label.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.9), 
        ncol=ncols
    )

    # set the title of the plot
    fig.suptitle(
        f'Eval Episode {episode_id} - Model {timestamp} - EE Trajectory Smoothness', 
        fontsize=16, 
        fontweight='bold'
    )

    # save the plots
    plt.savefig(os.path.join(out_path, f'episode_{episode_id}_plots.png'), dpi=300)

    # concatenate the stats
    rms_acc = np.concatenate(rms_acc)
    rms_jerk = np.concatenate(rms_jerk)
    rms_vel = np.concatenate(rms_vel)

    rms_acc_gt = np.concatenate(rms_acc_gt)
    rms_jerk_gt = np.concatenate(rms_jerk_gt)
    rms_vel_gt = np.concatenate(rms_vel_gt)

    rms_acc_ang = np.concatenate(rms_acc_ang)
    rms_jerk_ang = np.concatenate(rms_jerk_ang)
    rms_vel_ang = np.concatenate(rms_vel_ang)

    rms_acc_gt_ang = np.concatenate(rms_acc_gt_ang)
    rms_jerk_gt_ang = np.concatenate(rms_jerk_gt_ang)
    rms_vel_gt_ang = np.concatenate(rms_vel_gt_ang)

    ret_stats = {
        "rms_vel": np.sqrt(np.mean(vel_obs_norm**2)),
        "rms_acc": np.sqrt(np.mean(acc_obs_norm**2)),
        "rms_jerk": np.sqrt(np.mean(jerk_obs_norm**2)),
        "rms_vel_gt": np.sqrt(np.mean(vel_gt_norm**2)),
        "rms_acc_gt": np.sqrt(np.mean(acc_gt_norm**2)),
        "rms_jerk_gt": np.sqrt(np.mean(jerk_gt_norm**2)),
        "rms_vel_ang": np.sqrt(np.mean(ee_rot_obs_angular_vel**2)),
        "rms_acc_ang": np.sqrt(np.mean(ee_rot_obs_angular_acc**2)),
        "rms_jerk_ang": np.sqrt(np.mean(ee_rot_obs_angular_jerk**2)),
        "rms_vel_gt_ang": np.sqrt(np.mean(ee_rot_gt_angular_vel**2)),
        "rms_acc_gt_ang": np.sqrt(np.mean(ee_rot_gt_angular_acc**2)),
        "rms_jerk_gt_ang": np.sqrt(np.mean(ee_rot_gt_angular_jerk**2)),
        "target_distance_pick": target_distance[0],
        "target_distance_place": target_distance[1],
        "target_distance_retract": target_distance[2],
        "target_angular_distance_pick": target_angular_distance[0],
        "target_angular_distance_place": target_angular_distance[1],
        "target_angular_distance_retract": target_angular_distance[2],  
    }

    return ret_stats


def main(timestamp):
    """

    """
    
    base_path = os.path.join(root_path, 'data/thesis_eval', f'{timestamp}')
    out_path = os.path.join(base_path, "plots")

    nr_episodes = 101
    print(f"base_path: {base_path}")

    collisions = []
    failed_convergence = []

    nr_phases = 3

    # rms stats for each episode
    rms_acc = []
    rms_jerk = []
    rms_vel = []

    rms_acc_gt = []
    rms_jerk_gt = []
    rms_vel_gt = []

    rms_acc_ang = []
    rms_jerk_ang = []
    rms_vel_ang = []

    rms_acc_gt_ang = []
    rms_jerk_gt_ang = []
    rms_vel_gt_ang = []

    # target distance stats
    target_distance_pick = []
    target_distance_place = []
    target_distance_retract = []

    target_angular_distance_pick = []
    target_angular_distance_place = []
    target_angular_distance_retract = []


    # for episode_id, episode_df_path in enumerate(glob(os.path.join(base_path, 'episode_*.parquet'))):
    for i in range(0, nr_episodes, 1):
        episode_id = i
        episode_df_path = os.path.join(base_path, f'episode_{episode_id}_data.parquet')
        os.makedirs(out_path, exist_ok=True)
        episode_df = pd.read_parquet(episode_df_path)

        ret_stats = plot_episode(episode_df, episode_id, out_path, timestamp)

        # collect stats
        rms_acc.append(ret_stats["rms_acc"])
        rms_jerk.append(ret_stats["rms_jerk"])
        rms_vel.append(ret_stats["rms_vel"])

        rms_acc_gt.append(ret_stats["rms_acc_gt"])
        rms_jerk_gt.append(ret_stats["rms_jerk_gt"])
        rms_vel_gt.append(ret_stats["rms_vel_gt"])

        rms_acc_ang.append(ret_stats["rms_acc_ang"])
        rms_jerk_ang.append(ret_stats["rms_jerk_ang"])
        rms_vel_ang.append(ret_stats["rms_vel_ang"])

        rms_acc_gt_ang.append(ret_stats["rms_acc_gt_ang"])
        rms_jerk_gt_ang.append(ret_stats["rms_jerk_gt_ang"])
        rms_vel_gt_ang.append(ret_stats["rms_vel_gt_ang"])

        target_distance_pick.append(ret_stats["target_distance_pick"])
        target_distance_place.append(ret_stats["target_distance_place"])
        target_distance_retract.append(ret_stats["target_distance_retract"])

        target_angular_distance_pick.append(ret_stats["target_angular_distance_pick"])
        target_angular_distance_place.append(ret_stats["target_angular_distance_place"])
        target_angular_distance_retract.append(ret_stats["target_angular_distance_retract"])


        # sum collisions as indicator per phase
        episode_collisions = episode_df.groupby('phase')['collision'].any().sum()
        print(f"episode: {episode_id}: episode_collisions: {episode_collisions}")
        collisions.append((episode_id, episode_collisions))

        # sum all failed convergence from the episode
        episode_failed_convergence = episode_df['convergence_failed'].sum()
        failed_convergence.append((episode_id, episode_failed_convergence))

    # plot the number of collisions and failed convergence
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    # add space between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # add space between subplots and the edge of the figure
    plt.tight_layout(pad=9.0)
    # add space from top
    plt.subplots_adjust(top=0.8)

    # compute the average convergence fails
    avg_convergence_fails = (sum([x[1] for x in failed_convergence]) / (nr_episodes))

    # compute the average number of collisions
    avg_collisions = sum([x[1] for x in collisions]) / (nr_episodes)

    # set fig title
    fig.suptitle(
        f'Eval Dataset - Model {timestamp} - Collisions and Failed Convergence', 
        fontsize=21, 
        fontweight='bold'
    )

    # plot the number of collisions
    ax[0].set_title('Number of Collisions per Episode')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Number of Collisions')
    ax[0].bar([x[0] for x in collisions], [x[1] for x in collisions])

    # plot the average number of collisions
    ax[0].axhline(y=avg_collisions, color=[0.6, 0.0, 0.0], linestyle='--', label=f'Average Collisions: {avg_collisions:.2f}')
    
    # mark the average number of collisions on the axis
    ax[0].text(-4, avg_collisions + 0.1, f'{avg_collisions:.2f}', color=[0.6, 0.0, 0.0])

    # plot the number of failed convergence
    ax[1].set_title('Number of Failed Convergence per Episode')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Number of Failed Convergence')
    ax[1].bar([x[0] for x in failed_convergence], [x[1] for x in failed_convergence])

    # plot the avg convergence fails
    ax[1].axhline(y=avg_convergence_fails, color=[0.6, 0.0, 0.0], linestyle='--', label=f'Average Convergance Fails: {avg_convergence_fails:.2f}')

    # print the avg convergence fails next to the axis/on the side of the plot
    ax[1].text(-4, avg_convergence_fails + 0.1, f'{avg_convergence_fails:.2f}', color=[0.6, 0.0, 0.0])

    # add legend to the plots, centered at the top of the figure
    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0.9), 
        ncol=2
    )

    # save the plots
    stats_plot_path = os.path.join(out_path, f'{timestamp}_eval_dataset_success_rate_and_collision_rate_statistics.png')
    print(f"stats_plot_path: {stats_plot_path}")
    plt.savefig(stats_plot_path, dpi=300)

    # clear plot
    plt.clf()

    # plot the average rms values
    fig, ax = plt.subplots(6, 1, figsize=(15, 15))
    # add space between subplots
    plt.subplots_adjust(wspace=1.5, hspace=1.5)

    # add space between subplots and the edge of the figure
    # plt.tight_layout(pad=2.0)

    # add pad on top of figure
    # plt.subplots_adjust(top=10)

    # add padding to left, right, top, bottom
    fig.tight_layout(pad=4.0, h_pad=4.0, w_pad=4.0, rect=[0.05, 0.0, 0.95, 0.95])

    # set fig title
    fig.suptitle(
        f'Eval Dataset - Model {timestamp} - Average RMS Values', 
        fontsize=21, 
        fontweight='bold'
    )
    width = 0.2
    episodes = np.arange(nr_episodes)

    # plot the average rms values, rms gt, rms angular values, rms gt angular values as 4 bars per plot
    ax[0].set_title('Average RMS Velocity')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('RMS Velocity')
    
    ax[0].bar(episodes, rms_vel, alpha=0.5, color='green', label='RMS Pred')
    ax[0].bar(episodes, rms_vel_gt, alpha=0.5, color='orange', label='RMS GT')

    ax[1].set_title('Average RMS Angular Velocity')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('RMS Angular Velocity')

    ax[1].bar(episodes, rms_vel_ang, alpha=0.5, color='green', label='RMS Pred')
    ax[1].bar(episodes, rms_vel_gt_ang, alpha=0.5, color='orange', label='RMS GT')

    # plot the average rms values
    ax[2].set_title('Average RMS Acceleration')
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('RMS Acceleration')

    ax[2].bar(episodes, rms_acc, alpha=0.5, color='green', label='RMS Pred')
    ax[2].bar(episodes, rms_acc_gt, alpha=0.5, color='orange', label='RMS GT')

    # plot the average rms values
    ax[3].set_title('Average RMS Angular Acceleration')
    ax[3].set_xlabel('Episode')
    ax[3].set_ylabel('RMS Angular Acceleration')

    ax[3].bar(episodes, rms_acc_ang, alpha=0.5, color='green', label='RMS Pred')
    ax[3].bar(episodes, rms_acc_gt_ang, alpha=0.5, color='orange', label='RMS GT')

    # plot the average rms values
    ax[4].set_title('Average RMS Jerk')
    ax[4].set_xlabel('Episode')
    ax[4].set_ylabel('RMS Jerk')

    ax[4].bar(episodes, rms_jerk, alpha=0.5, color='green', label='RMS Pred')
    ax[4].bar(episodes, rms_jerk_gt, alpha=0.5, color='orange', label='RMS GT')

    # plot the average rms values
    ax[5].set_title('Average RMS Angular Jerk')
    ax[5].set_xlabel('Episode')
    ax[5].set_ylabel('RMS Angular Jerk')

    ax[5].bar(episodes, rms_jerk_ang, alpha=0.5, color='green', label='RMS Pred')
    ax[5].bar(episodes, rms_jerk_gt_ang, alpha=0.5, color='orange', label='RMS GT')

    # add average rms values to the plots
    ax[0].axhline(y=np.mean(rms_vel), color='green', linestyle='--', label=f'Mean RMS Pred')
    ax[0].axhline(y=np.mean(rms_vel_gt), color='orange', linestyle='--', label=f'Mean RMS GT')
    
    ax[1].axhline(y=np.mean(rms_vel_ang), color='green', linestyle='--', label=f'Mean RMS Pred')
    ax[1].axhline(y=np.mean(rms_vel_gt_ang), color='orange', linestyle='--', label=f'Mean RMS GT')
    
    ax[2].axhline(y=np.mean(rms_acc), color='green', linestyle='--', label=f'Mean RMS Pred')
    ax[2].axhline(y=np.mean(rms_acc_gt), color='orange', linestyle='--', label=f'Mean RMS GT')
    
    ax[3].axhline(y=np.mean(rms_acc_ang), color='green', linestyle='--', label=f'Mean RMS Pred')
    ax[3].axhline(y=np.mean(rms_acc_gt_ang), color='orange', linestyle='--', label=f'Mean RMS GT')

    ax[4].axhline(y=np.mean(rms_jerk), color='green', linestyle='--', label=f'Mean RMS Pred')
    ax[4].axhline(y=np.mean(rms_jerk_gt), color='orange', linestyle='--', label=f'Mean RMS GT')
    
    ax[5].axhline(y=np.mean(rms_jerk_ang), color='green', linestyle='--', label=f'Mean RMS')
    ax[5].axhline(y=np.mean(rms_jerk_gt_ang), color='orange', linestyle='--', label=f'Mean RMS GT')

    # filter out duplicate handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # sort the labels
    by_label = dict(zip(labels, handles))
    
    # add labels to the plots
    fig.legend(
        by_label.values(), 
        by_label.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95), 
        ncol=4
    )

    # save the plots
    rms_plot_path = os.path.join(out_path, f'{timestamp}_eval_dataset_average_rms_values.png')
    print(f"rms_plot_path: {rms_plot_path}")
    plt.savefig(rms_plot_path, dpi=300)

    # clear plot
    plt.clf()

    # plot the target distance and angular distance
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))
    # add space between subplots
    plt.subplots_adjust(wspace=1.5, hspace=1.5)

    # add padding to left, right, top, bottom
    fig.tight_layout(pad=4.0, h_pad=4.0, w_pad=4.0, rect=[0.05, 0.0, 0.95, 0.92])

    # set fig title
    fig.suptitle(
        f'Eval Dataset - Model {timestamp} - Target Distance', 
        fontsize=21, 
        fontweight='bold'
    )

    # angular distance on a second y axis
    angular_ax_0 = ax[0].twinx()
    angular_ax_1 = ax[1].twinx()
    angular_ax_2 = ax[2].twinx()

    ax[0].set_title('Pick Phase - Distance and Angular Distance')

    ax[1].set_title('Place Phase - Distance and Angular Distance')

    ax[2].set_title('Retract Phase - Distance and Angular Distance')

    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Distance')

    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Distance')
    
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('Distance')

    # set y axis as log scale
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')

    # set angular y axis to log scale
    angular_ax_0.set_yscale('log')
    angular_ax_1.set_yscale('log')
    angular_ax_2.set_yscale('log')

    angular_ax_0.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
    angular_ax_1.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
    angular_ax_2.grid(axis='y', which='both', linestyle='--', linewidth=0.5)

    # add horizontal lines

    angular_ax_0.set_ylabel('Angular Distance')

    angular_ax_1.set_ylabel('Angular Distance')

    angular_ax_2.set_ylabel('Angular Distance')
    
    mean_final_dist_pick = 0
    mean_final_dist_rot_pick = 0

    mean_final_dist_place = 0
    mean_final_dist_rot_place = 0

    mean_final_dist_retract = 0
    mean_final_dist_rot_retract = 0

    final_dist_pick = []
    final_dist_rot_pick = []

    final_dist_place = []
    final_dist_rot_place = []

    final_dist_retract = []
    final_dist_rot_retract = []

    target_distance_pick = np.array(target_distance_pick)
    target_distance_place = np.array(target_distance_place)
    target_distance_retract = np.array(target_distance_retract)
    
    for i in range(nr_episodes):

        target_angle_pick = np.array(target_angular_distance_pick[i]).squeeze()
        target_angle_place = np.array(target_angular_distance_place[i]).squeeze()
        target_angle_retract = np.array(target_angular_distance_retract[i]).squeeze()

        mean_final_dist_pick += target_distance_pick[i, -1]
        mean_final_dist_rot_pick += target_angle_pick[-1]

        mean_final_dist_place += target_distance_place[i, -1]
        mean_final_dist_rot_place += target_angle_place[-1]

        mean_final_dist_retract += target_distance_retract[i, -1]
        mean_final_dist_rot_retract += target_angle_retract[-1]

        final_dist_pick.append(target_distance_pick[i, -1])
        final_dist_rot_pick.append(target_angle_pick[-1])

        final_dist_place.append(target_distance_place[i, -1])
        final_dist_rot_place.append(target_angle_place[-1])

        final_dist_retract.append(target_distance_retract[i, -1])
        final_dist_rot_retract.append(target_angle_retract[-1])


        ax[0].plot(target_distance_pick[i], label='Distance', color='green', alpha=0.4, linewidth=0.7)
        angular_ax_0.plot(target_angle_pick, label='Angular Distance', color='orange', alpha=0.4, linewidth=0.7)

        ax[1].plot(target_distance_place[i], label='Distance', color='green', alpha=0.4, linewidth=0.7)
        angular_ax_1.plot(target_angle_place, label='Angular Distance', color='orange', alpha=0.4, linewidth=0.7)

        ax[2].plot(target_distance_retract[i], label='Distance', color='green', alpha=0.4, linewidth=0.7)
        angular_ax_2.plot(target_angle_retract, label='Angular Distance', color='orange', alpha=0.4, linewidth=0.7)


    mean_final_dist_rot_pick = mean_final_dist_rot_pick / nr_episodes
    mean_final_dist_rot_place = mean_final_dist_rot_place / nr_episodes
    mean_final_dist_rot_retract = mean_final_dist_rot_retract / nr_episodes

    mean_final_dist_pick = mean_final_dist_pick / nr_episodes
    mean_final_dist_place = mean_final_dist_place / nr_episodes
    mean_final_dist_retract = mean_final_dist_retract / nr_episodes

    # add labels with added mean values to the plots
    ax[0].axhline(y=mean_final_dist_pick, color='blue', linestyle='--', label=f'Pick Mean Distance: {mean_final_dist_pick:.3f} [m]')
    angular_ax_0.axhline(y=mean_final_dist_rot_pick, color='red', linestyle='--', label=f'Pick Mean Angular Distance: {mean_final_dist_rot_pick:.3f} [deg]')

    ax[1].axhline(y=mean_final_dist_place, color='blue', linestyle='--', label=f'Place Mean Distance: {mean_final_dist_place:.3f} [m]')
    angular_ax_1.axhline(y=mean_final_dist_rot_place, color='red', linestyle='--', label=f'Place Mean Angular Distance: {mean_final_dist_rot_place:.3f} [deg]')

    ax[2].axhline(y=mean_final_dist_retract, color='blue', linestyle='--', label=f'Retract Mean Distance: {mean_final_dist_retract:.3f} [m]')
    angular_ax_2.axhline(y=mean_final_dist_rot_retract, color='red', linestyle='--', label=f'Retract Mean Angular Distance: {mean_final_dist_rot_retract:.2f} [deg]')

    # get union of all axes lables
    handles_ax0, labels_ax0 = ax[0].get_legend_handles_labels()
    handles_ax0_angular, labels_ax0_angular = angular_ax_0.get_legend_handles_labels()

    handles_ax1, labels_ax1 = ax[1].get_legend_handles_labels()
    handles_ax1_angular, labels_ax1_angular = angular_ax_1.get_legend_handles_labels()

    handles_ax2, labels_ax2 = ax[2].get_legend_handles_labels()
    handles_ax2_angular, labels_ax2_angular = angular_ax_2.get_legend_handles_labels()

    # merge the labels and handles
    handles = handles_ax0 + handles_ax0_angular + handles_ax1 + handles_ax1_angular + handles_ax2 + handles_ax2_angular
    labels = labels_ax0 + labels_ax0_angular + labels_ax1 + labels_ax1_angular + labels_ax2 + labels_ax2_angular

    # create a dictionary with the labels and handles
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items(), key=lambda item: item[0]))

    # add legend to the plots
    fig.legend(
        by_label.values(), 
        by_label.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95), 
        ncol=4
    )

    # save the plots
    target_distance_plot_path = os.path.join(out_path, f'{timestamp}_eval_dataset_target_distance_and_angular_distance.png')
    print(f"target_distance_plot_path: {target_distance_plot_path}")
    plt.savefig(target_distance_plot_path, dpi=300)

    # clear plot
    plt.clf()

    # plot bar plot for final distance and angular distance for each phase
    fig, ax = plt.subplots(3, 2, figsize=(15, 10))
    # add space between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # add padding to left, right, top, bottom
    fig.tight_layout(pad=4.0, h_pad=4.0, w_pad=4.0, rect=[0.05, 0.0, 0.95, 0.92])

    # set fig title
    fig.suptitle(
        f'Eval Dataset - Model {timestamp} - Final Target Distance', 
        fontsize=21, 
        fontweight='bold'
    )

    # plot the final distance and angular distance for each phase and episode
    ax[0, 0].set_title('Pick Phase - Distance')
    ax[0, 0].set_xlabel('Episode')

    ax[0, 1].set_title('Pick Phase - Angular Distance')
    ax[0, 1].set_xlabel('Episode')

    ax[1, 0].set_title('Place Phase - Distance')
    ax[1, 0].set_xlabel('Episode')

    ax[1, 1].set_title('Place Phase - Angular Distance')
    ax[1, 1].set_xlabel('Episode')

    ax[2, 0].set_title('Retract Phase - Distance')
    ax[2, 0].set_xlabel('Episode')

    ax[2, 1].set_title('Retract Phase - Angular Distance')
    ax[2, 1].set_xlabel('Episode')

    # plot the final distance and angular distance for each phase and episode
    episodes = np.arange(nr_episodes)
    ax[0, 0].bar(episodes, final_dist_pick, alpha=0.5, color='green', label='Final Distance')
    ax[0, 1].bar(episodes, final_dist_rot_pick, alpha=0.5, color='orange', label='Final Angular Distance')

    ax[1, 0].bar(episodes, final_dist_place, alpha=0.5, color='green', label='Final Distance')
    ax[1, 1].bar(episodes, final_dist_rot_place, alpha=0.5, color='orange', label='Final Angular Distance')

    ax[2, 0].bar(episodes, final_dist_retract, alpha=0.5, color='green', label='Final Distance')
    ax[2, 1].bar(episodes, final_dist_rot_retract, alpha=0.5, color='orange', label='Final Angular Distance')

    # add average final distance and angular distance to the plots
    ax[0, 0].axhline(y=mean_final_dist_pick, color='blue', linestyle='--', label=f'Pick Mean Final Distance: {mean_final_dist_pick:.3f} [m]')
    ax[0, 1].axhline(y=mean_final_dist_rot_pick, color='red', linestyle='--', label=f'Pick Mean Final Angular Distance: {mean_final_dist_rot_pick:.3f} [deg]')

    ax[1, 0].axhline(y=mean_final_dist_place, color='blue', linestyle='--', label=f'Place Mean Final Distance: {mean_final_dist_place:.3f} [m]')
    ax[1, 1].axhline(y=mean_final_dist_rot_place, color='red', linestyle='--', label=f'Place Mean Final Angular Distance: {mean_final_dist_rot_place:.3f} [deg]')

    ax[2, 0].axhline(y=mean_final_dist_retract, color='blue', linestyle='--', label=f'Retract Mean Final Distance: {mean_final_dist_retract:.3f} [m]')
    ax[2, 1].axhline(y=mean_final_dist_rot_retract, color='red', linestyle='--', label=f'Retract Mean Final Angular Distance: {mean_final_dist_rot_retract:.3f} [deg]')

    # get union of all axes lables
    handles_ax0, labels_ax0 = ax[0, 0].get_legend_handles_labels()
    handles_ax0_angular, labels_ax0_angular = ax[0, 1].get_legend_handles_labels()
    
    handles_ax1, labels_ax1 = ax[1, 0].get_legend_handles_labels()
    handles_ax1_angular, labels_ax1_angular = ax[1, 1].get_legend_handles_labels()

    handles_ax2, labels_ax2 = ax[2, 0].get_legend_handles_labels()
    handles_ax2_angular, labels_ax2_angular = ax[2, 1].get_legend_handles_labels()

    # merge the labels and handles
    handles = handles_ax0 + handles_ax0_angular + handles_ax1 + handles_ax1_angular + handles_ax2 + handles_ax2_angular
    labels = labels_ax0 + labels_ax0_angular + labels_ax1 + labels_ax1_angular + labels_ax2 + labels_ax2_angular

    # create a dictionary with the labels and handles
    by_label = dict(zip(labels, handles))
    by_label = dict(sorted(by_label.items(), key=lambda item: item[0]))

    # add legend to the plots
    fig.legend(
        by_label.values(), 
        by_label.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95), 
        ncol=4
    )

    # save the plots
    final_target_distance_plot_path = os.path.join(out_path, f'{timestamp}_eval_dataset_final_target_distance_and_angular_distance.png')
    print(f"final_target_distance_plot_path: {final_target_distance_plot_path}")
    plt.savefig(final_target_distance_plot_path, dpi=300)



if __name__ == "__main__":
    eval_models = [ 
        # Thesis Experiments
        # U-Net Joint Space Model
        # "2024-04-18_00-49-51_unet_joint_space",

        # Transformer Joint Space Model
        # "2024-04-22_00-04-23_transformer_joint_space",

        # U-Net Cartesian Space Model
        # "2024-04-18_19-46-27_unet_cartesian_space",

        # Transformer Cartesian Space Model
        # "2024-04-26_02-07-12_transformer_cartesian_space",

        # interpolation
        # "2024-04-22_00-04-23_transformer_trajectory_interpolation",

        # unet joint space, weighted sampling
        "2024-05-10_23-40-20_unet_joint_space",
    ]

    for t in eval_models:
        main(t)