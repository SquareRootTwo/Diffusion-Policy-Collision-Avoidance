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

def dataset_augmentation_check():
    dataset = Panda_Diffusion_Policy_Full_Trajectory_Dataset_Loader(
        # "/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_robot_collision_dataset_fixed.parquet",
        "/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_robot_collision_dataset_fixed_augmented.parquet",
        fraction_to_use=0.01,
        eval=False,
        augment_data=False,
        load_dataset_to_memory=False
    )

    ee_pos_list = []
    ee_rot_list = []

    target_pos_list = []
    target_rot_list = []

    panda_collision_spheres_list = []

    for i in tqdm(range(len(dataset)), total=len(dataset), desc="Sanity Check for Dataset Augmentation"):
        data = dataset.__getitem__(i)

        obs = data["obs"]
        action = data["action"]

        target_pos = obs[-1, 14:17]
        target_rot = obs[-1, 17:21]
        final_joint = action[-1, 0:7]

        step = data["step"]

        # check that all values > step are 0
        assert torch.sum(obs[step+1:, :14]) == 0
        assert torch.sum(obs[step+1:, 21:]) == 0

        pk = PandaKinematics(device="cpu")

        joint_pos = action[:, 0:7]
        assert joint_pos.shape == (34, 7)
        
        panda_collision_spheres = obs[:(step+1), (265):(265+244)]
        fk_panda_collision_spheres = torch.zeros(((step+1), 61, 3))

        for step_i in range(step+1):
            fk_panda_collision_spheres_step_i = pk.get_panda_collision_spheres(joint_pos[step_i].unsqueeze(0).to(pk.device))
            fk_panda_collision_spheres[step_i] = fk_panda_collision_spheres_step_i


        for si in range(3, 61):
            fk_si_pos = fk_panda_collision_spheres[:, si].to(pk.device)
            si_pos = panda_collision_spheres[:, si*4:si*4+3].to(pk.device)

            si_diff = torch.norm(fk_si_pos - si_pos, dim=1)
            si_diff_mean = torch.mean(si_diff).to("cpu")

            panda_collision_spheres_list.append(si_diff_mean)

            collisions = 0

        ghost_collision_spheres = obs[:(step+1), 21:21+244]

        for si in range(61):
            for sj in range(61):
                panda_sphere = panda_collision_spheres[:, si*4:si*4+4]
                panda_sphere_radius = panda_sphere[:, 3]
                panda_sphere = panda_sphere[:, :3]
                ghost_sphere = ghost_collision_spheres[:, sj*4:sj*4+4]
                ghost_sphere_radius = ghost_sphere[:, 3]
                ghost_sphere = ghost_sphere[:, :3]

                dist = torch.norm(panda_sphere - ghost_sphere, dim=1)
                curr_collisions = torch.sum(dist < panda_sphere_radius + ghost_sphere_radius + 0.005)
                collisions += curr_collisions


        fk_ee_pos, fk_ee_rot = pk.get_ee_pose(joint_pos.to(pk.device))
        fk_ee_pos = fk_ee_pos.to("cpu")
        fk_ee_rot = fk_ee_rot.to("cpu")
        # w, i, j, k -> i, j, k, w
        fk_ee_rot_scipy = torch.cat([fk_ee_rot[:,1:], fk_ee_rot[:,:1]], dim=1).numpy()

        action_ee_pos = action[:, 7:10].to("cpu")
        action_ee_rot = action[:, 10:14]
        action_ee_rot_scipy = torch.cat([action_ee_rot[:,1:], action_ee_rot[:,:1]], dim=1).cpu().numpy()

        q_rel = R.from_quat(action_ee_rot_scipy).inv() * R.from_quat(fk_ee_rot_scipy)

        angle = R.from_quat(q_rel.as_quat()).as_rotvec()
        angle = np.linalg.norm(angle, axis=1)

        dist_ee_pos = torch.norm(fk_ee_pos - action_ee_pos, dim=1)

        dist_mean = torch.mean(dist_ee_pos)
        angle_mean = np.mean(angle)

        # print(f"step: {step}      pos: {dist_mean:.5f}      angle: {angle_mean:.5f}")

        ee_pos_list.append(dist_mean)
        ee_rot_list.append(angle_mean)

        fk_final_ee_pos, fk_final_ee_rot = pk.get_ee_pose(final_joint.to(pk.device).unsqueeze(0))
        fk_final_ee_pos = fk_final_ee_pos.to("cpu").squeeze(0)
        fk_final_ee_rot = fk_final_ee_rot.to("cpu").squeeze(0)
        # w, i, j, k -> i, j, k, w
        fk_final_ee_rot_scipy = torch.cat([fk_final_ee_rot[1:], fk_final_ee_rot[:1]], dim=0).numpy()
        target_rot_scipy = torch.cat([target_rot[1:], target_rot[:1]], dim=0).cpu().numpy()

        target_dist = torch.norm(fk_final_ee_pos - target_pos.to("cpu"))

        q_rel = R.from_quat(target_rot_scipy).inv() * R.from_quat(fk_final_ee_rot_scipy)

        angle_target = R.from_quat(q_rel.as_quat()).as_rotvec()
        angle_target = np.linalg.norm(angle_target)

        target_pos_list.append(target_dist)
        target_rot_list.append(angle_target)

    print(f"ee pos: {np.mean(target_pos_list):.10f}      ee angle: {np.mean(target_rot_list):.10f}")
    print(f"pos:    {np.mean(ee_pos_list):.10f}      angle: {np.mean(ee_rot_list):.10f}")
    print(f"panda sphere: {np.mean(panda_collision_spheres_list):.10f}")
    print(f"collisions: {collisions}")


def dataset_check():
    pk = PandaKinematics()

    df_path = "/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_robot_collision_dataset_fixed.parquet"
    # df_path = "/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_robot_collision_dataset_fixed_augmented.parquet"

    df = pd.read_parquet(df_path)

    # first check if all joint angles correspond to the ee pose
    # joint pose range
    dist_pos_list = []
    dist_rot_list = []
    angle_rot_list = []

    pick_target_distance = []
    place_target_distance = []
    retract_target_distance = []

    pick_target_angle = []
    place_target_angle = []
    retract_target_angle = []

    pick_target_distance = []
    place_target_distance = []
    retract_target_distance = []

    for e in tqdm(df["episode"].unique()):
        e_df = df[df["episode"] == e]

        # print(f"joint poses: {e_df.columns[16:23].values}")
        jp = e_df.iloc[:, 16:23].values

        # print(f"ee_pos: {e_df.columns[9:12].values}")
        ee_pos = torch.tensor(e_df.iloc[:, 9:12].values)

        assert len(jp) == len(ee_pos)

        # print(f"ee_rot: {e_df.columns[12:16].values}")
        ee_rot = torch.tensor(e_df.iloc[:, 12:16].values)

        th = torch.tensor(jp, device=pk.device, dtype=pk.dtype)
        fk_pos, fk_rot = pk.get_ee_pose(th)

        cond = fk_rot[:, 0] < 0
        fk_rot[:, 0] = torch.where(cond, -fk_rot[:, 0], fk_rot[:, 0])
        fk_rot[:, 1] = torch.where(cond, -fk_rot[:, 1], fk_rot[:, 1])
        fk_rot[:, 2] = torch.where(cond, -fk_rot[:, 2], fk_rot[:, 2])
        fk_rot[:, 3] = torch.where(cond, -fk_rot[:, 3], fk_rot[:, 3])

        cond = ee_rot[:, 0] < 0
        ee_rot[:, 0] = torch.where(cond, -ee_rot[:, 0], ee_rot[:, 0])
        ee_rot[:, 1] = torch.where(cond, -ee_rot[:, 1], ee_rot[:, 1])
        ee_rot[:, 2] = torch.where(cond, -ee_rot[:, 2], ee_rot[:, 2])
        ee_rot[:, 3] = torch.where(cond, -ee_rot[:, 3], ee_rot[:, 3])

        # print(fk_rot[0])
        # print(ee_rot[0])

        fk_ee_pos = fk_pos.to("cpu")
        fk_ee_rot = fk_rot.to("cpu")

        # w, i, j, k -> i, j, k, w
        fk_rot_scipy = np.stack([fk_ee_rot[:,1].numpy(), fk_ee_rot[:,2].numpy(), fk_ee_rot[:,3].numpy(), fk_ee_rot[:,0].numpy()], axis=1)
        ee_rot_scipy = np.stack([ee_rot[:,1], ee_rot[:,2], ee_rot[:,3], ee_rot[:,0]], axis=1)

        ee_rot_scipy /= np.repeat(np.linalg.norm(ee_rot_scipy, axis=1, keepdims=True), 4, axis=1)
        fk_rot_scipy /= np.repeat(np.linalg.norm(fk_rot_scipy, axis=1, keepdims=True), 4, axis=1)

        rel_angle = np.linalg.norm((R.from_quat(ee_rot_scipy).inv() * R.from_quat(fk_rot_scipy)).as_rotvec(degrees=False), axis=1)

        diff = torch.norm(fk_ee_pos - ee_pos, dim=1)
        dist_pos = torch.mean(diff)

        dist_pos_list.append(dist_pos)

        diff = torch.norm(fk_ee_rot - ee_rot, dim=1)
        dist_rot = torch.mean(diff)

        dist_rot_list.append(dist_rot)
        angle_rot_list.append(np.mean(rel_angle))

        # compute target distances
        pick_phase_df = e_df[e_df["phase"] == 1]
        place_phase_df = e_df[e_df["phase"] == 2]
        retract_phase_df = e_df[e_df["phase"] == 3]

        pick_target_pos = pick_phase_df.iloc[-1, 2:5].values
        pick_target_rot = pick_phase_df.iloc[-1, 5:9].values

        place_target_pos = place_phase_df.iloc[-1, 2:5].values
        place_target_rot = place_phase_df.iloc[-1, 5:9].values

        retract_target_pos = retract_phase_df.iloc[-1, 2:5].values
        retract_target_rot = retract_phase_df.iloc[-1, 5:9].values

        pick_last_ee_pos = pick_phase_df.iloc[-1, 9:12].values
        pick_last_ee_rot = pick_phase_df.iloc[-1, 12:16].values

        place_last_ee_pos = place_phase_df.iloc[-1, 9:12].values
        place_last_ee_rot = place_phase_df.iloc[-1, 12:16].values

        retract_last_ee_pos = retract_phase_df.iloc[-1, 9:12].values
        retract_last_ee_rot = retract_phase_df.iloc[-1, 12:16].values

        pick_target_distance.append(np.linalg.norm(pick_target_pos - pick_last_ee_pos))
        place_target_distance.append(np.linalg.norm(place_target_pos - place_last_ee_pos))
        retract_target_distance.append(np.linalg.norm(retract_target_pos - retract_last_ee_pos))

        # w, i, j, k -> i, j, k, w
        pick_target_rot_scipy = np.stack([pick_target_rot[1], pick_target_rot[2], pick_target_rot[3], pick_target_rot[0]], axis=0)
        place_target_rot_scipy = np.stack([place_target_rot[1], place_target_rot[2], place_target_rot[3], place_target_rot[0]], axis=0)
        retract_target_rot_scipy = np.stack([retract_target_rot[1], retract_target_rot[2], retract_target_rot[3], retract_target_rot[0]], axis=0)

        pick_last_ee_rot_scipy = np.stack([pick_last_ee_rot[1], pick_last_ee_rot[2], pick_last_ee_rot[3], pick_last_ee_rot[0]], axis=0)
        place_last_ee_rot_scipy = np.stack([place_last_ee_rot[1], place_last_ee_rot[2], place_last_ee_rot[3], place_last_ee_rot[0]], axis=0)
        retract_last_ee_rot_scipy = np.stack([retract_last_ee_rot[1], retract_last_ee_rot[2], retract_last_ee_rot[3], retract_last_ee_rot[0]], axis=0)

        pick_target_rel_angle = np.linalg.norm((R.from_quat(pick_target_rot_scipy).inv() * R.from_quat(pick_last_ee_rot_scipy)).as_rotvec(degrees=False))
        place_target_rel_angle = np.linalg.norm((R.from_quat(place_target_rot_scipy).inv() * R.from_quat(place_last_ee_rot_scipy)).as_rotvec(degrees=False))
        retract_target_rel_angle = np.linalg.norm((R.from_quat(retract_target_rot_scipy).inv() * R.from_quat(retract_last_ee_rot_scipy)).as_rotvec(degrees=False))

        pick_target_angle.append(pick_target_rel_angle)
        place_target_angle.append(place_target_rel_angle)
        retract_target_angle.append(retract_target_rel_angle)


    print(f"joint fk stats: pos: {np.mean(dist_pos_list):.5f}      angle: {np.mean(dist_rot_list):.5f}")
    print(f"pick stats:     pos: {np.mean(pick_target_distance):.5f}      angle: {np.mean(pick_target_angle):.5f}")
    print(f"place stats:    pos: {np.mean(place_target_distance):.5f}      angle: {np.mean(place_target_angle):.5f}")
    print(f"retract stats:  pos: {np.mean(retract_target_distance):.5f}      angle: {np.mean(retract_target_angle):.5f}")

    collisions = 0
    collision_episodes = []

    for idx, row in tqdm(df.iterrows()):
        # collision spheres as (x, y, z, radius)
        ghost_collision_spheres = row.iloc[25:269].values
        panda_collision_spheres = row.iloc[277:521].values

        for i in range(61):
            for j in range(61):
                panda_sphere = panda_collision_spheres[i*4:i*4+4]
                panda_sphere_radius = panda_sphere[3]
                panda_sphere = panda_sphere[:3]
                ghost_sphere = ghost_collision_spheres[j*4:j*4+4]
                ghost_sphere_radius = ghost_sphere[3]
                ghost_sphere = ghost_sphere[:3]

                dist = np.linalg.norm(panda_sphere - ghost_sphere)

                if dist < panda_sphere_radius + ghost_sphere_radius + 0.005:
                    collisions += 1
                    episode = row["episode"]
                    collision_episodes.append(episode)

    print(f"collisions: {collisions}")
    print(f"collision episodes: {collision_episodes}")


if __name__ == "__main__":
    dataset_check()
    # dataset_augmentation_check()