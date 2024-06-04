import pandas as pd


df = pd.read_parquet("/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_ghost_robot_dataset.parquet")


negative_indices = (df["target_rotation_w"] < 0).values()

df.loc[negative_indices, ["target_rotation_i", "target_rotation_j", "target_rotation_k"]] *= -1

negative_indices = (df["ee_rotation_w"] < 0).values()

df.loc[negative_indices, ["ee_rotation_i", "ee_rotation_j", "ee_rotation_k"]] *= -1

df.to_parquet("/mnt/sda1/Code/curobodataset/src/data/curobo_panda_pick_and_place_ghost_robot_dataset_quat_converted.parquet")