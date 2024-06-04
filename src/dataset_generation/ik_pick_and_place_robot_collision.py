import torch


a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")

parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)
args = parser.parse_args()

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "width": "1920",
        "height": "1080",
    }
)
from typing import Dict

import os
import carb
import pandas as pd
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.logger import log_warn
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

ISAAC_SIM_23 = False
try:
    # Third Party
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    # Third Party
    from omni.importer.urdf import _urdf  # isaac sim 2023.1

    ISAAC_SIM_23 = True

# Third Party
from omni.isaac.core.utils.extensions import enable_extension

# CuRobo
from curobo.util_file import get_assets_path, get_filename, get_path_of_dir, join_path

def quaternion_multiplication(q1, q2):
    # convention: (w, x, y, z)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])

def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    ext_list = []
    if headless_mode is not None:
        log_warn("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True


def add_robot_to_scene(
    robot_config: Dict,
    world: World,
    load_from_usd: bool = False,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.array = np.array([0, 0, 0]),
):
    urdf_interface = _urdf.acquire_urdf_interface()

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.make_default_prim = False
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 20000
    import_config.default_position_drive_damping = 500
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0
    asset_path = get_assets_path()
    if (
        "external_asset_path" in robot_config["kinematics"]
        and robot_config["kinematics"]["external_asset_path"] is not None
    ):
        asset_path = robot_config["kinematics"]["external_asset_path"]
    full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
    dest_path = subroot
    robot_path = urdf_interface.import_robot(
        robot_path,
        filename,
        imported_robot,
        import_config,
        dest_path,
    )

    robot_p = Robot(
        prim_path=robot_path,
        name=robot_name,
        position=position,
    )
    if ISAAC_SIM_23:
        robot_p.set_solver_velocity_iteration_count(4)
        robot_p.set_solver_position_iteration_count(44)

        world._physics_context.set_solver_type("PGS")

    robot = world.scene.add(robot_p)

    return robot, robot_path


def init_collision_objects(collision_df):
    # sample positions on a circle around the robot
    obstacle_positions = {}
    obstacles = []
    num_obstacles = len(collision_df.filter(regex='^robot_sphere_\d+_x$', axis=1).columns)
    random_row = collision_df.sample()

    print(f"Nr Obstacles: {num_obstacles}", flush=True)

    offset = np.array([1.0, 0.0, 0.0]) # offset for obstacle positions (since current panda is at location (0,0,0))

    for i in range(num_obstacles):
        sphere_center = offset + np.array([
            -1.0 * random_row[f"robot_sphere_{i}_x"], # mirror x axis for collision robot (since pick and place field is on the opposite side)
            random_row[f"robot_sphere_{i}_y"], 
            random_row[f"robot_sphere_{i}_z"]]
        ).reshape(3)
        sphere_radius = random_row[f"robot_sphere_{i}_radius"].item()

        obstacle = sphere.DynamicSphere(
            f"/World/obstacles/sphere_{i}",
            position=sphere_center,
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 1.0, 1.0]),
            radius=sphere_radius,
        )

        obstacle.disable_rigid_body_physics()
        obstacle.set_collision_enabled(True)
        obstacles.append(obstacle)

        obstacle_positions[f"sphere_{i}_x"] = sphere_center[0]
        obstacle_positions[f"sphere_{i}_y"] = sphere_center[1]
        obstacle_positions[f"sphere_{i}_z"] = sphere_center[2]
        obstacle_positions[f"sphere_{i}_radius"] = sphere_radius

    return obstacle_positions, obstacles


def get_new_collision_object_states(obstacles, collision_df):
    obstacle_positions = {}
    # sample random episode from collision_df
    episode = np.random.choice(collision_df["episode"].unique())
    # ensure it is not the last episode since we mgiht start at an offset and then don't have enough steps
    while episode == collision_df["episode"].max():
        episode = np.random.choice(collision_df["episode"].unique())
    
    # get all rows for this episode and the next 
    episode_df = collision_df[collision_df["episode"].isin([episode, episode+1])]
    # sort by episode, step
    episode_df = episode_df.sort_values(by=["episode", "step"])
    # sample random start index
    start_index = np.random.randint(0, 34)
    # get init row (index 0)
    init_row = episode_df.iloc[start_index]

    # get target x-y position -> offset for obstacle positions (since current panda is at location (0,0,0))
    offset = np.array([1.0, 0.0, 0.0]) # offset for obstacle positions (since current panda is at location (0,0,0))


    for i, obstacle in enumerate(obstacles):
        sphere_center = np.array(
            [
                -1.0 * init_row[f"robot_sphere_{i}_x"], # mirror x axis for collision robot (since pick and place field is on the opposite side)
                init_row[f"robot_sphere_{i}_y"], 
                init_row[f"robot_sphere_{i}_z"]
            ]
        ) + offset
        sphere_radius = init_row[f"robot_sphere_{i}_radius"]

        obstacle.set_world_pose(
            position=sphere_center.reshape(3),
            orientation=np.array([0, 1, 0, 0])
        )

        obstacle.set_radius(sphere_radius)

        # update obstacle positions
        obstacle_positions[f"sphere_{i}_x"] = sphere_center[0]
        obstacle_positions[f"sphere_{i}_y"] = sphere_center[1]
        obstacle_positions[f"sphere_{i}_z"] = sphere_center[2]
        obstacle_positions[f"sphere_{i}_radius"] = init_row[f"robot_sphere_{i}_radius"]

    return obstacle_positions, episode_df, start_index


def update_obstacle_positions(obstacles, episode_df, step_index):
    collision_objects_df = {}
    # get current row
    offset = np.array([1.0, 0.0, 0.0]) # offset for obstacle positions (since current panda is at location (0,0,0))
    current_row = episode_df.iloc[step_index]
    for i, obstacle in enumerate(obstacles):
        new_position = np.array([
            -1.0 * current_row[f"robot_sphere_{i}_x"], # mirror x axis
            current_row[f"robot_sphere_{i}_y"], 
            current_row[f"robot_sphere_{i}_z"]]
        ) + offset

        obstacle.set_world_pose(
            position=new_position.reshape(3),
            orientation=np.array([0, 1, 0, 0])
        )
        collision_objects_df[f"sphere_{i}_x"] = new_position[0]
        collision_objects_df[f"sphere_{i}_y"] = new_position[1]
        collision_objects_df[f"sphere_{i}_z"] = new_position[2]
        collision_objects_df[f"sphere_{i}_radius"] = obstacle.get_radius()

    return collision_objects_df

def check_collision(robot_sphere_list, obstacles, threshold=0.02):
    """
    Args:
        robot_sphere_list: list of robot collision spheres
        obstacles: list of obstacle spheres

    Returns:
        True if collision (plus 5 cm threshold), False otherwise
    """
    for si, robot_sphere in enumerate(robot_sphere_list):
        sphere_position = robot_sphere.pose[:3]
        sphere_radius = robot_sphere.radius
        for obstacle_sphere in obstacles:
            obstacle_position = obstacle_sphere.get_world_pose()[0]
            obstacle_radius = obstacle_sphere.get_radius()
            if np.linalg.norm(sphere_position - obstacle_position) < sphere_radius + obstacle_radius + threshold:
                print(f"Collision between robot sphere {si} and obstacle sphere {obstacle_sphere}", flush=True)
                return True
 
    return False


def main():
    # assuming obstacles are in objects_path:
    nr_episodes = 3
    missed_episodes = 0
    current_episode = 0
    out_path = "/root/dataset/curobo_panda_pick_and_place_robot_collision_dataset.parquet"

    collision_df_path = "/root/dataset/curobo_panda_pick_and_place_ghost_robot_dataset.parquet"

    # assuming obstacles are in objects_path:
    world = World(stage_units_in_meters=1.0)
    stage = world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    stage = world.stage

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )


    setup_curobo_logger("warn")
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    # warmup curobo instance
    usd_help = UsdHelper()

    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    # panda is at position (0.0, 0.0, 0.0)
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, world)

    collision_df = pd.read_parquet(collision_df_path)

    # import debug draw
    from omni.isaac.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()

    #  Robot at [0,0,0]
    #  collision robot will be at [1,0,0] (mirror y axis)
    #                   x
    #       __________  |  __________ 
    #      |          | | |          |
    #      |  pick    | | |  place   |
    #      |          | | |          |
    #      |__________| | |__________|
    # y --------------[0,0]--------------
    #                   |
    #                   |
    #                   |
    #                   |
    # 

    # for collision dataset this will be the place area
    pick_y_min = -0.6 # -0.8
    pick_y_max = -0.125 # -0.2
    pick_x_min = 0.3 # 0.2
    pick_x_max = 0.7 # 0.8

    pick_area = [
        (pick_x_min, pick_y_min, 0.0),
        (pick_x_max, pick_y_max, 0.0),
        (pick_x_min, pick_y_max, 0.0),
        (pick_x_max, pick_y_min, 0.0),
    ]

    # for collision dataset this will be the pick area
    place_y_min = 0.125 # 0.2
    place_y_max = 0.6 # 0.8
    place_x_min = 0.3 # 0.2
    place_x_max = 0.7 # 0.8

    place_area = [
        (place_x_min, place_y_min, 0.0),
        (place_x_max, place_y_max, 0.0),
        (place_x_min, place_y_max, 0.0),
        (place_x_max, place_y_min, 0.0),
    ]

    # draw blue corner points of pick area
    draw.draw_points(
        pick_area, 
        [(0.0, 0.0, 1.0, 1.0) for _ in range(4)],
        [20.0 for _ in range(4)],
    )

    # draw red corner points of place area
    draw.draw_points(
        place_area, 
        [(1.0, 0.0, 0.0, 1.0) for _ in range(4)],
        [20.0 for _ in range(4)],
    )

    pick_place_df = {}
    pick_place_df["pick_x_min"] = pick_x_min
    pick_place_df["pick_x_max"] = pick_x_max
    pick_place_df["pick_y_min"] = pick_y_min
    pick_place_df["pick_y_max"] = pick_y_max
    pick_place_df["place_x_min"] = place_x_min
    pick_place_df["place_x_max"] = place_x_max
    pick_place_df["place_y_min"] = place_y_min
    pick_place_df["place_y_max"] = place_y_max

    # add obstacle in between pick and place area
    # -> otherwise the gripper sometimes only moves around the ground plane
    obstacle_pick_place_depth = 0.6 # x
    obstacle_pick_place_width = 0.03 # y
    obstacle_pick_place_height = 0.05 # z

    obstacle_pick_place_position_x = (pick_x_max + pick_x_min) / 2.0
    obstacle_pick_place_position_y = (pick_y_min + place_y_max) / 2.0
    obstacle_pick_place_position_z = obstacle_pick_place_height / 2.0

    obstacle_pick_place = cuboid.DynamicCuboid(
        "/World/obstacle_pick_place",
        position=np.array([
            obstacle_pick_place_position_x, 
            obstacle_pick_place_position_y, 
            obstacle_pick_place_position_z
        ]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 1.0, 1.0]),
        scale=np.array([
            obstacle_pick_place_depth, 
            obstacle_pick_place_width, 
            obstacle_pick_place_height
        ]),
    )

    articulation_controller = robot.get_articulation_controller()

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    # experimental feature for reactive mode (non static robot state)
    trajopt_tsteps = 36
    trajopt_dt = 0.05
    max_attempts = 6
    optimize_dt = True
    trim_steps = [1, None]
    num_trajopt_seeds = 10
    num_graph_seeds = 10
    ik_seeds = 32
    noisy_trajopt_seeds = 4

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=num_trajopt_seeds,
        num_graph_seeds=num_graph_seeds,
        interpolation_dt=0.05,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )

    motion_gen_config.ik_seeds = ik_seeds
    motion_gen_config.noisy_trajopt_seeds = noisy_trajopt_seeds


    motion_gen = MotionGen(motion_gen_config)
    print("warming up...", flush=True)
    motion_gen.warmup(
        enable_graph=True, 
        warmup_js_trajopt=False, 
        parallel_finetune=True
    )

    print("Curobo is Ready", flush=True)

    add_extensions(simulation_app, "native")

    plan_config = MotionGenPlanConfig(
        enable_graph=True,
        enable_graph_attempt=4,
        max_attempts=max_attempts,
        enable_finetune_trajopt=True,
        parallel_finetune=True,
    )

    usd_help.load_stage(world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    world.scene.add_default_ground_plane()
    i = 0

    # create spheres:
    robot_spheres = {}
    retract_pose = torch.tensor(default_config).unsqueeze(0)
    cu_js = JointState(
        position=tensor_args.to_device(retract_pose),
        velocity=tensor_args.to_device(torch.zeros_like(retract_pose)), 
        acceleration=tensor_args.to_device(torch.zeros_like(retract_pose)),
        jerk=tensor_args.to_device(torch.zeros_like(retract_pose)),
        joint_names=joint_names,
    )

    sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
    for si, s in enumerate(sph_list[0]):
        robot_spheres[f"robot_sphere_{si}_x"] = s.pose[0]
        robot_spheres[f"robot_sphere_{si}_y"] = s.pose[1]
        robot_spheres[f"robot_sphere_{si}_z"] = s.pose[2]
        robot_spheres[f"robot_sphere_{si}_radius"] = s.radius

    # add obstacles
    collision_objects_df, collision_objects_prims = init_collision_objects(collision_df)


    if os.path.exists(out_path):
        df = pd.read_parquet(out_path)
        current_episode = df["episode"].max() + 1
        print(f"Resuming at episode {current_episode}", flush=True)
    else:
        df = pd.DataFrame(columns=[
            "episode", 
            "step", 
            "target_position_x", 
            "target_position_y", 
            "target_position_z", 
            "target_rotation_w", 
            "target_rotation_i", 
            "target_rotation_j", 
            "target_rotation_k", 
            "ee_position_x", 
            "ee_position_y", 
            "ee_position_z", 
            "ee_rotation_w", 
            "ee_rotation_i", 
            "ee_rotation_j", 
            "ee_rotation_k",
            *joint_names,
            *robot_spheres.keys(),
            *pick_place_df.keys(),
            *collision_objects_df.keys(),
            ]
        )


    # for forward kinematics
    config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))
    urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    ƒk_robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
    kin_model = CudaRobotModel(ƒk_robot_cfg.kinematics)

    # retract target pos and rot
    retract_state = kin_model.get_state(retract_pose.to(tensor_args.device))
    target_retract_pos = retract_state.ee_position[0].cpu().numpy()
    target_retract_rot = retract_state.ee_quaternion[0].cpu().numpy()

    if simulation_app.is_running():
        idx_list = [robot.get_dof_index(x) for x in joint_names]
        robot._articulation_view.set_max_efforts(
            values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
        )
        
        episode_attempts = 0
        while current_episode < nr_episodes:
            episode_attempts += 1

            # reset environment to random initial state
            world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in joint_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

            # reset pick and place locations to a random state
            target_pick_x = np.random.uniform(pick_x_min, pick_x_max)
            target_pick_y = np.random.uniform(pick_y_min, pick_y_max)
            target_pick_z = 0.15

            target_place_x = np.random.uniform(place_x_min, place_x_max)
            target_place_y = np.random.uniform(place_y_min, place_y_max)
            target_place_z = 0.15

            # Pick position and rotation
            target_pick_pos = np.array([target_pick_x, target_pick_y, target_pick_z])
            random_z_rotation_angle = np.random.uniform(0.0, 2*np.pi)
            random_z_rotation_quat = R.from_euler('z', random_z_rotation_angle).as_quat()
            # change scipy convention for quternions from: (i, j, k, w) to (w, i, j, k)
            random_z_rotation_quat = np.array([random_z_rotation_quat[3], random_z_rotation_quat[0], random_z_rotation_quat[1], random_z_rotation_quat[2]])

            # always down with random z rotation
            target_pick_rot = quaternion_multiplication(
                np.array([0.0, 1.0, 0.0, 0.0]),
                random_z_rotation_quat 
            )

            # Place position and rotation
            random_z_rotation_angle = np.random.uniform(0.0, 2*np.pi)
            random_z_rotation_quat = R.from_euler('z', random_z_rotation_angle).as_quat()
            # change scipy convention for quternions from: (i, j, k, w) to (w, i, j, k)
            random_z_rotation_quat = np.array([random_z_rotation_quat[3], random_z_rotation_quat[0], random_z_rotation_quat[1], random_z_rotation_quat[2]])

            target_place_pos = np.array([target_place_x, target_place_y, target_place_z])
            # always down with random z rotation
            target_place_rot = quaternion_multiplication(
                np.array([0.0, 1.0, 0.0, 0.0]),
                random_z_rotation_quat 
            )

            # reset robot to default pose
            idx_list = [robot.get_dof_index(x) for x in joint_names]
            robot.set_joint_positions(retract_pose, idx_list)

            # randomize obstacle positions
            collision_objects_df, episode_df, collision_df_row_start_index = get_new_collision_object_states(collision_objects_prims, collision_df)

            episode_data = []

            def trajectory_sampling_loop(motion_gen, target_pos, target_rot, start_pose, collision_df_row_start_index, phase): 
                step = 0
                # set target position for pick phase
                target.set_default_state(
                    position=target_pos,
                    orientation=target_rot,
                )
                target.post_reset()

                obstacles = usd_help.get_obstacles_from_stage(
                    reference_prim_path=robot_prim_path,
                    ignore_substring=[
                        robot_prim_path,
                        "/World/target",
                        "/World/defaultGroundPlane",
                        "/curobo",
                    ],
                ).get_collision_check_world()
                motion_gen.update_world(obstacles)

                # plan ik motion for pick phase
                ik_goal = Pose(
                    position=tensor_args.to_device(target_pos),
                    quaternion=tensor_args.to_device(target_rot),
                )

                result = motion_gen.plan_single(start_pose, ik_goal, plan_config)
                succ = result.success.item() 
                sim_js_names = robot.dof_names
                
                if succ:
                    cmd_plan = result.get_interpolated_plan()
                    cmd_plan = motion_gen.get_full_js(cmd_plan)

                    idx_list = []
                    common_js_names = []
                    for x in sim_js_names:
                        if x in cmd_plan.joint_names:
                            idx_list.append(robot.get_dof_index(x))
                            common_js_names.append(x)

                    cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                else:
                    cmd_plan = None
                    return False, None, None

                cmd_idx = 0
                past_cmd = None

                while cmd_idx < len(cmd_plan.position):
                    
                    # position and orientation of target virtual cube:
                    cube_position, cube_orientation = target.get_world_pose()

                    sim_js = robot.get_joints_state()
                    sim_js_names = robot.dof_names

                    cu_js = JointState(
                        position=tensor_args.to_device(sim_js.positions),
                        velocity=tensor_args.to_device(sim_js.velocities)* 0.0,
                        acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                        jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                        joint_names=joint_names,
                    )

                    collision_objects_df = update_obstacle_positions(collision_objects_prims, episode_df, step_index=collision_df_row_start_index)

                    if past_cmd is not None:
                        cu_js.position[:] = past_cmd.position
                        cu_js.velocity[:] = past_cmd.velocity
                        cu_js.acceleration[:] = past_cmd.acceleration
                    
                    cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)


                    # check if converged
                    if cmd_idx >= len(cmd_plan.position):
                        return True, cu_js, collision_df_row_start_index
                    
                    
                    cmd_state = cmd_plan[cmd_idx]
                    # state = motion_gen.compute_kinematics(cu_js)
                    state = kin_model.get_state(cmd_state.position)

                    replan = check_collision(
                        motion_gen.kinematics.get_robot_as_spheres(cmd_state.position)[0], 
                        collision_objects_prims
                    )

                    if replan:
                        print(f"Collision detected at step {step} of episode {current_episode}", flush=True)
                        return False, None, None

                    ee_pos = state.ee_position[0]
                    ee_rot = state.ee_quaternion[0]
                    past_cmd = cmd_state.clone()


                    # get full dof state
                    art_action = ArticulationAction(
                        cmd_state.position.cpu().numpy(),
                        cmd_state.velocity.cpu().numpy(),
                        joint_indices=idx_list,
                    )

                    # store executed data point
                    current_df = { **pick_place_df }
                    current_df["episode"] = current_episode
                    current_df["step"] = step

                    current_df["target_position_x"] = cube_position[0]
                    current_df["target_position_y"] = cube_position[1]
                    current_df["target_position_z"] = cube_position[2]

                    # quternion: w, x, y, z
                    current_df["target_rotation_w"] = cube_orientation[0]
                    current_df["target_rotation_i"] = cube_orientation[1]
                    current_df["target_rotation_j"] = cube_orientation[2]
                    current_df["target_rotation_k"] = cube_orientation[3]

                    current_df["ee_position_x"] = ee_pos.cpu().numpy()[0]
                    current_df["ee_position_y"] = ee_pos.cpu().numpy()[1]
                    current_df["ee_position_z"] = ee_pos.cpu().numpy()[2]

                    # quternion: w, x, y, z
                    current_df["ee_rotation_w"] = ee_rot.cpu().numpy()[0]
                    current_df["ee_rotation_i"] = ee_rot.cpu().numpy()[1]
                    current_df["ee_rotation_j"] = ee_rot.cpu().numpy()[2]
                    current_df["ee_rotation_k"] = ee_rot.cpu().numpy()[3]

                    # store phase
                    current_df["phase"] = phase

                    # store robot collision mesh
                    sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
                    
                    # store robot collision mesh
                    for si, s in enumerate(sph_list[0]):
                        current_df[f"robot_sphere_{si}_x"] = s.pose[0]
                        current_df[f"robot_sphere_{si}_y"] = s.pose[1]
                        current_df[f"robot_sphere_{si}_z"] = s.pose[2]
                        current_df[f"robot_sphere_{si}_radius"] = s.radius

                    # store obstacle collision mesh
                    for k, v in collision_objects_df.items():
                        current_df[k] = v

                    # store joint angles for robot
                    for joint_name in cmd_state.joint_names:
                        current_df[joint_name] = cmd_state.position[robot.get_dof_index(joint_name)].cpu().item()

                    episode_data.append(current_df)

                    # set desired joint angles obtained from IK:
                    articulation_controller.apply_action(art_action)
                    step += 1
                    collision_df_row_start_index += 1
                    cmd_idx += 1

                    # execute simulation
                    for _ in range(3):
                        world.step(render=False)

                # check if converged
                if cmd_idx >= len(cmd_plan.position):
                    return True, cu_js, collision_df_row_start_index
                else:
                    return False, None, None


            # update obstacles for motion_gen
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_cfg,
                world_cfg,
                tensor_args,
                collision_checker_type=CollisionCheckerType.MESH,
                num_trajopt_seeds=num_trajopt_seeds,
                num_graph_seeds=num_graph_seeds,
                interpolation_dt=0.05,
                collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
                optimize_dt=optimize_dt,
                trajopt_dt=trajopt_dt,
                trajopt_tsteps=trajopt_tsteps,
                trim_steps=trim_steps,
            )
            motion_gen_config.ik_seeds = ik_seeds
            motion_gen_config.noisy_trajopt_seeds = noisy_trajopt_seeds
            
            motion_gen = MotionGen(motion_gen_config)

            motion_gen.warmup(
                enable_graph=True, 
                warmup_js_trajopt=False, 
                parallel_finetune=True
            )

            # pick phase
            retract_cfg = motion_gen.get_retract_config()
            start_state = JointState.from_position(retract_cfg.view(1, -1))
            succ, cu_js, episode_df_index = trajectory_sampling_loop(motion_gen, target_pick_pos, target_pick_rot, start_state, collision_df_row_start_index, phase=1)

            # place phase
            if succ:
                print(f"success pick: {current_episode}, total episode attempts: {episode_attempts}", flush=True)
                
                start_state = cu_js.unsqueeze(0)
                succ, cu_js, episode_df_index = trajectory_sampling_loop(motion_gen, target_place_pos, target_place_rot, start_state, episode_df_index, phase=2)
            else: 
                print(f"failed in pick phase: {current_episode}, total episode attempts: {episode_attempts}", flush=True)
                missed_episodes += 1
                continue

            # retract phase
            if succ:
                print(f"success place: {current_episode}, total episode attempts: {episode_attempts}", flush=True)
                start_state = cu_js.unsqueeze(0)
                succ, cu_js, episode_df_index = trajectory_sampling_loop(motion_gen, target_retract_pos, target_retract_rot, start_state, episode_df_index, phase=3)
            else:
                print(f"failed in place phase: {current_episode}, total episode attempts: {episode_attempts}", flush=True)
                missed_episodes += 1
                continue

            if succ:
                current_episode += 1

                # store episode data
                df = pd.concat([df, pd.DataFrame(episode_data)], ignore_index=True)

                print(f"converged attempt: {current_episode}, total episode attempts: {episode_attempts}", flush=True)

                print(f"Store dataframe at iteration {i}", flush=True)
                df.to_parquet(out_path)

                print("Statisitics:", flush=True)
                steps_stat = df.groupby('episode').count()['step']
                print(f"Mean Nr Steps: {steps_stat.mean()}, Std Nr Steps: {steps_stat.std()}", flush=True)
                print(f"Min Nr Steps: {steps_stat.min()}, Max Nr Steps: {steps_stat.max()}", flush=True)

                print(f"Nr missed episodes: {missed_episodes}", flush=True)
                print(f"Nr stored episodes: {current_episode}", flush=True)
                print(f"Nr attempts: {episode_attempts}", flush=True)
            else: 
                print(f"failed in retract phase: {current_episode}, total episode attempts: {episode_attempts}", flush=True)
                missed_episodes += 1
        

    # print final stats:
    print("Final Statisitics:", flush=True)
    steps_stat = df.groupby('episode').count()['step']
    print(f"Mean Nr Steps: {steps_stat.mean()}, Std Nr Steps: {steps_stat.std()}", flush=True)
    print(f"Min Nr Steps: {steps_stat.min()}, Max Nr Steps: {steps_stat.max()}", flush=True)

    print(f"Nr missed episodes: {missed_episodes}", flush=True)
    print(f"Nr stored episodes: {current_episode}", flush=True)

    print(df.head())
    df.to_parquet(out_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
