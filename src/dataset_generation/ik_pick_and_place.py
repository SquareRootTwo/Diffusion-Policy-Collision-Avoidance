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

import carb
import pandas as pd
from typing import Optional
import numpy as np
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

def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        # "omni.isaac.asset_browser",
    ]
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

    
def main():
    # assuming obstacles are in objects_path:
    nr_episodes = 20  
    missed_episodes = 0
    current_episode = 0
    out_path="/root/dataset/curobo_panda_pick_and_place_dataset.parquet"

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

    # import debug draw
    from omni.isaac.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()

    # for collision dataset this will be the place area
    pick_x_min = -0.6
    pick_x_max = -0.1
    pick_y_min = 0.1
    pick_y_max = 0.6

    pick_area = [
        (pick_x_min, pick_y_min, 0.0),
        (pick_x_max, pick_y_max, 0.0),
        (pick_x_min, pick_y_max, 0.0),
        (pick_x_max, pick_y_min, 0.0),
    ]

    # for collision dataset this will be the pick area
    place_x_min = 0.6
    place_x_max = 0.1
    place_y_min = 0.1
    place_y_max = 0.6

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

    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4

    # experimental feature for reactive mode (non static robot state)
    trajopt_tsteps = 36
    trajopt_dt = 0.05
    optimize_dt = False
    max_attemtps = 1
    trim_steps = [1, None]


    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.05,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up...", flush=True)
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)

    print("Curobo is Ready", flush=True)

    add_extensions(simulation_app, "native")

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=True,
        parallel_finetune=True,
    )

    usd_help.load_stage(world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    cmd_plan = None
    cmd_idx = 0
    world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    past_cmd = None

    spheres = []
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
        robot_spheres[f"robot_sphere_{si}_x"] = s.position[0]
        robot_spheres[f"robot_sphere_{si}_y"] = s.position[1]
        robot_spheres[f"robot_sphere_{si}_z"] = s.position[2]
        robot_spheres[f"robot_sphere_{si}_radius"] = s.radius

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
        *robot_spheres.keys()
        ]
    )

    panda_pose_limits = {
        "panda_joint1": ((-166/360.0)*2*np.pi, (166/360.0)*2*np.pi),
        "panda_joint2": ((-101/360.0)*2*np.pi, (101/360.0)*2*np.pi),
        "panda_joint3": ((-166/360.0)*2*np.pi, (166/360.0)*2*np.pi),
        "panda_joint4": ((-176/360.0)*2*np.pi, (-4/360.0)*2*np.pi),
        "panda_joint5": ((-166/360.0)*2*np.pi, (166/360.0)*2*np.pi),
        "panda_joint6": ((-1/360.0)*2*np.pi, (215/360.0)*2*np.pi),
        "panda_joint7": ((-166/360.0)*2*np.pi, (166/360.0)*2*np.pi),
        "panda_finger_joint1": (0.0, 0.08),
        "panda_finger_joint2": (0.0, 0.08),
    }

    # robot lower and upper limits
    robot_joint_lower_limits = [panda_pose_limits[joint_name][0] for joint_name in joint_names]
    robot_joint_upper_limits = [panda_pose_limits[joint_name][1] for joint_name in joint_names]

    # for forward kinematics
    config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))
    urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    ƒk_robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
    kin_model = CudaRobotModel(ƒk_robot_cfg.kinematics)

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)


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

            # reset robot to a random state
            target_pick_x = np.random.uniform(pick_x_min, pick_x_max)
            target_pick_y = np.random.uniform(pick_y_min, pick_y_max)
            target_pick_z = 0.15

            target_place_x = np.random.uniform(place_x_min, place_x_max)
            target_place_y = np.random.uniform(place_y_min, place_y_max)
            target_place_z = 0.15

            target_pick_pos = np.array([target_pick_x, target_pick_y, target_pick_z])
            target_pick_rot = np.array([0.0, 1.0, 0.0, 0.0]) # always down TODO: rotate along x axis

            target_place_pos = np.array([target_place_x, target_place_y, target_place_z])
            target_place_rot = np.array([0.0, 1.0, 0.0, 0.0]) # always down TODO: rotate along x axis

            idx_list = [robot.get_dof_index(x) for x in joint_names]
            robot.set_joint_positions(retract_pose, idx_list)

            sim_js_names = robot.dof_names

            episode_data = []
            step = 0
            
            converged_place = False
            converged_pick = False

            # plan ik motion for pick phase
            motion_gen.reset()

            # cu_js = JointState(
            #     position=tensor_args.to_device(retract_pose),
            #     velocity=tensor_args.to_device(torch.zeros_like(retract_pose)), 
            #     acceleration=tensor_args.to_device(torch.zeros_like(retract_pose)),
            #     jerk=tensor_args.to_device(torch.zeros_like(retract_pose)),
            #     joint_names=joint_names,
            # )
            # cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

            print(f"plan ik motion {current_episode}", flush=True)

            # obstacles = usd_help.get_obstacles_from_stage(
            #     # only_paths=[obstacles_path],
            #     reference_prim_path=robot_prim_path,
            #     ignore_substring=[
            #         robot_prim_path,
            #         "/World/target",
            #         "/World/defaultGroundPlane",
            #         "/curobo",
            #     ],
            # ).get_collision_check_world()
            # motion_gen.update_world(obstacles)

            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(target_pick_pos),
                quaternion=tensor_args.to_device(target_pick_rot),
            )

            retract_cfg = motion_gen.get_retract_config()
            start_state = JointState.from_position(retract_cfg.view(1, -1))
            state = motion_gen.rollout_fn.compute_kinematics(
                JointState.from_position(retract_cfg.view(1, -1))
            )

            result = motion_gen.plan_single(start_state, ik_goal, plan_config)
            succ = result.success.item() 
            
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


            # set target position for pick phase
            target.set_default_state(
                position=target_pick_pos,
                orientation=target_pick_rot,
            )
            target.post_reset()

            if not succ:
                cmd_plan = None
                print(f"failed in pick phase: {current_episode}, total episode attempts: {episode_attempts}", flush=True)
                missed_episodes += 1
                continue
        
            print(f"Starting pick and place: {current_episode}, total episode attempts: {episode_attempts}", flush=True)
            while not converged_place:
                world.step(render=False)
                
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

                if past_cmd is not None:
                    cu_js.position[:] = past_cmd.position
                    cu_js.velocity[:] = past_cmd.velocity
                    cu_js.acceleration[:] = past_cmd.acceleration
                
                cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

                # state = motion_gen.compute_kinematics(cu_js)
                state = kin_model.get_state(cu_js.position)
                ee_pos = state.ee_position[0]
                ee_rot = state.ee_quaternion[0]

                if cmd_plan is not None:
                    cmd_state = cmd_plan[cmd_idx]
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

                    # store robot collision mesh
                    sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
                    
                    for si, s in enumerate(sph_list[0]):
                        current_df[f"robot_sphere_{si}_x"] = s.position[0]
                        current_df[f"robot_sphere_{si}_y"] = s.position[1]
                        current_df[f"robot_sphere_{si}_z"] = s.position[2]
                        current_df[f"robot_sphere_{si}_radius"] = s.radius

                    for joint_name in cmd_state.joint_names:
                        current_df[joint_name] = cmd_state.position[robot.get_dof_index(joint_name)].cpu().item()

                    episode_data.append(current_df)

                    # set desired joint angles obtained from IK:
                    articulation_controller.apply_action(art_action)
                    step += 1
                    cmd_idx += 1
                    for _ in range(2):
                        world.step(render=False)
                    
                    if (not converged_pick) and (cmd_idx >= len(cmd_plan.position)):
                        cmd_idx = 0
                        # replan for place phase
                        past_cmd = None 
                        converged_pick = True

                        # set target position for place phase
                        target.set_default_state(
                            position=target_place_pos,
                            orientation=target_place_rot,
                        )
                        target.post_reset()

                        # succ, cmd_plan = plan_ik_motion(
                        #     target_place_pos,
                        #     target_place_rot,
                        #     motion_gen,
                        #     plan_config,
                        #     tensor_args,
                        #     cu_js,
                        #     sim_js_names,
                        #     robot
                        # )

                        # compute curobo solution:
                        ik_goal = Pose(
                            position=tensor_args.to_device(target_place_pos),
                            quaternion=tensor_args.to_device(target_place_rot),
                        )

                        result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
                                
                        succ = result.success.item() 
                        
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
                            missed_episodes += 1
                            print(f"failed in place phase: {current_episode}, total episode attempts: {episode_attempts}", flush=True)
                            converged_place = True # skip this episode
                    
                    elif not converged_place and (cmd_idx >= len(cmd_plan.position)):
                        print(f"converged pick and place: {current_episode}, total episode attempts: {episode_attempts}", flush=True)
                        cmd_idx = 0
                        cmd_plan = None
                        current_episode += 1

                        converged_place = True

                        # store episode data
                        df = pd.concat([df, pd.DataFrame(episode_data)], ignore_index=True)

                        if current_episode % 50 == 0:
                            # print current_df statistics
                            print(f"Store dataframe at iteration {i}", flush=True)
                            df.to_parquet(out_path)

                            print("Statisitics:", flush=True)
                            steps_stat = df.groupby('episode').count()['step']
                            print(f"Mean Nr Steps: {steps_stat.mean()}, Std Nr Steps: {steps_stat.std()}", flush=True)
                            print(f"Min Nr Steps: {steps_stat.min()}, Max Nr Steps: {steps_stat.max()}", flush=True)

                            print(f"Nr missed episodes: {missed_episodes}", flush=True)
                            print(f"Nr stored episodes: {current_episode}", flush=True)
                            print(f"Nr attempts: {episode_attempts}", flush=True)

        

    # print final stats:
    print("Final Statisitics:", flush=True)
    steps_stat = df.groupby('episode').count()['step']
    print(f"Mean Nr Steps: {steps_stat.mean()}, Std Nr Steps: {steps_stat.std()}", flush=True)
    print(f"Min Nr Steps: {steps_stat.min()}, Max Nr Steps: {steps_stat.max()}", flush=True)

    print(f"Nr missed episodes: {missed_episodes}", flush=True)
    print(f"Nr stored episodes: {current_episode}", flush=True)


    print(df.head())
    df.to_parquet(out_path)
    # df.to_csv(out_path.replace(".parquet", ".csv"))


if __name__ == "__main__":
    main()
    simulation_app.close()
