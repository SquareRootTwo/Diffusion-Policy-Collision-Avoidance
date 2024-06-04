#!/usr/bin/env python

import numpy as np
import torch
from pytorch_kinematics.transforms import matrix_to_quaternion, quaternion_to_matrix
import pybullet_data
from scipy.spatial.transform import Rotation as R

class PandaKinematics():
    def __init__(
            self, 
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype: torch.dtype = torch.float32                          
        ):
        self.device = device
        self.dtype = dtype
        self.retract_pose = torch.tensor([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0], device=self.device, dtype=self.dtype)

        # Denavit-Hartenberg parameters of the Panda robot
        # https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
        self.alpha = torch.tensor([ 0,      -np.pi/2,   np.pi/2,    np.pi/2,    -np.pi/2,   np.pi/2,    np.pi/2,    0.0], device=self.device, dtype=self.dtype)
        self.a = torch.tensor([     0.0,    0.0,        0.0,        0.0825,     -0.0825,    0.0,        0.088,      0.0], device=self.device, dtype=self.dtype)
        self.d = torch.tensor([     0.333,  0.0,        0.316,      0.0,        0.384,      0.0,        0.0,        0.107], device=self.device, dtype=self.dtype)

        # collision sphere specs
        self.sphere_id_to_joint_id = torch.tensor(
            [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 
             4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
             7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            device=self.device, dtype=torch.int)
        
        # this sphere should be fixed and not move with joint 0
        self.sphere_1_pos_fix = torch.tensor([-0.1, 0.0, 0.085], device=self.device, dtype=self.dtype)
        
        self.realtive_offset = torch.tensor([
            [0.0, 0.0, -0.24799999594688416],
            [-0.10000000149011612, 0.0, -0.24799999594688416],
            [2.38418573772492e-09, -9.53674295089968e-09, -0.07999999821186066],
            [8.940696516468449e-10, -3.5762786065873797e-09, -0.029999999329447746],
            [0.0, 0.0, -0.12000000476837158],
            [0.0, 0.0, -0.17000000178813934],
            [-8.940696516468449e-10, 3.5762786065873797e-09, 0.029999999329447746],
            [-2.38418573772492e-09, 9.53674295089968e-09, 0.07999999821186066],
            [6.679031105960576e-09, 3.842079099314333e-09, -0.19600003957748413],
            [1.1625076012933278e-08, 5.442945649747344e-09, -0.14600002765655518],
            [1.7037763200278278e-08, 8.196142431415865e-09, -0.06000000610947609],
            [1.882418132481689e-08, 6.91525325891007e-09, -0.10000000894069672],
            [0.0020028622820973396, -0.001496167154982686, -0.06000000610947609],
            [0.0020028611179441214, -0.001496167154982686, -0.02000000700354576],
            [-5.96046212386625e-10, 0.0, 0.019999992102384567],
            [-1.788139081249085e-09, 0.0, 0.059999991208314896],
            [0.0025000073947012424, -4.7430717131646816e-09, -0.2890000343322754],
            [0.0025000469759106636, -2.0712895287822164e-10, -0.3240000307559967],
            [-3.2049925380306377e-08, 0.029999956488609314, -7.944313296093242e-08],
            [-3.144779725516855e-08, 0.08199995756149292, -7.270452329066757e-08],
            [-2.1888006962456075e-09, -1.3655328423567425e-08, -0.22000004351139069],
            [-9.053734983943684e-10, 0.051999982446432114, -0.18000006675720215],
            [0.009999998845160007, 0.07999997586011887, -0.14000004529953003],
            [0.009999974630773067, 0.08499997109174728, -0.11000004410743713],
            [0.01000000536441803, 0.0899999737739563, -0.08000006526708603],
            [0.00999997928738594, 0.09499996155500412, -0.050000064074993134],
            [-0.009999982081353664, 0.07999997586011887, -0.14000006020069122],
            [-0.010000004433095455, 0.08499997109174728, -0.11000005900859833],
            [-0.010000028647482395, 0.0899999737739563, -0.08000005036592484],
            [-0.009999998845160007, 0.09499996155500412, -0.05000007525086403],
            [-8.836239118181766e-08, -1.8298862869414734e-08, 0.009000041522085667],
            [0.08499988168478012, 0.03500000014901161, 4.956624266583276e-08],
            [-0.0030001059640198946, 4.884296345153416e-08, 4.285405452719715e-08],
            [-0.0030001031700521708, 4.894873129046573e-08, 0.0150000536814332],
            [-1.1294115864757259e-07, 4.9572477678339055e-08, -0.03699996694922447],
            [0.019999882206320763, 0.04000005125999451, -0.026999982073903084],
            [0.039999864995479584, 0.020000051707029343, -0.026999998837709427],
            [0.039999864995479584, 0.06000005081295967, -0.021999996155500412],
            [0.05999987572431564, 0.040000054985284805, -0.021999957039952278],
            [-0.05303310230374336, -0.05303296446800232, 0.010000042617321014],
            [-0.0318199023604393, -0.03181975707411766, 0.010000036098062992],
            [-0.010606706142425537, -0.010606551542878151, 0.010000031441450119],
            [0.010606488212943077, 0.010606651194393635, 0.01000002771615982],
            [0.03181969001889229, 0.03181985765695572, 0.010000021196901798],
            [0.05303289368748665, 0.05303306505084038, 0.01000001560896635],
            [-0.056568633764982224, -0.05656849592924118, 0.03000001795589924],
            [-0.031819913536310196, -0.03181975707411766, 0.02999999187886715],
            [-0.010606717318296432, -0.010606551542878151, 0.02999998815357685],
            [0.010606496594846249, 0.01060665212571621, 0.03000004030764103],
            [0.031819697469472885, 0.03181985765695572, 0.03000003471970558],
            [0.056568413972854614, 0.05656859651207924, 0.03000001236796379],
            [-0.056568630039691925, -0.05656849592924118, 0.04500002786517143],
            [-0.0318199023604393, -0.03181975707411766, 0.04500000178813934],
            [-0.010606707073748112, -0.010606551542878151, 0.044999998062849045],
            [0.010606488212943077, 0.01060665212571621, 0.044999994337558746],
            [0.03181969001889229, 0.03181985765695572, 0.04499998688697815],
            [0.05656841769814491, 0.05656859651207924, 0.045000020414590836],
            [0.03535522520542145, 0.03535538911819458, 0.1013999804854393],
            [0.04242628440260887, 0.0424264594912529, 0.07339997589588165],
            [-0.03535543009638786, -0.03535528481006622, 0.10140001028776169],
            [-0.04242650419473648, -0.042426351457834244, 0.07339999079704285],
            ], device=self.device, dtype=self.dtype
        )

        self.collision_sphere_radius = torch.tensor(
            [0.032499998807907104, 0.032499998807907104, 0.057500001043081284, 
             0.0625, 0.0625, 0.0625, 0.057500001043081284, 0.057500001043081284, 
             0.057500001043081284, 0.057500001043081284, 0.05250000208616257, 
             0.0625, 0.05450000241398811, 0.05450000241398811, 0.05450000241398811, 
             0.05450000241398811, 0.057500001043081284, 0.05450000241398811, 
             0.05250000208616257, 0.05250000208616257, 0.05250000208616257, 
             0.042500000447034836, 0.02449999935925007, 0.02449999935925007, 
             0.02449999935925007, 0.02449999935925007, 0.02449999935925007, 
             0.02449999935925007, 0.02449999935925007, 0.02449999935925007, 
             0.05250000208616257, 0.04750000312924385, 0.04750000312924385, 
             0.04750000312924385, 0.04750000312924385, 0.026499999687075615, 
             0.026499999687075615, 0.022499999031424522, 0.022499999031424522, 
             0.025499999523162842, 0.025499999523162842, 0.025499999523162842, 
             0.025499999523162842, 0.025499999523162842, 0.025499999523162842, 
             0.02449999935925007, 0.02449999935925007, 0.02449999935925007, 
             0.02449999935925007, 0.02449999935925007, 0.02449999935925007, 
             0.02449999935925007, 0.02449999935925007, 0.02449999935925007, 
             0.02449999935925007, 0.02449999935925007, 0.02449999935925007, 
             0.013499999418854713, 0.013499999418854713, 0.013499999418854713, 
             0.013499999418854713],
            device=self.device, dtype=self.dtype
        )

    def compute_dh_matrix(self, theta: torch.Tensor, bs: int) -> torch.Tensor:
        """
        Compute the Denavit-Hartenberg transformation matrix
            - theta.shape = (bs, 7)
        """
        theta = theta.to(self.device)
        theta = torch.cat([theta, torch.zeros((bs, 1), device=self.device, dtype=self.dtype)], dim=1)

        a: torch.Tensor = self.a.unsqueeze(0).repeat((bs, 1)).to(self.device)
        alpha: torch.Tensor = self.alpha.unsqueeze(0).repeat((bs, 1)).to(self.device)
        d: torch.Tensor = self.d.unsqueeze(0).repeat((bs, 1)).to(self.device)

        # dh.shape = (bs, 8, 4, 4)
        # Emika Panda DH parameters follow Craig's convention
        # https://frankaemika.github.io/docs/control_parameters.html
        # formula: https://de.wikipedia.org/wiki/Denavit-Hartenberg-Transformation
        dh = torch.stack([
            torch.stack([torch.cos(theta),                          -torch.sin(theta),                          torch.zeros((bs, 8), device=self.device),       a], dim=2),
            torch.stack([torch.sin(theta)*torch.cos(alpha),         torch.cos(theta)*torch.cos(alpha),          -torch.sin(alpha),                              -d * torch.sin(alpha)], dim=2),
            torch.stack([torch.sin(theta)*torch.sin(alpha),         torch.cos(theta)*torch.sin(alpha),          torch.cos(alpha),                               d * torch.cos(alpha)], dim=2),
            torch.stack([torch.zeros((bs, 8), device=self.device),  torch.zeros((bs, 8), device=self.device),   torch.zeros((bs, 8), device=self.device),       torch.ones((bs, 8), device=self.device)], dim=2)
        ], dim=2).to(device=self.device, dtype=self.dtype)

        # dh shape (bs, 8, 4, 4) -> (8, bs, 4, 4)
        dh = dh.permute(1, 0, 2, 3)

        return dh

    
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Forward kinematics of the Panda robot
            input: 
                - pose (torch.Tensor): shape = (bs, 7) batch of 7 joint angles of the Panda robot
            output: 
                - joint_positions (torch.Tensor): shape = (8, bs, 3) tensor of the joint positions (x, y, z)
                - joint_rotations (torch.Tensor): shape = (8, bs, 4)/(bs, 8, 3, 3) tensor of the joint rotations (quaternions: w, x, y, z)
        """
        pose = pose.to(self.device)
        bs = pose.shape[0]

        dh_mat = self.compute_dh_matrix(pose, bs)

        link_world_pose = torch.zeros((8, bs, 4, 4), device=self.device, dtype=self.dtype)

        link_world_pose[0] = dh_mat[0]
        for i in range(1, 8):
            link_world_pose[i] = torch.matmul(link_world_pose[i-1], dh_mat[i])

        # (7, bs, 4, 4)
        joint_positions = link_world_pose[:, :, :3, 3]
        # quaternion convention: w, x, y, z (same as the NN expects/cuRobo)
        joint_rotations = matrix_to_quaternion(link_world_pose[:, :, :3, :3].view((-1, 3, 3))).view((8, bs, 4))

        return joint_positions, joint_rotations
    

    def get_ee_pose(self, pose: torch.Tensor, return_quat: bool = True) -> torch.Tensor:
        """
        Computes the same as the forward kinematics but returns only the end effector pose (adjusted with a 45 degree rotation)
        what the NN expects (data coming from cuRobo is also in this format)
            input: 
                - pose (torch.Tensor): batch of 7 joint angles of the Panda robot
                - return_quat (bool): if True, return the end effector rotation as a quaternion, otherwise as a rotation matrix
            output:
                - ee_pos (torch.Tensor): shape = (bs, 3), batch of end effector positions (x, y, z)
                - ee_rot_adjusted (torch.Tensor): shape = (bs, 4)/(bs, 3, 3), batch of end effector rotations adjusted for cuRobo end effector orientation (quaternion: w, x, y, z or rotation matrices)
        """
        # pos.shape = (8, bs, 3), rot.shape = (8, bs, 4)
        bs = pose.shape[0]
        pos, rot = self.forward(pose)
        ee_pos = pos[7]
        ee_rot = rot[7]
        ee_rot_mat = quaternion_to_matrix(ee_rot)

        # https://en.wikipedia.org/wiki/Rotation_matrix
        teta = - np.pi / 4.0
        ee_rot_45 = torch.tensor(
            [[np.cos(teta), -np.sin(teta), 0.0],
             [np.sin(teta), np.cos(teta), 0.0],
             [0.0, 0.0, 1.0]]
        ).to(device=self.device, dtype=self.dtype).unsqueeze(0).repeat((bs, 1, 1))

        # 45 degree z rotation quaternion
        ee_rot_adjusted = torch.bmm(ee_rot_mat, ee_rot_45)
        if return_quat:
            ee_rot_adjusted = matrix_to_quaternion(ee_rot_adjusted)
            # TODO: only allow positive quaternions

        
        return ee_pos, ee_rot_adjusted

    def get_panda_collision_spheres(self, th, return_flattened: bool = False): 
        """
        Given a joint pose as input, computes the position of all 61 collision spheres of the Panda robot
        Input:
            - th (torch.Tensor): shape = (bs, 7), batch of joint angles of the Panda robot
            - return_flattened (bool): whether to return the sphere positions as a flattened tensor of shape (bs, 61*4)

        Output:
            - sphere_positions (torch.Tensor): shape = (bs, 61, 3), tensor of the positions of the 61 collision spheres
                if return_flattened is True, the sphere positions are concatenated with their radius and are returned as
                a flattened tensor of shape (bs, 61*4)
        """
        bs = th.shape[0]

        all_joint_pos, all_joint_rot = self.forward(th)
        all_joint_pos = all_joint_pos.permute(1, 0, 2)
        all_joint_rot = all_joint_rot.permute(1, 0, 2)

        all_joint_rot_mat = quaternion_to_matrix(all_joint_rot)

        sphere_positions = torch.zeros((bs, 61, 3), device=self.device, dtype=self.dtype)

        sphere_positions = all_joint_pos[:, self.sphere_id_to_joint_id] + torch.matmul(all_joint_rot_mat[:, self.sphere_id_to_joint_id], self.realtive_offset.unsqueeze(-1)).squeeze(-1)

        # fix sphere 1 position
        sphere_positions[:, 1, :] = self.sphere_1_pos_fix.unsqueeze(0).repeat((bs, 1))

        if return_flattened:
            sphere_radius_concat = torch.cat([sphere_positions, self.collision_sphere_radius.clone().unsqueeze(0).repeat((bs, 1, 1))], dim=-1)
            sphere_positions = sphere_radius_concat.flatten(start_dim=1)

        return sphere_positions
    
    def concat_sphere_with_radius(self, sphere_positions: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of sphere positions, concatenates the sphere positions with their radius
        Input:
            - sphere_positions (torch.Tensor): shape = (bs, 61, 3), tensor of sphere positions
        Output:
            - sphere_positions_concat (torch.Tensor): shape = (bs, 61*4), tensor of sphere positions concatenated with their radius
        """
        sphere_positions_concat = torch.cat([sphere_positions, self.collision_sphere_radius.clone().unsqueeze(0).repeat((sphere_positions.shape[0], 1, 1))], dim=-1)
        sphere_positions_concat = sphere_positions_concat.flatten(start_dim=-2)

        return sphere_positions_concat
    
    def rl_multi_agent_step(self, th):
        """
        Given a batch joint poses of shape (num_envs, num_agents, 7), returns the end effector pose plus the collision mesh 
        representation of the panda robot 

        Input:
            - th (torch.Tensor): shape = (num_envs, num_agents, 7), batch of joint angles of the Panda robot

        Output:
            - ee_pos (torch.Tensor): shape = (num_envs, num_agents, 7), batch of end effector positions and orientations
            - collision_mesh (torch.Tensor): shape = (num_envs, num_agents, 61*3), batch of collision mesh positions
        """
        pass


    def get_2_panda_obs(self, th, target):
        """
        Given a batch of 2 joint poses, computes the reoriented observation such that each panda robot gets its environment from
        its local base cooridantes

        obs dims: 
            joint_positions, # 0:7
            end_effector, # 7:14
            target, # 14:21
            obstacles_flattend, # 21:21+244
            robot_collision_spheres_flattend # 21+244:21+244+244

        Input:
            - th (torch.Tensor): shape = (2, 7) batch of size 2 of panda joint positions
            - target (torch.Tensor): shape = (2, 7) batch of size 2 of current panda targets in robots local coordinates

        Output:
            - obs (torch.Tensor): shape = (2, 1, 509)
    
        """
        target = target.to(self.device).to(dtype=self.dtype)
        # panda_2_transform_rotation_matrix = torch.tensor([
        #     [-1, 0, 0],
        #     [0, -1, 0],
        #     [0,  0, 1]
        # ], dtype=self.dtype, device=self.device)
        # panda_2_transform_offset = torch.tensor([1, 0, 0], dtype=self.dtype, device=self.device)

        # shape = (2, 1, 7)
        joint_positions = th.unsqueeze(1).to(self.device).to(dtype=self.dtype)

        # shape = (2, 3), (2, 4)
        # ee of panda 2 stays in world coordiantes since it needs to be local to the panda robots orientation
        ee_position, ee_rotation = self.get_ee_pose(th, return_quat=True)

        if ee_rotation[0, 0] < 0:
            ee_rotation[0, :] *= -1

        if ee_rotation[1, 0] < 0:
            ee_rotation[1, :] *= -1
            
        # shape = (2, 1, 7)
        end_effector = torch.cat([ee_position, ee_rotation], dim=-1).unsqueeze(1)

        # shape = (2, 1, 7)
        target = target.unsqueeze(1)
        if target[0, 0, 3] < 0:
            target[0, 0, 3:] *= -1

        if target[1, 0, 3] < 0:
            target[1, 0, 3:] *= -1

        # shape = (2, 61, 3)
        panda_collision_spheres = self.get_panda_collision_spheres(th)
        ghost_collision_spheres = panda_collision_spheres.clone()
        ghost_collision_spheres[:, :, 0] *= -1
        ghost_collision_spheres[:, :, 0] += 1
        # ghost_collision_spheres[:, :, 1] *= -1

        # ghost spheres -> rotated and offset added for both
        # shape = (2, 61, 3)
        # ghost_collision_spheres = torch.matmul(panda_collision_spheres, panda_2_transform_rotation_matrix) + panda_2_transform_offset.unsqueeze(0).unsqueeze(0).repeat((2, 61, 1))

        # shape  = (2, 1, 61*4)
        obstacles_flattend = torch.zeros((2, 1, 61*4), dtype=self.dtype, device=self.device)

        # shape  = (2, 1, 61*4)
        robot_collision_spheres_flattend = torch.zeros((2, 1, 61*4), dtype=self.dtype, device=self.device)

        for i in range(61):
            robot_collision_spheres_flattend[0, 0, 4*i:4*i+3] = panda_collision_spheres[0, i, :]
            robot_collision_spheres_flattend[1, 0, 4*i:4*i+3] = panda_collision_spheres[1, i, :]
            robot_collision_spheres_flattend[0, 0, 4*i+3] = self.collision_sphere_radius[i]
            robot_collision_spheres_flattend[1, 0, 4*i+3] = self.collision_sphere_radius[i]

            obstacles_flattend[1, 0, 4*i:4*i+3] = ghost_collision_spheres[0, i, :]
            obstacles_flattend[0, 0, 4*i:4*i+3] = ghost_collision_spheres[1, i, :]

            obstacles_flattend[0, 0, 4*i+3] = self.collision_sphere_radius[i]
            obstacles_flattend[1, 0, 4*i+3] = self.collision_sphere_radius[i]

        assert joint_positions.shape == (2, 1, 7)
        assert end_effector.shape == (2, 1, 7)
        assert target.shape == (2, 1, 7)
        assert obstacles_flattend.shape == (2, 1, 61*4)
        assert robot_collision_spheres_flattend.shape == (2, 1, 61*4)

        # shape = (2, 1, 509)
        return torch.cat([
            joint_positions, # 0:7
            end_effector, # 7:14
            target, # 14:21
            obstacles_flattend, # 21:21+244
            robot_collision_spheres_flattend # 21+244:21+244+244
        ], dim=2)


pk = PandaKinematics()

if __name__ == '__main__':
    # panda kinematics test with PyBullet
    import pybullet as p
    import os 
    physicsClient = p.connect(p.GUI, options='--width=3000 --height=2000')

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -9.81)

    # set initial camera pose
    p.resetDebugVisualizerCamera(
        cameraDistance=1.1, 
        cameraYaw=0, 
        cameraPitch=-45, 
        cameraTargetPosition=[0,0,0.3])
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
    p.setGravity(0,0,-9.81)
    pk = PandaKinematics()
    retract_pose = torch.tensor([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0], device=pk.device, dtype=pk.dtype)
    
    # used to get the mapping of the sphere to the joint correctly
    # sphere_id_to_joint_id = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],dtype=torch.int8)
    sphere_id_to_joint_id = pk.sphere_id_to_joint_id


    sphere_radius = pk.collision_sphere_radius

    # add trajectory points
    yellow = np.array([1.0, 1.0, 0.0, 0.3])
    red = np.array([1.0, 0.0, 0.0, 0.3])
    blue = np.array([0.0, 0.0, 1.0, 0.7])
    green = np.array([0.0, 1.0, 0.0, 0.7]) 

    colors_sphere = []
    for i in range(10):
        # c = (1 - i / 10) * blue + (i / 10) * green
        c = np.random.rand(4)
        c[3] = 1.0
        colors_sphere.append(c.tolist())

    # colors_sphere[5] = yellow

    color_gt = []
    for i in range(10):
        c = (1 - i / 10) * yellow + (i / 10) * red
        # random color
        color_gt.append(c.tolist())

    spheres = []

    realtive_offset = []

    joint_pose, joint_rot = pk.forward(retract_pose.unsqueeze(0))

    joint_pose = joint_pose.squeeze(1)
    joint_rot = joint_rot.squeeze(1)

    joint_rot_mat = quaternion_to_matrix(joint_rot)

    for i, rad in enumerate(sphere_radius.cpu().numpy()):
        # add sphere to pybullet
        color_index = int(sphere_id_to_joint_id[i].item())
        sphere = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=rad, rgbaColor=color_gt[color_index])
        sphere_id = p.createMultiBody(baseVisualShapeIndex=sphere,basePosition=[0, 0, i],baseOrientation=[0, 0, 0, 1]) 
        spheres.append(sphere_id)


    retract_pose = torch.tensor([0.0, 0, 0.0, 0, 0.0, 0.0, 0.0], device=pk.device, dtype=pk.dtype)
    panda_upper_limits = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], device=pk.device, dtype=pk.dtype)
    panda_lower_limits = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], device=pk.device, dtype=pk.dtype)
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=1)

    # start video recording in PyBullet
    # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "panda_fk_kinematics_collision_mesh.mp4")

    for i in range(10):
        random_pose = torch.rand(7, device=pk.device, dtype=pk.dtype) * (panda_upper_limits - panda_lower_limits) + panda_lower_limits
        sphere_pos = pk.get_panda_collision_spheres(random_pose.unsqueeze(0)) # realtive_offset, sphere_id_to_joint_id

        sphere_pos = sphere_pos.squeeze(0)

        for i, pos in enumerate(sphere_pos.cpu().numpy()):
            p.resetBasePositionAndOrientation(spheres[i], pos, [0, 0, 0, 1])

        # set joint angles
        p.setJointMotorControlArray(
            robot_id,
            np.arange(7),
            p.POSITION_CONTROL,
            targetPositions=random_pose
        )

        for _ in range(2000): 
            p.stepSimulation()


    # stop video recording
    # p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)

    assert False


    # joint pose range
    print(df.columns[16:23])
    jp = df.iloc[:, 16:23].values
    print(jp)

    ee_pos = torch.tensor(df.iloc[:, 9:12].values)
    print(df.columns[9:12])
    ee_rot = torch.tensor(df.iloc[:, 12:16].values)

    print(df.columns[12:16])

    th = torch.tensor(jp, device=pk.device, dtype=pk.dtype)
    fk_ee_pos, fk_ee_rot = pk.get_ee_pose(th)

    print(fk_ee_pos.shape)

    diff = torch.norm(fk_ee_pos - ee_pos, dim=1)
    dist_pos = torch.mean(diff)
    dist_pos_std = torch.std(diff)

    print(f"dist_pos m: {dist_pos}, std: {dist_pos_std}")

    diff = torch.norm(fk_ee_rot - ee_rot, dim=1)
    dist_rot = torch.mean(diff)
    dist_rot_std = torch.std(diff)

    print(f"dist_rot m: {dist_rot}, std: {dist_rot_std}")

    physicsClient = p.connect(p.GUI, options='--width=3000 --height=2000')

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -9.81)

    # set initial camera pose
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5, 
        cameraYaw=0, 
        cameraPitch=-45, 
        cameraTargetPosition=[0.5,-0.1,0.3])
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
    p.setGravity(0,0,-9.81)
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=1)

    index_range = list(range(len(jp)))
    # randomly shuffle the indices
    # np.random.shuffle(index_range)

    # start video recording in PyBullet
    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "isaac_sim_pybullet_difference.mp4")

    for i in index_range[:-1]:
        th = jp[i, :]


        # for o in [-1, 0, 1]:
        #     if i + o < 0 or i + o >= len(jp):
        #         continue

        curr_ee_pos = ee_pos[i + 1, :]
        curr_ee_rot = ee_rot[i + 1, :]

        # w, i, j, k to i, j, k, w
        curr_ee_rot_scipy = np.array([curr_ee_rot[1], curr_ee_rot[2], curr_ee_rot[3], curr_ee_rot[0]])

        ee_rot_mat = R.from_quat(curr_ee_rot_scipy).as_matrix()

        # debug draw the end effector pose
        p.addUserDebugLine(curr_ee_pos, curr_ee_pos + ee_rot_mat[:, 0], [1, 0, 0], 1)
        p.addUserDebugLine(curr_ee_pos, curr_ee_pos + ee_rot_mat[:, 1], [1, 0, 0], 1)
        p.addUserDebugLine(curr_ee_pos, curr_ee_pos + ee_rot_mat[:, 2], [1, 0, 0], 1)


        fk_ee_pos, fk_ee_rot = pk.get_ee_pose(torch.tensor(th, device=pk.device, dtype=pk.dtype).unsqueeze(0))
        fk_ee_rot_scipy = np.array([fk_ee_rot[0, 1], fk_ee_rot[0, 2], fk_ee_rot[0, 3], fk_ee_rot[0, 0]])
        ee_rot_mat_fk = R.from_quat(fk_ee_rot_scipy).as_matrix()
        
        (fk_ee_rot.unsqueeze(0)).squeeze(0)


        fk_ee_pos = fk_ee_pos[0].numpy()
        fk_ee_end_1 = fk_ee_pos + ee_rot_mat_fk[:, 0]
        fk_ee_end_2 = fk_ee_pos + ee_rot_mat_fk[:, 1]
        fk_ee_end_3 = fk_ee_pos + ee_rot_mat_fk[:, 2]

        p.addUserDebugLine(fk_ee_pos, fk_ee_end_1, [0, 1, 0], 1)
        p.addUserDebugLine(fk_ee_pos, fk_ee_end_2, [0, 1, 0], 1)
        p.addUserDebugLine(fk_ee_pos, fk_ee_end_3, [0, 1, 0], 1)


        # set joint angles
        p.setJointMotorControlArray(
            robot_id,
            np.arange(7),
            p.POSITION_CONTROL,
            targetPositions=th
        )

        for i in range(1000):
            p.stepSimulation()   

        # get end effecto pose
        link_state = p.getLinkState(robot_id, 7, computeForwardKinematics=True)
        pybullet_ee_pos = link_state[0]
        link_state = p.getLinkState(robot_id, 8, computeForwardKinematics=True)
        pybullet_ee_rot_scipy = link_state[1]

        pybullet_ee_rot_mat = R.from_quat(pybullet_ee_rot_scipy).as_matrix()

        # debug draw the end effector pose
        p.addUserDebugLine(pybullet_ee_pos, pybullet_ee_pos + pybullet_ee_rot_mat[:, 0], [0, 0, 1], 1)
        p.addUserDebugLine(pybullet_ee_pos, pybullet_ee_pos + pybullet_ee_rot_mat[:, 1], [0, 0, 1], 1)
        p.addUserDebugLine(pybullet_ee_pos, pybullet_ee_pos + pybullet_ee_rot_mat[:, 2], [0, 0, 1], 1)

        for i in range(40):
            p.stepSimulation()  
        # clear debug
        p.removeAllUserDebugItems()

    # end video recording
    p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)

    # th = torch.tensor([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0], device=pk.device, dtype=pk.dtype)
    # th = th.unsqueeze(0).repeat((1, 1))

    # jp, jr = pk.forward(th)
    # print(jp)
    # print(jr)

    # joint_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.0718, 2.8973], device=pk.device, dtype=pk.dtype)
    # joint_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], device=pk.device, dtype=pk.dtype)

    # physicsClient = p.connect(p.GUI, options='--width=3000 --height=2000')

    # p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # p.setGravity(0, 0, -9.81)

    # # set initial camera pose
    # p.resetDebugVisualizerCamera(
    #     cameraDistance=1.5, 
    #     cameraYaw=0, 
    #     cameraPitch=-45, 
    #     cameraTargetPosition=[0.5,-0.1,0.3])
    
    # p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
    # p.setGravity(0,0,-9.81)
    # p.setRealTimeSimulation(0)
    # num_steps = 34

    # n_x = 3
    # n_y = 3 
    # n_robots = n_x * n_y


    # blue = np.array([0, 0, 1, 0.7])
    # green = np.array([0, 1, 0, 0.7])

    # collision_sphere_colors = [
    #     i / 8 * blue + (8 - i) / 8 * green for i in range(8)
    # ]

    # robots = []
    # robot_spheres = []
    # robot_base = []

    # for nx_i in range(n_x):
    #     for ny_i in range(n_y):
    #         robot_id = p.loadURDF("franka_panda/panda.urdf", [2*nx_i, 2*ny_i, 0], useFixedBase=1)
    #         robots.append(robot_id)

    #         base_pos = torch.tensor([2*nx_i, 2*ny_i, 0])
    #         robot_base.append(base_pos)

    #         i = nx_i * n_y + ny_i
    #         robot_spheres.append([])

    #         for j in range(8):
    #             ee_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.07, rgbaColor=collision_sphere_colors[j])
    #             ee_sphere_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ee_sphere, basePosition=[2*nx_i, 2*ny_i, 1.0 + 0.05 * j])
    #             robot_spheres[i].append(ee_sphere_id)


    # while True:
    #     # Execute the trajectory by setting joint angles
    #     th = torch.rand((n_robots, 7), device=pk.device, dtype=pk.dtype) * (joint_max - joint_min) + joint_min

    #     jp, jr = pk.forward(th)

    #     # th_full = torch.concatenate([th.squeeze(0).cpu(), torch.tensor([0.4, 0.4])])

    #     for i in range(n_robots):
    #         robot_offset = robot_base[i].cpu()   
    #         for j in range(8):
    #             position = jp[j][i].cpu() + robot_offset

    #             p.resetBasePositionAndOrientation(robot_spheres[i][j], position.tolist(), [0, 0, 0, 1])

    #         p.setJointMotorControlArray(
    #             robots[i],
    #             np.arange(7),
    #             p.POSITION_CONTROL,
    #             targetPositions=th[i].tolist()
    #         )

                  
    #     for i in range(200):
    #         p.stepSimulation()      

        

    #     # p.resetBasePositionAndOrientation(ee_sphere_1, jp[0].squeeze(0).tolist(), [0, 0, 0, 1])
    #     # p.resetBasePositionAndOrientation(ee_sphere_2, jp[1].squeeze(0).tolist(), [0, 0, 0, 1])
    #     # p.resetBasePositionAndOrientation(ee_sphere_3, jp[2].squeeze(0).tolist(), [0, 0, 0, 1])
    #     # p.resetBasePositionAndOrientation(ee_sphere_4, jp[3].squeeze(0).tolist(), [0, 0, 0, 1])
    #     # p.resetBasePositionAndOrientation(ee_sphere_5, jp[4].squeeze(0).tolist(), [0, 0, 0, 1])
    #     # p.resetBasePositionAndOrientation(ee_sphere_6, jp[5].squeeze(0).tolist(), [0, 0, 0, 1])
    #     # p.resetBasePositionAndOrientation(ee_sphere_7, jp[6].squeeze(0).tolist(), [0, 0, 0, 1])
    #     # p.resetBasePositionAndOrientation(ee_sphere_8, jp[7].squeeze(0).tolist(), [0, 0, 0, 1])


    #     # # set joint angles
    #     # p.setJointMotorControlArray(
    #     #     robot_id,
    #     #     np.arange(7),
    #     #     p.POSITION_CONTROL,
    #     #     targetPositions=th.squeeze(0).tolist()
    #     # )
            
    #     # for i in range(1000):
    #     #     p.stepSimulation()