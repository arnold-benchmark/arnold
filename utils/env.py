import torch
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from peract.utils import get_obs_batch_dict
from .transforms import create_pcd_hardcode, quat_diff_rad, get_pose_relat, get_pose_world
CAMERAS = ['front', 'base', 'left', 'wrist_bottom', 'wrist']


class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 left_rgb: np.ndarray,
                 left_depth: np.ndarray,
                #  left_mask: np.ndarray,
                 left_point_cloud: np.ndarray,
                 base_rgb: np.ndarray,
                 base_depth: np.ndarray,
                #  base_mask: np.ndarray,
                 base_point_cloud: np.ndarray,
                #  overhead_rgb: np.ndarray,
                #  overhead_depth: np.ndarray,
                #  overhead_mask: np.ndarray,
                #  overhead_point_cloud: np.ndarray,
                 wrist_rgb: np.ndarray,
                 wrist_depth: np.ndarray,
                #  wrist_mask: np.ndarray,
                 wrist_point_cloud: np.ndarray,

                 wrist_bottom_rgb: np.ndarray,
                 wrist_bottom_depth: np.ndarray,
                #  wrist_bottom_mask: np.ndarray,
                 wrist_bottom_point_cloud: np.ndarray,

                 front_rgb: np.ndarray,
                 front_depth: np.ndarray,
                #  front_mask: np.ndarray,
                 front_point_cloud: np.ndarray,
                joint_velocities: np.ndarray,
                joint_positions: np.ndarray,
                #  joint_forces: np.ndarray,
                 gripper_open: float,
                 gripper_pose: np.ndarray,
                #  gripper_matrix: np.ndarray,
                 gripper_joint_positions: np.ndarray,
                #  gripper_touch_forces: np.ndarray,
                #  task_low_dim_state: np.ndarray,
                #  misc: dict
                bound_center: np.ndarray
                 ):
        self.left_rgb = left_rgb
        self.left_depth = left_depth
        # self.left_mask = left_mask
        self.left_point_cloud = left_point_cloud

        self.base_rgb = base_rgb
        self.base_depth = base_depth
        # self.base_mask = base_mask
        self.base_point_cloud = base_point_cloud
        # self.overhead_rgb = overhead_rgb
        # self.overhead_depth = overhead_depth
        # self.overhead_mask = overhead_mask
        # self.overhead_point_cloud = overhead_point_cloud

        self.wrist_rgb = wrist_rgb
        self.wrist_depth = wrist_depth
        # self.wrist_mask = wrist_mask
        self.wrist_point_cloud = wrist_point_cloud

        self.wrist_bottom_rgb = wrist_bottom_rgb
        self.wrist_bottom_depth = wrist_bottom_depth
        # self.wrist_bottom_mask = wrist_bottom_mask
        self.wrist_bottom_point_cloud = wrist_bottom_point_cloud

        self.front_rgb = front_rgb
        self.front_depth = front_depth
        # self.front_mask = front_mask
        self.front_point_cloud = front_point_cloud
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = None
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose
        self.gripper_matrix = None
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = None
        # self.task_low_dim_state = task_low_dim_state
        
        # self.misc = misc
        self.bound_center = bound_center


def get_ee(cspace_controller):
    from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats
    ee_pos, ee_rot = cspace_controller.get_motion_policy().get_end_effector_pose(
        cspace_controller.get_articulation_motion_policy().get_active_joints_subset().get_joint_positions()
    )
    ee_rot = rot_matrices_to_quats(ee_rot)
    return (ee_pos, ee_rot)


def get_obs(franka, cspace_controller, gt, type='rgb'):
    obs = {}

    robot_base = franka.get_world_pose()
    robot_base_pos = robot_base[0].copy()
    robot_forward_direction = R.from_quat(robot_base[1][[1,2,3,0]]).as_matrix()[:, 0]
    robot_forward_direction[1] = 0   # height
    robot_forward_direction = robot_forward_direction / np.linalg.norm(robot_forward_direction) * 50   # cm
    bound_center = robot_base_pos + robot_forward_direction

    position_rotation_world = get_ee(cspace_controller)
    # gripper_pose_trans = position_rotation_world[0] / 100.0   # cm to m
    gripper_pose_trans = position_rotation_world[0]
    quat = position_rotation_world[1][[1,2,3,0]]   # wxyz to xyzw
    gripper_pose = [*gripper_pose_trans, *quat.tolist()]

    gripper_joint_positions = franka.gripper.get_joint_positions()

    for camera_idx in [0,1,2,3,4]:
        if type == 'rgb':
            rgb = gt['images'][camera_idx]['rgb'][:,:,:3]
        elif type == 'mask':
            rgb = gt['images'][camera_idx]['semanticSegmentation'][:,:,np.newaxis].repeat(3,-1) * 50
        else:
            raise ValueError('observation type should be either rgb or mask')
        
        depth = np.clip(gt['images'][camera_idx]['depthLinear'], 0, 10)
        camera = gt['images'][camera_idx]['camera']
        point_cloud = create_pcd_hardcode(camera, depth, cm_to_m=True)
        obs[CAMERAS[camera_idx]+'_rgb'] = rgb
        obs[CAMERAS[camera_idx]+'_depth'] = depth
        obs[CAMERAS[camera_idx]+'_point_cloud'] = point_cloud - bound_center / 100
    
    gripper_open = (gripper_joint_positions[0] + gripper_joint_positions[1] > 7)

    ob = Observation(
        left_rgb=obs['left_rgb'], 
        left_depth=obs['left_depth'],
        left_point_cloud=obs['left_point_cloud'],
        front_rgb=obs['front_rgb'],
        front_depth=obs['front_depth'],
        front_point_cloud=obs['front_point_cloud'],
        base_rgb=obs['base_rgb'],
        base_depth=obs['base_depth'],
        base_point_cloud=obs['base_point_cloud'],
        wrist_rgb=obs['wrist_rgb'],
        wrist_depth=obs['wrist_depth'],
        wrist_point_cloud=obs['wrist_point_cloud'],
        wrist_bottom_rgb=obs['wrist_bottom_rgb'],
        wrist_bottom_depth=obs['wrist_bottom_depth'],
        wrist_bottom_point_cloud=obs['wrist_bottom_point_cloud'],
        gripper_open=gripper_open,
        gripper_pose=gripper_pose,
        gripper_joint_positions=gripper_joint_positions,
        bound_center=bound_center,
        joint_positions=franka.get_joint_positions(),
        joint_velocities=franka.get_joint_velocities(),
    )
    
    return ob


def position_reached(c_controller, target, robot, thres=1.5):
    if target is None:
        return True
    
    ee_pos, R = c_controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
    pos_diff = np.linalg.norm(ee_pos- target)
    # print(f'{pos_diff} = || {ee_pos} - {target} ||')
    if pos_diff < thres:
        return True
    else:
        return False


def rotation_reached(c_controller, target):
    if target is None:
        return True
    
    ee_pos, R = c_controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
    angle_diff = quat_diff_rad(R, target)[0]
    # print(f'angle diff: {angle_diff}')
    if angle_diff < 0.1:
        return True


@torch.no_grad()
def get_action(gt, agent, franka, c_controller, npz_file, offset, timestep, device, agent_type, obs_type='rgb', lang_embed_cache=None):
    obs = get_obs(franka, c_controller, gt, type=obs_type)
    bound_center = obs.bound_center

    instruction = npz_file['gt'][0]['instruction']

    if agent_type == 'cliport6d':
        bounds = offset / 100

        # y-up to z-up
        bounds = bounds[[0, 2, 1]]
        obs.front_point_cloud = obs.front_point_cloud[:, :, [0, 2, 1]]
        obs.base_point_cloud = obs.base_point_cloud[:, :, [0, 2, 1]]
        obs.left_point_cloud = obs.left_point_cloud[:, :, [0, 2, 1]]
        obs.wrist_bottom_point_cloud = obs.wrist_bottom_point_cloud[:, :, [0, 2, 1]]
        obs.wrist_point_cloud = obs.wrist_point_cloud[:, :, [0, 2, 1]]

        inp_img, lang_goal, p0, output_dict = agent.act(obs, [instruction], bounds=bounds, pixel_size=5.625e-3)
        
        trans = np.ones(3) * 100
        # m to cm
        trans[[0,2]] *= output_dict['place_xy']
        trans[1] *= output_dict['place_z']
        trans += bound_center

        rotation = np.array([output_dict['place_theta'], output_dict['pitch'], output_dict['roll']])
        rotation = R.from_euler('zyx', rotation, degrees=False).as_matrix()
        rot_transition = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).astype(float)
        rotation = R.from_matrix(rot_transition @ rotation).as_quat()

        # print(
        #     f'Model output: xy={output_dict["place_xy"]}, z={output_dict["place_z"]}, '
        #         f'theta={output_dict["place_theta"]/np.pi*180}, pitch={output_dict["pitch"]/np.pi*180}, roll={output_dict["roll"]/np.pi*180}'
        # )
    
    elif agent_type == 'peract':
        input_dict = {}

        obs_dict = get_obs_batch_dict(obs)
        input_dict.update(obs_dict)

        lang_goal_embs = lang_embed_cache.get_lang_embed(instruction)
        gripper_open = obs.gripper_open
        gripper_joint_positions = np.clip(obs.gripper_joint_positions, 0, 0.04)
        low_dim_state = np.array([gripper_open, *gripper_joint_positions, timestep]).reshape(1, -1)
        input_dict.update({
            'lang_goal_embs': lang_goal_embs,
            'low_dim_state': low_dim_state
        })

        for k, v in input_dict.items():
            if v is not None:
                if not isinstance(v, torch.Tensor):
                    v = torch.from_numpy(v)
                input_dict[k] = v.to(device)
        
        output_dict = agent.predict(input_dict)
        trans = output_dict['pred_action']['continuous_trans'].detach().cpu().numpy() * 100 + bound_center
        rotation = output_dict['pred_action']['continuous_quat']
    
    elif 'bc_lang' in agent_type:
        input_dict = {}

        obs_dict = get_obs_batch_dict(obs)
        input_dict.update(obs_dict)

        lang_goal_embs = lang_embed_cache.get_lang_embed(instruction)
        gripper_open = obs.gripper_open
        gripper_joint_positions = np.clip(obs.gripper_joint_positions, 0, 0.04)
        low_dim_state = np.array([gripper_open, *gripper_joint_positions, timestep]).reshape(1, -1)
        input_dict.update({
            'lang_goal_embs': lang_goal_embs,
            'low_dim_state': low_dim_state,
            'bound': np.array(offset).reshape(2, 3)
        })

        for k, v in input_dict.items():
            if v is not None:
                if not isinstance(v, torch.Tensor):
                    v = torch.from_numpy(v)
                input_dict[k] = v.to(device)
        
        trans, rotation = agent.act(step=None, replay_sample=input_dict)
        trans = trans * 100 + bound_center
    
    else:
        raise ValueError(f'{agent_type} agent not supported')

    # print(f'action prediction: trans={trans}, orient(euler XYZ)={R.from_quat(rotation).as_euler("XYZ", degrees=True)}')

    rotation = rotation[[3, 0, 1, 2]]   # xyzw to wxyz

    return trans, rotation


def get_pre_grasp_action(grasp_action, robot_base, task):
    """
    grasp_action: ( pos_world, rot_world (wxyz) )
    robot_base: ( robot_pos, robot_rot (wxyz) )
    position represented in cm
    return pre-grasping action ( pre_pos_world, pre_rot_world (wxyz) )
    """
    pos_world, rot_world = grasp_action
    robot_pos, robot_rot = robot_base

    # wxyz to xyzw
    rot_world = rot_world[[1,2,3,0]]
    robot_rot = robot_rot[[1,2,3,0]]
    # to matrix
    rot_world = R.from_quat(rot_world).as_matrix()
    robot_rot = R.from_quat(robot_rot).as_matrix()

    # relative action
    pre_pos_relat, pre_rot_relat = get_pose_relat(trans=pos_world, rot=rot_world, robot_pos=robot_pos, robot_rot=robot_rot)

    if task in ['pickup_object', 'open_drawer', 'close_drawer', 'open_cabinet', 'close_cabinet']:
        # x - 5cm
        pre_pos_relat[0] -= 5
    elif task in ['reorient_object']:
        # z at 15cm
        pre_pos_relat[2] = 15
    else:
        # water, z + 5cm
        pre_pos_relat[2] += 5
    
    # world action
    pre_pos_world, pre_rot_world = get_pose_world(trans_rel=pre_pos_relat, rot_rel=pre_rot_relat, robot_pos=robot_pos, robot_rot=robot_rot)
    pre_rot_world = R.from_matrix(pre_rot_world).as_quat()
    pre_rot_world = pre_rot_world[[3,0,1,2]]
    return pre_pos_world, pre_rot_world


def action_interpolation(trans_previous, rotation_previous, trans_target, rotation_target, alphas, task, joint_pos=None):
    action_list = []

    if 'drawer' in task:
        for alpha in alphas:
            trans_interp = alpha * trans_target + (1 - alpha) * trans_previous
            action_list.append((trans_interp, rotation_target))
    
    elif 'cabinet' in task:
        rotation_previous_xyzw = rotation_previous[[1, 2, 3, 0]]
        rotation_target_xyzw = rotation_target[[1, 2, 3, 0]]
        key_rots = R.from_quat(np.stack([rotation_previous_xyzw, rotation_target_xyzw]))
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(alphas).as_quat()
        interp_rots = interp_rots[:, [3, 0, 1, 2]]   # back to wxyz

        r_0 = trans_previous[[0, 2]] - joint_pos[[0, 2]]
        r_1 = trans_target[[0, 2]] - joint_pos[[0, 2]]

        radius_0 = np.linalg.norm(r_0)
        radius_1 = np.linalg.norm(r_1)
        radii = np.linspace(radius_0, radius_1, len(alphas)+1)[1:]

        theta_0 = np.arctan2(r_0[1], r_0[0])
        theta_1 = np.arctan2(r_1[1], r_1[0])
        thetas = np.linspace(theta_0 if theta_1 - theta_0 <= np.pi else theta_0 + 2 * np.pi, theta_1, len(alphas)+1)[1:]

        for alpha, radius, theta, interp_rot in zip(alphas, radii, thetas, interp_rots):
            trans_interp = np.array([
                joint_pos[0] + radius * np.cos(theta),
                alpha * trans_target[1] + (1 - alpha) * trans_previous[1],
                joint_pos[2] + radius * np.sin(theta)
            ])
            action_list.append((trans_interp, interp_rot))
    
    elif 'water' in task:
        rotation_previous_xyzw = rotation_previous[[1, 2, 3, 0]]
        rotation_target_xyzw = rotation_target[[1, 2, 3, 0]]
        key_rots = R.from_quat(np.stack([rotation_previous_xyzw, rotation_target_xyzw]))
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(alphas).as_quat()
        interp_rots = interp_rots[:, [3, 0, 1, 2]]   # back to wxyz
        for interp_rot in interp_rots:
            action_list.append((trans_target, interp_rot))
    
    else:
        raise ValueError(f'{task} does not need interpolation')
    
    return action_list
