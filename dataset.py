import os
import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from cliport6d.utils.utils import get_fused_heightmap
from utils.env import CAMERAS
from utils.transforms import get_pose_world, create_pcd_hardcode

pickle.DEFAULT_PROTOCOL=pickle.HIGHEST_PROTOCOL
RANDOM_SEED=1125
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def parse_state(filename):
    units = filename.split('-')
    if 'pickup_object' in filename:
        init_state = 0
        goal_state = int(units[-3]) / 40
    elif 'reorient_object' in filename:
        init_state = 0.5
        goal_state = int(filename.split(')')[0].split('_')[-1]) / 180
    elif 'pour_water' in filename:
        init_state = 0
        goal_state = (100 - int(units[-3])) / 100
    elif 'transfer_water' in filename:
        init_state = 0
        goal_state = int(units[-3]) / 100
    else:
        init_state = float(units[-4])
        goal_state = float(units[-3])
    
    return init_state, goal_state


class ArnoldDataset(Dataset):
    def __init__(self, data_path, task, obs_type='rgb', offset_bound=np.array([-0.63, 0, -0.63, 0.63, 1.26, 0.63])):
        """
        Dataset structure: {
            object_id: {
                'act1': List,
                'act2': List
            }
        }
        """
        super().__init__()
        self.data_path = data_path
        self.task = task
        self.pixel_size = 5.625e-3
        self.obs_type = obs_type
        self.task_offset = offset_bound.reshape(2, 3).transpose(1, 0)
        self.sample_weights = [0.2, 0.8]
        self.obj_ids = []
        self.episode_dict = {}
        self._load_keyframes()
    
    def _load_keyframes(self):
        """
        Generally, there are 4 frames in each demonstration:
            0: initial state
            1: pre-grasping state
            2: grasping state
            3: final goal state
        In this setting, two samples are extracted for training:
            - observation at initial state (0) as visual input, gripper state at grasping state (2) as action label
            - observation at grasping state (2) as visual input, gripper state at final goal state (3) as action label
        For tasks involving water, the difference is as follows:
            3: to the position before tilting
            4: to the orientation before reverting the cup
            5: final upwards state
            - combination of position at frame 3 and orientation at frame 4 as action label
        All template actions are in world frame.
        """
        for fname in os.listdir(self.data_path):
            if fname.endswith('npz'):
                obj_id = fname.split('-')[2]
                if obj_id not in self.episode_dict:
                    self.obj_ids.append(obj_id)
                    self.episode_dict.update({
                        obj_id: {
                            'act1': [],
                            'act2': []
                        }
                    })
                
                init_state, goal_state = parse_state(fname)

                gt_frames = np.load(os.path.join(self.data_path, fname), allow_pickle=True)['gt']
                language_instructions = gt_frames[0]['instruction']

                # pick phase
                step = gt_frames[0].copy()
                robot_base_pos = step['robot_base'][0] / 100
                robot_forward_direction = R.from_quat(step['robot_base'][1][[1,2,3,0]]).as_matrix()[:, 0]
                robot_forward_direction[1] = 0   # height
                robot_forward_direction = robot_forward_direction / np.linalg.norm(robot_forward_direction) * 0.5   # m
                bound_center = robot_base_pos + robot_forward_direction

                cmap, hmap, obs_dict = self.get_step_obs(step, self.task_offset[[0, 2, 1]], self.pixel_size, type=self.obs_type)
                hmap = np.tile(hmap[..., None], (1,1,3))
                img = np.concatenate([cmap, hmap], axis=-1)

                obj_pos = gt_frames[2]['position_rotation_world'][0] / 100
                obj_pos = obj_pos - bound_center

                act_pos = gt_frames[2]['position_rotation_world'][0].copy()
                act_rot = gt_frames[2]['position_rotation_world'][1].copy()
                act_pos /= 100
                act_rot = act_rot[[1,2,3,0]]   # wxyz to xyzw
                target_points = self.get_act_label_from_abs(pos_abs=act_pos, rot_abs=act_rot)
                target_points[:3] = target_points[:3] - bound_center

                gripper_open = 1
                gripper_joint_positions = step['gripper_joint_positions'] / 100
                gripper_joint_positions = np.clip(gripper_joint_positions, 0, 0.04)
                timestep = 0
                low_dim_state = np.array([gripper_open, *gripper_joint_positions, timestep])
                
                episode_dict1 = {
                    "img": img,   # [H, W, 6]
                    "obs_dict": obs_dict,   # { {camera_name}_{rgb/point_cloud}: [H, W, 3] }
                    "attention_points": obj_pos,   # [3,]
                    "target_points": target_points,   # [6,]
                    "target_gripper": gripper_open,   # binary
                    "low_dim_state": low_dim_state,   # [grip_open, left_finger, right_finger, timestep]
                    "language": language_instructions,   # str
                    "current_state": init_state,   # scalar
                    "goal_state": goal_state,   # scalar
                    "bounds": self.task_offset,   # [3, 2]
                    "pixel_size": self.pixel_size,   # scalar
                }

                self.episode_dict[obj_id]['act1'].append(episode_dict1)

                # place phase
                step = gt_frames[2].copy()
                robot_base_pos = step['robot_base'][0] / 100
                robot_forward_direction = R.from_quat(step['robot_base'][1][[1,2,3,0]]).as_matrix()[:, 0]
                robot_forward_direction[1] = 0   # height
                robot_forward_direction = robot_forward_direction / np.linalg.norm(robot_forward_direction) * 0.5   # m
                bound_center = robot_base_pos + robot_forward_direction

                cmap, hmap, obs_dict = self.get_step_obs(step, self.task_offset[[0, 2, 1]], self.pixel_size, type=self.obs_type)
                hmap = np.tile(hmap[..., None], (1,1,3))
                img = np.concatenate([cmap, hmap], axis=-1)

                obj_pos = step['position_rotation_world'][0] / 100 - bound_center

                act_pos = gt_frames[3]['position_rotation_world'][0].copy()
                if self.task in ['pour_water', 'transfer_water']:
                    # water, compose actions of two frames
                    act_rot = gt_frames[4]['position_rotation_world'][1].copy()
                else:
                    # default
                    act_rot = gt_frames[3]['position_rotation_world'][1].copy()
                
                act_pos /= 100
                act_rot = act_rot[[1,2,3,0]]   # wxyz to xyzw
                target_points = self.get_act_label_from_abs(pos_abs=act_pos, rot_abs=act_rot)
                target_points[:3] = target_points[:3] - bound_center

                gripper_open = 0
                gripper_joint_positions = step['gripper_joint_positions'] / 100
                gripper_joint_positions = np.clip(gripper_joint_positions, 0, 0.04)
                timestep = 1
                low_dim_state = np.array([gripper_open, *gripper_joint_positions, timestep])

                episode_dict2 = {
                    "img": img,   # [H, W, 6]
                    "obs_dict": obs_dict,   # { {camera_name}_{rgb/point_cloud}: [H, W, 3] }
                    "attention_points": obj_pos,   # [3,]
                    "target_points": target_points,   # [6,]
                    "target_gripper": gripper_open,   # binary
                    "low_dim_state": low_dim_state,   # [grip_open, left_finger, right_finger, timestep]
                    "language": language_instructions,   # str
                    "current_state": init_state,   # scalar
                    "goal_state": goal_state,   # scalar
                    "bounds": self.task_offset,   # [3, 2]
                    "pixel_size": self.pixel_size,   # scalar
                }

                self.episode_dict[obj_id]['act2'].append(episode_dict2)
        
        print(f'Loaded {[len(v["act1"]) for k, v in self.episode_dict.items()]} demos')

    def __len__(self):
        num_demos = 0
        for k, v in self.episode_dict.items():
            num_demos += len(v['act1'])
        return num_demos
    
    def __getitem__(self, index):
        obj_demos = [len(v['act1']) for k, v in self.episode_dict.items()]
        interval_upper = np.cumsum(obj_demos)
        interval_lower = np.array([0, *interval_upper[:-1]])

        obj_idx = ((index>=interval_lower) * (index<interval_upper)).tolist().index(True)
        demo_idx = index - interval_lower[obj_idx]
        obj_idx = self.obj_ids[obj_idx]

        act_idx = 1 + np.random.choice(2, size=1, p=self.sample_weights)[0]
        return self.episode_dict[obj_idx][f'act{act_idx}'][demo_idx]

        # sample
        # obj_idx = random.choice(list(self.episode_dict.keys()))
        # act_idx = 1 + np.random.choice(2, size=1, p=self.sample_weights)[0]
        # return random.choice(self.episode_dict[obj_idx][f'act{act_idx}'])
    
    def sample(self, batch_size):
        samples = []
        sampled_idx = []
        while len(samples) < batch_size:
            obj_idx = random.choice(self.obj_ids)
            act_idx = 1 + np.random.choice(2, size=1, p=self.sample_weights)[0]
            demo_idx = np.random.randint(len(self.episode_dict[obj_idx][f'act{act_idx}']), size=1)[0]
            obj_act_demo_tuple = (obj_idx, act_idx, demo_idx)
            if obj_act_demo_tuple not in sampled_idx:
                samples.append(self.episode_dict[obj_idx][f'act{act_idx}'][demo_idx])
                sampled_idx.append(obj_act_demo_tuple)
        
        return samples

    def get_step_obs(self, step, bounds, pixel_size, type='rgb'):
        # bounds: [3, 2], xyz (z-up)
        imgs = step['images']
        colors = []
        pcds = []
        obs_dict = {}
        for camera_name, camera_obs in zip(CAMERAS, imgs):
            camera = camera_obs['camera']

            if type == 'rgb':
                color = camera_obs['rgb'][:, :, :3]
            elif type == 'mask':
                color = camera_obs['semanticSegmentation'][:,:,np.newaxis].repeat(3,-1) * 50
            else:
                raise ValueError('observation type should be either rgb or mask')
            colors.append(color)

            depth = np.clip(camera_obs['depthLinear'], 0, 10)

            point_cloud = create_pcd_hardcode(camera, depth, cm_to_m=True)
            # here point_cloud is y-up
            robot_forward_direction = R.from_quat(step['robot_base'][1][[1,2,3,0]]).as_matrix()[:, 0]
            robot_forward_direction[1] = 0
            robot_forward_direction = robot_forward_direction / np.linalg.norm(robot_forward_direction) * 0.5   # m
            bound_center = step['robot_base'][0] / 100 + robot_forward_direction
            point_cloud = point_cloud - bound_center

            pcds.append(point_cloud[:, :, [0, 2, 1]])   # pcds is for cliport6d, which requires z-up

            obs_dict.update({
                f'{camera_name}_rgb': color,
                f'{camera_name}_point_cloud': point_cloud
            })   # obs_dict is for peract, which requires y-up
        
        # to fuse map, pcds and bounds are supposed to be xyz (z-up)
        cmap, hmap = get_fused_heightmap(colors, pcds, bounds, pixel_size)
        return cmap, hmap, obs_dict
    
    def get_act_label_from_rel(self, pos_rel, rot_rel, robot_base):
        robot_pos, robot_rot = robot_base
        robot_rot = R.from_quat(robot_rot).as_matrix()
        pos_world, rot_world = get_pose_world(pos_rel, rot_rel, robot_pos, robot_rot)
        if rot_world is None:
            return pos_world
        else:
            rot_world = R.from_matrix(rot_world).as_quat()
            return np.concatenate([pos_world, rot_world])
    
    def get_act_label_from_abs(self, pos_abs, rot_abs):
        if rot_abs is None:
            return pos_abs
        else:
            return np.concatenate([pos_abs, rot_abs])


class ArnoldMultiTaskDataset(Dataset):
    task_list = [
        'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
        'open_cabinet', 'close_cabinet', 'pour_water', 'transfer_water'
    ]
    def __init__(self, data_root, obs_type='rgb'):
        """
        Dataset structure: [
            {
                object_id: {
                    'act1': List,
                    'act2': List
                }
            }
        ]
        """
        super().__init__()
        self.data_root = data_root
        self.obs_type = obs_type
        self.task_dict = {}
        for task in self.task_list:
            print(f'Create dataset for {task}')
            self.task_dict[task] = ArnoldDataset(os.path.join(data_root, task, 'train'), task, obs_type)

    def __len__(self):
        num_demos = 0
        for k, v in self.task_dict.items():
            num_demos += len(v)
        return num_demos
    
    def __getitem__(self, index):
        demos_per_task = [len(v) for k, v in self.task_dict.items()]
        interval_upper = np.cumsum(demos_per_task)
        interval_lower = np.array([0, *interval_upper[:-1]])

        task_idx = ((index>=interval_lower) * (index<interval_upper)).tolist().index(True)
        demo_idx = index - interval_lower[task_idx]
        return self.task_dict[self.task_list[task_idx]][demo_idx]
    
    def sample(self, batch_size):
        samples = []
        while len(samples) < batch_size:
            task = random.choice(self.task_list)
            samples.append(self.task_dict[task].sample(1)[0])
        
        return samples


class InstructionEmbedding():
    def __init__(self, lang_encoder):
        self.cache = {}
        self.lang_encoder = lang_encoder
    
    def get_lang_embed(self, instructions):
        if self.lang_encoder is None:
            return None
        
        if isinstance(instructions, str):
            # a single sentence
            instructions = [instructions]
        
        lang_embeds = []
        for sen in instructions:
            if sen not in self.cache:
                sen_embed = self.lang_encoder.encode_text([sen])
                sen_embed = sen_embed[0]
                self.cache.update({sen: sen_embed})
            
            lang_embeds.append(self.cache[sen])
        
        lang_embeds = torch.stack(lang_embeds, dim=0)
        return lang_embeds
