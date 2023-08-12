import numpy as np
from scipy.spatial.transform import Rotation as R
CAMERAS = ['front', 'base', 'left', 'wrist_bottom', 'wrist']


def point_to_voxel_index(points: np.ndarray, voxel_size: np.ndarray, coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indices = np.minimum(np.floor((points - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one)
    return voxel_indices


def normalize_quaternion(quat):
    quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat.reshape(-1, 4)
    for i in range(quat.shape[0]):
        if quat[i, -1] < 0:
            quat[i] = -quat[i]
    return quat


def quaternion_to_discrete_euler(quaternion, resolution):
    euler = R.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return R.from_euler('xyz', euluer, degrees=True).as_quat()


def preprocess_inputs(replay_sample):
    obs, pcds = [], []
    for n in CAMERAS:
        rgb = replay_sample[f'{n}_rgb']
        pcd = replay_sample[f'{n}_point_cloud']

        rgb = (rgb.float() / 255.0) * 2.0 - 1.0

        obs.append([rgb, pcd]) # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd) # only pointcloud
    return obs, pcds


def get_obs_batch_dict(obs):
    rgb_list = [obs.front_rgb, obs.base_rgb, obs.left_rgb, obs.wrist_bottom_rgb, obs.wrist_rgb]

    pcd_list = [obs.front_point_cloud, obs.base_point_cloud, obs.left_point_cloud, obs.wrist_bottom_point_cloud, obs.wrist_point_cloud]

    obs_dict = {}
    for n, rgb, pcd in zip(CAMERAS, rgb_list, pcd_list):
        rgb = rgb[np.newaxis, ...]
        pcd = pcd[np.newaxis, ...]

        # peract requires input as [C, H, W]
        obs_dict[f'{n}_rgb'] = rgb.transpose(0, 3, 1, 2)
        obs_dict[f'{n}_point_cloud'] = pcd.transpose(0, 3, 1, 2)
    
    return obs_dict
