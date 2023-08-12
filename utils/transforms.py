import math
import numpy as np
from pxr import Gf
from .compute_points import compute_points


def get_pose_world(trans_rel, rot_rel, robot_pos, robot_rot):
    if rot_rel is not None:
        rot = robot_rot @ rot_rel
    else:
        rot = None

    if trans_rel is not None:
        trans = robot_rot @ trans_rel + robot_pos
    else:
        trans = None

    return trans, rot


def get_pose_relat(trans, rot, robot_pos, robot_rot):
    inv_rob_rot = robot_rot.T

    if trans is not None:
        trans_rel = inv_rob_rot @ (trans - robot_pos)
    else:
        trans_rel = None

    if rot is not None:
        rot_rel = inv_rob_rot @ rot
    else:
        rot_rel = None
    
    return trans_rel, rot_rel


def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([w, x, y, z], axis=-1).reshape(shape)

    return quat


def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return np.concatenate((a[:, 0:1], -a[:, 1:]), axis=-1).reshape(shape)


def quat_diff_rad(a, b):
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    return 2.0 * np.arcsin(np.clip(np.linalg.norm(mul[:, 1:], axis=-1), 0, 1))


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    # might need to be normalized
    rotm = Gf.Matrix3f(Gf.Quatf(*quat.tolist())).GetTranspose()
    return np.array(rotm)


_FLOAT_EPS = np.finfo(np.float32).eps
_EPS4 = _FLOAT_EPS * 4.0
def matrix_to_euler_angles(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Euler XYZ angles.

    Args:
        mat (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: Euler XYZ angles (in radians).
    """
    cy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
    singular = cy < _EPS4
    if not singular:
        roll = math.atan2(mat[2, 1], mat[2, 2])
        pitch = math.atan2(-mat[2, 0], cy)
        yaw = math.atan2(mat[1, 0], mat[0, 0])
    else:
        roll = math.atan2(-mat[1, 2], mat[1, 1])
        pitch = math.atan2(-mat[2, 0], cy)
        yaw = 0
    return np.array([roll, pitch, yaw])


def euler_angles_to_quat(euler_angles: np.ndarray, degrees: bool = False) -> np.ndarray:
    """Convert Euler XYZ angles to quaternion. Adapted from omni.isaac.core

    Args:
        euler_angles (np.ndarray):  Euler XYZ angles.
        degrees (bool, optional): Whether input angles are in degrees. Defaults to False.

    Returns:
        np.ndarray: quaternion (w, x, y, z).
    """
    roll, pitch, yaw = euler_angles
    if degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    w = (cr * cp * cy) + (sr * sp * sy)
    x = (sr * cp * cy) - (cr * sp * sy)
    y = (cr * sp * cy) + (sr * cp * sy)
    z = (cr * cp * sy) - (sr * sp * cy)
    return np.array([w, x, y, z])


def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    return euler_angles_to_quat(matrix_to_euler_angles(mat))


def create_pcd_hardcode(camera, depth, cm_to_m=True):
    height = camera['resolution']['height']
    width =  camera['resolution']['width']
    focal_length = camera['focal_length'] 
    horiz_aperture = camera['horizontal_aperture']
    vert_aperture = height / width * horiz_aperture
    
    fx = width * focal_length / horiz_aperture
    fy = height * focal_length / vert_aperture
    
    cx = cy = height/2

    points_cam = compute_points(height, width, depth, fx, fy, cx, cy)
    
    T = camera['pose'].T
    Rotation = T[:3, :3]
    t = T[:3, 3]
    points_world = points_cam @ np.transpose(Rotation) + t
    points_world = np.array(points_world).reshape(height, width,3)
    
    return points_world / 100 if cm_to_m else points_world
