from .base_task import BaseTask
from typing import List
from environment.parameters import *
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_at_path, delete_prim, get_all_matching_child_prims
from omni.isaac.core.utils.semantics import add_update_semantics

import omni
from omni.isaac.core.prims import XFormPrim
from environment.physics_utils import set_physics_properties
from utils.env import position_reached, rotation_reached, get_pre_grasp_action, action_interpolation
from omni.isaac.core.simulation_context import SimulationContext
from environment.fluid_utils import set_particle_system_for_cup
from utils.transforms import get_pose_relat, euler_angles_to_quat, quat_to_rot_matrix, matrix_to_quat, quat_diff_rad

from pxr import Gf
import logging
logger = logging.getLogger(__name__)


class PourWater(BaseTask):
    def __init__(self, num_stages, horizon, stage_properties, cfg) -> None:
        super().__init__(num_stages, horizon, stage_properties, cfg)

        self.task = 'pour_water'
        self.grip_open = cfg.gripper_open[self.task]
        self.use_gpu_physics = True
        self.iso_surface = cfg.iso_surface

    def reset(self, robot_parameters, 
              scene_parameters, 
              object_parameters,
              robot_base,
              gt_actions,
        ):

        super().stop()

        self.robot_parameters: RobotParameters = robot_parameters
        self.object_parameter: ObjectParameters = object_parameters[0]
        self.stage = omni.usd.get_context().get_stage()
        self.checker = None

        self.robot_base = robot_base

        obs = super().reset(
            robot_parameters = robot_parameters,
            scene_parameters = scene_parameters
        )
        # simulation_context = SimulationContext.instance() 
        # while True:
        #         simulation_context.step(render=True)
        # used for max mumber of steps (grasp, raise up)
        self.current_stage = 0
        self.end_stage = 0

        self.time_step = 0
        self.is_success = 0
        self.gt_actions = gt_actions

        return obs

    def set_up_task(self) -> None:
        self.load_object()

    def load_object(self):
        # TODO:
        # For now only supports one environment, we will use cloner in the future
        index = 0
        self.objects_list = []
        param = self.object_parameter
        
        object_prim_path = find_unique_string_name(
            initial_name = f"/World_{index}/{param.object_type}",
            is_unique_fn = lambda x: not is_prim_path_valid(x)
        )

        object_prim = add_reference_to_stage(param.usd_path, object_prim_path)

        cup_water_init_holder = object_prim_path
        cup_water_final_holder = object_prim_path

        particle_system_path = '/World_0/Fluid'

        particle_instance_str = "/World_0/Particles"
        
        volume_mesh_path = object_prim.GetPath().AppendPath(f"cup_volume").pathString
        set_particle_system_for_cup(
            self.stage, Gf.Vec3f(param.object_position[0], param.object_position[1], param.object_position[2]),
            volume_mesh_path, particle_system_path, particle_instance_str, param.fluid_properties,
            asset_root=self.cfg.asset_root, enable_iso_surface=self.iso_surface
        )

        self._wait_for_loading()

        self.objects_list.append(object_prim)
        
        position = param.object_position
        rotation = param.orientation_quat
        
        # use this to set relative position, orientation and scale
        xform_prim = XFormPrim(object_prim_path, translation= position, orientation = rotation, scale = np.array(param.scale))
        self._wait_for_loading()

        if param.object_physics_properties:
            set_physics_properties(self.stage, object_prim, param.object_physics_properties)
        
        add_update_semantics(object_prim, param.object_type)

        if param.part_physics_properties:
            for keyword, properties in param.part_physics_properties.items():
                
                prim_list = get_all_matching_child_prims(object_prim_path, properties.properties[PREDICATE])
                for sub_prim_path in prim_list:
                    try:
                        sub_prim = get_prim_at_path(sub_prim_path)
                    except:
                        # since 2022.1.1 get_prim_at_path returns a prim instead of a path
                        sub_prim = get_prim_at_path(sub_prim_path.GetPath().pathString)
                    set_physics_properties(self.stage, sub_prim, properties)
                    add_update_semantics(sub_prim, keyword)
        
        if param.object_timeline_management is not None:
            self.checker = param.object_timeline_management
            self.checker.pre_initialize(cup_water_init_holder, cup_water_final_holder, particle_instance_str, self.iso_surface)
    
    def remove_objects(self):
        for prim in self.objects_list:
            delete_prim(prim.GetPath().pathString)
        if is_prim_path_valid('/World_0/Fluid'):
            delete_prim('/World_0/Fluid')
        if is_prim_path_valid('/World_0/Particles'):
            delete_prim("/World_0/Particles")
        if is_prim_path_valid('/Looks/Water'):
            delete_prim("/Looks/Water")
        if is_prim_path_valid('/World'):
            delete_prim('/World')
        if is_prim_path_valid('/Looks'):
            delete_prim('/Looks')
        if is_prim_path_valid('/lula'):
            delete_prim('/lula')
        self.objects_list = []
        self._wait_for_loading()

    def step(self, act_pos, act_rot, render, use_gt):
        """
        `act_pos`: np.ndarray (3,)
        `act_rot`: np.ndarray (4,) (wxyz)
        `render`: bool
        `use_gt`: bool
        `step` is called twice, first for grasping object and second for manipulating object
        """
        simulation_context = SimulationContext.instance()
        position_rotation_interp_list = None
        current_target = None

        if self.current_stage == 0:
            self.end_stage = 2
            if use_gt:
                self.trans_pick, self.rotat_pick = self.gt_actions[1]
            else:
                self.trans_pick = act_pos
                self.rotat_pick = act_rot
        else:
            self.end_stage = self.num_stages
            if use_gt:
                self.trans_target, self.rotat_target = self.gt_actions[2]
            else:
                self.trans_target = act_pos
                self.rotat_target = act_rot

            # interpolation for manipulation
            up_rot_quat = euler_angles_to_quat(np.array([np.pi, 0, 0]))
            _, down_rot_mat = get_pose_relat(
                trans=None, rot=quat_to_rot_matrix(self.rotat_target),
                robot_pos=self.robot_base[0],
                robot_rot=quat_to_rot_matrix(self.robot_base[1])
            )
            down_rot_quat = matrix_to_quat(down_rot_mat)
            quat_diff = quat_diff_rad(up_rot_quat, down_rot_quat)
            num_interpolation = int(200 * quat_diff / (0.7*np.pi))
            alphas = np.linspace(start=0, stop=1, num=num_interpolation)[1:]
            position_rotation_interp_list = action_interpolation(
                self.trans_pick, self.rotat_pick, self.trans_target, self.rotat_target, alphas, self.task
            )
            position_rotation_interp_iter = iter(position_rotation_interp_list)

        while self.current_stage < self.end_stage:
            if self.time_step % 120 == 0:
                logger.info(f"tick: {self.time_step}")
            
            if self.time_step >= self.horizon:
                self.is_success = -1
                break

            if current_target is None:
                grip_open = self.grip_open[self.current_stage]

                if self.current_stage == 0:
                    if use_gt:
                        trans_pre, rotation_pre = self.gt_actions[0]
                    else:
                        trans_pre, rotation_pre = get_pre_grasp_action(
                            grasp_action=(self.trans_pick, self.rotat_pick),
                            robot_base=self.robot_base, task=self.task
                        )
                    current_target = (trans_pre, rotation_pre, grip_open)

                elif self.current_stage == 1:
                    current_target = (self.trans_pick, self.rotat_pick, grip_open)
                
                elif self.current_stage == 2:
                    current_target = (
                        np.array([self.trans_pick[0], self.trans_target[1], self.trans_pick[2]]),
                        self.rotat_pick,
                        grip_open
                    )
                
                elif self.current_stage == 3:
                    current_target = (self.trans_target, self.rotat_pick, grip_open)
                
                elif self.current_stage == 4:
                    try:
                        trans_interp, rotation_interp = next(position_rotation_interp_iter)
                        current_target = (trans_interp, rotation_interp, grip_open)
                    except:
                        # finish interpolation
                        position_rotation_interp_iter_back = iter(
                            position_rotation_interp_list[::-50] + position_rotation_interp_list[0:1]
                        )
                        self.current_stage += 1
                        continue

                elif self.current_stage == 5:
                    # cup return to upward orientation
                    try:
                        trans_interp, rotation_interp = next(position_rotation_interp_iter_back)
                        current_target = (trans_interp, rotation_interp, grip_open)
                    except:
                        # finish interpolation
                        position_rotation_interp_list = None
                        self.current_stage += 1
                        continue

            if ( position_reached(self.c_controller, current_target[0], self.robot, thres=(0.1 if self.current_stage == 1 else 0.5)) or (self.current_stage in [4,5]) ) \
            and rotation_reached(self.c_controller, current_target[1]):
                gripper_state = self.gripper_controller.get_joint_positions()
                current_gripper_open = (gripper_state[0] + gripper_state[1] > 7)

                if current_target[2] != current_gripper_open:
                    if current_target[2] < 0.5:
                        target_joint_positions_gripper = self.gripper_controller.forward(action="close")
                        for _ in range(self.cfg.gripper_trigger_period):
                            articulation_controller = self.robot.get_articulation_controller()
                            articulation_controller.apply_action(target_joint_positions_gripper)
                            simulation_context.step(render=render)
                        
                    else:
                        target_joint_positions_gripper = self.gripper_controller.forward(action="open")
                        for _ in range(self.cfg.gripper_trigger_period):
                            articulation_controller = self.robot.get_articulation_controller()
                            articulation_controller.apply_action(target_joint_positions_gripper)
                            simulation_context.step(render=render)

                current_target = None
                if self.current_stage < 4:
                    self.current_stage += 1
                    logger.info(f"enter stage {self.current_stage}")
            
            else:
                target_joint_positions = self.c_controller.forward(
                    target_end_effector_position=current_target[0], target_end_effector_orientation=current_target[1]
                )
                if self.current_stage >= 2:
                    # close force
                    target_joint_positions.joint_positions[-2:] = -1
                
                articulation_controller = self.robot.get_articulation_controller()
                articulation_controller.apply_action(target_joint_positions)

            simulation_context.step(render=render)
            self.time_step += 1

        if self.current_stage == self.num_stages:
            # stages exhausted, success check
            for _ in range(self.cfg.success_check_period):
                simulation_context.step(render=False)
                if self.checker.success:
                    self.is_success = 1
                    break
        
        return self.render(), self.is_success
