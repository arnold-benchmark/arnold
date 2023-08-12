from .base_task import BaseTask
from typing import List
from environment.parameters import *
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import is_prim_path_valid,get_prim_at_path, get_all_matching_child_prims
from omni.isaac.core.utils.semantics import add_update_semantics

import omni
from omni.isaac.core.prims import XFormPrim
from environment.physics_utils import set_physics_properties
from utils.env import position_reached, rotation_reached, get_pre_grasp_action, action_interpolation
from omni.isaac.core.simulation_context import SimulationContext

import logging
logger = logging.getLogger(__name__)


class OpenCabinet(BaseTask):
    def __init__(self, num_stages, horizon, stage_properties, cfg) -> None:
        super().__init__(num_stages, horizon, stage_properties, cfg)
        self.task = 'open_cabinet'
        self.grip_open = cfg.gripper_open[self.task]
        self.use_gpu_physics = False

    def reset(self, robot_parameters, 
              scene_parameters, 
              object_parameters,
              robot_base,
              gt_actions
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

        # TODO
        # use issac sim default collision for handle processing. This is not needed if we apply Su Hao's method first
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
            self.checker.pre_initialize(object_prim_path)

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
            num_interpolation = int(10 * np.linalg.norm(self.trans_target - self.trans_pick))
            alphas = np.linspace(start=0, stop=1, num=num_interpolation)[1:]
            joint_pos = self.checker.joint_checker.get_joint_position()
            position_rotation_interp_list = action_interpolation(
                self.trans_pick, self.rotat_pick, self.trans_target, self.rotat_target, alphas, self.task, joint_pos=joint_pos
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
                
                else:
                    try:
                        trans_interp, rotation_interp = next(position_rotation_interp_iter)
                        current_target = (trans_interp, rotation_interp, grip_open)
                    except:
                        # finish interpolation
                        position_rotation_interp_list = None
                        self.current_stage += 1
                        continue

            if position_reached( self.c_controller, current_target[0], self.robot, thres=(0.1 if self.current_stage == 1 else 0.5) ) \
            and ( rotation_reached(self.c_controller, current_target[1]) or (self.current_stage == 2) ):
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
                if self.current_stage < 2:
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
