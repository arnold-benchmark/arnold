from .pour_water import PourWater
from typing import List
from environment.parameters import *
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import is_prim_path_valid,get_prim_at_path, get_all_matching_child_prims
from omni.isaac.core.utils.semantics import add_update_semantics

import omni
from omni.isaac.core.prims import XFormPrim
from environment.physics_utils import set_physics_properties
from environment.fluid_utils import set_particle_system_for_cup
from pxr import Gf


class TransferWater(PourWater):
    def __init__(self, num_stages, horizon, stage_properties, cfg) -> None:
        super().__init__(num_stages, horizon, stage_properties, cfg)

        self.task = 'transfer_water'
        self.grip_open = cfg.gripper_open[self.task]
    
    def reset(self, robot_parameters, 
              scene_parameters, 
              object_parameters,
              robot_base,
              gt_actions
        ):
        self.objects_parameters: List[ObjectParameters] = object_parameters
        obs = super().reset(
            robot_parameters = robot_parameters,
            scene_parameters = scene_parameters,
            object_parameters = object_parameters,
            robot_base = robot_base,
            gt_actions = gt_actions
        )

        return obs

    def load_object(self):
        # TODO:
        # For now only supports one environment, we will use cloner in the future
        index = 0
        self.objects_list = []
        for param in self.objects_parameters:
            object_prim_path = find_unique_string_name(
                initial_name = f"/World_{index}/{param.object_type}",
                is_unique_fn = lambda x: not is_prim_path_valid(x)
            )
            object_prim = add_reference_to_stage(param.usd_path, object_prim_path)
            volume_mesh_path = object_prim.GetPath().AppendPath(f"cup_volume").pathString
            if param.fluid_properties:
                cup_water_init_holder = object_prim_path
                
                particle_system_path = '/World_0/Fluid'

                particle_instance_str = "/World_0/Particles"
                
                volume_mesh_path = object_prim.GetPath().AppendPath(f"cup_volume").pathString
                set_particle_system_for_cup(
                    self.stage, Gf.Vec3f(param.object_position[0], param.object_position[1], param.object_position[2]),
                    volume_mesh_path, particle_system_path, particle_instance_str, param.fluid_properties,
                    asset_root=self.cfg.asset_root, enable_iso_surface=self.iso_surface
                )
            
            else:
                cup_water_final_holder = object_prim_path
                self.stage.GetPrimAtPath(volume_mesh_path).SetActive(False)
            
            self._wait_for_loading()

            self.objects_list.append(object_prim)
            
            position = param.object_position
            rotation = param.orientation_quat
            
            # use this to set relative position, orientation and scale
            xform_prim = XFormPrim(object_prim_path, translation= position, orientation = rotation, scale = np.array(param.scale))
            self._wait_for_loading()
            
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
            
        for param in self.objects_parameters:
            if param.object_timeline_management is not None:
                self.checker = param.object_timeline_management
                self.checker.pre_initialize(cup_water_final_holder, cup_water_init_holder, particle_instance_str, self.iso_surface)
