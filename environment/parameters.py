import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from .constants import *
OBJECTS = [BOTTLE, CABINET, DRAWER, FAUCET, CUP, BASIN]
OBJECT_PHYSICS_PROPERTIES = [HAS_PHYSICS_MATERIAL, STATIC_FRICTION, DYNAMIC_FRICTION, RESTITUTION, COLLISION, MASS, IS_RIGID_BODY, PREDICATE, DAMPING_COEFFICIENT]
LIQUID_PHYSICS_PROPERTIES = [FLUID_SPHERE_DIAMETER, PARTICLE_SYSTEM_SCHEMA_PARAMETERS, PARTICLE_MASS, PARTICLE_COLOR]

class Parameters():
    def jsonify(self):
        return json.dumps(self.__dict__)
    def __eq__(self, other) : 
        return self.__dict__ == other.__dict__

class RobotParameters(Parameters):
    def __init__(self, usd_path, robot_position, robot_orientation_quat):
        self.usd_path = str(Path(usd_path).resolve())
        self.robot_position = robot_position
        self.robot_orientation_quat = robot_orientation_quat


class SceneParameters(Parameters):
    def __init__(
        self, usd_path, traj_dir, task_type, floor_material_url, wall_material_url, 
        floor_path = "floors", wall_path = "roomStruct", furniture_path = "furniture"
    ):
        self.usd_path = str(Path(usd_path).resolve())
        self.task_type = task_type
        self.traj_dir = traj_dir
        self.floor_path = floor_path
        self.wall_path = wall_path
        self.furniture_path = furniture_path
        self.floor_material_url = floor_material_url
        self.wall_material_url = wall_material_url
        

class StageProperties(Parameters):
    def __init__(self, light_usd_path, up_axis, stage_unit, gravity_direction, gravity_magnitude):
        self.light_usd_path = str(Path(light_usd_path).resolve())
        self.scene_up_axis = up_axis
        self.scene_stage_unit = stage_unit
        self.gravity_direction = gravity_direction
        self.gravity_magnitude = gravity_magnitude


class ObjectPhysicsProperties(Parameters):
    def __init__(self, **kwargs):
        self.properties = {}
        for key, value in kwargs.items():
            assert key in OBJECT_PHYSICS_PROPERTIES, f"{key} not in supported object physics properties"
            self.properties[key] = value


class FluidPhysicsProperties(Parameters):
    def __init__(self, **kwargs) -> None:
        self.properties = {}
        for key, value in kwargs.items():
            assert key in LIQUID_PHYSICS_PROPERTIES, f"{key} not in supported liquid physics properties"
            self.properties[key] = value


class ObjectParameters(Parameters):
    def __init__(
        self, usd_path, object_type, object_position, orientation_quat, scale,
        object_physics_properties: ObjectPhysicsProperties = None, part_physics_properties: Optional[Dict[str, ObjectPhysicsProperties]] = None, 
        fluid_properties: Optional[FluidPhysicsProperties] = None, object_timeline_management = None, args = None
    ):
        self.usd_path = str(Path(usd_path).resolve())
        self.scale = np.array(scale)
        self.object_position = object_position
        self.orientation_quat = orientation_quat
        assert object_type in OBJECTS, f"{object_type} not in supported objects"
        self.object_type = object_type
        self.object_physics_properties = object_physics_properties
        self.part_physics_properties = part_physics_properties
        self.fluid_properties = fluid_properties
        self.object_timeline_management = object_timeline_management
        self.args = args


class CheckerParameters(Parameters):
    def __init__(self, init_state, target_state, target_joint=None, language_description=None):
        self.init_state = init_state
        self.target_state = target_state
        self.language_description = language_description
        self.target_joint = target_joint
