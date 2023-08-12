from .base_task import BaseTask
from .pour_water import PourWater
from .transfer_water import TransferWater
from .close_cabinet import CloseCabinet
from .open_cabinet import OpenCabinet
from .close_drawer import CloseDrawer
from .open_drawer import OpenDrawer
from .pickup_object import PickupObject
from .reorient_object import ReorientObject
from environment.parameters import *
from omni.isaac.core.utils.prims import get_prim_at_path


def handle_part_predicate(prim_path: str):
    prim = get_prim_at_path(prim_path)
    if "handle" in prim.GetPath().pathString and prim.GetTypeName() == "Mesh":
        return True
                            
    return False


def joint_part_predicate(prim_path:str):
    prim = get_prim_at_path(prim_path)
    if "joint" in prim.GetPath().pathString and \
        (prim.GetTypeName() == "PhysicsPrismaticJoint" or prim.GetTypeName() == "PhysicsRevoluteJoint"):
        return True
                            
    return False


def cup_shape_predicate(prim_path: str):
    prim = get_prim_at_path(prim_path)
    if "cupShape" in prim.GetPath().pathString and prim.GetTypeName() == "Mesh":
        return True
                            
    return False


def load_task(asset_root, npz, cfg):
    import os
    from tasks.checkers import BaseChecker, PickupChecker, OrientChecker, JointChecker, WaterChecker
    info = npz['info'].item()

    scene_parameters = [SceneParameters(**info['scene_parameters'])]
    scene_parameters[0].usd_path = os.path.abspath(scene_parameters[0].usd_path).split(os.path.sep)
    path_idx = scene_parameters[0].usd_path.index('VRKitchen2.0')
    path_idx += 1
    scene_parameters[0].usd_path = os.path.join(asset_root, os.path.sep.join(scene_parameters[0].usd_path[path_idx:]))

    floor_material_url = scene_parameters[0].floor_material_url
    if 'omniverse' in floor_material_url:
        floor_material_url = floor_material_url.split(os.path.sep)
        
        path_idx = floor_material_url.index('Base')
        path_idx += 1

        scene_parameters[0].floor_material_url = os.path.join(
            asset_root, 'materials', os.path.sep.join(floor_material_url[path_idx:])
        )
    
    elif 'wasabi' in floor_material_url:
        floor_material_url = floor_material_url.split(os.path.sep)
        path_idx = floor_material_url.index('materials')
        scene_parameters[0].floor_material_url = os.path.join(
            asset_root, os.path.sep.join(floor_material_url[path_idx:])
        )
        
    else:
        floor_material_url = floor_material_url.split(os.path.sep)
        path_idx = floor_material_url.index('VRKitchen2.0')
        path_idx += 1

        scene_parameters[0].floor_material_url = os.path.join(
            asset_root, os.path.sep.join(floor_material_url[path_idx:])
        )

    wall_material_url = scene_parameters[0].wall_material_url
    if 'omniverse' in wall_material_url:
        wall_material_url = wall_material_url.split(os.path.sep)
        path_idx = wall_material_url.index('Base')
        path_idx += 1

        scene_parameters[0].wall_material_url = os.path.join(
            asset_root, 'materials', os.path.sep.join(wall_material_url[path_idx:])
        )
    
    elif 'wasabi' in wall_material_url:
        wall_material_url = wall_material_url.split(os.path.sep)
        path_idx = wall_material_url.index('materials')
        scene_parameters[0].wall_material_url = os.path.join(
            asset_root, os.path.sep.join(wall_material_url[path_idx:])
        )
    
    else:
        wall_material_url = wall_material_url.split(os.path.sep)
        path_idx = wall_material_url.index('VRKitchen2.0')
        path_idx += 1

        scene_parameters[0].wall_material_url = os.path.join(
            asset_root, os.path.sep.join(wall_material_url[path_idx:])
        )

    robot_parameters = [RobotParameters(**info['robot_parameters'])]
    robot_parameters[0].usd_path = os.path.abspath(robot_parameters[0].usd_path).split(os.path.sep)
    path_idx = robot_parameters[0].usd_path.index('VRKitchen2.0')
    path_idx += 1
    robot_parameters[0].usd_path = os.path.join(
        asset_root, os.path.sep.join(robot_parameters[0].usd_path[path_idx:])
    )

    objects_parameters = [[]]
    for i in range(len(info['objects_parameters'])):
        object_parameters = {
            'usd_path': info['objects_parameters'][i]['usd_path'],
            'scale': info['objects_parameters'][i]['scale'],
            'object_position': info['objects_parameters'][i]['object_position'],
            'orientation_quat': info['objects_parameters'][i]['orientation_quat'],
            'object_type': info['objects_parameters'][i]['object_type'],
            'args': info['objects_parameters'][i]['args']
        }

        if info['objects_parameters'][i]['object_physics_properties'] is not None:
            object_parameters.update({
                'object_physics_properties': ObjectPhysicsProperties(**info['objects_parameters'][i]['object_physics_properties'])
            })
        else:
            object_parameters.update({
                'object_physics_properties': None
            })
        
        if info['objects_parameters'][i]['part_physics_properties'] is not None:
            object_parameters['part_physics_properties'] = {}
            for k, v in info['objects_parameters'][i]['part_physics_properties'].items():
                object_parameters['part_physics_properties'][k] = ObjectPhysicsProperties(**v)
                if k == 'handle':
                    object_parameters['part_physics_properties'][k].properties[PREDICATE] = handle_part_predicate
                elif k == 'joint':
                    object_parameters['part_physics_properties'][k].properties[PREDICATE] = joint_part_predicate
                elif k == 'cup_shape':
                    object_parameters['part_physics_properties'][k].properties[PREDICATE] = cup_shape_predicate
        else:
            object_parameters.update({
                'part_physics_properties': None
            })
        
        if info['objects_parameters'][i]['fluid_properties'] is not None:
            object_parameters.update({
                'fluid_properties': FluidPhysicsProperties(**info['objects_parameters'][i]['fluid_properties'])
            })
        else:
            object_parameters.update({
                'fluid_properties': None
            })
        
        if info['objects_parameters'][i]['object_timeline_management'] is not None:
            checker_parameters = CheckerParameters(**info['objects_parameters'][i]['object_timeline_management'])
            if 'pickup' in object_parameters['args']['task_type']:
                object_parameters.update({
                    'object_timeline_management': PickupChecker(checker_parameters=checker_parameters)
                })
            elif 'reorient' in object_parameters['args']['task_type']:
                object_parameters.update({
                    'object_timeline_management': OrientChecker(checker_parameters=checker_parameters)
                })
            elif 'water' in object_parameters['args']['task_type']:
                object_parameters.update({
                    'object_timeline_management': WaterChecker(checker_parameters=checker_parameters)
                })
            else:
                object_parameters.update({
                    'object_timeline_management': JointChecker(checker_parameters=checker_parameters)
                })
        else:
            object_parameters.update({
                'object_timeline_management': None
            })

        objects_parameters[0].append(ObjectParameters(**object_parameters))
        objects_parameters[0][i].usd_path = os.path.abspath(objects_parameters[0][i].usd_path).split(os.path.sep)
        path_idx = objects_parameters[0][i].usd_path.index('VRKitchen2.0')
        path_idx += 1
        objects_parameters[0][i].usd_path = os.path.join(
            asset_root, os.path.sep.join(objects_parameters[0][i].usd_path[path_idx:])
        )

    robot_shift = info['robot_shift']
    light_usd_path = os.path.join(cfg.asset_root, 'sample/light/skylight.usd')
    stage_properties = StageProperties(light_usd_path, "y", 0.01, gravity_direction=[0,-1,0], gravity_magnitude=981)

    task_name = object_parameters['args']['task_type']

    if object_parameters['args']['task_type'] == 'pickup_object':
        env = PickupObject(cfg.num_stages[task_name], cfg.horizon, stage_properties=stage_properties, cfg=cfg)
    
    elif  object_parameters['args']['task_type'] == 'reorient_object':
        env = ReorientObject(cfg.num_stages[task_name], cfg.horizon, stage_properties=stage_properties, cfg=cfg)

    elif object_parameters['args']['task_type'] == 'open_drawer':
        env = OpenDrawer(cfg.num_stages[task_name], cfg.horizon, stage_properties=stage_properties, cfg=cfg)

    elif  object_parameters['args']['task_type'] == 'close_drawer':
        env = CloseDrawer(cfg.num_stages[task_name], cfg.horizon, stage_properties=stage_properties, cfg=cfg)

    elif  object_parameters['args']['task_type'] == 'open_cabinet':
        env = OpenCabinet(cfg.num_stages[task_name], cfg.horizon, stage_properties=stage_properties, cfg=cfg)
        
    elif object_parameters['args']['task_type'] == 'close_cabinet':
        env = CloseCabinet(cfg.num_stages[task_name], cfg.horizon, stage_properties=stage_properties, cfg=cfg)
    
    elif object_parameters['args']['task_type'] == 'pour_water':
        env = PourWater(cfg.num_stages[task_name], cfg.horizon, stage_properties=stage_properties, cfg=cfg)
    elif object_parameters['args']['task_type'] == 'transfer_water':
        env = TransferWater(cfg.num_stages[task_name], cfg.horizon, stage_properties=stage_properties, cfg=cfg)
    
    else:
        raise Exception(f"task not implemented: {object_parameters['args']['task_type']}")

    # if scene_loader is None:
    #     scene_loader = SceneLoader(simulation_app,scene_parameters, robot_parameters, objects_parameters, scene_properties, sensor_resolution=(128,128), use_gpu_physics=use_gpu_physics)
    # else:
    #     scene_loader.reinitialize(scene_parameters, robot_parameters, objects_parameters, new_stage=False, use_gpu_physics=use_gpu_physics)

    # franka = scene_loader.robots[0]
    # franka_pose_init = franka.get_world_pose()
    # franka.set_world_pose(franka_pose_init[0] + np.array(robot_shift), franka_pose_init[1])
    robot_parameters[0].robot_position += np.array(robot_shift)

    # if 'object_angle' in info:
    #     object_angle = info['object_angle']
    #     object = XFormPrim(scene_loader.objects[0][0].GetPath().pathString)
    #     object_pose_init = object.get_world_pose()
    #     object.set_world_pose(object_pose_init[0], euler_angles_to_quat(object_angle))

    # return scene_loader
    return env, objects_parameters[0], robot_parameters, scene_parameters