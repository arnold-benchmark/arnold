from environment.runner_utils import get_simulation, pop, load_task_infos_all_missions
import hydra
import os
import glob
from environment.parameters import StageProperties
from environment.sceneloader import SceneLoader
from record_utilities import record_extra
from copy import deepcopy
# config_files = str(os.getcwd()) + '/configs/*.json'
config_files = '/home/nikepupu/Desktop/translate/**/*.json'
files = sorted(list(glob.glob(  config_files, recursive=True)))

task_infos = load_task_infos_all_missions(files, ['open_drawer'])

light_usd_path = str(os.path.join("sample", "light", "skylight.usd"))
scene_properties = StageProperties(light_usd_path, "y", 0.01, gravity_direction=[0,-1,0], gravity_magnitude=981)

import omni
import numpy as np
from utils.converter import compute_num
from omni.isaac.franka.controllers import RMPFlowController

from language.language_generation2 import *
from state_machine.utils_state_machine import  compute_feature_points
from state_machine.open_drawer_planner import OpenDrawerController
from omni.isaac.core.prims import XFormPrim
import queue
to_write = queue.Queue(4)
import time

stats = {
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[],
    8:[],
    9:[],
}
stats_object = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: []
}

robot_height_adjustment = {
    0:0,
    1:0,
    2:0,
    3:10,
    4:0,
    5:0,
    6:0,
    7:0,
    8:0,
    9:0
}

object_scale_adjustment = {
    0:1.0,
    1:0.7,
    2:0.7,
    3:0.7,
    4:0.7,
    5:0.7,
    6:0.7,
    7:0.7,
    8:0.7,
    9:0.7
}

missions_count = []

scene_loader : SceneLoader = None
cnt = 0
correct = 0
configs_set = {}
init_fail = 0
random.seed(0)
skipped = 0
# cnt = 0
adj_table_drawer = {
    0: 'top left',
    1: 'top left',
    2: 'top',
    3: 'top',
    4: 'top left',
    5: 'top',
    6: 'middle',
    7: 'top',
    8: 'top',
    9: 'top'
}


def extract_init_info(scene_parameters, robot_parameters, objects_parameters, config, robot_shift, object_angle=None):
    initialization_info = {
        'scene_parameters': {
            'usd_path': scene_parameters[0].usd_path,
            'task_type': scene_parameters[0].task_type,
            'traj_dir': scene_parameters[0].traj_dir,
            'floor_path': scene_parameters[0].floor_path,
            'wall_path': scene_parameters[0].wall_path,
            'furniture_path': scene_parameters[0].furniture_path,
            'floor_material_url': scene_parameters[0].floor_material_url,
            'wall_material_url': scene_parameters[0].wall_material_url
        },
        'robot_parameters': {
            'usd_path': robot_parameters[0].usd_path,
            'robot_position': robot_parameters[0].robot_position,
            'robot_orientation_quat': robot_parameters[0].robot_orientation_quat
        },
        'objects_parameters': [],
        'config': deepcopy(config),
        'robot_shift': robot_shift
    }

    # objects parameters
    for i in range(len(objects_parameters[0])):
        initialization_info['objects_parameters'].append({
            'usd_path': objects_parameters[0][i].usd_path,
            'scale': objects_parameters[0][i].scale,
            'object_position': objects_parameters[0][i].object_position,
            'orientation_quat': objects_parameters[0][i].orientation_quat,
            'object_type': objects_parameters[0][i].object_type,
            'args': objects_parameters[0][i].args
        })

        if objects_parameters[0][i].object_physics_properties is not None:
            initialization_info['objects_parameters'][i].update({
                'object_physics_properties': objects_parameters[0][i].object_physics_properties.properties
            })
        else:
            initialization_info['objects_parameters'][i].update({
                'object_physics_properties': None
            })
        
        if objects_parameters[0][i].part_physics_properties is not None:
            initialization_info['objects_parameters'][i]['part_physics_properties'] = {}
            for k, v in objects_parameters[0][i].part_physics_properties.items():
                initialization_info['objects_parameters'][i]['part_physics_properties'].update({k: deepcopy(v.properties)})
                if 'predicate' in initialization_info['objects_parameters'][i]['part_physics_properties'][k]:
                    del initialization_info['objects_parameters'][i]['part_physics_properties'][k]['predicate']
        else:
            initialization_info['objects_parameters'][i].update({
                'part_physics_properties': None
            })
        
        if objects_parameters[0][i].fluid_properties is not None:
            initialization_info['objects_parameters'][i].update({
                'fluid_properties': objects_parameters[0][i].fluid_properties.properties
            })
        else:
            initialization_info['objects_parameters'][i].update({
                'fluid_properties': None
            })

        if objects_parameters[0][i].object_timeline_management is not None:
            initialization_info['objects_parameters'][i].update({
                'object_timeline_management': {
                    'init_state': objects_parameters[0][i].object_timeline_management.checker_parameters.init_state,
                    'target_state': objects_parameters[0][i].object_timeline_management.checker_parameters.target_state,
                    'language_description': objects_parameters[0][i].object_timeline_management.checker_parameters.langauge_description,
                    'target_joint': objects_parameters[0][i].object_timeline_management.checker_parameters.target_joint
                } 
            })
        else:
            initialization_info['objects_parameters'][i].update({
                'object_timeline_management': None
            })
    
    # object angle
    if object_angle is not None:
        initialization_info['object_angle'] = object_angle
    
    return initialization_info


while True:
    # args = (config['task_type'], config['task_id'], config['robot_id'], config['mission_id'], config['annotator'])
    scene_parameters, robot_parameters, objects_parameters = pop(task_infos, 1)
    
    if len(scene_parameters) == 0:
        break
    
    config = objects_parameters[0][0].args
    mission_id = int(config['mission_id'])
    task_id = int(config['task_id'])
    robot_id = int(config['robot_id'])
    house_id = int(config['house_id'])
    annotator = config['annotator']
    task_type = config['task_type']
    # if task_id <= 5:
    #     continue

    state = config['state']
    
    missions_count.append((task_id, ) + state)
    objects_parameters[0][0].scale *= object_scale_adjustment[task_id]
    

    identifier = '-'.join([ annotator, task_type, str(task_id), str(house_id), str(state)])
    if identifier in configs_set:
        continue
    
    success = False
    robot_shifts = []
    for i in [-10, 0,10]:
        for j in [-10, 0, 10]:
            # for k in [-10, 0, 10]:
                robot_shift = [i, robot_height_adjustment[task_id], j]
                robot_shifts.append(robot_shift)
    # robot_shifts = [[-10, 40,0], [0, 40,0], [10, 40, 0]]
    franka_pos_param = robot_parameters[0].robot_position
    drawer_pos_param = objects_parameters[0][0].object_position

    
    subcorrect = 0
    for robot_shift in robot_shifts:
        if subcorrect >= 4:
            break
        shifted_franka_pos = np.array(franka_pos_param + np.array(robot_shift))
        dist = np.abs(shifted_franka_pos[[0,2]]- np.array(drawer_pos_param)[[0,2]])
        
        # if np.any(dist < 55):
        #         continue
        
        for grasp_pose_index in [1]:

            current_stage = -1
            gts = []

            cnt += 1
            print('initialize')
            if scene_loader is None:
                scene_loader = SceneLoader(simulation_app,scene_parameters, robot_parameters, objects_parameters, scene_properties, set_sensors=True)
                # vpif = omni.kit.viewport_legacy.get_viewport_interface()
                # vpw = vpif.get_viewport_window()
                # vpw.set_active_camera("/World_0/franka/FrontCamera")    
            else:
            
                scene_loader.reinitialize(scene_parameters, robot_parameters, objects_parameters, new_stage=False)
            franka = scene_loader.robots[0]
            franka_pose_init = franka.get_world_pose()
            
            franka.set_world_pose(franka_pose_init[0] + np.array(robot_shift), franka_pose_init[1])
            
           
            storage_path = scene_loader.objects[0][0].GetPath().pathString
            storage = XFormPrim(storage_path)
            storage_pose_init = storage.get_world_pose()
            
            
            checker = scene_loader.time_line_callbacks[0][0] 
            # if cnt == 1:
            #     while True:
            #         simulation_context.render()
            print("try start")
            scene_loader.start(simulation_context)
            
            print("physics settled")
            
            franka_translation, franka_rotation = franka.get_world_pose()
            print('compute percentage')
            init_value = checker.init_value
            percentage = checker.joint_checker.compute_percentage()
            if abs(init_value * 100 - percentage) > 1 or np.isnan(percentage):
                    init_fail += 1
                    print("init failed next")
                    continue
                
            stage = omni.usd.get_context().get_stage()
            pre_grasp_position, grasp_position_final, pull_position, pose = compute_feature_points(stage, storage_path, checker, task_id)


            _controller = RMPFlowController(name="cspace_controller",
                    robot_articulation = franka, physics_dt = 1.0 / 120.0)
            gripper_controller = franka.gripper

            articulation_controller = franka.get_articulation_controller()
            opendrawer_controller = OpenDrawerController(
                "open_drawer_controller",
                _controller, 
                gripper_controller)
            seed = compute_num(config['task_type'], config['task_id'], 
                        config['robot_id'], config['mission_id'], config['house_id'], 0, config['annotator'])

            language_annotation = eval(config['task_type'])(state[1], seed, adj_table_drawer[int(task_id)])[0]

            diffs = []
            print('start running ', task_id)
            for step in range(2000):
                if opendrawer_controller._event == 0:
                    target_position = pre_grasp_position
                    
                    endeffector_orientation = pose[0][grasp_pose_index].as_quat()
                    endeffector_orientation[[0,1,2,3]] = endeffector_orientation[[3,0,1,2]]
                elif opendrawer_controller._event == 1:
                    target_position = grasp_position_final
                    endeffector_orientation = pose[0][grasp_pose_index].as_quat()
                    endeffector_orientation[[0,1,2,3]] = endeffector_orientation[[3,0,1,2]]

                elif opendrawer_controller._event  in [2,3]:
                    target_position = None
                    endeffector_orientation = None
                elif opendrawer_controller._event == 4:
                    target_position = deepcopy(pull_position)
                    endeffector_orientation = pose[0][grasp_pose_index].as_quat()
                    endeffector_orientation[[0,1,2,3]] = endeffector_orientation[[3,0,1,2]]
                    
                else:
                    target_position = None
                    endeffector_orientation = None
                
                diff = abs(checker.get_diff()) * 100
                # print('diff: ', diff)
                if diff > 2:
                    target_joint_positions = opendrawer_controller.forward(None, None, target_position, 
                                        franka.get_joints_state().positions, end_effector_orientation = endeffector_orientation )
                    
                    
                    articulation_controller.apply_action(target_joint_positions)

                planner_stage = opendrawer_controller._event
                if not planner_stage == current_stage and opendrawer_controller._t > 0:
                    
                    if planner_stage in [0, 1, 2, 3]:
                        percentage = checker.joint_checker.compute_percentage()
                        if abs(init_value * 100 - percentage) > 2 or np.isnan(percentage):
                            init_fail += 1
                            print("pre grasp and grasp failed next  init value: ", init_value , " percentage: ",  percentage)
                            break

                    simulation_context.step(render=True)
                    simulation_context.step(render=True)
                    diffs.append(checker.get_diff())

                    keyframe = False
                    if  planner_stage in [1, 4]:
                        keyframe = True

                    gt = None
                    gt = scene_loader.render(simulation_context)
                    gt['keyframe'] = keyframe
                    gripper_state = franka.gripper.get_joint_positions()
                    gripper_open = gripper_state[0] > 3.9 and gripper_state[1] > 3.9
                    if planner_stage == 0 and not gripper_open:
                        print('something wrong')
                        break
                        
                    
                    language_annotation_with_low = language_annotation #+ f' step {len(gts)}' 
                    
                    if planner_stage in [0, 1, 4]:
                        gt = record_extra(gt, franka, language_annotation_with_low, 
                            opendrawer_controller.applied_action, gripper_open, config['state'], \
                            opendrawer_controller.get_current_positon_world(), checker.get_diff()  )
                            

                        gts.append(deepcopy(gt))
                    current_stage = planner_stage 
                    
                    
                else:
                    simulation_context.step(render=False)

                if checker.success:
                    diffs.append(checker.get_diff())
                    monotone = True
                    for ii in range(len(diffs)-1):
                        if diffs[ii] > diffs[ii+1] and abs(diffs[ii] - diffs[ii+1]) > 0.01:
                            monotone = False
                            break
                    
                    if not monotone:

                        print('not monotone skip', diffs)
                        break

                    if len(gts) != 3:
                        print('wrong not 3 frames', len(gts))
                        break

                    if planner_stage < 4:
                        print('did not reach stage 4')
                        break


                    success = True
                    print("success") 
                   
                    subcorrect += 1
                   
                    save_file = '-'.join([ annotator, task_type, str(task_id), str(robot_id), str(state[0]), str(state[1]),
                            str(house_id), str(time.asctime())])

                    if identifier in configs_set:
                        configs_set[identifier] += 1
                    else:
                        configs_set[identifier] = 1
                    correct += 1

                    initialization_info['robot_shift'] = robot_shift

                    gripper_state = franka.gripper.get_joint_positions()
                    gripper_open = gripper_state[0] > 3.9 and gripper_state[1] > 3.9
                    gt = scene_loader.render(simulation_context)
                    language_annotation_with_low = language_annotation #+ f' step {len(gts)}' 

                    gt = record_extra(gt, franka, language_annotation_with_low, \
                        opendrawer_controller.applied_action, gripper_open, config['state'], \
                        opendrawer_controller.get_current_positon_world(),checker.get_diff())

                    gts.append(deepcopy(gt))
                    
                    from os.path import exists
                    split = 'not_used'
                    if task_id < 8 and house_id <= 15 and abs(float(state[1]) - 0.75) > 1e-6:
                        split = 'train'
                        
                    elif task_id >= 8 and house_id <= 15 and  abs(float(state[1]) - 0.75) > 1e-6:
                        split = 'novel_object'
                    elif task_id < 8 and house_id > 15 and abs(float(state[1]) - 0.75) > 1e-6:
                        split = 'novel_scene'
                    elif task_id < 8 and house_id <= 15 and abs(float(state[1]) - 0.75) <= 1e-6:
                        split = 'novel_state'
                    
                    if split is not None:
                        folder_path = os.path.join('/mnt/data', 'data', task_type, split)
                        folder_exists = exists(folder_path)
                        if not folder_exists:
                            os.makedirs(folder_path)

                        save_path = os.path.join(folder_path, save_file)
                        np.savez(save_path, gt=gts, info=initialization_info)
                    break
            stats[int(mission_id)].append(success)
            stats_object[int(task_id)].append(success)
            print("correct: ", correct, "init_fail: ", init_fail, "total: ", cnt)


@hydra.main(config_path='./configs', config_name='default')
def main(cfg):
    simulation_app, simulation_context, _ = get_simulation(headless=True, gpu_id=0)

    simulation_app.close()


if __name__ == '__main__':
    main()
