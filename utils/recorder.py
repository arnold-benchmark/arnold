import os
import omni.ext
import omni.appwindow
import numpy as np
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction
import gzip

class DataRecorder():
    def __init__(self, robot_path, target_paths, frankabot, task_type):
        self.replayBuffer = []
        self.replayBufferObj = []
        self.record = False
        self.replay_start = False
        self.target_paths = target_paths
        self.robot_path = robot_path
        self.target_objs = []

        for t_path in self.target_paths:
            self.target_objs.append(XFormPrim(t_path))

        self.traj_dir = None
        self._frankabot = frankabot
        self.legacy = False
        self.task_type = task_type
        self.checker = None
        self.ptcl_frame_skip = 20
        self.buffer = {"robot": [], "object": [], "particles": []}

    def get_replay_status(self):
        return self.replay_start

    def start_record(self, traj_dir, checker):
        self.replay_start = False
        self.record = True
        self.traj_dir = traj_dir
        self.checker = checker

        self.buffer = {"robot": [], "object": [], "particle": []}

    def stop_record(self):
        self.record = False

    def save_buffer(self, success, abs_info=None):
        print("write:", self.traj_dir)
        if not os.path.exists(self.traj_dir):
            os.makedirs(self.traj_dir)

        # Save CSV files with gzip compression
        def save_csv_gzip(file_name, data):
            with gzip.open(os.path.join(self.traj_dir, f'{file_name}.csv.gz'), 'wt') as file:
                for line in data:
                    file.write(line)

        save_csv_gzip('record_robot', self.buffer['robot'])
        save_csv_gzip('record_object', self.buffer['object'])
        save_csv_gzip('record_particle', self.buffer['particle'])

        
        with open(os.path.join(self.traj_dir, 'success.txt'), 'w') as file:
            if success:
                file.write('success')
            else:
                file.write('fail')
        


        if abs_info is not None:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return super().default(obj)

            config_path = os.path.join(self.traj_dir, 'object_info.json')
            with open(config_path, "w") as f:
                # Minimize JSON file size by removing indentation
                json.dump(abs_info, f, cls=NpEncoder)

        self.buffer = {"robot": [], "object": [], "particle": []}

    def delete_traj_folder(self):
        import shutil
        from pathlib import Path
        path = Path(self.traj_dir)
        path = path.parent.absolute()

        try:
            shutil.rmtree(path)
            print("replay failed, delete this traj: ", str(path))
        except OSError as e:
            print("Error: %s : %s" % (path, e.strerror))
    
    def record_data(self, robot_states, actions, time_step):
        self.obj_data = [{'pos': None, 'rot': None, 'joint': None, 'path': t_path} for t_path in self.target_paths]
        self.robot_data = {
            'joint_pos': robot_states['pos'].tolist(),
            'joint_vel': robot_states['vel'].tolist(),
            # 'joint_effort': robot_states['effort'].tolist(),
            'actions': str(actions),
        }
        self.ptcl_data = {}
        
        if self.task_type in ["pickup_object", "reorient_object"]:
            self.record_obj_pose()
        elif self.task_type in ["open_drawer","open_cabinet", "close_drawer", "close_cabinet"]:
            self.record_obj_joint()
        elif self.task_type in ["pour_water", "transfer_water"]:
            # record both pose of the glass and particle positions
            self.record_obj_pose()
            self.record_ptcl(time_step)
        else:
            raise RuntimeError("data recording for %s not implemented" % self.task_type)
        if self.record:
            self.buffer['robot'].append(str(self.robot_data).replace("\n", ' ') + '\n')
            self.buffer['object'].append(str(self.obj_data).replace("\n", ' ') + '\n')
            self.buffer['particle'].append(str(self.ptcl_data).replace("\n", ' ') + '\n')

    def record_obj_pose(self):
        for idx, t_obj in enumerate(self.target_objs):
            obj_trans, obj_rot = t_obj.get_world_pose()
            self.obj_data[idx]["pos"] = obj_trans.tolist()
            self.obj_data[idx]["rot"] = obj_rot.tolist()

    def record_obj_joint(self):
        assert(self.checker is not None)
        # record target joint angles for the specific joint object
        for idx, t_path in enumerate(self.target_paths):
            if t_path == self.checker.target_prim_path:
                self.obj_data[idx]["joint"] = self.checker.joint_checker.compute_percentage()

    def record_ptcl(self, time_step):
        assert(self.checker is not None)
        # print("time step:", time_step, self.ptcl_frame_skip)
        # print("get all particle:", self.checker.liquid_checker.get_all_particles())
        if time_step % self.ptcl_frame_skip == 0:
            self.ptcl_data = self.checker.liquid_checker.get_all_particles()

    def start_replay(self, traj_dir, checker):
        self.traj_dir = traj_dir
        self.checker = checker

        # Check for legacy record file
        if os.path.exists(os.path.join(self.traj_dir, 'record.csv')):
            with open(os.path.join(self.traj_dir, 'record.csv'), 'r') as file1:
                Lines = file1.readlines()

            try:
                self.replayBuffer = [eval(line) for line in Lines]
            except:
                self.replayBuffer = []
            self.legacy = True
        else:
            self.legacy = False
            self.replayBuffer = []
            self.replayBufferObj = []
            self.replayBufferPtcl = []

            # Function to read gzip compressed CSV files
            def read_gzip_csv(file_name):
                try:
                    with gzip.open(os.path.join(self.traj_dir, f'{file_name}.csv.gz'), 'rt') as file:
                        return [eval(line) for line in file]
                except FileNotFoundError:
                    return []
            

            # Load the data from the new gzip compressed files
            self.replayBuffer = read_gzip_csv('record_robot')
            self.replayBufferObj = read_gzip_csv('record_object')
            self.replayBufferPtcl = read_gzip_csv('record_particle')

            assert(len(self.replayBuffer) == len(self.replayBufferObj))

        self.replay_start = True

    def replay_data(self):
        taskDone = False
        # print("len(self.replayBuffer) ", len(self.replayBuffer) )
        # print("len(self.replayBufferObj)", len(self.replayBufferObj))
        if self.legacy == True:
            robot_data = self.replayBuffer.pop(0)
            action = robot_data
            if action is not None:
                actions = ArticulationAction(joint_positions=action["joint_positions"])
                # print("actions: ", actions)
                _articulation_controller = self._frankabot.get_articulation_controller()
                _articulation_controller.apply_action(actions)
            if len(self.replayBuffer) == 0:
                # print("Task Done")
                taskDone = True
        else:
            if len(self.replayBuffer) > 0 and len(self.replayBufferObj) > 0:
                robot_data = self.replayBuffer.pop(0)
                obj_data = self.replayBufferObj.pop(0)
                
                if self.replayBufferPtcl is not None and len(self.replayBufferPtcl) > 0:
                    ptcl_data = self.replayBufferPtcl.pop(0)
                
                if len(self.replayBuffer) % 1000 == 0:
                    print("len buffer: ", len(self.replayBuffer))

                if "joint_pos" in robot_data and robot_data["joint_pos"] is not None:
                    self._frankabot.set_joint_positions(robot_data["joint_pos"])
                    self._frankabot.set_joint_velocities(robot_data["joint_vel"])
                    # self._frankabot.set_joint_efforts(robot_data["joint_effort"])

                if 'actions' in robot_data:
                    action = eval(robot_data["actions"])
                    if action is not None:
                        actions = ArticulationAction(joint_positions=action["joint_positions"])
                        _articulation_controller = self._frankabot.get_articulation_controller()
                        _articulation_controller.apply_action(actions)
                
                if self.task_type in ["pickup_object", "reorient_object"]:
                    self.replay_obj_pose(obj_data)
                elif self.task_type in ["open_drawer","open_cabinet", "close_drawer", "close_cabinet"]:
                    self.replay_obj_joint(obj_data)
                elif self.task_type in ["pour_water", "transfer_water"]:
                    self.replay_obj_pose(obj_data)
                    self.replay_ptcl(ptcl_data)
                else:
                    raise RuntimeError("data replay for %s not implemented" % self.task_type)
                    
            if len(self.replayBuffer) == 0 or len(self.replayBufferObj) == 0:
                # print("Task Done")
                taskDone = True

        return taskDone

    def replay_obj_pose(self, all_obj_data):
        for idx, obj_data in enumerate(all_obj_data):
            if "pos" in obj_data and "rot" in obj_data and \
                obj_data["pos"] is not None and obj_data["rot"] is not None:
                self.target_objs[idx].set_local_pose(translation=np.array(obj_data["pos"]), 
                    orientation=np.array(obj_data["rot"]))

    def replay_obj_joint(self, all_obj_data):
        for idx, obj_data in enumerate(all_obj_data):
            if self.target_paths[idx] == self.checker.target_prim_path and \
                "joint" in obj_data and obj_data["joint"] is not None:
                self.checker.joint_checker.set_joint(np.array(obj_data["joint"]))

    def replay_ptcl(self, ptcl_data):
        if ptcl_data:
            self.checker.liquid_checker.set_all_particles(ptcl_data)
