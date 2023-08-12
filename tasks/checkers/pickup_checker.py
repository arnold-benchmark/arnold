import omni
from omni.isaac.core.prims import XFormPrim
from .base_checker import BaseChecker
from environment.parameters import CheckerParameters


class PickupChecker(BaseChecker):
    def __init__(self, checker_parameters: CheckerParameters, tolerance = 5) -> None:
        self.checker_parameters = checker_parameters
        self.tolerance = tolerance

    def pre_initialize(self, target_prim_path):
        super().__init__()

        self.target_prim_path =  target_prim_path
        self.target_delta_y = self.checker_parameters.target_state
        self.targetRigid = XFormPrim(self.target_prim_path)
        self.previous_pos = None
        self.vel = None

        self.check_freq = 1

        self.target_prim = self.stage.GetPrimAtPath(self.target_prim_path)
        if not self.target_prim:
            raise Exception(f"Target prim must exist at path {self.target_prim_path}")
    
    def initialization_step(self):
        # get transform
        mat = omni.usd.utils.get_world_transform_matrix(self.target_prim) 
        self.target_prim_init_y = mat.ExtractTranslation()[1] # extract y axis
        self.is_init = True
        self.create_task_callback()
        
    def get_height(self):
        mat = omni.usd.utils.get_world_transform_matrix(self.target_prim) 
        target_prim_current_y = mat.ExtractTranslation()[1]
        return target_prim_current_y

    def get_diff(self):
        mat = omni.usd.utils.get_world_transform_matrix(self.target_prim) 
        target_prim_current_y = mat.ExtractTranslation()[1]
        need_delta_y = target_prim_current_y - (self.target_delta_y + self.target_prim_init_y)

        return need_delta_y
    
    def start_checking(self):
        if not self.is_init:
            return 
        
        self.total_step += 1
        if self.total_step % self.check_freq == 0:
            mat = omni.usd.utils.get_world_transform_matrix(self.target_prim) 
            target_prim_current_y = mat.ExtractTranslation()[1]
            
            pos, rot = self.targetRigid.get_world_pose()
            pos = pos[1]
            
            if self.previous_pos is not None:
                self.vel  = abs(pos - self.previous_pos)
            
            target_height = (self.target_delta_y + self.target_prim_init_y)
            need_delta_y = abs(target_prim_current_y - target_height)
            if self.total_step % self.print_every == 0:
                print("target height %s current height %s" %(target_height, target_prim_current_y))

            # success condition
            if  need_delta_y < self.tolerance and self.vel is not None and self.vel < 0.1 :
                self.success_steps += self.check_freq
                self._on_success_hold()
            else:
                # self.success = False
                self._on_not_success()
            self.previous_pos = pos
            super().start_checking()
