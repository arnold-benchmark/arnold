import math
import omni
from omni.isaac.core.prims import XFormPrim
from .base_checker import BaseChecker
from environment.parameters import CheckerParameters


class OrientChecker(BaseChecker):
    def __init__(self, checker_parameters: CheckerParameters, tolerance = 20) -> None:
        self.checker_parameters = checker_parameters
        self.tolerance = tolerance
    
    def pre_initialize(self, target_prim_path):
        super().__init__()
       
        self.target_prim_path =  target_prim_path
        self.target_delta_y = self.checker_parameters.target_state

        self.targetRigid = XFormPrim(self.target_prim_path)
        self.previous_pos = None
        self.vel = None

        self.target_prim = self.stage.GetPrimAtPath(self.target_prim_path)
        if not self.target_prim:
            raise Exception(f"Target prim must exist at path {self.target_prim_path}")
        
        self.check_freq = 1
    
    def get_prim_y_angle(self):
        """
        Get prim at angle difference from [0,1,0]
        """
        
        mat = omni.usd.utils.get_world_transform_matrix(self.target_prim) 

        y = mat.GetColumn(1)
        # print("mat", mat, "\n column", y)
        cos_angle = y[1] / math.sqrt(y[0]**2 + y[1]**2 + y[2]**2)
        return math.degrees(math.acos(cos_angle))

    def get_diff(self):
        delta_angle = self.get_prim_y_angle() - self.target_delta_y

        return delta_angle

    def start_checking(self):
        if self.is_init == False:
            return
        # success condition
        self.total_step += 1
        if self.total_step % self.check_freq == 0:
            delta_angle = abs(self.get_prim_y_angle() - self.target_delta_y)

            pos, rot = self.targetRigid.get_world_pose()
            pos = pos[1]
            
            if self.previous_pos is not None:
                self.vel  = abs(pos - self.previous_pos)
            
            if self.total_step % self.print_every == 0:
                print("delta_angle", delta_angle)

            if delta_angle < self.tolerance and self.vel is not None and self.vel < 0.1:
                self.success_steps += self.check_freq
                self._on_success_hold()
            else:
                self._on_not_success()
            self.previous_pos = pos
            super().start_checking()
