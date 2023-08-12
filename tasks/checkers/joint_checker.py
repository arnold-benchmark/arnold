import math
import omni
from omni.isaac.core.prims import XFormPrim
from omni.isaac.dynamic_control import _dynamic_control
from .base_checker import BaseChecker
from environment.parameters import CheckerParameters


class JointCheck():
    def __init__(self, joint_prim, joint_name) -> None:
        self.joint_name = joint_name
        self.stage = omni.usd.get_context().get_stage()

        self.prim_list = list(self.stage.TraverseAll())
        # print("self.prim_list: ", self.prim_list)
        self.prim_list = [ item for item in self.prim_list if joint_name in  
            item.GetPath().pathString and item.GetPath().pathString.startswith(joint_prim) and item.GetPath().pathString.endswith(joint_name)]

        assert len(self.prim_list) == 1, "len of " + str(len(self.prim_list))
        self.prim = self.prim_list[0]

        self.type = self.prim.GetTypeName()
        self.full_name = self.prim.GetPath().pathString
        self.joint = self.stage.GetPrimAtPath(self.full_name)

    def get_joint_position(self):
        body1 = self.joint.GetRelationship("physics:body1").GetTargets()[0]
        
        return XFormPrim(body1).get_world_pose()[0]
    
    def get_joint_link(self):
        body0 = self.joint.GetRelationship("physics:body0").GetTargets()[0]
        body1 = self.joint.GetRelationship("physics:body1").GetTargets()[0]
        return body1

    @property
    def upper(self):
        return self.joint.GetAttribute("physics:upperLimit").Get()
    
    @property
    def lower(self):
        return self.joint.GetAttribute("physics:lowerLimit").Get()
        
    def compute_percentage(self):

        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.art = self.dc.get_articulation(self.full_name)

        dof_ptr = self.dc.find_articulation_dof(self.art, self.joint_name)
        dof_pos = self.dc.get_dof_position(dof_ptr)
        
        if self.type == 'PhysicsPrismaticJoint':
            tmp = dof_pos
        else:
            tmp = math.degrees(dof_pos)
            
        pertentage = (tmp - self.lower)/(self.upper - self.lower) * 100

        # print("upper lower percentage", tmp, self.upper, self.lower, pertentage)

        if pertentage > 100:
            pertentage = 100
        elif pertentage < 0:
            pertentage = 0

        return pertentage 
    
    def compute_distance(self):
        return abs(self.compute_percentage() - self.initial_percentage)

    def set_joint(self, percentage):
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.art = self.dc.get_articulation(self.full_name)
        dof_ptr = self.dc.find_articulation_dof(self.art, self.joint_name)
       
        
        tmp = percentage / 100.0 *(self.upper-self.lower) + self.lower
        if self.type == 'PhysicsPrismaticJoint':
            dof_pos = tmp
        else:
            dof_pos = math.radians(tmp)
      
        self.dc.wake_up_articulation(self.art)
        self.dc.set_dof_position(dof_ptr, dof_pos)


class JointChecker(BaseChecker):
    def __init__(self, checker_parameters: CheckerParameters, tolerance = 0.1) -> None:
        self.checker_parameters = checker_parameters
        self.tolerance = tolerance
    
    def pre_initialize(self, target_prim_path):
        super().__init__()
        self.target_joint = self.checker_parameters.target_joint

        self.init_value = self.checker_parameters.init_state
        self.target_value = self.checker_parameters.target_state
        self.target_prim_path = target_prim_path
      
        self.joint_checker = JointCheck(self.target_prim_path, self.target_joint)

        self.check_joint_direction()
        
        # set joint at start
        self.set_joint_at_start = True if self.init_value != -1 else False

        self.previous_percentage = None
        self.vel = None
        self.check_freq = 1
        
    
    def check_joint_direction(self):
        """
        Check joint positive rotation to upper or negative rotation to lower
        """
        is_upper = abs(self.joint_checker.upper) > abs(self.joint_checker.lower)
        if not is_upper:
            # if is lower, reverse init_value and target value
            self.init_value = 1 - self.init_value if self.init_value != -1 else -1
            self.target_value = 1 - self.target_value

    def get_diff(self):
        percentage =  self.joint_checker.compute_percentage()
        return percentage/100  - self.target_value
        
    def start_checking(self):
        if self.is_init == False:
            return
        
        self.total_step += 1
        if self.total_step % self.check_freq == 0:
            if self.set_joint_at_start:
                self.joint_checker.set_joint(self.init_value*100)
                self.set_joint_at_start = False

            percentage =  self.joint_checker.compute_percentage()
        
            if self.previous_percentage is not None:
                self.vel  = abs(percentage - self.previous_percentage)

            # log
            if self.total_step % self.print_every == 0:
                print("current: {:.1f}; target: {:.1f}; delta percentage: {:.1f}:".format(percentage, self.target_value*100, self.target_value*100 - percentage) )
            
            if abs(percentage/100 - self.target_value) < self.tolerance and self.vel is not None and self.vel < 0.05:
                self.success_steps += self.check_freq
                self._on_success_hold()
            else:
                self._on_not_success()

            self.previous_percentage = percentage
            
            super().start_checking()
