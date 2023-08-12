import pxr
import omni
import math
import numpy as np
from pxr import UsdGeom
from typing import List, Union
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_stage_up_axis
from .base_checker import BaseChecker
from environment.parameters import CheckerParameters


class liquid_cup_check():
    def __init__(self,  cup_path: str, particle_paths: List[str], iso_surface = False) -> None:
        self.iso_surface = iso_surface
        print('iso surface: ', iso_surface)
        
        self.cup_path = cup_path
        self.cup_shape_path = cup_path+'/cupShape'
        self.particle_paths = particle_paths
        self.stage = omni.usd.get_context().get_stage()

        self.cup_prim = self.stage.GetPrimAtPath(self.cup_shape_path)
        self.cup_xform = UsdGeom.Xformable(self.cup_prim)
        self.iso_surface = iso_surface
    
    def get_bbox(self):
        bboxes = self.cup_xform.ComputeWorldBound(0, UsdGeom.Tokens.default_)
        prim_bboxes = np.array([bboxes.ComputeAlignedRange().GetMin(), bboxes.ComputeAlignedRange().GetMax()])
        return prim_bboxes

    def height_percentage(self):
        particle_positions = self.get_particle_positions()
        up_axis = get_stage_up_axis()
        axises = [pxr.UsdGeom.Tokens.x, pxr.UsdGeom.Tokens.y, pxr.UsdGeom.Tokens.z]
        box = self.get_bbox()
   
        def inside(particle):
            inA = particle[0] > box[0][0] and particle[0] < box[1][0]
            inB = particle[1] > box[0][1] and particle[1] < box[1][1]
            inC = particle[2] > box[0][2] and particle[2] < box[1][2]
            if inA and inB and inC:
                return True
            return False
        
        res = list(map(inside, particle_positions))
        if not np.any(res):
            return 0

        inside_particles_max = np.max([particle_position[axises.index(up_axis)] for particle_position in particle_positions[res]])
        
        height_percentage = (inside_particles_max-box[0][ axises.index(up_axis) ])/( box[1][ axises.index(up_axis)]- box[0][ axises.index(up_axis)])
        return height_percentage * 100.0
    
    def percentage_inside(self):
        particle_positions = self.get_particle_positions().tolist()
        box = self.get_bbox()

        def inside(particle):
            inA = particle[0] > box[0][0] and particle[0] < box[1][0]
            inB = particle[1] > box[0][1] and particle[1] < box[1][1]
            inC = particle[2] > box[0][2] and particle[2] < box[1][2]
            if inA and inB and inC:
                return True
            return False

        res = list(map(inside, particle_positions))    
        return sum(res)/len(res) * 100.0

    def get_particle_positions(self, paths :Union[None, List[str] ] = None ):
        positions = []
        if paths is not None:
            path_in_use = paths
        else:
            path_in_use = self.particle_paths

        for particle_path in path_in_use:
            tmp = self.get_particle_position_list(particle_path=particle_path)
            positions.append(tmp)
        
        particle_positions = np.vstack(positions)
        return particle_positions
    
    def get_all_particles(self, paths :Union[None, List[str] ] = None ):
        ptcl_dict = {}
        if paths is not None:
            path_in_use = paths
        else:
            path_in_use = self.particle_paths

        for particle_path in path_in_use:
            particle_prim = self.stage.GetPrimAtPath(particle_path)
            particles = pxr.UsdGeom.PointInstancer(particle_prim)
            pos = np.around(np.array(particles.GetPositionsAttr().Get()), 3).tolist()
            vel = np.around(np.array(particles.GetVelocitiesAttr().Get()), 3).tolist()
            ptcl_dict[particle_path] = [pos, vel]
        
        return ptcl_dict

    def get_particle_position_list(self, particle_path):
        self.particle_prim = self.stage.GetPrimAtPath(particle_path)

        mat = omni.usd.utils.get_world_transform_matrix(self.particle_prim) 
        translation = mat.ExtractTranslation()
        rotation_matrix = mat.ExtractRotationMatrix()
        particles = pxr.UsdGeom.PointInstancer(self.particle_prim)

        positions = np.array(particles.GetPositionsAttr().Get())

        positions = positions @  rotation_matrix + translation

        return positions #+ translation
    
    def set_all_particles(self, ptcl_dict):
        path_in_use = list(ptcl_dict.keys())
        for particle_path in path_in_use:
            if particle_path in self.particle_paths:
                particle_prim = self.stage.GetPrimAtPath(particle_path)
                particles = pxr.UsdGeom.PointInstancer(particle_prim)
                pos = ptcl_dict[particle_path][0]
                particles.CreatePositionsAttr().Set([pxr.Gf.Vec3f(*p) for p in pos])
                vel = ptcl_dict[particle_path][1]
                particles.CreateVelocitiesAttr().Set([pxr.Gf.Vec3f(*v) for v in vel])


class WaterChecker(BaseChecker):
    def __init__(self, checker_parameters: CheckerParameters, tolerance = 10) -> None:
        self.checker_parameters = checker_parameters
        self.tolerance = tolerance
    
    def pre_initialize(self, target_prim_path, constrained_cup_prim_path, particle_path, iso_surface):
        super().__init__()
        
        self.iso_surface = iso_surface
        self.target_prim_path = target_prim_path

        self.constrained_cup_prim_path = constrained_cup_prim_path
        self.target_volume = self.checker_parameters.target_state
        self.particle_path = particle_path
        
        self.liquid_checker = liquid_cup_check(self.target_prim_path, [self.particle_path], iso_surface)

        self.target_prim = self.stage.GetPrimAtPath(self.target_prim_path)
        self.constrained_cup_prim = get_prim_at_path(self.constrained_cup_prim_path)

        
        if target_prim_path != self.constrained_cup_prim_path:
            self.liquid_checker2 = liquid_cup_check(self.constrained_cup_prim_path, [self.particle_path], iso_surface)
        else:
            self.liquid_checker2 = None

        if not self.target_prim:
            raise Exception(f"Target prim must exist at path {self.target_prim_path}")

        self.check_freq = 1
    
    def initialize(self):
        self.create_task_callback()
        self.is_init = True
        
    def diff_to_upright(self):
       
        """
        Get prim at angle difference from [0,1,0]
        """

        mat = omni.usd.utils.get_world_transform_matrix(self.constrained_cup_prim) 

        y = mat.GetColumn(1)
        # print("mat", mat, "\n column", y)
        cos_angle = y[1] / math.sqrt(y[0]**2 + y[1]**2 + y[2]**2)
        return math.degrees(math.acos(cos_angle))
    
    def get_percentage(self):
        """
        Get water percentage in original container
        """
        return self.liquid_checker.percentage_inside()
    
    def get_diff(self):
        percentage = self.get_percentage() 
        return percentage -  self.target_volume
    
    def start_checking(self):
        if self.is_init == False:
            return
        # success condition
        self.total_step += 1
        if self.total_step % self.check_freq == 0:
            percentage = self.get_percentage() 

            if self.liquid_checker2 is not None:
                remain_percentage = self.liquid_checker2.percentage_inside()
                spill = 100 - percentage - remain_percentage
            else:
                spill = 0

            if self.total_step % 60 == 0:
                print(f'percentage: {percentage} target {self.target_volume}')
            
            if abs(percentage -  self.target_volume) < (self.tolerance) and abs(self.diff_to_upright()) < 30 and spill < 10:
                self.success_steps += self.check_freq
                self._on_success_hold()
            else:
                self._on_not_success()
           
            super().start_checking()
