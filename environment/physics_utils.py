from pxr import Usd, UsdPhysics, UsdShade, UsdGeom
from omni.physx.scripts import physicsUtils
from omni.physx.scripts.utils import setRigidBody
from .constants import *
from .parameters import ObjectPhysicsProperties


def set_collision(stage, prim : Usd.Prim, approximationShape:str):
    collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
    if not collision_api:
        collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    
    collision_api.CreateApproximationAttr().Set(approximationShape)


def set_mass(stage, prim: Usd.Prim, mass:float):
    mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
    if not mass_api:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(mass)
    else:
        mass_api.GetMassAttr().Set(mass)


def set_physics_material(stage, prim: Usd.Prim, object_physics_properties : ObjectPhysicsProperties):
    """
    Set up physic material for prim at Path
    """
    # def _setup_physics_material(self, path: Sdf.Path):
    object_physics_properties = object_physics_properties.properties # get dict

    _material_static_friction = object_physics_properties[STATIC_FRICTION]
    _material_dynamic_friction = object_physics_properties[DYNAMIC_FRICTION]
    _material_restitution = object_physics_properties[RESTITUTION]

    _physicsMaterialPath = prim.GetPath().AppendChild("physicsMaterial")
    # print("physics_material_path: ", _physicsMaterialPath)
    
    UsdShade.Material.Define(stage, _physicsMaterialPath)
    material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(_physicsMaterialPath))
    material.CreateStaticFrictionAttr().Set(_material_static_friction)
    material.CreateDynamicFrictionAttr().Set(_material_dynamic_friction)
    material.CreateRestitutionAttr().Set(_material_restitution)

    # apply material
    physicsUtils.add_physics_material_to_prim(stage, prim, _physicsMaterialPath)


def set_physics_properties(stage, prim : Usd.Prim, properties: ObjectPhysicsProperties):

    if COLLISION in properties.properties:
        if  IS_RIGID_BODY in properties.properties and properties.properties[IS_RIGID_BODY]:
            setRigidBody(prim, properties.properties[COLLISION], False)
        else:
            set_collision(stage, prim, properties.properties[COLLISION])
    
    if HAS_PHYSICS_MATERIAL in properties.properties and  properties.properties[HAS_PHYSICS_MATERIAL]:
        set_physics_material(stage, prim, properties)

    if MASS in properties.properties:
        set_mass(stage, prim, properties.properties[MASS])
    
    if DAMPING_COEFFICIENT in properties.properties:
        set_joint_properties(stage, prim, properties.properties[DAMPING_COEFFICIENT] )


def set_joint_properties(stage, prim, damping_cofficient):
    joint_driver = UsdPhysics.DriveAPI.Get(prim, "linear")
    if joint_driver:
        joint_driver.CreateDampingAttr(damping_cofficient)

    # find linear drive
    joint_driver = UsdPhysics.DriveAPI.Get(prim, "angular")
    if joint_driver:
        joint_driver.CreateDampingAttr(damping_cofficient)
    
    # find linear joint upperlimit, this assumes that lower limit is 0
    joint = UsdPhysics.PrismaticJoint.Get(stage, prim.GetPath())	
    if joint:
        upper_limit = joint.GetUpperLimitAttr().Get() #GetAttribute("xformOp:translate").Get()
        mobility_prim = prim.GetParent().GetParent()
        mobility_xform = UsdGeom.Xformable.Get(stage, mobility_prim.GetPath())
        scale_factor = mobility_xform.GetOrderedXformOps()[2].Get()[0]
        joint.CreateUpperLimitAttr(upper_limit * scale_factor / 100 )
