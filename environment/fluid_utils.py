import math
import numpy as np
import os
from .constants import *
from .parameters import FluidPhysicsProperties
from omni.physx.scripts import particleUtils
from pxr import UsdGeom, Sdf, Gf, Vt, PhysxSchema, Usd

from omni.physx.scripts import physicsUtils, particleUtils
import omni
from environment.fluid_constants import PARTICLE_PROPERTY


def point_sphere(samples, scale):
    indices = [x + 0.5 for x in range(0, samples)]

    phi = [math.acos(1 - 2 * x / samples) for x in indices]
    theta = [math.pi * (1 + 5**0.5) * x for x in indices]

    x = [math.cos(th) * math.sin(ph) * scale for (th, ph) in zip(theta, phi)]
    y = [math.sin(th) * math.sin(ph) * scale for (th, ph) in zip(theta, phi)]
    z = [math.cos(ph) * scale for ph in phi]
    points = [Gf.Vec3f(x, y, z) for (x, y, z) in zip(x, y, z)]
    return points


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    try:
        from scipy.spatial import Delaunay
    except:
        import omni
        omni.kit.pipapi.install("scipy")
        from scipy.spatial import Delaunay
        
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


# Generates hexagonal close packed samples inside an axis aligned bounding box
def generate_hcp_samples(boxMin: Gf.Vec3f, boxMax: Gf.Vec3f, sphereDiameter: float):

    layerDistance = math.sqrt(2.0 / 3.0) * sphereDiameter
    rowShift = math.sqrt(3.0) / 2.0 * sphereDiameter

    result = []
    layer1Offset = (1.0 / 3.0) * (
        Gf.Vec2f(0, 0) + Gf.Vec2f(0.5 * sphereDiameter, rowShift) + Gf.Vec2f(sphereDiameter, 0)
    )

    zIndex = 0
    while True:

        z = boxMin[2] + zIndex * layerDistance
        if z > boxMax[2]:
            break

        yOffset = layer1Offset[1] if zIndex % 2 == 1 else 0

        yIndex = 0
        while True:
            y = boxMin[1] + yIndex * rowShift + yOffset
            if y > boxMax[1]:
                break

            xOffset = 0
            if zIndex % 2 == 1:
                xOffset += layer1Offset[0]
                if yIndex % 2 == 1:
                    xOffset -= 0.5 * sphereDiameter
            elif yIndex % 2 == 1:
                xOffset += 0.5 * sphereDiameter

            xIndex = 0
            while True:
                x = boxMin[0] + xIndex * sphereDiameter + xOffset
                if x > boxMax[0]:
                    break

                result.append(Gf.Vec3f(x, y, z))
                xIndex += 1
            yIndex += 1
        zIndex += 1

    return result


def generate_inside_point_cloud(sphereDiameter, cloud_points, scale = 1, max_particles = 3000):
    """
    Generate sphere packs inside a point cloud
    """
    offset = 2
    min_x = np.min(cloud_points[:, 0]) + offset
    min_y = np.min(cloud_points[:, 1]) + offset
    min_z = np.min(cloud_points[:, 2]) + offset

    max_x = np.max(cloud_points[:, 0]) 
    max_y = np.max(cloud_points[:, 1]) 
    max_z = np.max(cloud_points[:, 2]) 

    
    min_bound = [min_x, min_y, min_z]
    max_bound = [max_x, max_y, max_z]
    
    min_bound = [item * scale for item in min_bound]
    max_bound = [item * scale for item in max_bound]

    samples = generate_hcp_samples(Gf.Vec3f(*min_bound), Gf.Vec3f(*max_bound), sphereDiameter)
    
    finalSamples = []
    contains = in_hull(samples, cloud_points)

    for contain, sample in zip(contains, samples):
        if contain and len(finalSamples) < max_particles:
            finalSamples.append(sample)

    
    print("number of particles created: ", len(finalSamples) )
    print("max particles: ", max_particles)
    return finalSamples



# TODO
# fix this function, this is harded coded for cup data
def set_particle_system_for_cup(
        stage, mug_init_position, volume_mesh_path: str, particle_system_path: str, particle_instance_str,
        fluid_physis_properties: FluidPhysicsProperties, asset_root,
        simulation_owner_path: str = "/physicsScene", enable_iso_surface = False
    ):

    volume_mesh = UsdGeom.Mesh.Get(stage, Sdf.Path(volume_mesh_path))

    particle_system_path = Sdf.Path(particle_system_path)

    particle_system = particleUtils.add_physx_particle_system(
        stage, particle_system_path, **fluid_physis_properties.properties["particle_system_schema_parameters"], simulation_owner=Sdf.Path(simulation_owner_path)
    )
    particle_instance_path = Sdf.Path(particle_instance_str)

    

    positions_list = []
    velocities_list = []
    
    particle_rest_offset = fluid_physis_properties.properties["particle_system_schema_parameters"]["fluid_rest_offset"]

    cloud_points = np.array(volume_mesh.GetPointsAttr().Get())

    positions_list  = generate_inside_point_cloud(sphereDiameter=particle_rest_offset * (2.0 + 0.08), cloud_points = cloud_points, scale=1.0)

    stage.GetPrimAtPath(volume_mesh_path).SetActive(False)

    for _ in range(len(positions_list)):
        velocities_list.append(Gf.Vec3f(0, 0, 0))
   
    positions = Vt.Vec3fArray(positions_list)
    velocities = Vt.Vec3fArray(velocities_list)
    
    # TODO
    # this is not tested do not use
    if enable_iso_surface:
        fluidRestOffset = 0.22
        particleSystemPath = particle_system_path

        #########################################
        from omni.kit.material.library import get_material_prim_path, create_mdl_material
        
        # water_url = 'http://localhost:8080/omniverse://127.0.0.1/NVIDIA/Materials/Base/Natural/Water.mdl'
        water_url = str(os.path.join(asset_root, "materials", "Natural", "Water.mdl"))

        water_mtl_name = water_url.split("/")[-1][:-4]

        # print("material dict: ", self.material_dict)
        water_material_prim_path = get_material_prim_path(water_mtl_name)

        def on_create(path):
            pass
        water_path = create_mdl_material(omni.usd.get_context().get_stage(), water_url, water_mtl_name, on_create)
        # Render material
        omni.kit.commands.execute("BindMaterial", prim_path=particleSystemPath, material_path=water_path)

        particleUtils.add_pbd_particle_material(stage, water_path, 
                                                **PARTICLE_PROPERTY._particleMaterialAttributes)
        physicsUtils.add_physics_material_to_prim(stage, particle_system.GetPrim(), water_path)

        particle_system.CreateMaxVelocityAttr().Set(40)

        # add particle smoothing
        smoothingAPI = PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.GetPrim())
        smoothingAPI.CreateParticleSmoothingEnabledAttr().Set(True)
        smoothingAPI.CreateStrengthAttr().Set(0.5)
        
        # apply isosurface params
        isosurfaceAPI = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.GetPrim())
        isosurfaceAPI.CreateIsosurfaceEnabledAttr().Set(True)
        isosurfaceAPI.CreateMaxVerticesAttr().Set(1024 * 1024)
        isosurfaceAPI.CreateMaxTrianglesAttr().Set(2 * 1024 * 1024)
        isosurfaceAPI.CreateMaxSubgridsAttr().Set(1024 * 4)
        isosurfaceAPI.CreateGridSpacingAttr().Set(fluidRestOffset * 1.5)
        isosurfaceAPI.CreateSurfaceDistanceAttr().Set(fluidRestOffset * 1.6)
        isosurfaceAPI.CreateGridFilteringPassesAttr().Set("")
        isosurfaceAPI.CreateGridSmoothingRadiusAttr().Set(fluidRestOffset * 2)

        isosurfaceAPI.CreateNumMeshSmoothingPassesAttr().Set(1)

        primVarsApi = UsdGeom.PrimvarsAPI(particle_system)
        primVarsApi.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

        stage.SetInterpolationType(Usd.InterpolationTypeHeld)

        particleSpacing = 0.2
        widths = [particleSpacing] * len(positions)

        particleUtils.add_physx_particleset_pointinstancer(
            stage,
            particle_instance_path,
            positions,
            velocities,
            particle_system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=fluid_physis_properties.properties["particle_mass"],
            density=0.0,
        )

    else:        
        particleUtils.add_physx_particleset_pointinstancer(
            stage,
            particle_instance_path,
            positions,
            velocities,
            particle_system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=fluid_physis_properties.properties["particle_mass"],
            density=0.0,
        )

        particle_color = fluid_physis_properties.properties["particle_color"]
        color = Vt.Vec3fArray([Gf.Vec3f(particle_color[0], particle_color[1], particle_color[2])])
        colorPathStr = f"{particle_instance_str}/particlePrototype0"
        gprim = UsdGeom.Sphere.Define(stage, Sdf.Path(colorPathStr))
        gprim.CreateDisplayColorAttr(color)
        gprim.GetRadiusAttr().Set(fluid_physis_properties.properties["fluid_sphere_diameter"])

    particles = UsdGeom.Xformable(stage.GetPrimAtPath(particle_instance_str))
    particles.AddTranslateOp().Set(mug_init_position)

    return particles
