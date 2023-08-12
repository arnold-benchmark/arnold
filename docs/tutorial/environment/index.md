# Environment

<tt>ARNOLD</tt> is built on [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim), a robotic simulation application that provides photo-realistic and physically-accurate simulations for robotics research and development. In <tt>ARNOLD</tt>, the photo-realistic rendering is powered by GPU-enabled ray tracing, and the physics simulation is based on PhysX 5.0. The figure below gives an illustration of the simulation and rendering.

```{image} multi_view.png
---
width: 75%
alt: Multi-view rendering
---
```

<tt>ARNOLD</tt> contains 20 diverse scenes and 40 distinct objects. Their compositions amount to abundant environments. In `environment/parameters.py` we define some basic classes for scene construction and physics simulation, *e.g.*, `SceneParameters` and `ObjectPhysicsProperties`. They make the scene components disentangled and can be modified friendly.

## Scenes

### Loading
<tt>ARNOLD</tt> scenes originate from [3D-Front dataset](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset). We formulate a pipeline to parse the 3D-Front scenes to USD format (preferred in Omniverse). Make sure you have prepared the **assets** (see [Setup](../setup/index.md)). Besides, you can refer to [USDify](https://github.com/arnold-benchmark/Usdify) if you are interested in the pipeline of converting the scene format.

Loading scenes mainly encompasses the `add_reference_to_stage` function:
```python
def add_reference_to_stage(usd_path: str, prim_path: str, prim_type: str = "Xform") -> Usd.Prim:
```

### Collision
We set collisions and other attributes for different objects inside the scene:
```python
def setStaticCollider(prim, approximationShape="none", custom_execute_fn=None):
    setColliderSubtree(prim, approximationShape, custom_execute_fn)

# set collisions for a room
setStaticCollider(furniture_prim, approximationShape=CONVEXHULL)
setStaticCollider(room_struct_prim, approximationShape="none")
```
`furniture_prim` holds all the furniture and `room_struct_prim` defines all the walls. Currently, the wall geometry is very complicated. And the default collision mechanism might cause issues (huge amount of VRAM *etc.*). Thus, we use the convex hull for furniture and default collision for the walls.

You can disable the wall collision. Or you can decompose the walls into simple geometries like a collection of blocks. For example, you can refer to [collision processing](https://github.com/SarahWeiii/CoACD).

### Texture
After setting collisions, we set the scene materials. Make sure you have prepared the **materials**. Or you can use custom `mtl` files. We use the following command to create a material in <tt>Isaac Sim</tt>:
```python
omni.kit.commands.execute(
    "CreateMdlMaterialPrim",
    mtl_url=floor_material_url,
    mtl_name=floor_mtl_name,
    mtl_path=floor_material_prim_path,
    select_new_prim=False,
)
```

Then we can bind the materials to prims through:
```python
omni.kit.commands.execute(
    "BindMaterial",
    prim_path=floor_prim.GetPath(),
    material_path=floor_material_prim_path,
    strength=UsdShade.Tokens.strongerThanDescendants
)
```

Here we leverage some functions from low-level API which is not documented very well. For more information, you can refer to [USD documentation](https://openusd.org/release/api/index.html). Even though it is a <tt>CPP</tt> documentation, you can find equivalent <tt>python</tt> functions.

## Objects
Loading objects is similar to loading scenes. In particular, when we load drawers and cabinets, we re-process the handle for better collision. For fluid simulation, you can enable or disable isosurface for realistic simulation. Detailed settings about the objects are introduced in [Physics](#physics).

## Robot

We use a 7-DoF Franka Emika Panda manipulator with a parallel gripper in <tt>ARNOLD</tt> for task execution. The agent has direct control over its seven joints and its gripper. We represent end-effector actions with three spatial coordinates for translation and quaternion for rotation. We utilize the built-in motion planner of </tt>Isaac Sim</tt> to transform the end-effector action back to the space of robot joints for execution. Currently, our tasks do not involve navigation, *i.e.*, the robot base remains fixed during task execution.

## Observation

<tt>ARNOLD</tt> provides five cameras around the robot for visual input. Their placements are shown in the illustrative rendering example at the top of this page. Each camera renders RGB-D image at a resolution of 128×128 by default. Notably, the rendering in <tt>ARNOLD</tt> is stochastic due to the ray tracing sampling process. In addition to the visual observation, other auxiliary observations can be accessed, *e.g.*, camera parameters, robot base pose, and part-level semantic mask. See how to access them in [Data](../data/index.md).

## Rendering
There are two types of rendering modes in <tt>Isaac Sim</tt>: ray tracing and path tracing. Ray tracing can work in real-time. Path tracing is quite slow but renders at a much higher quality. To switch rendering mode, check the `get_simulation()` function in `environment/runner_utils.py`. Check [Omniverse RTX Renderer Overview](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_overview.html) for more details on rendering. Two rendering modes are visualized as below, left is ray tracing and right is path tracing.

```{image} ray_tracing.png
---
width: 49%
alt: Ray tracing
---
```

```{image} path_tracing.png
---
width: 49%
alt: Path tracing
---
```

## Physics

To ensure physically-realistic simulation, we assign physics parameters to objects, including weight and friction for rigid-body objects, and cohesion, surface tension, and viscosity for fluids. Fluids are simulated using the GPU-accelerated position-based-dynamics (PBD) method. And we provide an optional surface construction process using marching cubes for higher rendering quality. This may cause slower rendering and is controlled by a `bool` argument `iso_surface`.

In `environment/physics_utils.py` we define some helper functions to set up physics properties. For example, as grasping is done through friction in <tt>Issac Sim</tt>, we need to apply a physical material for all graspable parts. You can refer to the `set_physics_material` function.
```python
def set_physics_material(stage, prim: Usd.Prim, object_physics_properties: ObjectPhysicsProperties):
    """
    Set up physic material for prim at Path
    """
```

Articulation simulation depends on joint properties. To set joint properties for prismatic joint or revolute joint, you can refer to the `set_joint_properties` function in `environment/physics_utils.py`.
```
def set_joint_properties(stage, prim, damping_cofficient):
```

If you want to add new properties, you can refer to [USD documentation](https://openusd.org/release/api/index.html) and make educated guesses about what functions to use.

The fluid simulation involves two files `environment/fluid_constants.py` and `environment/fluid_utils.py`, where fluid parameters and construction process are defined. In addition, the isosurface mode can be enabled for more realistic fluid simulation by setting `iso_surface` to `True` in `configs/default.yaml`.
