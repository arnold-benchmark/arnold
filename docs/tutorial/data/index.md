# Data

As mentioned in [Tasks](../tasks/index.md), each task consists of several stages to complete. A corresponding motion planner is pre-defined for each task to complete the stages. The motion planner operates according to stage-wise keypoints, on which the diversity of demonstrations is highly dependent. To address this, we collect about 2k human annotations of task configurations (*e.g.*, object positions). Moreover, we broaden the data variations with additional relative positions and robot shifts. After the data generation process, we run inference with ground-truth keypoints to curate valid demonstrations. Finally, we collect 10k valid demonstrations for <tt>ARNOLD</tt> benchmark, with each demonstration containing 4–6 keyframes. See data statistics below:

```{image} data.png
---
width: 75%
alt: Data overview
---
```

## Split

Generalization is a major focus of <tt>ARNOLD</tt>. We randomly split the objects, scenes, and goal states into seen and unseen subsets, respectively. We create the *Normal* split by gathering data with seen objects, scenes, and states. This split is further shuffled and divided into *Train/Val/Test* sets proportioned at 70%/15%/15%. Furthermore, we create the *Generalization* splits *Novel Object/Scene/State* by gathering data with exactly one of the three components (*i.e.*, objects, scenes, and goal states) unseen; *e.g.*, the *Novel Object* split comprises data of unseen objects and seen scenes/states. In addition, we create an extra evaluation split *Any State*, which incorporates seen objects/scenes and arbitrary goal states within a continuous range, *e.g.*, 0%–100%.

## Language

For each demonstration, we sample a template-based language instruction with our language generation engine. We design several instruction templates with blanks for each task, and each template can be lexicalized with various phrase candidates. For example, the template "*pull the* [position] [object] [percentage] *open*" may be lexicalized into "*pull the top drawer 50% open*". In addition to the representation with explicit numbers, we also prepare a candidate pool of equivalent phrases (*e.g.*, "*fifty percent*", "*half*", "*two quarters*") for random replacement. We present a few examples of instruction templates as follows:

```{image} language.png
---
width: 85%
alt: Language templates
---
```

## Format

Each demonstration is saved in `npz` format, which is structured as below (here only present important elements for simplicity):
- <tt>demonstration</tt>: `numpy.lib.npyio.NpzFile`
  - `gt: numpy.ndarray (list)`
    - `dict` :: recorded information of each keyframe
      - `images -> list`
        - `dict` :: RGB-D observation from each camera
          - `rgb -> numpy.ndarray`
          - `depthLinear -> numpy.ndarray`
          - `camera -> dict` :: camera parameters
          - `...`
      - `instruction -> str`
      - `position_rotation_world -> tuple` :: end effector pose in world frame
        - `numpy.ndarray` :: position (xyz, y axis upward, in cm)
        - `numpy.ndarray` :: rotation (quaternion, wxyz)
      - `gripper_open -> bool`
      - `gripper_joint_positions -> numpy.ndarray` :: gripper joint values
      - `robot_base -> tuple` :: robot base pose in world frame
        - `numpy.ndarray` :: position (xyz, y axis upward, in cm)
        - `numpy.ndarray` :: rotation (quaternion, wxyz)
      - `diff -> float` :: the difference between current state and goal state
      - `...`
  - `info: numpy.ndarray (dict)` :: environment configurations, access the `dict` via `item()`
    - `scene_parameters -> dict` :: arguments of `SceneParameters`
    - `robot_parameters -> dict` :: arguments of `RobotParameters`
    - `objects_parameters -> list`
      - `dict` :: arguments of `ObjectParameters`
    - `config -> dict` :: misc configurations
    - `robot_shift -> list` :: robot position shift (xyz, y axis upward, in cm)

## Dataloader

We provide a single-task dataset class `ArnoldDataset` and a multi-task dataset class `ArnoldMultiTaskDataset` in `dataset.py`.
- `ArnoldDataset`. For each task, the demonstrations are maintained in `episode_dict`, which is a `dict` organized by categorizing different objects and phases.
  ```python
  MetaData: {
    'img': img,   # [H, W, 6], rgbddd
    'obs_dict': obs_dict,   # { {camera_name}_{rgb/point_cloud}: [H, W, 3] }
    'attention_points': obj_pos,   # [3,]
    'target_points': target_points,   # [6,]
    'target_gripper': gripper_open,   # binary
    'low_dim_state': [gripper_open, left_finger, right_finger, timestep]
    'language': language_instructions,   # str
    'current_state': init_state,   # scalar
    'goal_state': goal_state,   # scalar
    'bounds': task_offset,   # [3, 2]
    'pixel_size': pixel_size,   # scalar
  }

  ArnoldDataset.episode_dict: {
    obj_id: {
      'act1': List[MetaData],
      'act2': List[MetaData],
    }
  }
  ```

  Referring to the structure of `episode_dict`, fetching a piece of data requires three values: an `obj_id`, the phase (`act1` or `act2`), and an index in the corresponding `List[MetaData]`. There are two modes to fetch data from the dataset: *index* and *sample*. In both modes, the phase (`act1` or `act2`) is sampled according to `sample_weights` because of the phase imbalance. In *index* mode, we get the data by calling `__getitem__()` and passing an index to retrieve data sequentially. The provided index maps to a unique `MetaData`. In *sample* mode, we get the data by calling `sample()`, where both the `obj_id` and `MetaData` index are uniformly sampled.

- `ArnoldMultiTask`. This is a wrapper upon the single-task class `ArnoldDataset`. Specifically, it contains a `dict` named `task_dict`, whose keys are task names and corresponding values are `ArnoldDataset`.

Since the language encoding modules are fixed, we create a class `InstructionEmbedding` in `dataset.py` to store the embedding caches. When forwarding, the embedding of language instruction will be computed and added to the cache unless it already exists in the cache.
