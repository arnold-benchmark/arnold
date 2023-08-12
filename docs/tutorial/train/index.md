# Training

Remember to check the configurations in the `yaml` file before training the manipulation models. Currently <tt>ARNOLD</tt> supports the training of three existing language-conditioned manipulation models: [6D-CLIPort](https://sites.google.com/ucsc.edu/vlmbench/home), [PerAct](https://peract.github.io/), and [BC-Z](https://sites.google.com/view/bc-z/home). The corresponding training scripts are: `train_cliport6d.py`, `train_peract.py`, and `train_bc_lang.py`. For example, run the following commands:

```bash
# cliport6d
python train_cliport6d.py task=pickup_object model=cliport6d mode=train batch_size=8 steps=100000

# single-task peract
python train_peract.py task=pickup_object model=peract lang_encoder=clip mode=train batch_size=8 steps=100000

# multi-task peract
python train_peract.py task=multi model=peract lang_encoder=clip mode=train batch_size=8 steps=200000

# single-task bc-lang-cnn
python train_bc_lang.py task=pickup_object model=bc_lang_cnn mode=train batch_size=8 steps=100000

# single-task bc-lang-vit
python train_bc_lang.py task=pickup_object model=bc_lang_vit mode=train batch_size=8 steps=100000

# multi-task bc-lang-vit
python train_bc_lang.py task=multi model=bc_lang_vit mode=train batch_size=8 steps=200000
```

We unify the execution of tasks into a two-phase procedure: grasping the target object and then manipulating it towards the goal state. Each phase is directed by a keypoint, which we train the models to predict. The learning objectives and training settings of each model follow the original paper. All of our training is conducted on a single NVIDIA A100 GPU with batch size 8.

The model implementations can be found in `cliport6d/`, `peract/`, and `bc_z/`, respectively. **Any other manipulation models can be implemented and adapted on your own. The model implementation itself is a separate part. To train different models on <tt>ARNOLD</tt>, the main difference lies in preparing the input batch from fetched `MetaData` for forwarding.**

## 6D-CLIPort

6D-CLIPort takes as input an RGB-D image from a top-down view and predicts an image-calibrated end effector pose. The 6D-CLIPort requires such an input batch:
```python
{
    'img': img,   # (N, H, W, 6)
    'lang_goal': language_instructions,   # List[str], len=N
    'p0': p0,   # 2D coordinates of attention points, (N, 2)
    'p0_z': p0_z,   # height of attention points, (N,)
    'p1': p1,   # 2D coordinates of target points, (N, 2)
    'p1_z': p1_z,   # height of target points, (N,)
    'p1_rotation': p1_rotation,   # Euler angles of target pose, (N, 3)
}
```

## PerAct

PerAct takes RGB-D images as input to generate a voxelized representation, and predicts a voxel-calibrated end effector pose. The PerAct requires such an input batch:
```python
{
    f'{camera_name}_rgb': color,   # (N, C, H, W)
    f'{camera_name}_point_cloud': point_cloud   # (N, 3, H, W)
    'trans_action_indices': trans_action_indices,   # target voxel indices, (N, 3)
    'rot_grip_action_indices': rot_grip_action_indices,   # rotation and gripper_open of target pose, (N, 4)
    'states': states,   # current state and goal state, (N, 2)
    'lang_goal_embs': lang_goal_embs,   # instruction embedding, (N, T, D)
    'low_dim_state': low_dim_state   # [gripper_open, left_finger, right_finger, timestep], (N, 4)
}
```

## BC-Z

BC-Z has two model variants: CNN and ViT. Regardless of the architecture difference, the BC-Z model takes RGB-D images as input and directly regresses an end effector pose, whose translation and rotation are both continuous values (coordinates and quaternions). The input batch is as below:
```python
{
    f'{camera_name}_rgb': color,   # (N, C, H, W)
    f'{camera_name}_point_cloud': point_cloud   # (N, 3, H, W)
    'action': action,   # [translation, quaternion, gripper_open], (N, 8)
    'lang_goal_embs': lang_goal_embs,   # instruction global embedding, (N, D)
    'low_dim_state': low_dim_state,   # [gripper_open, left_finger, right_finger, timestep], (N, 4)
    'bound': offset_bound,   # (2, 3)
}
```

## Validation
We do not determine the best model checkpoint during training. Instead, the model checkpoint is saved every `save_interval` steps. With these checkpoints, a particular script will be used to select the best checkpoint (see [Eval](../eval/index.md)).
