# Evaluation

## Sanity Check

Different from the training, the evaluation has a dependency on <tt>Isaac Sim</tt>. So you should make sure that your <tt>Isaac Sim</tt> works well before running evaluation. You can run a toy example for a sanity check:
```bash
python ${Isaac_Sim_Root}/standalone_examples/api/omni.isaac.franka/pick_place.py
# e.g., python ~/.local/share/ov/pkg/isaac_sim-2022.1.1/standalone_examples/api/omni.isaac.franka/pick_place.py
```

## Run

### Checkpoint Selection

As mentioned in [Training](../train/index.md), the best checkpoint is not selected during the training. Instead, you should run `ckpt_selection.py` to select the best one among the saved checkpoints. For example, to select the best checkpoint for PerAct (with CLIP language encoder) on `PickupObject`, you can run the following command:
```bash
python ckpt_selection.py task=pickup_object model=peract lang_encoder=clip mode=eval visualize=0
```

The argument `mode` should always be `eval` except for training. The assignment `visualize=0` disables rendering and accelerates the simulation. Notably, you should ensure the existence of the checkpoints that correspond to the specified setting (*e.g.*, `model`, `lang_encoder` and `state_head`). Because it will automatically locate the checkpoints and load them for evaluation. Similarly, to select the best checkpoint for multi-task PerAct (with CLIP language encoder), run the following command:
```bash
python ckpt_selection.py task=multi model=peract lang_encoder=clip mode=eval visualize=0
```

As the selection process goes on, it will automatically record the evaluation logs in a file `select.log` in the hydra output directory. After all the checkpoints are enumerated, the best checkpoint will be determined and renamed with a `best` mark, which will be identified by the subsequent evaluation process.

### Evaluation

After the checkpoint selection completes, you can evaluate the model with the selected best checkpoint. For example, to evaluate PerAct (with CLIP language encoder), you can run the following commands:
```bash
# single-task pickup_object
python eval.py task=pickup_object model=peract lang_encoder=clip mode=eval use_gt=[0,0] visualize=0

# multi-task
python eval.py task=multi model=peract lang_encoder=clip mode=eval use_gt=[0,0] visualize=0
```

The argument `use_gt` specifies whether to use ground-truth action for the two phases, respectively. This argument has two usages: 1) conduct ablation with the first-phase ground truth (`use_gt=[1,0]`); 2) replay demonstrations with both-phase ground truth (`use_gt=[1,1]`). Similarly, running `eval.py` will also automatically generate evaluation logs in a file in the hydra output directory. The file name will be `eval_wo_gt.log` if no first-phase ground truth is provided else `eval_w_gt.log`.

### Automatic Pipeline

As the simulation takes a long time and sometimes may encounter errors, an end-to-end and auto-resume pipeline is preferred for evaluation. Based on the training checkpoints, you can wrap together the checkpoint selection, evaluation without first-phase ground truth, and evaluation with first-phase ground truth processes via a simple command:
```bash
bash eval_pipeline.sh python open_drawer bc_lang_vit clip 0 1
```

The shell script is executed with five arguments:
- Interpreter. Use the `python.sh` in <tt>Isaac Sim</tt> root for docker-based setup and `python` for conda-based setup.
- Task. Use the task name for single-task and `multi` for multi-task.
- Model. Currently, one of `cliport6d`, `peract`, `bc_lang_cnn`, `bc_lang_vit`.
- Language encoder. Currently, one of `clip`, `none`, `t5`, `roberta`.
- State head. For PerAct in particular, `1` if a state head is added else `0`.
- First-phase ground truth. If specified to `1`, the pipeline will continue to run the evaluation with first-phase ground truth after the evaluation without first-phase ground truth. If specified to `0`, the pipeline will only run the evaluation without first-phase ground truth.

## Inference

When not using ground truth action, the robot performs inference via `get_action()`, which returns an end effector pose (translation + rotation) given task instruction and current-frame RGB-D observation. There are several notable arguments:
- `gt`. Rendered RGB-D observation (the name may be misleading).
- `agent`. Model, `torch.nn.Module`.
- `npz_file`. The loaded `npz` demonstration for evaluating.
- `offset`. Perception bound, which is centered at robot base.
- `timestep`. Which phase, `0` for first-phase and `1` for second-phase.
- `agent_type`. Model name, one of `cliport6d`, `peract`, `bc_lang_cnn` and `bc_lang_vit`.
- `lang_embed_cache`. The cache of instruction embedding.

In `get_action()`, different models require different inputs.
- <tt>6D-CLIPort</tt>. It performs inference via the `act()` method, which takes as input:
  - `obs`. RGB image and unprojected point cloud from each view.
  - `instruction`. Task instruction, raw text.
  - `bounds`. Perception bound.
  - `pixel_size`. Calibrated size of each pixel.
- <tt>PerAct</tt>. It performs inference via the `predict()` method, which takes as input:
  - `obs`. RGB image and unprojected point cloud from each view.
  - `lang_goal_embs`. Instruction embedding.
  - `low_dim_state`. Proprioception, including gripper state and `timestep`.
- <tt>BC-Z</tt>. It performs inference via the `predict()` method, which takes perception bound as an extra input in addition to the same input as <tt>PerAct</tt>.

Subsequently, the end effector pose derived from `get_action()` will be sent to `step()` for task execution. The task execution details can be found in [Tasks](../tasks/index.md).

## Metrics

A task instance is regarded as a success when the success condition is satisfied continually for two seconds. The success condition requires the current state to be within a tolerance threshold from the goal state; *i.e.*, the success range (see details in [Tasks](../tasks/index.md)). Note that `TransferWater` imposes an extra condition that only 10% or less of the water can be spilled. To avoid shortcuts, we check the success condition only after all stages are completed. For example, given the task "*pour half of the water out of the cup*", the robot succeeds if 40% ∼ 60% of the water remains in the cup for two seconds after the robot reorients the cup upright. We use success rate as our evaluation metrics.
