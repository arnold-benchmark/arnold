# Configs

We adopt [hydra](https://hydra.cc/) and provide a configuration file `configs/default.yaml` to configure the settings before running experiments. Here we explain some notable terms.

- Path
  - `base_root`. Base directory.
  - `data_root`. The directory of data.
  - `asset_root`. The directory of asset samples and materials.
  - `output_root`. The directory to save experiment outputs.
- Task and model
  - `mode`. Either `train` or `eval`.
  - `task`. Task name for single-task, `multi` for multi-task.
  - `model`. One of `cliport6d`, `peract`, `bc_lang_cnn`, `bc_lang_vit`.
  - `lang_encoder`. One of `clip`, `none`, `t5`, `roberta`.
  - `state_head`. Whether using an additional state head, either `0` or `1`.
- Running arguments
  - `batch_size`. `8` for training and `1` for evaluation.
  - `steps`. Training steps, default at 100k.
  - `log_interval`. The step interval between logging behavior during training.
  - `save_interval`. The step interval between checkpoint saving during training.
  - `use_gt`. For evaluation, two `bool` values indicating whether to use ground-truth keypoints for each phase.
  - `visualize`: For evaluation, keep rendering if `True`. Setting to `False` can accelerate evaluation.
- Environment
  - `offset_bound`. The perception bound, used to crop a cube centered at the robot, represented in [x1, y1, z1, x2, y2, z2], in m.
  - `iso_surface`. Whether enabling realistic fluid simulation.
- PerAct
  - `t5_cfg`. The path of pre-trained T5 model.
  - `roberta_cfg`. The path of pre-trained RoBERTa model.

In addition to configuring the settings in `yaml` file, we can also configure them when running commands, *e.g.*,
```bash
python eval.py task=pickup_object model=peract lang_encoder=clip mode=eval use_gt=[0,0] visualize=0
```
