"""
For example, run:
    Single-task:
        python train_peract.py task=pickup_object model=peract lang_encoder=clip \
                               mode=train batch_size=8 steps=100000
    Multi-task:
        python train_peract.py task=multi model=peract lang_encoder=clip \
                               mode=train batch_size=8 steps=200000
"""

import os
import time
import hydra
import torch
import numpy as np
from math import floor
from torch.utils.tensorboard import SummaryWriter
from dataset import ArnoldDataset, ArnoldMultiTaskDataset, InstructionEmbedding
from peract.agent import CLIP_encoder, T5_encoder, RoBERTa, PerceiverIO, PerceiverActorAgent
from peract.utils import point_to_voxel_index, normalize_quaternion, quaternion_to_discrete_euler


def create_lang_encoder(cfg, device):
    if cfg.lang_encoder == 'clip':
        return CLIP_encoder(device)
    elif cfg.lang_encoder == 't5':
        return T5_encoder(cfg.t5_cfg, device)
    elif cfg.lang_encoder == 'roberta':
        return RoBERTa(cfg.roberta_cfg, device)
    elif cfg.lang_encoder == 'none':
        return None
    else:
        raise ValueError('Language encoder key not supported')


def create_agent(cfg, device):
    perceiver_encoder = PerceiverIO(
        depth=6,
        iterations=1,
        voxel_size=cfg.voxel_size,
        initial_dim=3 + 3 + 1 + 3,
        low_dim_size=4,
        layer=0,
        num_rotation_classes=72,
        num_grip_classes=2,
        num_state_classes=2,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        activation='lrelu',
        input_dropout=0.1,
        attn_dropout=0.1,
        decoder_dropout=0.0,
        voxel_patch_size=5,
        voxel_patch_stride=5,
        final_dim=64,
        lang_embed_dim=cfg.lang_embed_dim[cfg.lang_encoder],
        with_language=(cfg.lang_encoder != 'none')
    )

    peract_agent = PerceiverActorAgent(
        coordinate_bounds=cfg.offset_bound,
        perceiver_encoder=perceiver_encoder,
        camera_names=cfg.cameras,
        batch_size=cfg.batch_size,
        voxel_size=cfg.voxel_size,
        voxel_feature_size=3,
        num_rotation_classes=72,
        rotation_resolution=5,
        lr=0.0001,
        image_resolution=[cfg.img_size, cfg.img_size],
        lambda_weight_l2=0.000001,
        transform_augmentation=False,
        optimizer_type='lamb',
        state_head=cfg.state_head
    )

    peract_agent.build(training=(cfg.mode == 'train'), device=device)

    return peract_agent


def prepare_batch(batch_data, cfg, lang_embed_cache, device):
    obs_dict = {}
    language_instructions = []
    target_points = []
    gripper_open = []
    low_dim_state = []
    current_states = []
    goal_states = []
    for data in batch_data:
        for k, v in data['obs_dict'].items():
            if k not in obs_dict:
                obs_dict[k] = [v]
            else:
                obs_dict[k].append(v)
        
        target_points.append(data['target_points'])
        gripper_open.append(data['target_gripper'])
        language_instructions.append(data['language'])
        low_dim_state.append(data['low_dim_state'])
        current_states.append(data['current_state'])
        goal_states.append(data['goal_state'])

    for k, v in obs_dict.items():
        v = np.stack(v, axis=0)
        obs_dict[k] = v.transpose(0, 3, 1, 2)   # peract requires input as [C, H, W]
    
    bs = len(language_instructions)
    target_points = np.stack(target_points, axis=0)
    gripper_open = np.array(gripper_open).reshape(bs, 1)
    low_dim_state = np.stack(low_dim_state, axis=0)

    current_states = np.array(current_states).reshape(bs, 1)
    goal_states = np.array(goal_states).reshape(bs, 1)
    states = np.concatenate([current_states, goal_states], axis=1)   # [bs, 2]

    trans_action_coords = target_points[:, :3]
    trans_action_indices = point_to_voxel_index(trans_action_coords, cfg.voxel_size, cfg.offset_bound)

    rot_action_quat = target_points[:, 3:]
    rot_action_quat = normalize_quaternion(rot_action_quat)
    rot_action_indices = quaternion_to_discrete_euler(rot_action_quat, cfg.rotation_resolution)
    rot_grip_action_indices = np.concatenate([rot_action_indices, gripper_open], axis=-1)

    lang_goal_embs = lang_embed_cache.get_lang_embed(language_instructions)

    inp = {}
    inp.update(obs_dict)
    inp.update({
        'trans_action_indices': trans_action_indices,
        'rot_grip_action_indices': rot_grip_action_indices,
        'states': states,
        'lang_goal_embs': lang_goal_embs,
        'low_dim_state': low_dim_state
    })

    for k, v in inp.items():
        if v is not None:
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(v)
            inp[k] = v.to(device)
    
    return inp


@hydra.main(config_path='./configs', config_name='default')
def main(cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg.task != 'multi':
        train_dataset = ArnoldDataset(data_path=os.path.join(cfg.data_root, cfg.task, 'train'), task=cfg.task, obs_type=cfg.obs_type)
    else:
        train_dataset = ArnoldMultiTaskDataset(data_root=cfg.data_root, obs_type=cfg.obs_type)
    
    writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tb_logs'))

    agent = create_agent(cfg, device=device)

    lang_encoder = create_lang_encoder(cfg, device)
    lang_embed_cache = InstructionEmbedding(lang_encoder)

    start_step = 0
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print(f'=> loading checkpoint {cfg.resume}')
            start_step = agent.load_model(cfg.resume)
            print(f'=> loaded checkpoint {cfg.resume} (step {start_step})')
        else:
            print(f'=> no checkpoint found at {cfg.resume}')

    cfg.save_interval = floor(cfg.steps/10)
    print(f'Training {cfg.steps} steps, {len(train_dataset)} demos, {cfg.batch_size} batch_size')
    start_time = time.time()
    for iteration in range(start_step, cfg.steps):
        # train
        batch_data = train_dataset.sample(cfg.batch_size)
        inp = prepare_batch(batch_data, cfg, lang_embed_cache, device)
        update_dict = agent.update(iteration, inp)
        running_loss = update_dict['total_loss']

        if iteration % cfg.log_interval == 0:
            elapsed_time = (time.time() - start_time) / 60.0
            print(f'Iteration: {iteration} | Total Loss: {running_loss} | Elapsed Time: {elapsed_time} mins')
            writer.add_scalar('total_loss', running_loss, iteration)
        
        if (iteration+1) % cfg.save_interval == 0:
            # save for model selection
            ckpt_path = os.path.join(cfg.checkpoint_dir, f'peract_{cfg.task}_{cfg.obs_type}_{cfg.lang_encoder}_{iteration+1}.pth')
            print('Saving checkpoint')
            agent.save_model(ckpt_path, iteration)
    
    writer.close()


if __name__ == '__main__':
    main()
