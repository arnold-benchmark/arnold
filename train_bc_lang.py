"""
For example, run:
    Single-task:
        python train_bc_lang.py task=pickup_object model=bc_lang_vit \
                                mode=train batch_size=8 steps=100000
    Multi-task:
        python train_bc_lang.py task=multi model=bc_lang_vit \
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
from peract.utils import normalize_quaternion
from bc_z.blocks import SiameseNet, CNNLangAndFcsNet, ViT, ViTLangAndFcsNet
from bc_z.agent import CLIP_encoder, BCLangAgent


def create_lang_encoder(cfg, device):
    return CLIP_encoder(device)


def create_agent(cfg, device: str):
    if 'cnn' in cfg.model:
        siamese_net = SiameseNet(
            input_channels=[3, 3],
            filters=[16],
            kernel_sizes=[5],
            strides=[1],
            activation=cfg.bc_lang.activation,
            norm=None,
        )
        siamese_net.build()

        actor_net = CNNLangAndFcsNet(
            siamese_net=siamese_net,
            input_resolution=cfg.bc_lang.image_resolution,
            filters=[32, 64, 64],
            kernel_sizes=[3, 3, 3],
            strides=[2, 2, 2],
            norm=None,
            activation=cfg.bc_lang.activation,
            fc_layers=[128, 64, 3 + 4 + 1],
            low_dim_state_len=4
        )
    
    elif 'vit' in cfg.model:
        vit = ViT(
            image_size=128,
            patch_size=8,
            num_classes=16,
            dim=64,
            depth=6,
            heads=8,
            mlp_dim=64,
            dropout=0.1,
            emb_dropout=0.1,
            channels=6,
        )

        actor_net = ViTLangAndFcsNet(
            vit=vit,
            input_resolution=cfg.bc_lang.image_resolution,
            filters=[64, 96, 128],
            kernel_sizes=[1, 1, 1],
            strides=[1, 1, 1],
            norm=None,
            activation=cfg.bc_lang.activation,
            fc_layers=[128, 64, 3 + 4 + 1],
            low_dim_state_len=4
        )
    
    actor_net.build()

    bc_agent = BCLangAgent(
        actor_network=actor_net,
        camera_name=cfg.bc_lang.camera_name,
        lr=cfg.bc_lang.lr,
        weight_decay=cfg.bc_lang.weight_decay,
        grad_clip=cfg.bc_lang.grad_clip)
    
    bc_agent.build(training=(cfg.mode == 'train'), device=device)

    return bc_agent


def prepare_batch(batch_data, cfg, lang_embed_cache, device):
    obs_dict = {}
    language_instructions = []
    target_points = []
    gripper_open = []
    low_dim_state = []
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

    for k, v in obs_dict.items():
        v = np.stack(v, axis=0)
        obs_dict[k] = v.transpose(0, 3, 1, 2)   # peract requires input as [C, H, W]
    
    bs = len(language_instructions)
    target_points = np.stack(target_points, axis=0)
    gripper_open = np.array(gripper_open).reshape(bs, 1)
    low_dim_state = np.stack(low_dim_state, axis=0)

    trans_action_coords = target_points[:, :3]

    rot_action_quat = target_points[:, 3:]
    rot_action_quat = normalize_quaternion(rot_action_quat)

    lang_goal_embs = lang_embed_cache.get_lang_embed(language_instructions)

    inp = {}
    inp.update(obs_dict)
    inp.update({
        'action': np.concatenate([trans_action_coords, rot_action_quat, gripper_open], axis=-1),
        'lang_goal_embs': lang_goal_embs,
        'low_dim_state': low_dim_state,
        'bound': np.array(cfg.offset_bound).reshape(2, 3)
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
            start_step = agent.load_weights(cfg.resume)
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
            ckpt_path = os.path.join(cfg.checkpoint_dir, f'{cfg.model}_{cfg.task}_{cfg.obs_type}_{cfg.lang_encoder}_{iteration+1}.pth')
            print('Saving checkpoint')
            agent.save_weights(ckpt_path, iteration)
    
    writer.close()


if __name__ == '__main__':
    main()
