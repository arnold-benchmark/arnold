"""
Script for model selection. For example, run:
    Single-task:
        python ckpt_selection.py task=pickup_object model=peract lang_encoder=clip \
                                 mode=eval visualize=0
    Multi-task:
        python ckpt_selection.py task=multi model=peract lang_encoder=clip \
                                 mode=eval visualize=0
"""
import hydra
import json
import logging
import numpy as np
import os
import shutil
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from environment.runner_utils import get_simulation
simulation_app, simulation_context, _ = get_simulation(headless=True, gpu_id=0)

from dataset import InstructionEmbedding
from tasks import load_task
from utils.env import get_action

logger = logging.getLogger(__name__)


def load_data(data_path):
    demos = list(Path(data_path).iterdir())
    demo_path = sorted([str(item) for item in demos if not item.is_dir()])
    data = []
    for npz_path in demo_path:
        data.append(np.load(npz_path, allow_pickle=True))
    return data


def make_agent(cfg, device):
    lang_embed_cache = None
    if cfg.model == 'cliport6d':
        from cliport6d.agent import TwoStreamClipLingUNetLatTransporterAgent
        agent = TwoStreamClipLingUNetLatTransporterAgent(name='cliport_6dof', device=device, cfg=cfg.cliport6d, z_roll_pitch=True)
        agent.eval()
        agent.to(device)
    
    elif cfg.model == 'peract':
        from train_peract import create_agent, create_lang_encoder
        agent = create_agent(cfg, device=device)

        lang_encoder = create_lang_encoder(cfg, device=device)
        lang_embed_cache = InstructionEmbedding(lang_encoder)
    elif 'bc_lang' in cfg.model:
        from train_bc_lang import create_agent, create_lang_encoder
        agent = create_agent(cfg, device=device)

        lang_encoder = create_lang_encoder(cfg, device=device)
        lang_embed_cache = InstructionEmbedding(lang_encoder)
    
    else:
        raise ValueError(f'{cfg.model} agent not supported')
    
    return agent, lang_embed_cache


def load_ckpt(cfg, agent, ckpt_path, device):
    if cfg.model == 'cliport6d':
        checkpoint = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(checkpoint['state_dict'])
        agent.eval()
        agent.to(device)
    elif cfg.model == 'peract':
        agent.load_model(ckpt_path)
    elif 'bc_lang' in cfg.model:
        agent.load_weights(ckpt_path)
    else:
        raise ValueError(f'{cfg.model} agent not supported')
    
    logger.info(f"Loaded {cfg.model} from {ckpt_path}")
    return agent


@hydra.main(config_path='./configs', config_name='default')
def main(cfg):
    cfg.checkpoint_dir = cfg.checkpoint_dir.split(os.path.sep)
    cfg.checkpoint_dir[-2] = cfg.checkpoint_dir[-2].replace('eval', 'train')
    cfg.checkpoint_dir = os.path.sep.join(cfg.checkpoint_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    render = cfg.visualize

    offset = cfg.offset_bound
    use_gt = cfg.use_gt
    agent, lang_embed_cache = make_agent(cfg, device=device)
    
    if cfg.task != 'multi':
        task_list = [cfg.task]
    else:
        task_list = [
            'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
            'open_cabinet', 'close_cabinet', 'pour_water', 'transfer_water'
        ]

    ckpts_list = [f for f in os.listdir(cfg.checkpoint_dir) if f.endswith('pth')]
    for ckpt_name in ckpts_list:
        if 'best' in ckpt_name:
            logger.info('Best checkpoint already recognized')
            simulation_app.close()
            return 1
    ckpts_list = sorted(ckpts_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    log_path = os.path.join(cfg.exp_dir, 'select_log.json')
    """
    val log structure:
    {
        'ckpt_name': {
            'task': {
                # 'stats': {
                #     'fname': int (1, 0, -1)
                # },
                'score': float
            }
        }
    }
    """

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            val_log = json.load(f)
    else:
        val_log = {}
    
    for ckpt_name in ckpts_list:
        agent = load_ckpt(cfg, agent, os.path.join(cfg.checkpoint_dir, ckpt_name), device)
        if ckpt_name not in val_log:
            val_log[ckpt_name] = {}
        for task_name in task_list:
            if task_name not in val_log[ckpt_name]:
                val_log[ckpt_name][task_name] = {}
            elif 'score' in val_log[ckpt_name][task_name]:
                continue

            logger.info(f'Evaluating {ckpt_name} {task_name}')
            data = load_data(data_path=os.path.join(cfg.data_root, task_name, 'val'))
            correct = 0
            total = 0
            while len(data) > 0:
                anno = data.pop(0)
                gt_frames = anno['gt']
                robot_base = gt_frames[0]['robot_base']
                gt_actions = [
                    gt_frames[1]['position_rotation_world'], gt_frames[2]['position_rotation_world'],
                    gt_frames[3]['position_rotation_world'] if 'water' not in task_name \
                    else (gt_frames[3]['position_rotation_world'][0], gt_frames[4]['position_rotation_world'][1])
                ]

                env, object_parameters, robot_parameters, scene_parameters = load_task(cfg.asset_root, npz=anno, cfg=cfg)

                obs = env.reset(robot_parameters, scene_parameters, object_parameters, 
                                robot_base=robot_base, gt_actions=gt_actions)

                logger.info(f'Instruction: {gt_frames[0]["instruction"]}')
                logger.info('Ground truth action:')
                for gt_action, grip_open in zip(gt_actions, cfg.gripper_open[task_name]):
                    act_pos, act_rot = gt_action
                    act_rot = R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)
                    logger.info(f'trans={act_pos}, orient(euler XYZ)={act_rot}, gripper_open={grip_open}')

                try:
                    for i in range(2):
                        if use_gt[i]:
                            obs, suc = env.step(act_pos=None, act_rot=None, render=render, use_gt=True)
                        else:
                            act_pos, act_rot = get_action(
                                gt=obs, agent=agent, franka=env.robot, c_controller=env.c_controller, npz_file=anno, offset=offset, timestep=i,
                                device=device, agent_type=cfg.model, obs_type=cfg.obs_type, lang_embed_cache=lang_embed_cache
                            )

                            logger.info(
                                f"Prediction action {i}: trans={act_pos}, orient(euler XYZ)={R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)}"
                            )

                            obs, suc = env.step(act_pos=act_pos, act_rot=act_rot, render=render, use_gt=False)

                        if suc == -1:
                            break
                
                except:
                    suc = -1

                env.stop()
                if suc == 1:
                    correct += 1
                total += 1
                log_str = f'correct: {correct} | total: {total} | remaining: {len(data)}'
                logger.info(f'{log_str}\n')
            
            logger.info(f'{ckpt_name} {task_name}: {correct/total*100:.2f}\n\n')
            val_log[ckpt_name][task_name]['score'] = correct / total
            
            with open(log_path, 'w') as f:
                json.dump(val_log, f, indent=2)

    ckpt_scores = [np.mean([val_log[ckpt_name][task_name]['score'] for task_name in task_list]) for ckpt_name in ckpts_list]
    selected_idx = ckpt_scores.argmax()
    selected_name = ckpts_list[selected_idx]

    for ckpt_name in ckpts_list:
        if ckpt_name == selected_name:
            new_name = ckpt_name.split('_')
            new_name[-1] = 'best.pth'
            new_name = '_'.join(new_name)
            shutil.move(os.path.join(cfg.checkpoint_dir, ckpt_name), os.path.join(cfg.checkpoint_dir, new_name))
            logger.info(f'Select {selected_name} as best')
        # else:
        #     os.remove(os.path.join(cfg.checkpoint_dir, ckpt_name))
        #     logger.info(f'Remove {ckpt_name}')

    simulation_app.close()


if __name__ == '__main__':
    main()
