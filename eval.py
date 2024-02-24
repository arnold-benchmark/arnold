"""
For example, run:
    Single-task:
        python eval.py task=pickup_object model=peract lang_encoder=clip \
                       mode=eval use_gt=[0,0] visualize=0
    Multi-task:
        python eval.py task=multi model=peract lang_encoder=clip \
                       mode=eval use_gt=[0,0] visualize=0
"""
import hydra
import json
import logging
import numpy as np
import os
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from environment.runner_utils import get_simulation
simulation_app, simulation_context, _ = get_simulation(headless=False, gpu_id=0)

from dataset import InstructionEmbedding
from tasks import load_task
from utils.env import get_action

logger = logging.getLogger(__name__)


def load_data(data_path):
    demos = list(Path(data_path).iterdir())
    demo_path = sorted([str(item) for item in demos if not item.is_dir()])
    data = []
    fnames = []

    for npz_path in demo_path:
        data.append(np.load(npz_path, allow_pickle=True))
        fnames.append(npz_path)
    return data, fnames


def load_agent(cfg, device):
    checkpoint_path = None
    for fname in os.listdir(cfg.checkpoint_dir):
        if fname.endswith('best.pth'):
            checkpoint_path = os.path.join(cfg.checkpoint_dir, fname)
    
    assert checkpoint_path is not None, "best checkpoint not found"
    lang_embed_cache = None
    if cfg.model == 'cliport6d':
        from cliport6d.agent import TwoStreamClipLingUNetLatTransporterAgent
        agent = TwoStreamClipLingUNetLatTransporterAgent(name='cliport_6dof', device=device, cfg=cfg.cliport6d, z_roll_pitch=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(checkpoint['state_dict'])
        agent.eval()
        agent.to(device)
    
    elif cfg.model == 'peract':
        from train_peract import create_agent, create_lang_encoder
        agent = create_agent(cfg, device=device)
        agent.load_model(checkpoint_path)

        lang_encoder = create_lang_encoder(cfg, device=device)
        lang_embed_cache = InstructionEmbedding(lang_encoder)

    elif 'bc_lang' in cfg.model:
        from train_bc_lang import create_agent, create_lang_encoder
        agent = create_agent(cfg, device=device)
        agent.load_weights(checkpoint_path)

        lang_encoder = create_lang_encoder(cfg, device=device)
        lang_embed_cache = InstructionEmbedding(lang_encoder)

    else:
        raise ValueError(f'{cfg.model} agent not supported')
    
    logger.info(f"Loaded {cfg.model} from {checkpoint_path}")
    return agent, lang_embed_cache


@hydra.main(config_path='./configs', config_name='default')
def main(cfg):
    cfg.checkpoint_dir = cfg.checkpoint_dir.split(os.path.sep)
    cfg.checkpoint_dir[-2] = cfg.checkpoint_dir[-2].replace('eval', 'train')
    cfg.checkpoint_dir = os.path.sep.join(cfg.checkpoint_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    render = cfg.visualize

    offset = cfg.offset_bound
    use_gt = cfg.use_gt
    if not (use_gt[0] and use_gt[1]):
        agent, lang_embed_cache = load_agent(cfg, device=device)

    if cfg.task != 'multi':
        task_list = [cfg.task]
    else:
        task_list = [
            'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
            'open_cabinet', 'close_cabinet', 'pour_water', 'transfer_water'
        ]

    if use_gt[0]:
        if use_gt[1]:
            eval_setting = '2gt'
        else:
            eval_setting = '1gt'
    else:
        eval_setting = '0gt'
    log_path = os.path.join(cfg.exp_dir, f'eval_{eval_setting}_log.json')

    """
    eval log structure:
    {
        'task_name': {
            'split': {
                'stats': {
                    'fname': int (1, 0, -1)
                },
                'score': float
            }
        }
    }
    """

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            eval_log = json.load(f)
    else:
        eval_log = {}

    for task in task_list:
        if task not in eval_log:
            eval_log[task] = {}
        for eval_split in cfg.eval_splits:
            if eval_split not in eval_log[task]:
                eval_log[task][eval_split] = {}
            elif 'score' in eval_log[task][eval_split]:
                continue
            
            if os.path.exists(os.path.join(cfg.data_root, task, eval_split)):
                logger.info(f'Evaluating {task} {eval_split}')
                data, fnames = load_data(data_path=os.path.join(cfg.data_root, task, eval_split))
            else:
                logger.info(f'{eval_split} not exist')
                continue

            correct = 0
            total = 0
            stats = {}
            while len(data) > 0:
                anno = data.pop(0)
                fname = fnames.pop(0)
                gt_frames = anno['gt']
                robot_base = gt_frames[0]['robot_base']

                gt_actions = [gt_frames[1]['position_rotation_world'], gt_frames[2]['position_rotation_world']]
                if gt_frames[3]['position_rotation_world'] is not None:
                    gt_actions.append(
                        gt_frames[3]['position_rotation_world'] if 'water' not in task \
                        else (gt_frames[3]['position_rotation_world'][0], gt_frames[4]['position_rotation_world'][1])
                    )
                else:
                    gt_actions.append(None)

                if use_gt[0]:
                    assert gt_actions[0] is not None and gt_actions[1] is not None, "Use first gt action but it is missing"
                if use_gt[1]:
                    assert gt_actions[2] is not None, "Use second gt action but it is missing"

                env, object_parameters, robot_parameters, scene_parameters = load_task(cfg.asset_root, npz=anno, cfg=cfg)

                obs = env.reset(robot_parameters, scene_parameters, object_parameters, 
                                robot_base=robot_base, gt_actions=gt_actions)

                logger.info(f'Instruction: {gt_frames[0]["instruction"]}')
                logger.info('Ground truth action:')
                for gt_action, grip_open in zip(gt_actions, cfg.gripper_open[task]):
                    if gt_action is None:
                        continue
                    act_pos, act_rot = gt_action
                    act_rot = R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)
                    logger.info(f'trans={act_pos}, orient(euler XYZ)={act_rot}, gripper_open={grip_open}')

                if cfg.record:
                    env.recorder.start_record(
                        traj_dir=os.path.join(cfg.exp_dir, f'traj_{eval_setting}', os.path.split(fname)[-1]),
                        checker=env.checker,
                    )

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
                else:
                    logger.info(f'{fname}: {suc}')
                total += 1
                log_str = f'correct: {correct} | total: {total} | remaining: {len(data)}'
                logger.info(f'{log_str}\n')
                stats[fname] = suc

            logger.info(f'{task} {eval_split} score: {correct/total*100:.2f}\n\n')
            eval_log[task][eval_split]['stats'] = stats
            eval_log[task][eval_split]['score'] = correct / total

            with open(log_path, 'w') as f:
                json.dump(eval_log, f, indent=2)

    simulation_app.close()


if __name__ == '__main__':
    main()
