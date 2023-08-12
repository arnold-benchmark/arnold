"""
For example, run:
    python train_cliport6d.py task=pickup_object model=cliport6d \
                              mode=train batch_size=8 steps=100000
"""

import os
import time
import hydra
import torch
import datetime
import warnings
import numpy as np
from math import ceil, floor
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R
from dataset import ArnoldDataset
from cliport6d.agent import TwoStreamClipLingUNetLatTransporterAgent
warnings.filterwarnings('ignore')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def sec_to_str(delta):
    t = datetime.timedelta(seconds=delta)
    s = str(t)
    return s.split(".")[0] + "s"


def train(dataset, model, optimizer, scheduler, epoch, losses, cfg, timer, writer):
    batch_time = timer["batch_time"]
    data_time = timer["data_time"]
    model.train()
    end = time.time()
    steps_per_epoch = ceil(len(dataset)/cfg.batch_size)
    for batch_step in range(steps_per_epoch):
        batch_data = dataset.sample(cfg.batch_size)
        data_time.update(time.time() - end)
        loss_dict = {}
        if len(batch_data)==0:
            continue

        pixel_size = batch_data[0]['pixel_size']
        bounds = []
        img, language_instructions = [], []
        attention_points, target_points = [], []
        for data in batch_data:
            bounds.append(data['bounds'])
            img.append(data['img'])
            language_instructions.append(data['language'])
            attention_points.append(data['attention_points'])
            target_points.append(data['target_points'])
        img = np.stack(img, axis=0)
        attention_points = np.stack(attention_points, axis=0)
        target_points = np.stack(target_points, axis=0)
        bounds = np.stack(bounds, axis=0)

        # y,z axis swapped
        p0 = np.int16((attention_points[:, [0,2]]-bounds[:, [0,2], 0])/pixel_size)
        p0_z = attention_points[:, 1]-bounds[:, 1, 0]

        p1 = np.int16((target_points[:, [0,2]]-bounds[:, [0,2], 0])/pixel_size)
        p1_z = target_points[:, 1]-bounds[:, 1, 0]

        p0 = p0[:,::-1]
        p1 = p1[:,::-1]

        p1_rotation = target_points[:, 3:]
        # [yaw, pitch, roll] (zyx) in z-up to [yaw, pitch, roll] in y-up
        p1_rotation = R.from_quat(p1_rotation).as_matrix()
        rot_transition = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float)
        p1_rotation = R.from_matrix(rot_transition @ p1_rotation).as_euler('zyx', degrees=True)

        inp = {
            'img': img, 'lang_goal': language_instructions,
            'p0': p0, 'p0_z': p0_z, 'p1': p1, 'p1_z': p1_z, 'p1_rotation': p1_rotation
        }
        loss_dict = model(inp)

        if losses == {}:
            for loss_term in loss_dict:
                losses[loss_term] = AverageMeter(loss_term)
        for loss_term in loss_dict:
            losses[loss_term].update(loss_dict[loss_term].item(), cfg.batch_size)
        loss = sum(l for l in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        batch_time.update(time.time() - end)
        end = time.time()
        # Calculate time remaining.
        time_per_epoch = batch_time.avg * steps_per_epoch
        epochs_left = cfg.epochs - epoch - 1
        batches_left = steps_per_epoch - batch_step - 1

        time_left = sec_to_str(batches_left * batch_time.avg + epochs_left * time_per_epoch)
        time_elapsed = sec_to_str(batch_time.sum)
        time_estimate = sec_to_str(cfg.epochs * time_per_epoch)

        if batch_step % cfg.log_interval == 0:
            tmp_str = 'Epoch: [{}/{}] Batch: [{}/{}]  ' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  ' \
                        'Elapsed: {}  ' \
                        'ETA: {} / {}  ' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\n'.format(
                epoch+1, cfg.epochs, batch_step, steps_per_epoch, time_elapsed, time_left, time_estimate,
                batch_time=batch_time, data_time=data_time)
            
            tmp_str += f'total_loss: {loss.item():.4f}  '
            writer.add_scalar('total_loss', loss.item(), batch_time.count)

            for loss_term in losses:
                tmp_str += '{}: {loss.val:.4f} ({loss.avg:.4f})  '.format(loss_term, loss=losses[loss_term])
                writer.add_scalar(loss_term, losses[loss_term].val, batch_time.count)
            
            print(tmp_str)


def val(data_loader, model, cfg, epoch):
    losses= {}
    total_loss = []
    model.eval()
    for batch_step, batch_data in enumerate(data_loader):
        if len(batch_data)==0:
            continue

        pixel_size = batch_data[0]['pixel_size']
        bounds = []
        img, language_instructions = [], []
        attention_points, target_points = [], []
        for data in batch_data:
            bounds.append(data['bounds'])
            img.append(data['img'])
            language_instructions.append(data['language'])
            attention_points.append(data['attention_points'])
            target_points.append(data['target_points'])
        img = np.stack(img, axis=0)
        attention_points = np.stack(attention_points, axis=0)
        target_points = np.stack(target_points, axis=0)
        bounds = np.stack(bounds, axis=0)
        
        # y,z axis swapped
        p0 = np.int16((attention_points[:, [0,2]]-bounds[:, [0,2], 0])/pixel_size)
        p0_z = attention_points[:, 1]-bounds[:, 1, 0]

        p1 = np.int16((target_points[:, [0,2]]-bounds[:, [0,2], 0])/pixel_size)
        p1_z = target_points[:, 1]-bounds[:, 1, 0]

        p0 = p0[:,::-1]
        p1 = p1[:,::-1]

        p1_rotation = target_points[:, 3:]
        # [yaw, pitch, roll] (zyx) in z-up to [yaw, pitch, roll] in y-up
        p1_rotation = R.from_quat(p1_rotation).as_matrix()
        rot_transition = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float)
        p1_rotation = R.from_matrix(rot_transition @ p1_rotation).as_euler('zyx', degrees=True)

        inp = {
            'img': img, 'lang_goal': language_instructions,
            'p0': p0, 'p0_z': p0_z, 'p1': p1, 'p1_z': p1_z, 'p1_rotation': p1_rotation
        }
        with torch.no_grad():
            loss_dict = model(inp)
        if losses == {}:
            for loss_term in loss_dict:
                losses[loss_term] = AverageMeter(loss_term)
        for loss_term in loss_dict:
            losses[loss_term].update(loss_dict[loss_term].item(), cfg.batch_size)
        loss = sum(l.item() for l in loss_dict.values())
        total_loss.append(loss)
    avg_loss = torch.tensor(total_loss).mean(0, keepdim=True).cuda()

    all_avg_loss = torch.tensor(avg_loss).mean().item()
    tmp_str = 'Epoch [{}/{}] Val_loss: {:.4f} '.format(epoch + 1, cfg.epochs, all_avg_loss)
    for loss_term in losses:
        tmp_str += '{}: {loss.val:.4f} ({loss.avg:.4f})  '.format(loss_term, loss=losses[loss_term])
    print(f'{tmp_str}\n\n')
    return all_avg_loss


@hydra.main(config_path='./configs', config_name='default')
def main(cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TwoStreamClipLingUNetLatTransporterAgent(name='cliport_6dof', device=device, cfg=cfg.cliport6d, z_roll_pitch=True)
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    start_epoch = 0
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print(f'=> loading checkpoint {cfg.resume}')
            checkpoint = torch.load(cfg.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print(f'=> loaded checkpoint {cfg.resume} (epoch {start_epoch})')
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print(f'=> no checkpoint found at {cfg.resume}')
    
    cudnn.benchmark = True
    train_dataset = ArnoldDataset(data_path=os.path.join(cfg.data_root, cfg.task, 'train'), task=cfg.task, obs_type=cfg.obs_type)
    
    writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tb_logs'))

    losses = {}
    timer = {
        "batch_time": AverageMeter('Time', ':6.3f'),
        "data_time": AverageMeter('Data', ':6.3f')
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    cfg.epochs = ceil( cfg.steps / (len(train_dataset) / cfg.batch_size) )
    save_interval = floor(cfg.epochs/10)
    saved_ckpt = 0
    print(f'Training epochs: {cfg.steps} steps / ({len(train_dataset)} demos / {cfg.batch_size} batch_size) = {cfg.epochs}')
    for epoch in range(start_epoch, cfg.epochs):
        # train for one epoch
        train(train_dataset, model, optimizer, scheduler, epoch, losses, cfg, timer, writer)

        if (epoch+1) % save_interval == 0:
            save_name = os.path.join(cfg.checkpoint_dir, f'cliport6d_{cfg.task}_{cfg.obs_type}_{epoch}.pth')
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'train_tasks': cfg.task
            }, save_name)
            saved_ckpt += 1
        
        if saved_ckpt >= 10:
            break
    
    writer.close()


if __name__== '__main__':
    main()
