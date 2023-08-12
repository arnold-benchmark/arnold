import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cliport6d.utils.utils as utils
from cliport6d.models.core.attention import Attention
from cliport6d.models.core.transport import Transport6Dof
from cliport6d.models.streams.one_stream_attention_lang_fusion import OneStreamAttentionLangFusion
from cliport6d.models.streams.one_stream_transport_lang_fusion import OneStreamTransportLangFusion
from cliport6d.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat
from cliport6d.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLat


class TransporterAgent(nn.Module):
    def __init__(self, name, device, cfg):
        super().__init__()
        # self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this is bad for PL :(
        self.device_type = torch.device(device)
        self.name = name
        self.cfg = cfg

        self.total_steps = 0
        if not hasattr(self, 'crop_size'):
            self.crop_size = 64
        if not hasattr(self, 'n_rotations'):
            self.n_rotations = cfg['train']['n_rotations']

        # self.pix_size = 0.005
        self.in_shape = (160, 128, 6)
        # self.bounds = np.array([[-0.075,0.575],[-0.45, 0.45], [0.75, 0.85]])
        self._build_model()

    def _build_model(self):
        pass

    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='sum'):
        # Lucas found that both sum and mean work equally well
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()/labels.shape[0]
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']

        output = self.attention.forward(inp_img, softmax=softmax)
        return output

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']

        inp = {'inp_img': inp_img}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta):
        # Get label.
        # theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        # theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:3] + (self.attention.n_rotations,)
        label = np.zeros(label_size)
        # label[p[0], p[1], theta_i] = 1
        label[:, p[:, 1], p[:, 0]] = 1
        label = label.reshape(label_size[0], -1)
        # label = label.transpose((2, 0, 1))
        # label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)

        # Get loss.
        loss = self.cross_entropy_with_logits(out, label, reduction='sum')

        # # Backpropagate.
        # if backprop:
        #     attn_optim = self._optimizers['attn']
        #     self.manual_backward(loss, attn_optim)
        #     attn_optim.step()
        #     attn_optim.zero_grad()

        # # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            pick_conf = pick_conf[..., 0]
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax
            # p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(p) - p0_pix, ord=1),
                # 'theta': np.absolute((theta - p0_theta) % np.pi)
            }
        return loss#, err

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']

        output = self.transport.forward(inp_img, p0, softmax=softmax)
        return output

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']

        inp = {'inp_img': inp_img, 'p0': p0}
        output = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, output, p0, p1, p1_theta)
        return loss, err

    def transport_criterion(self, backprop, compute_err, inp, output, p, q, theta):
        itheta = theta / (2 * np.pi / self.transport.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.transport.n_rotations

        # Get one-hot pixel label map.
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
        output = output.reshape(1, np.prod(output.shape))
        loss = self.cross_entropy_with_logits(output, label)
        if backprop:
            transport_optim = self._optimizers['trans']
            self.manual_backward(loss, transport_optim)
            transport_optim.step()
            transport_optim.zero_grad()
 
        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            place_conf = self.trans_forward(inp)
            place_conf = place_conf.permute(1, 2, 0)
            place_conf = place_conf.detach().cpu().numpy()
            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(q) - p1_pix, ord=1),
                'theta': np.absolute((theta - p1_theta) % np.pi)
            }
        self.transport.iters += 1
        return err, loss

    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()

        frame, _ = batch

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame)
        else:
            loss1, err1 = self.transport_training_step(frame)
        total_loss = loss0 + loss1
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/loss', total_loss)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def check_save_iteration(self):
        global_step = self.trainer.global_step
        if (global_step + 1) in self.save_steps:
            self.trainer.run_evaluation()
            val_loss = self.trainer.callback_metrics['val_loss']
            steps = f'{global_step + 1:05d}'
            filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
            checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if (global_step + 1) % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
        )

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)['state_dict'])
        self.to(device=self.device_type)

class TransporterAgent_6Dof(TransporterAgent):
    def __init__(self, name, device, cfg):
        self.crop_size = 32
        super().__init__(name, device, cfg)
        self.loss_fuc = torch.nn.HuberLoss(reduction='sum', delta=1.0)
        self.original_loss = False
    
    def _build_model(self):
        stream_fcn = 'plain_resnet'
        self.attention = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = Transport6Dof(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

        self.rpz = Transport6Dof(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
            z_roll_pitch=True
        )
    
    def forward(self, frame, epoch = None, train_xytheta=True, train_rpz=True):
        if epoch is not None:
            utils.set_seed(epoch, torch=True)
        loss = {}
        if train_xytheta:
            loss0 = self.attn_training_step(frame)
            loss1 = self.transport_training_step(frame)
            loss.update(loss0)
            loss.update(loss1)
        if train_rpz:
            loss2 = self.rpz_training_step(frame)
            loss.update(loss2)
        return loss
    
    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0)

    def attn_criterion(self, backprop, compute_err, inp, out, p):
        # Get label.
        # theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        # theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        batch_size = inp_img.shape[0]
        label_size = inp_img.shape[:3] + (self.attention.n_rotations,)
        label = np.zeros(label_size)
        # label = np.ones(label_size)*(-1e-6)
        # label[p[0], p[1], theta_i] = 1
        for i in range(batch_size):
            label[i, p[i, 0], p[i, 1]] = 1
        label = label.reshape(label_size[0], -1)
        # label = label.transpose((2, 0, 1))
        # label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)

        # log
        pred = out.detach().cpu().numpy()
        pred = np.unravel_index(pred.argmax(1), shape=inp_img.shape[1:3])
        pred = np.stack(pred, axis=-1)
        # print(f'training attention:\npred\n{pred}\ngt\n{p}')

        # Get loss.
        loss = self.cross_entropy_with_logits(out, label)
        loss_dict = {"attention_loss": loss}

        return loss_dict
    
    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p1= frame['p0'], frame['p1']
        # p0_z, p0_rotation = frame['p0_z'], frame['p0_rotation']
        p1_z, p1_rotation = frame['p1_z'], frame['p1_rotation']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'p0': p0, 'lang_goal': lang_goal}
        out = self.trans_forward(inp, softmax=False)
        # err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
        loss= {}
        # pick_loss = self.transport_criterion('pick', self.pick_transport, inp, pick_out, p0, p0_z, p0_rotation)
        place_loss = self.transport_criterion(inp, out, p1, p1_z, p1_rotation)
        # loss.update(pick_loss)
        loss.update(place_loss)
        return loss#, err

    def transport_criterion(self, inp, output, q, z, rotation):
        angle = 360//self.n_rotations
        rotation = np.round(rotation/angle).astype('int16')
        rotation[rotation<0] += self.n_rotations
        itheta = rotation[:, 0]
        xy_theta_tensors, z_tensors, roll_tensors, pitch_tensors = output
        inp_img = inp['inp_img']
        batch_size = inp_img.shape[0]

        if not self.original_loss:
            label_size = inp_img.shape[:3]
            label = np.zeros(label_size)
            # label = np.ones(label_size)*(-1e-6)
            for i in range(batch_size):
                label[i, q[i, 0], q[i, 1]] = 1
            theta_label = np.zeros([batch_size, self.n_rotations])
            # theta_label = np.ones([batch_size, self.n_rotations])*(-1e-6)
            for i in range(batch_size):
                theta_label[i, itheta[i]] = 1
            theta_label = torch.from_numpy(theta_label).to(dtype=torch.float, device=xy_theta_tensors.device)

            # log
            pred = xy_theta_tensors.detach().cpu().numpy()
            pred_xy = np.unravel_index(pred.mean(1).reshape(pred.shape[0],-1).argmax(1), shape=inp_img.shape[1:3])
            pred_xy = np.stack(pred_xy, axis=-1)
            pred_theta = pred.mean((2,3)).argmax(1)
            # print(f'training x,y,theta:\npred\n{pred_xy}\n{pred_theta}\ngt\n{q}\n{itheta}')
            
            # Get loss.
            label = label.reshape(label.shape[0], -1)
            label = torch.from_numpy(label).to(dtype=torch.float, device=xy_theta_tensors.device)
            xy_loss = self.cross_entropy_with_logits(xy_theta_tensors.mean(1).reshape(xy_theta_tensors.shape[0],-1), label, reduction='sum')
            xy_loss += self.cross_entropy_with_logits(xy_theta_tensors.mean([2,3]), theta_label, reduction='sum')
        else:
            label_size = inp_img.shape[:3] + (self.n_rotations,)
            label = np.zeros(label_size)
            label = label.transpose((0, 3, 1, 2))
            for i in range(batch_size):
                label[i, itheta[i], q[i, 0], q[i, 1]] = 1
            label = label.reshape(label.shape[0], -1)
            label = torch.from_numpy(label).to(dtype=torch.float, device=xy_theta_tensors.device)
            xy_theta_tensors = xy_theta_tensors.reshape(xy_theta_tensors.shape[0],-1)
            xy_loss = self.cross_entropy_with_logits(xy_theta_tensors, label)
        loss_dict={"xy_loss": xy_loss}
        return loss_dict
    
    def rpz_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p1= frame['p0'], frame['p1']
        # p0_z, p0_rotation = frame['p0_z'], frame['p0_rotation']
        p1_z, p1_rotation = frame['p1_z'], frame['p1_rotation']

        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'p0': p0, 'lang_goal': lang_goal}
        out = self.rpz_forward(inp, softmax=False)
        # err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
        loss= {}
        # pick_loss = self.transport_criterion('pick', self.pick_transport, inp, pick_out, p0, p0_z, p0_rotation)
        place_loss = self.rpz_criterion(inp, out, p1, p1_z, p1_rotation)
        # loss.update(pick_loss)
        loss.update(place_loss)
        return loss#, err
    
    def rpz_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']

        output = self.rpz.forward(inp_img, p0, softmax=softmax)
        return output
    
    def rpz_criterion(self, inp, output, q, z, rotation):
        angle = 360//self.n_rotations
        rotation = np.round(rotation/angle).astype('int16')
        rotation[rotation<0] += self.n_rotations
        itheta = rotation[:, 0]
        xy_theta_tensors, z_tensors, roll_tensors, pitch_tensors = output
        inp_img = inp['inp_img']
        batch_size = inp_img.shape[0]

        u_window = 7
        v_window = 7
        theta_window = 1
        u_min = np.maximum(q[:, 0] - u_window, 0)
        u_max = np.minimum(q[:, 0] + u_window + 1, z_tensors.shape[2])
        v_min = np.maximum(q[:, 1] - v_window, 0)
        v_max = np.minimum(q[:, 1] + v_window + 1, z_tensors.shape[3])
        theta_min = np.maximum(itheta - theta_window, 0)
        theta_max = np.minimum(itheta + theta_window + 1, z_tensors.shape[1])
        z_label = torch.from_numpy(z).to(dtype=torch.float, device=z_tensors.device)
        roll_label = np.zeros((batch_size, self.rpz.n_rotations))
        roll_label[range(batch_size), rotation[range(batch_size),2]] = 1
        roll_label = torch.from_numpy(roll_label).to(dtype=torch.float, device=roll_tensors.device)
        pitch_label = np.zeros((batch_size, self.rpz.n_rotations))
        pitch_label[range(batch_size), rotation[range(batch_size),1]] = 1
        pitch_label = torch.from_numpy(pitch_label).to(dtype=torch.float, device=pitch_tensors.device)
        z_losses, roll_losses, pitch_losses = [],[],[]
        for i in range(batch_size):
            z_i = z_tensors[i, theta_min[i]:theta_max[i], 
                                u_min[i]:u_max[i], v_min[i]:v_max[i]]
            z_i = self.rpz.z_regressor(z_i.reshape(-1, 1)).mean()
            z_loss = self.loss_fuc(100*z_i, 100*z_label[i:i+1])
            z_losses.append(z_loss)
            #roll loss
            roll_i = roll_tensors[i, theta_min[i]:theta_max[i], 
                                u_min[i]:u_max[i], v_min[i]:v_max[i]]
            roll_i = self.rpz.roll_regressor(roll_i.reshape(-1, 1))#.mean(0, keepdims=True)
            roll_loss = self.cross_entropy_with_logits(roll_i, roll_label[i:i+1], reduction='sum')
            roll_losses.append(10*roll_loss)
            #pitch loss
            pitch_i = pitch_tensors[i, theta_min[i]:theta_max[i], 
                                u_min[i]:u_max[i], v_min[i]:v_max[i]]
            pitch_i = self.rpz.pitch_regressor(pitch_i.reshape(-1, 1))#.mean(0, keepdims=True)
            pitch_loss = self.cross_entropy_with_logits(pitch_i, pitch_label[i:i+1], reduction='sum')
            pitch_losses.append(10*pitch_loss)

            # log
            # if i == batch_size - 1:
            #     print(f'training z,pitch,roll: pred=({z_i.item():.4f}, {pitch_i.mean(0).argmax().item()}, {roll_i.mean(0).argmax().item()}), gt=({z_label[i].item():.4f}, {rotation[i,1]}, {rotation[i,2]})\n')
            # else:
            #     print(f'training z,pitch,roll: pred=({z_i.item():.4f}, {pitch_i.mean(0).argmax().item()}, {roll_i.mean(0).argmax().item()}), gt=({z_label[i].item():.4f}, {rotation[i,1]}, {rotation[i,2]})')

        z_losses = torch.stack(z_losses).mean()
        roll_losses = torch.stack(roll_losses).mean()
        pitch_losses = torch.stack(pitch_losses).mean()

        loss_dict={
            "z_loss": z_losses,
            "roll_loss":roll_losses,
            "pitch_losses":pitch_losses
        }
        return loss_dict
    
    def obs_preprocess(self, obs, bounds, pixel_size):
        front_rgb = obs.front_rgb
        base_rgb = obs.base_rgb
        left_rgb = obs.left_rgb
        wrist_bottom_rgb = obs.wrist_bottom_rgb
        wrist_rgb = obs.wrist_rgb

        front_point_cloud = obs.front_point_cloud
        base_point_cloud = obs.base_point_cloud
        left_point_cloud = obs.left_point_cloud
        wrist_bottom_point_cloud = obs.wrist_bottom_point_cloud
        wrist_point_cloud = obs.wrist_point_cloud

        colors = [front_rgb, base_rgb, left_rgb, wrist_bottom_rgb, wrist_rgb]
        pcds = [front_point_cloud, base_point_cloud, left_point_cloud, wrist_bottom_point_cloud, wrist_point_cloud]

        cmap, hmap = utils.get_fused_heightmap(colors, pcds, bounds, pixel_size)
        hmap = np.tile(hmap[..., None], (1,1,3))
        img = np.concatenate([cmap, hmap], axis=-1)
        img = img[None,...]
        return img

    @torch.no_grad()
    def act(self, obs, lang_goal, goal=None, bounds=np.array([[-0.11,0.61],[-0.45, 0.45], [0.7, 1.5]]), pixel_size=5.625e-3, draw_result=True):
        img = self.obs_preprocess(obs, bounds, pixel_size)
        # Attention model forward pass.
        attn_inp = {'inp_img': img, 'lang_goal': [lang_goal[0]]}
        attn_conf = self.attn_forward(attn_inp)
        attn_conf = attn_conf.detach().cpu().numpy()
        attn_conf = attn_conf[0,:,:,0]
        argmax = np.argmax(attn_conf)
        argmax = np.unravel_index(argmax, shape=attn_conf.shape)

        p0 = np.array(argmax[:2])[None,...]
        place_inp = {'inp_img': img, 'p0': p0, 'lang_goal': [lang_goal[0]]}
        trans_config = self.trans_forward(place_inp, False)
        rpz_config = self.rpz_forward(place_inp, False)
        # place_xy_theta_tensors, place_z_tensors, place_roll_tensors, place_pitch_tensors = trans_config
        xy_theta_tensors = trans_config[0]
        _, z_tensors, roll_tensors, pitch_tensors = rpz_config
        xy_theta_tensors = xy_theta_tensors[0].permute(1, 2, 0).detach().cpu().numpy()

        angle = 2 * np.pi / xy_theta_tensors.shape[2]
        if self.original_loss:
            place_argmax = np.argmax(xy_theta_tensors)
            place_argmax = np.unravel_index(place_argmax, shape=xy_theta_tensors.shape)
            place_xy = (np.array(place_argmax[:2])*pixel_size)[::-1]+bounds[:2,0]
            place_theta = place_argmax[2] * angle
            place_theta_argmax = [place_argmax[2]]
        else:
            predict_xy = xy_theta_tensors.mean(-1)
            place_argmax = np.argmax(predict_xy)
            place_argmax = np.unravel_index(place_argmax, shape=predict_xy.shape)
            place_xy = (np.array(place_argmax[:2])*pixel_size)[::-1]+bounds[:2,0]
            predict_theta = xy_theta_tensors.mean((0,1))
            place_theta_argmax = np.argmax(predict_theta)
            place_theta_argmax = np.unravel_index(place_theta_argmax, shape=predict_theta.shape)
            place_theta = place_theta_argmax[0] * angle
        output_dict = {
            "place_xy":place_xy,
            "place_theta":place_theta
        }
        
        u_window = 7
        v_window = 7
        theta_window = 1
        u_min = np.maximum(place_argmax[0] - u_window, 0)
        u_max = np.minimum(place_argmax[0] + u_window + 1, xy_theta_tensors.shape[0])
        v_min = np.maximum(place_argmax[1] - v_window, 0)
        v_max = np.minimum(place_argmax[1] + v_window + 1, xy_theta_tensors.shape[1])
        theta_min = np.maximum(place_theta_argmax[0] - theta_window, 0)
        theta_max = np.minimum(place_theta_argmax[0] + theta_window + 1, xy_theta_tensors.shape[2])
        z_i = z_tensors[0, theta_min:theta_max, u_min:u_max, v_min:v_max]
        # z_i = place_z_tensors[0:1, place_theta_argmax[0], place_argmax[0], place_argmax[1]]
        z_i = self.rpz.z_regressor(z_i.reshape(-1, 1)).mean()
        place_z = z_i+bounds[2,0]
        output_dict.update({"place_z": place_z.item()})

        roll_i =  roll_tensors[0, theta_min:theta_max, u_min:u_max, v_min:v_max]
        roll_i = self.rpz.roll_regressor(roll_i.reshape(-1, 1)).mean(0)
        roll_i = torch.argmax(roll_i).item()*angle
        pitch_i =  pitch_tensors[0, theta_min:theta_max, u_min:u_max, v_min:v_max]
        pitch_i = self.rpz.pitch_regressor(pitch_i.reshape(-1,1)).mean(0)
        pitch_i = torch.argmax(pitch_i).item()*angle
        output_dict.update({"roll": roll_i, "pitch": pitch_i})
        if draw_result:
            index = lang_goal[0].split("Step")[-1][:-1].strip()
            attn_conf = (attn_conf-attn_conf.min())/(attn_conf.max()-attn_conf.min())
            cv2.imwrite(f'./results/atten_map_predict_{index}.png', np.uint8(attn_conf*255))
            center_coordinates = (argmax[1], argmax[0])
            # print("predict:{}".format(center_coordinates))
            # Radius of circle
            radius = 4
            # Blue color in BGR
            color = (0, 255, 255)
            # Line thickness of 2 px
            thickness = -1
            # Using cv2.circle() method
            # Draw a circle with blue line borders of thickness of 2 px
            image = np.uint8(img[0,:,:,:3])
            image = cv2.circle(image, center_coordinates, radius, color, thickness)
            predict_xy = xy_theta_tensors.mean(-1)
            predict_img = (predict_xy-predict_xy.min())/(predict_xy.max()-predict_xy.min())
            cv2.imwrite(f'./results/place_map_predict_{index}.png', np.uint8(predict_img*255))

            center_coordinates = (place_argmax[1],place_argmax[0])
            color = (140,140,255)
            radius=2
            image_place = cv2.circle(image, center_coordinates, radius, color, thickness)
            cv2.imwrite(f'./results/place_map_{index}.png', cv2.cvtColor(image_place,cv2.COLOR_RGB2BGR))
        return img, [lang_goal[0]], None, output_dict

class TwoStreamClipLingUNetLatTransporterAgent(TransporterAgent_6Dof):
    def __init__(self, name, device, cfg, z_roll_pitch):
        self.crop_size = 32
        # self.n_rotations = 1
        # utils.set_seed(0)
        self.z_roll_pitch = z_roll_pitch
        super().__init__(name, device, cfg)
    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type
        )
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

        self.rpz = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
            z_roll_pitch=True
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']
        lang_goal = inp['lang_goal']

        out = self.transport.forward(inp_img, p0, lang_goal, softmax=softmax)
        return out
    
    def rpz_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']
        lang_goal = inp['lang_goal']

        output = self.rpz.forward(inp_img, p0, lang_goal, softmax=softmax)
        return output

class ImgDepthAgent_6dof(TwoStreamClipLingUNetLatTransporterAgent):
    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = ["Step"+l.split('Step')[-1] for l in inp['lang_goal']]

        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']
        lang_goal = ["Step"+l.split('Step')[-1] for l in inp['lang_goal']]

        out = self.transport.forward(inp_img, p0, lang_goal, softmax=softmax)
        return out
    
    def rpz_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']
        lang_goal = ["Step"+l.split('Step')[-1] for l in inp['lang_goal']]

        output = self.rpz.forward(inp_img, p0, lang_goal, softmax=softmax)
        return output

class BlindLangAgent_6Dof(TransporterAgent_6Dof):
    def __init__(self, name, device, cfg):
        super().__init__(name, device, cfg)
    
    def _build_model(self):
        stream_fcn = 'clip_lingunet'
        self.attention = OneStreamAttentionLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.rpz = OneStreamTransportLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
            z_roll_pitch=True
        )
    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        inp_img = np.zeros_like(inp_img)
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        inp_img = np.zeros_like(inp_img)
        p0 = inp['p0']
        lang_goal = inp['lang_goal']

        out = self.transport.forward(inp_img, p0, lang_goal, softmax=softmax)
        return out
    
    def rpz_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        inp_img = np.zeros_like(inp_img)
        p0 = inp['p0']
        lang_goal = inp['lang_goal']

        output = self.rpz.forward(inp_img, p0, lang_goal, softmax=softmax)
        return output
