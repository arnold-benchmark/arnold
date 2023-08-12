import clip
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger(__name__)
NAME = 'BCLangAgent'


class CLIP_encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, preprocess = clip.load("RN50", device=device, jit=False)
    
    @torch.no_grad()
    def encode_text(self, text):
        tokens = clip.tokenize(text)
        tokens = tokens.to(self.device)
        return self.model.encode_text(tokens)


class BCLangAgent(nn.Module):
    def __init__(self,
                 actor_network: nn.Module,
                 camera_name: str,
                 lr: float = 0.01,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 20.0):
        super().__init__()
        self._camera_name = camera_name
        self._actor = actor_network
        self._lr = lr
        self._weight_decay = weight_decay
        self._grad_clip = grad_clip

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')
        self._actor = self._actor.to(device).train(training)
        if training:
            self._actor_optimizer = torch.optim.Adam(
                self._actor.parameters(), lr=self._lr,
                weight_decay=self._weight_decay)
            logger.info('# Actor Params: %d' % sum(
                p.numel() for p in self._actor.parameters() if p.requires_grad))
        else:
            for p in self._actor.parameters():
                p.requires_grad = False

        self._device = device
    
    def _norm_rgb_(self, x):
        return (x.float() / 255.0) * 2.0 - 1.0

    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()

    def update(self, step: int, replay_sample: dict) -> dict:
        lang_goal_embs = replay_sample['lang_goal_embs'].float() if replay_sample['lang_goal_embs'] is not None else None
        robot_state = replay_sample['low_dim_state'].float()

        rgb = replay_sample['%s_rgb' % self._camera_name].float() / 255 * 2 - 1
        pcd = replay_sample['%s_point_cloud' % self._camera_name].float()
        if torch.isnan(pcd).any():
            bound = replay_sample['bound'].float()
            pcd = torch.nan_to_num(pcd)
            pcd = torch.max(pcd, bound[0].reshape(1, 3, 1, 1))
            pcd = torch.min(pcd, bound[1].reshape(1, 3, 1, 1))
        
        observations = [rgb, pcd]

        mu = self._actor(observations, robot_state, lang_goal_embs)

        loss_pos = F.huber_loss(mu[:, :3], replay_sample['action'][:, :3].float(), reduction='none').sum(1)
        loss_rot = F.huber_loss(mu[:, 3:7], replay_sample['action'][:, 3:7].float(), reduction='none').sum(1)

        # 100:10:0.5, here we do not supervise gripper value
        loss = loss_pos * 10 + loss_rot
        loss = loss.mean()
        
        self._grad_step(loss, self._actor_optimizer,
                        self._actor.parameters(), self._grad_clip)
        self._summaries = {
            'pi/loss': loss,
            'pi/mu': mu.mean(),
        }
        return {'total_loss': loss}

    def _normalize_quat(self, x):
        return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)

    @torch.no_grad()
    def act(self, step: int, replay_sample: dict):
        lang_goal_embs = replay_sample['lang_goal_embs'].float() if replay_sample['lang_goal_embs'] is not None else None
        robot_state = replay_sample['low_dim_state'].float().to(self._device)

        rgb = replay_sample['%s_rgb' % self._camera_name].float().to(self._device) / 255 * 2 - 1
        pcd = replay_sample['%s_point_cloud' % self._camera_name].float().to(self._device)
        if torch.isnan(pcd).any():
            bound = replay_sample['bound']
            pcd = torch.nan_to_num(pcd)
            pcd = torch.max(pcd, bound[0].reshape(1, 3, 1, 1))
            pcd = torch.min(pcd, bound[1].reshape(1, 3, 1, 1))
        
        observations = [rgb, pcd]

        mu = self._actor(observations, robot_state, lang_goal_embs)
        return mu[:, :3].squeeze().cpu().numpy(), self._normalize_quat(mu[:, 3:7]).squeeze().cpu().numpy()

    def update_summaries(self):
        summaries = []
        for n, v in self._summaries.items():
            summaries.append({'%s/%s' % (NAME, n): v})

        for tag, param in self._actor.named_parameters():
            summaries.append({'%s/gradient/%s' % (NAME, tag): param.grad})
            summaries.append({'%s/weight/%s' % (NAME, tag): param.data})

        return summaries

    def act_summaries(self):
        return []

    def load_weights(self, savepath: str):
        loaded = torch.load(savepath, map_location=torch.device('cpu'))
        self._actor.load_state_dict(loaded['state_dict'])
        iteration = loaded['iteration']
        logger.info(f'Loaded weights from {savepath} at iteration {iteration}')
        return iteration

    def save_weights(self, savepath: str, iteration: int):
        torch.save({
            'state_dict': self._actor.state_dict(),
            'iteration': iteration,
        }, savepath)
