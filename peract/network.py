# Adapted from https://github.com/stepjam/ARM/blob/main/arm/network_utils.py

import copy
from functools import wraps
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

LRELU_SLOPE = 0.02


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def act_layer(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == 'elu':
        return nn.ELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError('%s not recognized.' % act)


def norm_layer2d(norm, channels):
    if norm == 'batch':
        return nn.BatchNorm2d(channels)
    elif norm == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == 'layer':
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == 'group':
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError('%s not recognized.' % norm)


def norm_layer1d(norm, num_channels):
    if norm == 'batch':
        return nn.BatchNorm1d(num_channels)
    elif norm == 'instance':
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == 'layer':
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError('%s not recognized.' % norm)


class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)

        x = gamma * x + beta

        return x


class Conv2DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, strides,
                 norm=None, activation=None, padding_mode='replicate'):
        super(Conv2DBlock, self).__init__()
        padding = kernel_sizes // 2 if isinstance(kernel_sizes, int) else (
            kernel_sizes[0] // 2, kernel_sizes[1] // 2)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv2d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv2d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv2d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv2d.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv2d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer2d(norm, out_channels)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv2DFiLMBlock(Conv2DBlock):

    def __init__(self, in_channels, out_channels, kernel_sizes, strides,
                 norm=None, activation=None, padding_mode='replicate'):
        super(Conv2DFiLMBlock, self).__init__(
            in_channels, out_channels, kernel_sizes, strides, norm, activation,
            padding_mode)

        self.film = FiLMBlock()

    def forward(self, x, gamma, beta):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.film(x, gamma, beta)
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_sizes: Union[int, list]=3, strides=1,
                 norm=None, activation=None, padding_mode='replicate',
                 padding=None):
        super(Conv3DBlock, self).__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv3d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.conv3d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv3d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            raise NotImplementedError('Norm not implemented.')
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class ConvTranspose3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_sizes: Union[int, list], strides,
                 norm=None, activation=None, padding_mode='zeros',
                 padding=None):
        super(ConvTranspose3DBlock, self).__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv3d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.conv3d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv3d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer3d(norm, out_channels)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv2DUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, strides,
                 norm=None, activation=None):
        super(Conv2DUpsampleBlock, self).__init__()
        layer = [Conv2DBlock(
            in_channels, out_channels, kernel_sizes, 1, norm, activation)]
        if strides > 1:
            layer.append(nn.Upsample(
                scale_factor=strides, mode='bilinear',
                align_corners=False))
        convt_block = Conv2DBlock(
            out_channels, out_channels, kernel_sizes, 1, norm, activation)
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class Conv3DUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, strides, kernel_sizes=3,
                 norm=None, activation=None):
        super(Conv3DUpsampleBlock, self).__init__()
        layer = [Conv3DBlock(
            in_channels, out_channels, kernel_sizes, 1, norm, activation)]
        if strides > 1:
            layer.append(nn.Upsample(
                scale_factor=strides, mode='trilinear',
                align_corners=False))
        convt_block = Conv3DBlock(
            out_channels, out_channels, kernel_sizes, 1, norm, activation)
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class DenseBlock(nn.Module):

    def __init__(self, in_features, out_features, norm=None, activation=None):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        if activation is None:
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.linear.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.linear.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.linear.weight, a=LRELU_SLOPE, nonlinearity='leaky_relu')
            nn.init.zeros_(self.linear.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
            nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer1d(norm, out_features)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class SiameseNet(nn.Module):

    def __init__(self,
                 input_channels: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 norm: str = None,
                 activation: str = 'relu'):
        super(SiameseNet, self).__init__()
        self._input_channels = input_channels
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self.output_channels = filters[-1] #* len(input_channels)

    def build(self):
        self._siamese_blocks = nn.ModuleList()
        for i, ch in enumerate(self._input_channels):
            blocks = []
            for i, (filt, ksize, stride) in enumerate(
                    zip(self._filters, self._kernel_sizes, self._strides)):
                conv_block = Conv2DBlock(
                    ch, filt, ksize, stride, self._norm, self._activation)
                blocks.append(conv_block)
            self._siamese_blocks.append(nn.Sequential(*blocks))
        self._fuse = Conv2DBlock(self._filters[-1] * len(self._siamese_blocks),
                                 self._filters[-1], 1, 1, self._norm,
                                 self._activation)

    def forward(self, x):
        if len(x) != len(self._siamese_blocks):
            raise ValueError('Expected a list of tensors of size %d.' % len(
                self._siamese_blocks))
        self.streams = [stream(y) for y, stream in zip(x, self._siamese_blocks)]
        y = self._fuse(torch.cat(self.streams, 1))
        return y


class CNNAndFcsNet(nn.Module):

    def __init__(self,
                 siamese_net: SiameseNet,
                 low_dim_state_len: int,
                 input_resolution: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 norm: str = None,
                 fc_layers: List[int] = None,
                 activation: str = 'relu'):
        super(CNNAndFcsNet, self).__init__()
        self._siamese_net = copy.deepcopy(siamese_net)
        self._input_channels = self._siamese_net.output_channels + low_dim_state_len
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._fc_layers = [] if fc_layers is None else fc_layers
        self._input_resolution = input_resolution

    def build(self):
        self._siamese_net.build()
        layers = []
        channels = self._input_channels
        for i, (filt, ksize, stride) in enumerate(
                list(zip(self._filters, self._kernel_sizes, self._strides))[
                :-1]):
            layers.append(Conv2DBlock(
                channels, filt, ksize, stride, self._norm, self._activation))
            channels = filt
        layers.append(Conv2DBlock(
            channels, self._filters[-1], self._kernel_sizes[-1],
            self._strides[-1]))
        self._cnn = nn.Sequential(*layers)
        self._maxp = nn.AdaptiveMaxPool2d(1)

        channels = self._filters[-1]
        dense_layers = []
        for n in self._fc_layers[:-1]:
            dense_layers.append(
                DenseBlock(channels, n, activation=self._activation))
            channels = n
        dense_layers.append(
            DenseBlock(channels, self._fc_layers[-1]))
        self._fcs = nn.Sequential(*dense_layers)

    def forward(self, observations, low_dim_ins):
        x = self._siamese_net(observations)
        _, _, h, w = x.shape
        low_dim_latents = low_dim_ins.unsqueeze(
            -1).unsqueeze(-1).repeat(1, 1, h, w)
        combined = torch.cat([x, low_dim_latents], dim=1)
        x = self._cnn(combined)
        x = self._maxp(x).squeeze(-1).squeeze(-1)
        return self._fcs(x)


class CNNLangAndFcsNet(nn.Module):

    def __init__(self,
                 siamese_net: SiameseNet,
                 low_dim_state_len: int,
                 input_resolution: List[int],
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 norm: str = None,
                 fc_layers: List[int] = None,
                 activation: str = 'relu'):
        super(CNNLangAndFcsNet, self).__init__()
        self._siamese_net = copy.deepcopy(siamese_net)
        self._input_channels = self._siamese_net.output_channels + low_dim_state_len
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._fc_layers = [] if fc_layers is None else fc_layers
        self._input_resolution = input_resolution

        self._lang_feat_dim = 1024

    def build(self):
        self._siamese_net.build()
        layers = []
        channels = self._input_channels

        self.conv1 = Conv2DFiLMBlock(
            channels, self._filters[0], self._kernel_sizes[0],
            self._strides[0])
        self.gamma1 = nn.Linear(self._lang_feat_dim, self._filters[0])
        self.beta1 = nn.Linear(self._lang_feat_dim, self._filters[0])

        self.conv2 = Conv2DFiLMBlock(
            self._filters[0], self._filters[1], self._kernel_sizes[1],
            self._strides[1])
        self.gamma2 = nn.Linear(self._lang_feat_dim, self._filters[1])
        self.beta2 = nn.Linear(self._lang_feat_dim, self._filters[1])

        self.conv3 = Conv2DFiLMBlock(
            self._filters[1], self._filters[2], self._kernel_sizes[2],
            self._strides[2])
        self.gamma3 = nn.Linear(self._lang_feat_dim, self._filters[2])
        self.beta3 = nn.Linear(self._lang_feat_dim, self._filters[2])

        self._maxp = nn.AdaptiveMaxPool2d(1)

        channels = self._filters[-1]
        dense_layers = []
        for n in self._fc_layers[:-1]:
            dense_layers.append(
                DenseBlock(channels, n, activation=self._activation))
            channels = n
        dense_layers.append(
            DenseBlock(channels, self._fc_layers[-1]))
        self._fcs = nn.Sequential(*dense_layers)

    def forward(self, observations, low_dim_ins, lang_goal_feats):
        x = self._siamese_net(observations)
        _, _, h, w = x.shape
        low_dim_latents = low_dim_ins.unsqueeze(
            -1).unsqueeze(-1).repeat(1, 1, h, w)
        combined = torch.cat([x, low_dim_latents], dim=1)

        g1 = self.gamma1(lang_goal_feats)
        b1 = self.beta1(lang_goal_feats)
        x = self.conv1(combined, g1, b1)

        g2 = self.gamma2(lang_goal_feats)
        b2 = self.beta2(lang_goal_feats)
        x = self.conv2(x, g2, b2)

        g3 = self.gamma3(lang_goal_feats)
        b3 = self.beta3(lang_goal_feats)
        x = self.conv3(x, g3, b3)

        x = self._maxp(x).squeeze(-1).squeeze(-1)
        return self._fcs(x)



class Conv3DInceptionBlockUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 norm=None, activation=None, residual=False):
        super(Conv3DInceptionBlockUpsampleBlock, self).__init__()
        layer = []

        convt_block = Conv3DInceptionBlock(
            in_channels, out_channels, norm, activation)
        layer.append(convt_block)

        if scale_factor > 1:
            layer.append(nn.Upsample(
                scale_factor=scale_factor, mode='trilinear',
                align_corners=False))

        convt_block = Conv3DInceptionBlock(
            out_channels, out_channels, norm, activation)
        layer.append(convt_block)

        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class Conv3DInceptionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm=None, activation=None,
                 residual=False):
        super(Conv3DInceptionBlock, self).__init__()
        self._residual = residual
        cs = out_channels // 4
        assert out_channels % 4 == 0
        latent = 32
        self._1x1conv = Conv3DBlock(
            in_channels, cs * 2, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)

        self._1x1conv_a = Conv3DBlock(
            in_channels, latent, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)
        self._3x3conv = Conv3DBlock(
            latent, cs, kernel_sizes=3, strides=1,
            norm=norm, activation=activation)

        self._1x1conv_b = Conv3DBlock(
            in_channels, latent, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)
        self._5x5_via_3x3conv_a = Conv3DBlock(
            latent, latent, kernel_sizes=3, strides=1, norm=norm,
            activation=activation)
        self._5x5_via_3x3conv_b = Conv3DBlock(
            latent, cs, kernel_sizes=3, strides=1, norm=norm,
            activation=activation)
        self.out_channels = out_channels + (in_channels if residual else 0)

    def forward(self, x):
        yy = []
        if self._residual:
            yy = [x]
        return torch.cat(yy + [self._1x1conv(x),
                               self._3x3conv(self._1x1conv_a(x)),
                               self._5x5_via_3x3conv_b(self._5x5_via_3x3conv_a(
                                   self._1x1conv_b(x)))], 1)

class ConvTransposeUp3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, strides=2, padding=0,
                 norm=None, activation=None, residual=False):
        super(ConvTransposeUp3DBlock, self).__init__()
        self._residual = residual

        self._1x1conv = Conv3DBlock(
            in_channels, out_channels, kernel_sizes=1, strides=1, norm=norm,
            activation=activation)
        self._3x3conv = ConvTranspose3DBlock(
            out_channels, out_channels, kernel_sizes=2, strides=strides, norm=norm,
            activation=activation, padding=padding)
        self._1x1conv_a = Conv3DBlock(
            out_channels, out_channels, kernel_sizes=1, strides=1, norm=norm,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self._1x1conv(x)
        x = self._3x3conv(x)
        x = self._1x1conv_a(x)
        return x


class SpatialSoftmax3D(torch.nn.Module):

    def __init__(self, depth, height, width, channel):
        super(SpatialSoftmax3D, self).__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.channel = channel
        self.temperature = 0.01
        pos_x, pos_y, pos_z = np.meshgrid(
            np.linspace(-1., 1., self.depth),
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        pos_x = torch.from_numpy(
            pos_x.reshape(self.depth * self.height * self.width)).float()
        pos_y = torch.from_numpy(
            pos_y.reshape(self.depth * self.height * self.width)).float()
        pos_z = torch.from_numpy(
            pos_z.reshape(self.depth * self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)

    def forward(self, feature):
        feature = feature.view(
            -1, self.height * self.width * self.depth)  # (B, c*d*h*w)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1,
                               keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1,
                               keepdim=True)
        expected_z = torch.sum(self.pos_z * softmax_attention, dim=1,
                               keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y, expected_z], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 3)
        return feature_keypoints


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module): # is all you need. Living up to its name. 
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
