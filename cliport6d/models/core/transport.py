import numpy as np
import cliport6d.models as models
import cliport6d.utils.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from cliport6d.utils.utils import mlp

class Transport(nn.Module):

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        """Transport (a.k.a Place) module."""
        super().__init__()

        self.iters = 0
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        in_shape = np.array(in_shape)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        # Crop before network (default from Transporters CoRL 2020).
        self.kernel_shape = (self.crop_size, self.crop_size, self.in_shape[2])

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        model = models.names[stream_one_fcn]
        self.key_resnet = model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_resnet = model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        print(f"Transport FCN: {stream_one_fcn}")

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:])
        return output

    def transport(self, in_tensor, crop):
        logits = self.key_resnet(in_tensor)
        kernel = self.query_resnet(crop)
        return logits, kernel

    def forward(self, inp_img, p, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape) # [B W H D]
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default from Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2) # [B D W H]

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        logits, kernel = self.transport(in_tensor, crop)

        return self.correlate(logits, kernel, softmax)

class Transport6Dof(Transport):
    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device, z_roll_pitch=False, joint_all=False):
        self.output_dim = 3
        self.kernel_dim = 3
        if z_roll_pitch:
            if joint_all:
                self.output_dim = 16
                self.kernel_dim = 16
                self.z_start = 4
            else:
                self.output_dim = 12
                self.kernel_dim = 12
                self.z_start = 0
        self.z_roll_pitch = z_roll_pitch
        self.joint_all = joint_all
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)
        if z_roll_pitch:
            self.z_regressor = mlp(input_dim=1, hidden_dim=32, output_dim=1, hidden_depth=3, output_mod=None, device=device)
            self.roll_regressor = mlp(input_dim=1, hidden_dim=32, output_dim=n_rotations, hidden_depth=3, output_mod=None, device=device)
            self.pitch_regressor = mlp(input_dim=1, hidden_dim=32, output_dim=n_rotations, hidden_depth=3, output_mod=None, device=device)
    
    def forward(self, inp_img, p, softmax=True):
        """Forward pass."""
        padding = np.zeros((4,2),dtype=int)
        padding[1:,:] = self.padding
        img_unprocessed = np.pad(inp_img, padding, mode='constant')
        input_data = img_unprocessed
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = p + self.pad_size

        # Crop before network (default from Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2).contiguous()
        b, c, w, h = in_tensor.shape
        crop = in_tensor.unsqueeze(1).repeat(1, self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=np.flip(pv,axis=1).copy(), reverse=True)
        crop = torch.stack(crop, dim=1)
        crop = [crop[i, :, :, pv[i,0]-hcrop:pv[i,0]+hcrop, pv[i,1]-hcrop:pv[i,1]+hcrop] for i in range(crop.shape[0])]
        crop = torch.cat(crop, dim=0)

        logits, kernels = self.transport(in_tensor, crop)
        kernels = kernels.reshape(torch.Size([-1, self.n_rotations])+kernels.shape[1:])

        return self.correlate(logits, kernels, softmax)
    
    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        assert in0.shape[0] == in1.shape[0]
        outputs, z_tensors, roll_tensors, pitch_tensors = [],[],[],[]
        if not self.z_roll_pitch:
            for i in range(in0.shape[0]):
                output = F.conv2d(in0[i:i+1], in1[i], padding=(self.pad_size, self.pad_size))
                output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
                # output = in0[i:i+1]
                output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
            if softmax:
                outputs_shape = outputs.shape
                outputs = outputs.reshape((1, np.prod(outputs.shape)))
                outputs = F.softmax(outputs, dim=-1)
                outputs = outputs.reshape(outputs_shape[1:])
        else:
            dim = self.kernel_dim//3
            for i in range(in0.shape[0]):

                z_tensor = F.conv2d(in0[i:i+1, :dim], in1[i, :, :dim], padding=(self.pad_size, self.pad_size))
                z_tensor = F.interpolate(z_tensor, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
                z_tensor = z_tensor[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
                z_tensors.append(z_tensor)

                roll_tensor = F.conv2d(in0[i:i+1, dim:2*dim], in1[i, :, dim:2*dim], padding=(self.pad_size, self.pad_size))
                roll_tensor = F.interpolate(roll_tensor, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
                roll_tensor = roll_tensor[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
                roll_tensors.append(roll_tensor)

                pitch_tensor = F.conv2d(in0[i:i+1, 2*dim:3*dim], in1[i, :, 2*dim:3*dim], padding=(self.pad_size, self.pad_size))
                pitch_tensor = F.interpolate(pitch_tensor, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
                pitch_tensor = pitch_tensor[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
                pitch_tensors.append(pitch_tensor)
            z_tensors = torch.cat(z_tensors, dim=0)
            roll_tensors = torch.cat(roll_tensors, dim=0)
            pitch_tensors = torch.cat(pitch_tensors, dim=0)
            if self.joint_all:
                for i in range(in0.shape[0]):
                    output = F.conv2d(in0[i:i+1, 3*dim:], in1[i, :, 3*dim:], padding=(self.pad_size, self.pad_size))
                    output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
                    # output = in0[i:i+1]
                    output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
                    outputs.append(output)
                outputs = torch.cat(outputs, dim=0)
        return outputs, z_tensors, roll_tensors, pitch_tensors