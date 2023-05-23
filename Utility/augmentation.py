"""データ拡張
"""
import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)
from Utility.type_hint import *
from Utility.log_conf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

import math
import random

import torch
from torch import nn
import torch.nn.functional as F


class SegmentationAugmentation(nn.Module):

    def __init__(self, 
                 flip : Optional[float] = None, 
                 offset : Optional[float] = None,
                 scale : Optional[float] = None, 
                 rotate : Optional[float] = None, 
                 noise : Optional[float] = None):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise


    def forward(self, input_g, label_g):
        transform_t = self._build2TransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(
            transform_t[:, :2], input_g.size(), align_corners=False
        )

        # input : 4D (N,C,Hin,Win) or 5D (N,C,Din,Hin,Win)
        # grid : (N,Hout,Wout,2) or (N,Dout,Hout,Wout,3)
        # output: (N,C,Hout,Wout) or (N,C,Dout,Hout,Wout)
        augmented_input_g = F.grid_sample( # input
            input=input_g, 
            grid=affine_t, 
            padding_mode="border", 
            align_corners=False
        )
        augmented_label_g = F.grid_sample( # label
            input=label_g.to(torch.float32),
            grid=affine_t,
            padding_mode="border",
            align_corners=False
        )

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

        return augmented_input_g, augmented_label_g > 0.5
    

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i,i] *= -1
            
            if self.offset:
                offset_float = self.offset
                random_float = random.random() * 2 - 1 # [-1,1]
                transform_t[2,i] = offset_float * random_float # [0, offset_float]

            if self.scale:
                scale_float = self.scale
                random_float = random.random() * 2 - 1 # [-1,1]
                transform_t[i,i] *= 1.0 + scale_float * random_float # [1.0-scale_float,1.0+scale_float]
            
        if self.rotate:
            angle_rad = random.random() * math.pi * 2 # [0,2pi]
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])
            
            transform_t @= rotation_t
        
        return transform_t
