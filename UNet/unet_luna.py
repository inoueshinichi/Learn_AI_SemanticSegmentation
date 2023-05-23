"""UNetモデルのラッパー
"""
import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)
from Utility.type_hint import *
from Utility.log_conf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ThirdParty.unet import Unet


class UNetWrapper(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        # BatchNorm2d は入力のチャンネル数を必要とする
        # その情報をキーワード引数から取り出す
        self.input_batchnorm = nn.BatchNorm2d(kwargs["in_chanenls"])
        # U-Netの取り込み部分はこれだけだが、ほとんどの処理はここで行われる
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        # 第11章と同じように独自の重み初期化を行う
        self._init_weights()

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.Convtranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu", a=0)
            
            if m.bias is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                nn.init.normal_(m.bias, -bound, bound)

        # nn.init.constant_(self.unet.last.bias, -4)
        # nn.init.constant_(self.unet.last.bias, 4)

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output





