"""Lunaデータセットのキャッシング
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

import argparse

from Utility.enumerate_with_estimate import enumerate_with_estimate
from DatasetLuna.dataset_luna import LunaDataset

import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader


class LunaPrepCacheApp:

    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=1024,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        # データセットディレクトリの指定
        parser.add_argument('--datasetdir',
            help="Luna dataset directory",
            default='',
            type=str,
        )

        self.cli_args = parser.parse_args(sys_argv)


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        self.prep_dl = DataLoader(
            LunaDataset(
                datasetdir=self.cli_args.datasetdir,
                sortby_str='series_uid',
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iter = enumerate_with_estimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for _ in batch_iter:
            pass


if __name__ == '__main__':
    LunaPrepCacheApp().main()
