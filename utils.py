"""ユーティリティ
"""
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
# print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

from .log_conf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

import datetime
import time


def test_name_deco(f):
    def _wrapper(*args, **kargs):

        print(f"{datetime.datetime.now()} [Start] {f.__name__}")

        v = f(*args, **kargs)

        print(f"{datetime.datetime.now()} [End] {f.__name__}")

        return v
    
    return _wrapper
