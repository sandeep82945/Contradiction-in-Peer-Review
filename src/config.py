# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.


import os
from pathlib import Path

'''
The code provided defines SRC_ROOT as the path to the directory containing the current script file. Let's break it down step by step:

os.path.realpath(__file__): This retrieves the absolute path of the current script file (__file__ refers to the script's file name). os.path.realpath resolves any symbolic links and returns the canonical path.

os.path.dirname: This extracts the directory name from the absolute path obtained in the previous step, giving you the parent directory of the script file.

Path: This is a class from the pathlib module used to handle file system paths. It is used here to create a Path object.

Putting it all together, the code assigns SRC_ROOT as a Path object representing the directory containing the current script file.
'''

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PRO_ROOT = SRC_ROOT.parent

if __name__ == '__main__':
    pass