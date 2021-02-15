from typing import List

import tensorflow as tf

def ensure_dir_exists(path: str):
    if not tf.io.gfile.exists(path):
        tf.io.gfile.mkdir(path)

def ensure_dirs_exist(paths: List[str]):
    for path in paths:
        ensure_dir_exists(path)