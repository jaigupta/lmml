from typing import List

import tensorflow as tf

gfile = tf.io.gfile


def ensure_dir_exists(path: str) -> None:
    if not gfile.exists(path):
        gfile.mkdir(path)


def ensure_dirs_exist(paths: List[str]) -> None:
    for path in paths:
        ensure_dir_exists(path)


def clear_dir(path: List[str]) -> None:
    if gfile.exists(path):
        gfile.rmtree(path)
    gfile.mkdir(path)