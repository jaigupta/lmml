import tensorflow as tf

def ensure_dir_exists(path):
    if not tf.io.gfile.exists(path):
        tf.io.gfile.mkdir(path)