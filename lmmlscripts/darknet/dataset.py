import os

import tensorflow as tf
from lmmlscripts.core import dataset

def mapper(example, image_size):
    image = example['image']
    image = tf.image.resize(image, (image_size, image_size))
    image = image/255.
    label = example['label']
    return image, label


def load_dataset(ds_type: str, split: str, batch_size: int, image_size: int) -> tf.data.Dataset:
    ds = dataset.load_dataset(ds_type, split)
    return ds.repeat().map(lambda example: mapper(example, image_size), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(8)