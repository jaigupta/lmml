import os

import gin
import tensorflow as tf
from lmmlscripts.core import dataset

@gin.configurable
def mapper(example, image_size, img_resize_fn=gin.REQUIRED):
    image = example['image']
    if img_resize_fn == 'resize':
        image = tf.image.resize(image, (image_size, image_size))
    elif img_resize_fn == 'resize_with_pad':
        image = tf.image.resize_with_pad(image, image_size, image_size)
    else:
        raise ValueError('Unknown img_resize_fn')
    image = image/255.
    label = example['label']
    return image, label


def load_dataset(ds_type: str, split: str, batch_size: int, image_size: int) -> tf.data.Dataset:
    ds = dataset.load_dataset(ds_type, split)
    return ds.repeat().map(
        lambda example: mapper(example, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(8)