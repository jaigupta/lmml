"""Load dataset utils."""

import os

import tensorflow as tf

def load_tfds_dataset(ds_type, split) -> tf.data.Dataset:
    import tensorflow_datasets as tfds
    return tfds.load(ds_type, split=split, data_dir=os.environ.get('TFDS_DATA_DIR', None))


def load_dataset(ds_type: str, split: str) -> tf.data.Dataset:
    if ds_type.startswith('tfds://'):
        tfds_type = ds_type[len('tfds://'):]
        return load_tfds_dataset(tfds_type, split)

    raise ValueError(f'Dataset {ds_type} not handled.')
