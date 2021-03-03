"""Load dataset utils."""

import os

import tensorflow as tf

_TFDS_PREFIX = 'tfds://'

def load_tfds_dataset(ds_type, split) -> tf.data.Dataset:
    import tensorflow_datasets as tfds
    return tfds.load(ds_type, split=split, data_dir=os.environ.get('TFDS_DATA_DIR', None))


def load_dataset(ds_type: str, split: str) -> tf.data.Dataset:
    if ds_type.startswith(_TFDS_PREFIX):
        tfds_type = ds_type[len(_TFDS_PREFIX):]
        return load_tfds_dataset(tfds_type, split)

    raise ValueError(f'Dataset {ds_type} not handled.')


def is_tfds(ds_type: str, tfds_name: str=None) -> bool:
    if not ds_type.startswith(_TFDS_PREFIX):
        return False
    ds_type = ds_type[len(_TFDS_PREFIX):]
    return ds_type == tfds_name or ds_type.startswith(tfds_name + '/')