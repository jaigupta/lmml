import tensorflow as tf

IMAGENET_DS_TYPES = [
    'imagenet2012',
    'imagenet_resized',
    'imagenette',
    'imagewang',
]

def mapper(example, image_size):
    image = example['image']
    image = tf.image.resize(image, (image_size, image_size))
    image = image/255.
    label = example['label']
    return (image, label)


def load_imagenet_ds(ds_type, split, batch_size, image_size):
    import tensorflow_datasets as tfds
    ds = tfds.load(ds_type, split=split)
    return ds.map(lambda example: mapper(example, image_size)).batch(batch_size)


def load_dataset(ds_type, split, batch_size, image_size):
    if '/' in ds_type and ds_type.split('/')[0] in IMAGENET_DS_TYPES:
        return load_imagenet_ds(ds_type, split, batch_size, image_size)

    raise ValueError(f'Dataset {ds_type} not handled.')