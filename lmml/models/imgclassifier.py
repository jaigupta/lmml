"""A generic image classifier model."""

from lmml.models.darknet import (
    Darknet,
    DarknetTiny
)
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    AveragePooling2D
)

def get_backbone_model(backbone: str):
    if backbone == 'darknet':
        return Darknet(name=backbone)
    if backbone == 'darknet_tiny':
        return DarknetTiny(name=backbone)

    raise ValueError(f'Unknown backbone: {backbone}')


def image_classifier(num_classes, backbone_model):
    x = inputs = Input([None, None, 3])
    x = features = backbone_model(x)[-1]
    x = AveragePooling2D(pool_size=(8,8))(x)
    x = tf.squeeze(x, axis=[1, 2])
    x = logits = Dense(num_classes)(x)
    return tf.keras.Model(inputs, logits, name='classifier')

