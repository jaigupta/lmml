import os

from absl import app
from absl import flags
from absl import logging
from dataclasses import dataclass, field
import gin
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

from lmml.models.darknet import Darknet
from lmml.models.imgclassifier import image_classifier, get_backbone_model
from lmmlscripts.core import files
from lmmlscripts.core.trainer import BaseTrainer, BaseTrainerConfig
from lmmlscripts.darknet import dataset


@gin.configurable
class Trainer(BaseTrainer):
    def __init__(self, config, image_size, dataset, num_classes, backbone, output_dir):
        super().__init__(config)
        self.image_size = image_size
        self.dataset = dataset
        self.num_classes = num_classes
        self.backbone = backbone
        self.output_dir = output_dir

    def build(self):
        logging.info('Using image size: %s', self.image_size)
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)

        self.backbone_model = get_backbone_model(self.backbone)
        self.model = image_classifier(self.num_classes, self.backbone_model)
        self.checkpointer = tf.train.Checkpoint(model=self.model, backbone_model=self.backbone_model, total_steps=self.total_steps, epoch=self.epoch)

        self.ds_train = dataset.load_dataset(self.dataset, 'train', self.config.train_batch_size, self.image_size)
        self.ds_val = dataset.load_dataset(self.dataset, 'validation', self.config.val_batch_size, self.image_size)

        self.optimizer = keras.optimizers.Adam(lr=self.config.learning_rate)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    @tf.function
    def train_step(self, batch):
        images, labels = batch
        with tf.GradientTape() as tape:
            output_logits = self.model(images, training=True)
            regularization_loss = tf.reduce_sum(self.model.losses)
            pred_loss = self.loss_fn(labels, output_logits)
            total_loss = pred_loss + regularization_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.avg_loss.update_state(total_loss)

        return total_loss, pred_loss

    def train_step_end(self, result):
        total_loss, pred_loss = result
        tf.summary.scalar('avg_loss', self.avg_loss.result(), self.total_steps)
        tf.summary.scalar('total_loss', total_loss, self.total_steps)
        tf.summary.scalar('pred_loss', pred_loss, self.total_steps)

    @tf.function
    def eval_step(self, batch):
        images, labels = batch
        output_logits = self.model(images)
        regularization_loss = tf.reduce_sum(self.model.losses)
        pred_loss = self.loss_fn(labels, output_logits)
        total_loss = pred_loss + regularization_loss
        self.avg_val_loss.update_state(total_loss)
        return total_loss, pred_loss

    def eval_end(self):
        logging.info("{}, train: {}, val: {}".format(
            self.epoch.numpy(),
            self.avg_loss.result().numpy(),
            self.avg_val_loss.result().numpy()))
        tf.summary.scalar('avg_loss', self.avg_val_loss.result(), self.total_steps)

        self.avg_loss.reset_states()
        self.avg_val_loss.reset_states()

        tf.saved_model.save(self.backbone_model, os.path.join(self.output_dir, 'saved_model/backbone'))
        tf.saved_model.save(self.model, os.path.join(self.output_dir, 'saved_model/classifier'))

def main(_argv):
    gin.parse_config_file('lmmlscripts/gin/darknet.gin')
    config = BaseTrainerConfig()
    t = Trainer(config)
    t.start()

if __name__ == '__main__':
    app.run(main)