import os

from absl import app
from absl import flags
from absl import logging
import attr
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

from lmml.core.oop import overrides
from lmml.models.darknet import Darknet
from lmml.models.imgclassifier import image_classifier, get_backbone_model
from lmmlscripts.core import files
from lmmlscripts.core.trainer import BaseTrainer, BaseTrainerConfig
from lmmlscripts.darknet import dataset

flags.DEFINE_multi_string('gin_param', None, 'Repeated Gin parameter bindings.')
flags.DEFINE_string('model_config', 'base', 'Repeated Gin parameter bindings.')


@gin.configurable
class Trainer(BaseTrainer):
    def __init__(
            self,
            image_size=gin.REQUIRED, dataset=gin.REQUIRED, num_classes=gin.REQUIRED,
            backbone=gin.REQUIRED, output_dir=gin.REQUIRED):
        super().__init__(output_dir)
        self.image_size = image_size
        self.dataset = dataset
        self.num_classes = num_classes
        self.backbone = backbone
        logging.info('Using image size: %s', self.image_size)
        for d in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(d, True)

    @overrides(BaseTrainer)
    def build(self):
        self.ds_train = dataset.load_dataset(
            self.dataset, 'train', self.config.train_batch_size, self.image_size)
        self.ds_val = dataset.load_dataset(
            self.dataset, 'validation', self.config.val_batch_size, self.image_size)

        with self.mirrored_strategy.scope():
            self.backbone_model = get_backbone_model(self.backbone)
            self.model = image_classifier(self.num_classes, self.backbone_model)
            self.optimizer = keras.optimizers.Adam(lr=self.config.learning_rate)
            self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

            self.avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
            self.avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
            self.accuracy = tf.keras.metrics.Accuracy()
            self.val_accuracy = tf.keras.metrics.Accuracy()

        self.checkpointer = tf.train.Checkpoint(
            model=self.model, backbone_model=self.backbone_model,
            optimizer=self.optimizer,
            total_steps=self.total_steps, epoch=self.epoch)

    @overrides(BaseTrainer)
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            output_logits = self.model(images, training=True)
            regularization_loss = tf.reduce_sum(self.model.losses)
            pred_loss = self.loss_fn(labels, output_logits)
            total_loss = pred_loss + regularization_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        self.avg_loss.update_state(total_loss)
        predicted_class = tf.argmax(output_logits, axis=-1)
        self.accuracy.update_state(
            tf.expand_dims(predicted_class, axis=-1), tf.expand_dims(labels, axis=-1))

    @overrides(BaseTrainer)
    def train_step_end(self):
        tf.summary.scalar('accuracy', self.accuracy.result(), self.total_steps)
        tf.summary.scalar('avg_loss', self.avg_loss.result(), self.total_steps)

    @overrides(BaseTrainer)
    def train_epoch_end(self):
        self.avg_loss.reset_states()
        self.accuracy.reset_states()
        tf.saved_model.save(self.backbone_model, os.path.join(
            self.output_dir, 'saved_model/backbone'))
        tf.saved_model.save(self.model, os.path.join(
            self.output_dir, 'saved_model/classifier'))

    @overrides(BaseTrainer)
    @tf.function
    def eval_step(self, images, labels):
        output_logits = self.model(images)
        regularization_loss = tf.reduce_sum(self.model.losses)
        pred_loss = tf.reduce_mean(self.loss_fn(labels, output_logits)) / len(self.devices)
        total_loss = pred_loss + regularization_loss
        self.avg_val_loss.update_state(total_loss)
        predicted_class = tf.argmax(output_logits, axis=-1)
        self.val_accuracy.update_state(
            tf.expand_dims(predicted_class, axis=-1), tf.expand_dims(labels, axis=-1))
        return total_loss, pred_loss

    @overrides(BaseTrainer)
    def eval_end(self):
        tf.summary.scalar('avg_loss', self.avg_val_loss.result(), self.total_steps)
        tf.summary.scalar('accuracy', self.val_accuracy.result(), self.total_steps)

        self.avg_val_loss.reset_states()
        self.val_accuracy.reset_states()


def main(_argv):
    gin.parse_config_files_and_bindings(
        [f'lmmlscripts/gin/darknet/{flags.FLAGS.model_config}.gin'],
        flags.FLAGS.gin_param)
    t = Trainer()
    t.start()


if __name__ == '__main__':
    app.run(main)
